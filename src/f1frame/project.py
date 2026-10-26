"""Project position samples onto a reference line.

The public timing feed snaps cars to one path, and the start/finish samples of
a closed lap sit next to both s = 0 and s = length. A global nearest-segment
search follows the chord across a hairpin or snaps the first sample to the end
of the lap. Projection therefore walks forward: each sample may only land in a
window ahead of the previous one.
"""

from __future__ import annotations

import numpy as np

from f1frame.types import F1FrameError, ReferenceLine


def project_s(
    x: np.ndarray,
    y: np.ndarray,
    reference: ReferenceLine,
    *,
    lookahead_m: float = 250.0,
    lookback_m: float = 20.0,
    max_dist_m: float = 40.0,
    max_advance_m: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(s_m, distance_m)`` for each point.

    ``distance_m`` is the distance from the raw point to the segment it was
    matched to. Samples farther than ``max_dist_m`` from the window, or with
    no segment in the window, are NaN in both arrays. The previous accepted
    s is kept, so one rejected sample does not stop the rest of the lap.

    When a sample is closer to a later part of the circuit than to the
    continuation of the current one, the match within ``max_advance_m`` of the
    previous s wins. Raise ``max_advance_m`` if position samples are sparse
    enough that a car can legally travel farther than that between samples.
    """

    if lookahead_m <= 0 or lookback_m < 0 or max_dist_m <= 0 or max_advance_m <= 0:
        raise F1FrameError("projection windows must be positive")
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.ndim != 1 or xs.shape != ys.shape:
        raise F1FrameError("x and y must be 1-D arrays of equal length")
    if xs.shape[0] == 0:
        raise F1FrameError("cannot project an empty position stream")
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise F1FrameError("position coordinates contain NaN or infinity")

    ax = reference.x[:-1]
    ay = reference.y[:-1]
    bx = reference.x[1:]
    by = reference.y[1:]
    sab = reference.s_m[:-1]
    seg_len = np.diff(reference.s_m)

    s_hat = np.full(xs.shape[0], np.nan)
    dist = np.full(xs.shape[0], np.nan)

    s0, d0 = _search(xs[0], ys[0], ax, ay, bx, by, sab, seg_len, 0.0, lookahead_m, prefer_low_s=True)
    if not np.isfinite(d0) or d0 > max_dist_m:
        s0, d0 = _search(
            xs[0],
            ys[0],
            ax,
            ay,
            bx,
            by,
            sab,
            seg_len,
            0.0,
            reference.length_m,
            prefer_low_s=True,
        )
    if np.isfinite(d0) and d0 <= max_dist_m:
        s_hat[0] = s0
        dist[0] = d0
        prev = float(s0)
    else:
        prev = 0.0

    for i in range(1, xs.shape[0]):
        s_lo = max(0.0, prev - lookback_m)
        s_i, d_i = _search(
            xs[i],
            ys[i],
            ax,
            ay,
            bx,
            by,
            sab,
            seg_len,
            s_lo,
            prev + lookahead_m,
            prefer_s=prev,
        )
        if np.isfinite(s_i) and s_i > prev + max_advance_m:
            s_near, d_near = _search(
                xs[i],
                ys[i],
                ax,
                ay,
                bx,
                by,
                sab,
                seg_len,
                s_lo,
                prev + max_advance_m,
                prefer_s=prev,
            )
            # Keep the continuation when it is a real match. If the best
            # continuation sits on the advance limit and a later segment is
            # much closer, the limit is cutting through a normal step.
            clipped = np.isfinite(s_near) and s_near >= prev + max_advance_m - 1.0 and d_i + 1.0 < d_near
            if np.isfinite(d_near) and d_near <= max_dist_m and not clipped:
                s_i, d_i = s_near, d_near
        if not np.isfinite(s_i) or not np.isfinite(d_i) or d_i > max_dist_m:
            continue
        s_hat[i] = s_i
        dist[i] = d_i
        prev = float(s_i)
    return s_hat, dist


def _search(
    px: float,
    py: float,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    sab: np.ndarray,
    seg_len: np.ndarray,
    s_lo: float,
    s_hi: float,
    *,
    prefer_low_s: bool = False,
    prefer_s: float | None = None,
) -> tuple[float, float]:
    mask = (sab < s_hi) & ((sab + seg_len) > s_lo) & (seg_len > 1e-9)
    if not np.any(mask):
        return np.nan, np.nan
    t, d = _project_segments(px, py, ax[mask], ay[mask], bx[mask], by[mask])
    s_cand = sab[mask] + t * seg_len[mask]
    best = float(np.min(d))
    close = d <= best + 0.5
    close_index = np.flatnonzero(close)
    if prefer_low_s:
        chosen = close_index[int(np.argmin(s_cand[close_index]))]
    elif prefer_s is not None:
        chosen = close_index[int(np.argmin(np.abs(s_cand[close_index] - prefer_s)))]
    else:
        chosen = int(np.argmin(d))
    return float(s_cand[chosen]), float(d[chosen])


def _project_segments(
    px: float,
    py: float,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
    qx = ax + t * abx
    qy = ay + t * aby
    return t, np.hypot(px - qx, py - qy)
