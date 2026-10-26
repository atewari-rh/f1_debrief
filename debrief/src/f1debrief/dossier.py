"""Text and HTML dossiers. Both are generated from an ApproachCard."""

from __future__ import annotations

from html import escape

from f1debrief.audit import ApproachCard


def render_text(card: ApproachCard) -> str:
    lines = [
        f"{card.title}  {card.start_m:.0f}–{card.end_m:.0f} m",
        "",
        f"{'Driver':<8}{'Brake m':>10}{'Apex km/h':>12}{'Apex m':>10}",
    ]
    for driver in card.drivers:
        brake = "—" if driver.brake_m is None else f"{driver.brake_m:.0f}"
        speed = "—" if driver.apex_speed_kph is None else f"{driver.apex_speed_kph:.0f}"
        apex = "—" if driver.apex_s_m is None else f"{driver.apex_s_m:.0f}"
        lines.append(f"{driver.lap_id:<8}{brake:>10}{speed:>12}{apex:>10}")
    lines.append("")
    lines.append(_comparison(card))
    lines.extend(_gap_lines(card))
    lines.append("")
    lines.append(f"Confidence: {card.confidence.level}")
    for reason in card.confidence.reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(
        "The public position stream is one path, so this card does not say who was on the inside."
    )
    return "\n".join(lines) + "\n"


def render_html(card: ApproachCard) -> str:
    rows = []
    for driver in card.drivers:
        rows.append(
            "<tr>"
            f"<td>{escape(driver.lap_id)}</td>"
            f"<td>{_num(driver.brake_m)}</td>"
            f"<td>{_num(driver.apex_speed_kph)}</td>"
            f"<td>{_num(driver.apex_s_m)}</td>"
            "</tr>"
        )
    reasons = "".join(f"<li>{escape(reason)}</li>" for reason in card.confidence.reasons)
    level = escape(card.confidence.level)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(card.title)}</title>
  <style>
    body {{ font-family: "Iowan Old Style", Palatino, Georgia, serif; max-width: 38rem; margin: 2.5rem auto; color: #1c1c1c; padding: 0 1rem; }}
    h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 0.2rem; }}
    p.meta {{ color: #444; margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1.2rem 0; }}
    th, td {{ text-align: right; padding: 0.4rem 0.5rem; border-bottom: 1px solid #ccc; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .confidence {{ font-weight: 600; }}
    .high {{ color: #0d5c2e; }}
    .limited {{ color: #8a4b08; }}
    .low {{ color: #8a1c1c; }}
    footer {{ font-size: 0.95rem; color: #333; }}
  </style>
</head>
<body>
  <h1>{escape(card.title)}</h1>
  <p class="meta">{card.start_m:.0f}–{card.end_m:.0f} m on a {card.step_m:.0f} m grid</p>
  <table>
    <thead>
      <tr><th>Driver</th><th>Brake m</th><th>Apex km/h</th><th>Apex m</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <p>{escape(_comparison(card))}</p>
  {''.join(f'<p>{escape(line)}</p>' for line in _gap_lines(card))}
  <p class="confidence {level}">Confidence: {level}</p>
  <ul>
    {reasons}
  </ul>
  <footer>The public position stream is one path, so this card does not say who was on the inside.</footer>
</body>
</html>
"""


def _gap_lines(card: ApproachCard) -> list[str]:
    lines: list[str] = []
    if card.pass_s_m is not None:
        lines.append(f"Pass at {card.pass_s_m:.0f} m.")
    if card.corner_index is not None:
        radius = ""
        if card.corner_radius_m is not None:
            radius = f", map radius about {card.corner_radius_m:.0f} m"
        direction = card.corner_direction or "bend"
        lines.append(
            f"Corner {card.corner_index}, {direction}{radius}. "
            "The radius is measured on the timing map."
        )
    if (
        card.window_delta_s is not None
        and card.delta_reference_id
        and card.delta_other_id
    ):
        gap = card.window_delta_s
        if abs(gap) < 0.005:
            lines.append(
                f"In this window, {card.delta_other_id} and {card.delta_reference_id} are level on the map clock."
            )
        elif gap > 0:
            lines.append(
                f"In this window, {card.delta_other_id} is {gap:.2f} s behind {card.delta_reference_id} on the map."
            )
        else:
            lines.append(
                f"In this window, {card.delta_other_id} is {-gap:.2f} s ahead of {card.delta_reference_id} on the map."
            )
    if card.closure_error_ms is not None:
        lines.append(
            f"Lap-time closure: {card.closure_error_ms:.0f} ms "
            "(map integral versus the official lap-time gap)."
        )
    elif card.delta_reference_id is not None:
        lines.append("Lap-time closure: unavailable.")
    if card.note:
        lines.append(card.note)
    return lines


def _comparison(card: ApproachCard) -> str:
    later = card.later_brake
    if later is None:
        return "The window does not contain a brake onset for two drivers."
    earlier, late, gap = later
    if gap <= card.step_m:
        sentence = f"{earlier.lap_id} and {late.lap_id} brake on the same {card.step_m:.0f} m step."
    else:
        sentence = f"{late.lap_id} brakes {gap:.0f} m later than {earlier.lap_id}."
    speeds = [driver for driver in card.drivers if driver.apex_speed_kph is not None]
    if len(speeds) >= 2:
        slow = min(speeds, key=lambda driver: driver.apex_speed_kph or 0.0)
        fast = max(speeds, key=lambda driver: driver.apex_speed_kph or 0.0)
        assert slow.apex_speed_kph is not None and fast.apex_speed_kph is not None
        delta = fast.apex_speed_kph - slow.apex_speed_kph
        if delta >= 1:
            sentence += f" {slow.lap_id}'s minimum speed in the window is {delta:.0f} km/h lower."
    return sentence


def _num(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}"
