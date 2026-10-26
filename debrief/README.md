# f1-debrief

f1-debrief reads laps that `f1frame` has already placed on one meter grid and writes an approach card: where each driver braked, and how slow they were at the apex of that window.

It is installed with the library. From the repository root:

```bash
pip install -e ".[dev]"
f1-debrief pass --recorded
```

`f1-debrief pass --recorded` is lap 41 of the 2024 Monza race: Norris passes Verstappen. The corner window is chosen from the shared line. The card prints the brake meters and the lap-time closure. Pit stops and penalty messages next to the pass are left out.

```bash
f1-debrief pass --year 2024 --race Monza --drivers NOR VER
f1-debrief approach --demo
```

The live command reads OpenF1. The demo compares two synthetic cars into the same corner. The card's brake meters stay on the geometric grid. A speed integral with an 8% high channel, printed underneath, moves the second driver's brake point by well over a grid step.

```bash
f1-debrief approach --frames ver.parquet nor.parquet --window 780 920 --html dossier.html
```

Those parquet files must come from one `align_laps` call. Aligning each driver alone builds a different ruler, and the command rejects that pair.

## What the card says

Brake onset is the first meter in the window where brake turns on. Apex speed is the minimum speed in that window. Confidence is `high`, `limited`, or `low`, taken from the f1frame report:

- a clock more than 1 s away from the official lap time, a projection worse than 30 m, coverage under 90%, or more than 5% of position samples missing the reference → `low`, and the brake points are marked unverified
- a clock more than 0.4 s off, a projection worse than 15 m, or coverage under 97% → `limited`
- otherwise `high`

A 0.1–0.25 s clock error is normal for this feed. The recorded Monza race laps sit nearer 0.3 s and stay `high`. A clock past 0.4 s is `limited`.

The card does not say who was on the inside. The public position stream is one path. It also does not draw the pit lane: that path is not in the coordinates. Pit entry is a time in the timing feed, which you can look up on this same ruler.
