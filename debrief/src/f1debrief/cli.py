"""Command line for an approach dossier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from f1frame import read_frame

from f1debrief.audit import approach_card
from f1debrief.demo import WINDOW, demo_pass
from f1debrief.dossier import render_html, render_text
from f1debrief.openf1 import DebriefError
from f1debrief.race import card_from_race, card_from_recorded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1-debrief", description="Approach dossiers on a shared f1frame grid.")
    sub = parser.add_subparsers(dest="command", required=True)
    approach = sub.add_parser("approach", help="Brake point and apex speed for two or more laps.")
    approach.add_argument("--demo", action="store_true", help="Run the built-in synthetic pass.")
    approach.add_argument("--frames", nargs="+", type=Path, help="f1frame parquet files aligned together.")
    approach.add_argument("--window", nargs=2, type=float, metavar=("START_M", "END_M"))
    approach.add_argument("--title", default="Approach")
    approach.add_argument("--html", type=Path, help="Write an HTML dossier to this path.")
    passed = sub.add_parser("pass", help="One on-track pass, with the corner chosen from the shared line.")
    passed.add_argument("--recorded", action="store_true", help="Use the committed 2024 Monza race pass.")
    passed.add_argument("--year", type=int, help="Season, for a live OpenF1 load.")
    passed.add_argument("--race", help="Circuit or country, for a live OpenF1 load. Example: Monza.")
    passed.add_argument("--drivers", nargs=2, metavar=("A", "B"), help="The two drivers, either order.")
    passed.add_argument("--index", type=int, default=0, help="Which pass between that pair, in time order.")
    passed.add_argument("--html", type=Path, help="Write an HTML dossier to this path.")
    args = parser.parse_args(argv)

    if args.command == "approach":
        return _approach(args)
    if args.command == "pass":
        return _pass(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _approach(args: argparse.Namespace) -> int:
    if args.demo:
        demo = demo_pass()
        start, end = args.window if args.window else WINDOW
        card = approach_card(demo.frames, start, end, title=args.title)
        text = render_text(card)
        text += "\nSpeed-integral brake point, for comparison:\n"
        for lap_id, meters in demo.integrated_brake_m.items():
            text += f"  {lap_id}: {meters:.0f} m\n"
    elif args.frames:
        if not args.window:
            print("approach --frames requires --window START_M END_M", file=sys.stderr)
            return 2
        frames = [read_frame(path) for path in args.frames]
        card = approach_card(frames, args.window[0], args.window[1], title=args.title)
        text = render_text(card)
    else:
        print("approach needs --demo or --frames", file=sys.stderr)
        return 2

    sys.stdout.write(text)
    if args.html:
        args.html.write_text(render_html(card), encoding="utf-8")
    return 0


def _pass(args: argparse.Namespace) -> int:
    try:
        if args.recorded:
            card = card_from_recorded()
        elif args.year and args.race and args.drivers:
            card = card_from_race(args.year, args.race, (args.drivers[0], args.drivers[1]), index=args.index)
        else:
            print("pass needs --recorded, or --year, --race, and --drivers", file=sys.stderr)
            return 2
    except DebriefError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(render_text(card))
    if args.html:
        args.html.write_text(render_html(card), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
