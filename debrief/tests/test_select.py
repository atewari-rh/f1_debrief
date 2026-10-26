"""Pit stops and penalty messages are removed before a pass becomes a card."""

from datetime import datetime, timezone

from f1debrief.select import PassEvent, Penalty, PitStop, on_track_passes, passes_between


def _at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2024, 9, 1, hour, minute, second, tzinfo=timezone.utc)


def test_pit_and_penalty_swaps_are_dropped():
    overtakes = [
        PassEvent(_at(13, 11, 40), "NOR", "VER", 4),
        PassEvent(_at(14, 0, 37), "NOR", "VER", 4),
        PassEvent(_at(14, 10, 10), "VER", "NOR", 3),
    ]
    pits = [PitStop(_at(13, 11, 37), "NOR", 24.0)]
    penalties = [Penalty(_at(14, 10, 0), "VER")]

    kept = on_track_passes(overtakes, pits, penalties)

    assert [(event.date, event.overtaking) for event in kept] == [(_at(14, 0, 37), "NOR")]
    pair = passes_between(kept, "ver", "nor")
    assert len(pair) == 1
    assert pair[0].overtaking == "NOR"


def test_a_swap_on_the_way_into_the_pits_is_dropped():
    overtakes = [
        PassEvent(_at(13, 23, 36), "VER", "NOR", 5),
        PassEvent(_at(14, 0, 37), "NOR", "VER", 4),
    ]
    pits = [PitStop(_at(13, 23, 51), "NOR", 24.1)]

    kept = on_track_passes(overtakes, pits, [])

    assert [event.overtaking for event in kept] == ["NOR"]


def test_a_penalty_for_someone_else_stays():
    overtakes = [PassEvent(_at(14, 0, 37), "NOR", "VER", 4)]
    penalties = [Penalty(_at(14, 0, 37), "RIC")]
    assert len(on_track_passes(overtakes, [], penalties)) == 1
