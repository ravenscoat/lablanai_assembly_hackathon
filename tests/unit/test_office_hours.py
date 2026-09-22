"""Deterministic office-hours check."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

TASHKENT = ZoneInfo("Asia/Tashkent")


def _cfg(office_days=None, start=9, end=18):
    from config.schema import TransferConfig

    return TransferConfig.model_validate(
        {
            "office_hours_start": start,
            "office_hours_end": end,
            "office_days": office_days or [0, 1, 2, 3, 4, 5],
        }
    )


def test_weekday_midday_is_within_hours():
    from tools.platform.escalate import _is_within_office_hours

    # Monday 10:00 Tashkent
    assert _is_within_office_hours(_cfg(), datetime(2026, 4, 20, 10, 0, tzinfo=TASHKENT)) is True


def test_weekday_evening_is_out_of_hours():
    from tools.platform.escalate import _is_within_office_hours

    # Monday 22:00 Tashkent
    assert _is_within_office_hours(_cfg(), datetime(2026, 4, 20, 22, 0, tzinfo=TASHKENT)) is False


def test_weekday_early_morning_is_out_of_hours():
    from tools.platform.escalate import _is_within_office_hours

    # Monday 08:00 Tashkent (before 09:00 open)
    assert _is_within_office_hours(_cfg(), datetime(2026, 4, 20, 8, 0, tzinfo=TASHKENT)) is False


def test_sunday_is_out_of_hours():
    from tools.platform.escalate import _is_within_office_hours

    # Sunday 2026-04-19 14:00 — weekday=6, not in Mon-Sat default
    assert _is_within_office_hours(_cfg(), datetime(2026, 4, 19, 14, 0, tzinfo=TASHKENT)) is False


def test_start_boundary_inclusive():
    from tools.platform.escalate import _is_within_office_hours

    # Monday 09:00 exactly — should be inside
    assert _is_within_office_hours(_cfg(), datetime(2026, 4, 20, 9, 0, tzinfo=TASHKENT)) is True


def test_end_boundary_exclusive():
    from tools.platform.escalate import _is_within_office_hours

    # Monday 18:00 exactly — should be outside (end is exclusive)
    assert _is_within_office_hours(_cfg(), datetime(2026, 4, 20, 18, 0, tzinfo=TASHKENT)) is False


def test_defaults_to_now_when_no_dt_passed():
    from tools.platform.escalate import _is_within_office_hours

    # Just verify the helper doesn't crash when `now` is omitted — the
    # return value depends on real wall-clock time, so we only assert
    # it's a bool.
    result = _is_within_office_hours(_cfg())
    assert isinstance(result, bool)
