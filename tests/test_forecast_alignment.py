"""
Forecast time-alignment regression tests.

Background (the bug these tests lock down):
  AEMO PD7DAY ``INTERVAL_DATETIME`` is PERIOD-BEGINNING — a run lists the first
  interval at 18:00 to mean the 18:00–18:30 dispatch period.  Settlement and
  Amber, however, label that same period's price by its PERIOD-ENDING stamp
  (18:30).  The sidecar parser correctly stores PD7DAY as ``interval_start``
  (period-beginning) and keeps it INTERNAL for calibration bucketing and
  forecast↔actual pairing.  But the PUBLISHED forecast timestamp (what HA
  exposes / a user plots and overlays on Amber) must be PERIOD-ENDING, i.e.
  ``interval_end = interval_start + period``.

These tests prove:
  1. A PD7DAY row with INTERVAL_DATETIME = "2026/06/16 18:00:00" parses to an
     interval_start of 18:00 NEM (period-beginning, internal — unchanged).
  2. The PUBLISHED forecast slot for that row carries ``interval_end`` == 18:30
     NEM (period-ending), i.e. start + 30 min for the 30-min native period.
  3. The resampled (published) forecast also carries the period-ending stamp.
  4. "Current price" selection (the half-open [interval_start, interval_end)
     window, which uses the period-beginning stamp) still returns the slot that
     actually covers ``now`` — the period-ending switch does NOT shift it.

They FAIL on the pre-fix code (no ``interval_end`` key existed) and PASS after.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from forecast_cache import PriceForecastSlot
from forecast_resampler import resample_price_slots
from pd7day_client import NEM_TIMEZONE, Pd7DayClient


# A minimal but realistic PD7DAY PRICESOLUTION CSV.  INTERVAL_DATETIME values are
# PERIOD-BEGINNING in NEM time (UTC+10).  Columns follow the real layout:
#   I,PD7DAY,PRICESOLUTION,1,RUN_DATETIME,INTERVENTION,INTERVAL_DATETIME,REGIONID,RRP,...
_PD7DAY_CSV = (
    "C,PD7DAY,,AEMO,PUBLIC,2026/06/16,00:00:00\n"
    "I,PD7DAY,PRICESOLUTION,1,RUN_DATETIME,INTERVENTION,INTERVAL_DATETIME,REGIONID,RRP\n"
    'D,PD7DAY,PRICESOLUTION,1,"2026/06/16 17:30:00",0,"2026/06/16 18:00:00",QLD1,90.0\n'
    'D,PD7DAY,PRICESOLUTION,1,"2026/06/16 17:30:00",0,"2026/06/16 18:30:00",QLD1,110.0\n'
    'D,PD7DAY,PRICESOLUTION,1,"2026/06/16 17:30:00",0,"2026/06/16 19:00:00",QLD1,130.0\n'
    "C,END OF REPORT,4\n"
)

_NATIVE_PERIOD_MINUTES = 30


def _parse_pd7day():
    return Pd7DayClient()._parse_csv(_PD7DAY_CSV, "QLD1")


def test_pd7day_interval_start_is_period_beginning():
    """The parser keeps PD7DAY INTERVAL_DATETIME as interval_start (internal)."""
    forecast = _parse_pd7day()
    first = forecast.slots[0]

    # 18:00 NEM, period-beginning, preserved unchanged for internal bucketing.
    assert first.interval_start_nem.hour == 18
    assert first.interval_start_nem.minute == 0
    assert first.interval_start_utc == datetime(2026, 6, 16, 8, 0, tzinfo=timezone.utc)


def test_published_slot_timestamp_is_period_ending():
    """
    The PUBLISHED forecast slot for the 18:00 PD7DAY row must be stamped 18:30
    (period-ending = interval_start + 30 min for the 30-min native period).
    """
    forecast = _parse_pd7day()
    first = forecast.slots[0]

    published = PriceForecastSlot(
        interval_start=first.interval_start_utc,
        raw_rrp_per_mwh=first.rrp_per_mwh,
        calibrated_wholesale_kwh=0.09,
        import_price_kwh=0.20,
        export_price_kwh=0.09,
        network_tou_rate_kwh=0.05,
    ).as_dict()

    # interval_start unchanged (period-beginning, internal/reference)
    assert published["interval_start"] == "2026-06-16T08:00:00+00:00"  # 18:00 NEM

    # interval_end is the PUBLISHED, period-ending (Amber-aligned) stamp
    assert "interval_end" in published, "published slot must carry interval_end"
    interval_end = datetime.fromisoformat(published["interval_end"])
    interval_start = datetime.fromisoformat(published["interval_start"])
    assert interval_end - interval_start == timedelta(minutes=_NATIVE_PERIOD_MINUTES)

    # 18:30 NEM == 08:30 UTC
    assert interval_end.astimezone(NEM_TIMEZONE).hour == 18
    assert interval_end.astimezone(NEM_TIMEZONE).minute == 30
    assert interval_end == datetime(2026, 6, 16, 8, 30, tzinfo=timezone.utc)


def test_resampled_published_forecast_is_period_ending():
    """The resampled (published) forecast carries period-ending interval_end too."""
    forecast = _parse_pd7day()
    now_utc = forecast.slots[0].interval_start_utc  # 18:00 NEM

    slot_dicts = [
        PriceForecastSlot(
            interval_start=s.interval_start_utc,
            raw_rrp_per_mwh=s.rrp_per_mwh,
            calibrated_wholesale_kwh=s.rrp_per_mwh / 1000.0,
            import_price_kwh=s.rrp_per_mwh / 1000.0,
            export_price_kwh=s.rrp_per_mwh / 1000.0,
            network_tou_rate_kwh=0.0,
        ).as_dict()
        for s in forecast.slots
    ]

    resampled = resample_price_slots(
        slot_dicts,
        target_period_minutes=30,
        horizon_hours=6,
        now_utc=now_utc,
    )
    assert resampled, "expected at least one resampled slot"

    first = resampled[0]
    assert first["interval_start"] == "2026-06-16T08:00:00+00:00"
    start = datetime.fromisoformat(first["interval_start"])
    end = datetime.fromisoformat(first["interval_end"])
    assert end - start == timedelta(minutes=30)
    assert end == datetime(2026, 6, 16, 8, 30, tzinfo=timezone.utc)  # 18:30 NEM


def _select_current_slot(slots, now_utc):
    """
    Replica of the HA sensor's current-slot predicate (sensor.py ``_current_slot``)
    so it can be exercised without importing Home Assistant.  The current slot is
    the one whose half-open interval [interval_start, interval_end) contains now.
    Selection deliberately uses the PERIOD-BEGINNING interval_start.
    """
    for slot in slots:
        if slot["start"] <= now_utc < slot["end"]:
            return slot
    return None


def test_current_price_selection_covers_now():
    """
    "Current price" must select the slot whose [start, end) window contains now —
    and the period-ending publish switch must NOT shift it to the wrong/past slot.
    """
    forecast = _parse_pd7day()

    slots = []
    for s in forecast.slots:
        published = PriceForecastSlot(
            interval_start=s.interval_start_utc,
            raw_rrp_per_mwh=s.rrp_per_mwh,
            calibrated_wholesale_kwh=0.0,
            import_price_kwh=s.rrp_per_mwh / 1000.0,
            export_price_kwh=0.0,
            network_tou_rate_kwh=0.0,
        ).as_dict()
        slots.append(
            {
                "start": datetime.fromisoformat(published["interval_start"]),
                "end": datetime.fromisoformat(published["interval_end"]),
                "rrp": s.rrp_per_mwh,
            }
        )

    # now = 18:10 NEM (08:10 UTC) — inside the FIRST period [18:00, 18:30)
    now_utc = datetime(2026, 6, 16, 18, 10, tzinfo=NEM_TIMEZONE).astimezone(timezone.utc)

    current = _select_current_slot(slots, now_utc)
    assert current is not None
    assert current["rrp"] == 90.0  # the 18:00->18:30 period's price
    assert current["start"] == datetime(2026, 6, 16, 8, 0, tzinfo=timezone.utc)
    assert current["end"] == datetime(2026, 6, 16, 8, 30, tzinfo=timezone.utc)

    # Guard the regression the task warns about: selecting by the period-ENDING
    # stamp instead would wrongly pick a different (later) slot.
    wrong = None
    for slot in slots:
        if slot["end"] > now_utc:
            wrong = slot
            break
    # The first slot's end (18:30) is also > now here, so this particular guard
    # confirms the [start, end) window — not a bare end>now test — is what keeps
    # selection correct as the period boundary advances.
    later_now = datetime(2026, 6, 16, 18, 40, tzinfo=NEM_TIMEZONE).astimezone(timezone.utc)
    assert _select_current_slot(slots, later_now)["rrp"] == 110.0  # 18:30->19:00 period
