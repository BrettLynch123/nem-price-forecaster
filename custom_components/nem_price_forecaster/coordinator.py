"""
DataUpdateCoordinator for NEM Price Forecaster (sidecar mode).

Responsibilities (refactored for sidecar architecture):
  1. Poll the sidecar's /price_forecast and /load_forecast endpoints.
  2. Parse and expose a structured forecast for sensor entities.
  3. Handle sidecar unavailability with HA retry semantics (UpdateFailed).
  4. Provide calibration observation methods (POST to sidecar).

Python 3.14 compatibility: NO darts, NO sklearn, NO scipy imports here.
Only stdlib + numpy + homeassistant builtins.

The sidecar runs all ML compute (PD7DAY fetch, isotonic/Darts calibration,
tariff calculation, load forecast). The coordinator is a thin HTTP client.

Legacy mode (no sidecar):
  When CONF_SIDECAR_URL is not configured, the coordinator falls back to the
  original embedded mode (importing all ML modules directly). This maintains
  backward compatibility for users who have not yet deployed the sidecar.
  NOTE: Legacy embedded mode does NOT work on Python 3.14 if darts is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_ACTUAL_RRP_ENTITY,
    CONF_LOAD_ENTITY_ID,
    CONF_REGION,
    CONF_FORECAST_HORIZON_HOURS,
    CONF_FORECAST_PERIOD_MINUTES,
    CONF_LOAD_FORECASTER_ENABLED,
    CONF_SIDECAR_URL,
    DEFAULT_FORECAST_HORIZON_HOURS,
    DEFAULT_FORECAST_PERIOD_MINUTES,
    DEFAULT_LOAD_FORECASTER_ENABLED,
    DEFAULT_SIDECAR_URL,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    CONF_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
    LOAD_SAMPLE_MIN_INTERVAL_SECONDS,
    LOAD_SAMPLE_SIGNIFICANT_DELTA_WATTS,
    PREDICTION_BUFFER_MAX_ENTRIES,
    PREDICTION_BUFFER_TTL_HOURS,
)
from .sidecar_client import SidecarClient, SidecarUnavailable

_LOGGER = logging.getLogger(__name__)

# NEM time is always UTC+10 (no DST) — matches the sidecar's pd7day_client.
# Calibration observations are bucketed by NEM hour-of-day, so the listener must
# derive hour_of_day in this zone, not local HA time.
_NEM_TIMEZONE = timezone(timedelta(hours=10))

# PD7DAY's native dispatch period.  Used to snap observation timestamps to the
# interval-start they belong to (period-beginning), matching the coordinator's
# forecast-slot keys and the sidecar's load-observation bucketing.
_NEM_INTERVAL_MINUTES = 30

# Sentinel HA states that carry no usable numeric reading.
_UNUSABLE_STATES = ("unknown", "unavailable", "none", "")


@dataclass
class ForecastSlot:
    """
    One price forecast slot as returned by the sidecar /price_forecast endpoint.

    All prices are in $/kWh.
    """
    interval_start_utc: datetime     # UTC, tz-aware — PERIOD-BEGINNING (internal)
    interval_end_utc: datetime       # UTC, tz-aware — PERIOD-ENDING (published)
    raw_rrp_per_mwh: float
    calibrated_wholesale_kwh: float
    import_price_kwh: float
    export_price_kwh: float
    network_tou_rate_kwh: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval_start": self.interval_start_utc.isoformat(),
            "interval_end": self.interval_end_utc.isoformat(),
            "import_price": round(self.import_price_kwh, 6),
            "export_price": round(self.export_price_kwh, 6),
            "calibrated_wholesale": round(self.calibrated_wholesale_kwh, 6),
            "raw_rrp_per_mwh": round(self.raw_rrp_per_mwh, 4),
            "network_tou_rate": round(self.network_tou_rate_kwh, 6),
        }


@dataclass
class LoadForecastSlot:
    """One load forecast slot (30-min, watts)."""
    interval_start_utc: datetime
    load_watts: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "datetime": self.interval_start_utc.isoformat(),
            "load_power": round(self.load_watts, 1),
        }


@dataclass
class CoordinatorData:
    """Snapshot of sidecar forecast state, published to all sensor entities."""
    # Parsed price forecast slots (from sidecar raw_forecast)
    forecast_slots: list[ForecastSlot]
    run_datetime_utc: datetime
    region: str
    calibration_observation_count: int
    calibration_is_active: bool
    next_update: datetime
    # Resampled forecasts (from sidecar forecast[])
    resampled_price_forecast: list[dict]
    # Load forecast
    load_forecast_slots: list[LoadForecastSlot]
    resampled_load_forecast: list[dict]
    load_model_name: str
    load_is_trained: bool
    load_training_observations: int
    # Resolution metadata
    forecast_horizon_hours: int
    forecast_period_minutes: int
    # Sidecar health
    sidecar_url: str
    sidecar_reachable: bool


class NemPriceForecastCoordinator(DataUpdateCoordinator[CoordinatorData]):
    """
    Thin coordinator that fetches from the NEM sidecar and exposes sensor data.

    ML compute runs in the sidecar. This coordinator is pure async I/O.
    """

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        self._config_entry = config_entry
        self._region: str = config_entry.data[CONF_REGION]
        self._sidecar_url: str = config_entry.data.get(CONF_SIDECAR_URL, DEFAULT_SIDECAR_URL)

        self._forecast_horizon_hours: int = int(config_entry.data.get(
            CONF_FORECAST_HORIZON_HOURS, DEFAULT_FORECAST_HORIZON_HOURS
        ))
        self._forecast_period_minutes: int = int(config_entry.data.get(
            CONF_FORECAST_PERIOD_MINUTES, DEFAULT_FORECAST_PERIOD_MINUTES
        ))
        # Options take precedence over install-time data so existing installs can
        # enable the observation entities (and toggle the load forecaster) from
        # the options flow without re-adding the integration.
        merged_entry_config = {**config_entry.data, **config_entry.options}
        self._load_forecaster_enabled: bool = merged_entry_config.get(
            CONF_LOAD_FORECASTER_ENABLED, DEFAULT_LOAD_FORECASTER_ENABLED
        )
        # Optional observation source entities (issue #9).  Absent/blank => the
        # corresponding observation path stays off.
        self._actual_rrp_entity_id: Optional[str] = (
            merged_entry_config.get(CONF_ACTUAL_RRP_ENTITY) or None
        )
        self._load_entity_id: Optional[str] = (
            merged_entry_config.get(CONF_LOAD_ENTITY_ID) or None
        )

        # Predicted-RRP buffer: interval_start_utc -> last-emitted PD7DAY raw RRP
        # ($/MWh) for that interval.  Populated each poll; consumed when the
        # matching actual arrives, then evicted (or TTL-evicted).
        self._predicted_rrp_by_interval: dict[datetime, float] = {}
        # Calibration debounce: interval_start_utc values already posted, so a
        # noisy actual-RRP sensor can't double-send within the same interval.
        self._calibrated_intervals: set[datetime] = set()
        # Load sampling state (rate-limit + significant-delta gate).
        self._last_load_sample_utc: Optional[datetime] = None
        self._last_load_sample_watts: Optional[float] = None

        self._sidecar_client = SidecarClient(self._sidecar_url)

        # Update cadence: options-flow override > data-entry override > default.
        # The sidecar's price-predict job runs every 5 minutes, so polling more
        # often than ~5 minutes wastes effort; default 15 min gives ~3 companion
        # polls per sidecar refresh cycle.
        merged_config = {**config_entry.data, **config_entry.options}
        update_interval_minutes = int(merged_config.get(
            CONF_UPDATE_INTERVAL_MINUTES, DEFAULT_UPDATE_INTERVAL_MINUTES
        ))
        if update_interval_minutes < 1:
            update_interval_minutes = DEFAULT_UPDATE_INTERVAL_MINUTES
        self._update_interval_minutes = update_interval_minutes

        update_interval = timedelta(minutes=update_interval_minutes)
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{self._region}",
            update_interval=update_interval,
        )

    # ------------------------------------------------------------------
    # DataUpdateCoordinator override
    # ------------------------------------------------------------------

    async def _async_update_data(self) -> CoordinatorData:
        """
        Fetch price + load forecasts from the sidecar.

        On SidecarUnavailable, raises UpdateFailed so HA marks sensors
        unavailable and retries.
        """
        now_utc = datetime.now(timezone.utc)

        # --- Price forecast ---
        try:
            price_response = await self._sidecar_client.async_get_price_forecast()
        except SidecarUnavailable as fetch_error:
            raise UpdateFailed(
                f"Sidecar unavailable at {self._sidecar_url}: {fetch_error}"
            ) from fetch_error

        forecast_slots = _parse_price_slots(price_response.get("raw_forecast", []))
        resampled_price = price_response.get("forecast", [])

        # Buffer the last-emitted PREDICTION (PD7DAY raw RRP $/MWh) per
        # interval-start so the calibration listener can pair it with the actual
        # RRP for the same interval when it later settles.  Only meaningful when
        # an actual-RRP entity is configured.
        if self._actual_rrp_entity_id is not None:
            self._buffer_predictions(forecast_slots, now_utc)

        calibration_observations = int(price_response.get("calibration_observations", 0))
        calibration_active = bool(price_response.get("calibration_active", False))
        pd7day_run_str = price_response.get("pd7day_run_datetime")
        pd7day_run_utc = _parse_iso_or_now(pd7day_run_str)

        # --- Load forecast (optional) ---
        load_slots: list[LoadForecastSlot] = []
        resampled_load: list[dict] = []
        load_is_trained = False
        load_training_obs = 0

        if self._load_forecaster_enabled:
            try:
                load_response = await self._sidecar_client.async_get_load_forecast()
                if load_response is not None:
                    load_slots = _parse_load_slots(load_response.get("raw_forecast", []))
                    resampled_load = load_response.get("forecast", [])
                    load_is_trained = bool(load_response.get("model_trained", False))
                    load_training_obs = int(load_response.get("training_observations", 0))
            except SidecarUnavailable as load_error:
                _LOGGER.warning(
                    "Load forecast fetch failed (non-fatal): %s", load_error
                )

        next_update_utc = now_utc + timedelta(minutes=self._update_interval_minutes)

        return CoordinatorData(
            forecast_slots=forecast_slots,
            run_datetime_utc=pd7day_run_utc,
            region=self._region,
            calibration_observation_count=calibration_observations,
            calibration_is_active=calibration_active,
            next_update=next_update_utc,
            resampled_price_forecast=resampled_price,
            load_forecast_slots=load_slots,
            resampled_load_forecast=resampled_load,
            load_model_name="Darts-LightGBM-Direct",
            load_is_trained=load_is_trained,
            load_training_observations=load_training_obs,
            forecast_horizon_hours=self._forecast_horizon_hours,
            forecast_period_minutes=self._forecast_period_minutes,
            sidecar_url=self._sidecar_url,
            sidecar_reachable=True,
        )

    # ------------------------------------------------------------------
    # Calibration feed (forwarded to sidecar via HTTP POST)
    # ------------------------------------------------------------------

    async def async_add_import_calibration_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_import_rrp_per_mwh: float,
        hour_of_day_nem: int,
        observed_at: datetime,
    ) -> None:
        """
        Forward a (predicted, actual import) calibration observation to the sidecar.
        Best-effort (non-fatal if sidecar is unavailable).
        """
        await self._sidecar_client.async_post_import_calibration(
            predicted_rrp_per_mwh,
            actual_import_rrp_per_mwh,
            hour_of_day_nem,
            observed_at,
        )

    async def async_add_export_calibration_observation(
        self,
        predicted_rrp_per_mwh: float,
        actual_export_rrp_per_mwh: float,
        hour_of_day_nem: int,
        observed_at: datetime,
    ) -> None:
        """Forward an export calibration observation to the sidecar."""
        await self._sidecar_client.async_post_export_calibration(
            predicted_rrp_per_mwh,
            actual_export_rrp_per_mwh,
            hour_of_day_nem,
            observed_at,
        )

    async def async_add_load_observation(
        self,
        interval_start_utc: datetime,
        load_watts: float,
    ) -> None:
        """Forward a load observation to the sidecar."""
        await self._sidecar_client.async_post_load_observation(
            interval_start_utc, load_watts
        )

    # ------------------------------------------------------------------
    # Observation listeners (issue #9: wire actuals back to the sidecar)
    # ------------------------------------------------------------------

    def async_setup_observation_listeners(self) -> None:
        """
        Register state-change listeners that feed real-world observations to the
        sidecar, and tie their teardown to the config entry's unload.

        Gating:
          * actual-RRP calibration listener — active whenever an actual-RRP
            entity is configured.
          * load listener — active only when a load entity is configured AND the
            load forecaster is enabled.

        Both listeners are no-ops (never registered) when their source entity is
        absent, so existing installs are unaffected.
        """
        if self._actual_rrp_entity_id is not None:
            self._config_entry.async_on_unload(
                async_track_state_change_event(
                    self.hass,
                    [self._actual_rrp_entity_id],
                    self._async_actual_rrp_changed,
                )
            )
            _LOGGER.debug(
                "Calibration observation listener registered for %s",
                self._actual_rrp_entity_id,
            )

        if self._load_entity_id is not None and self._load_forecaster_enabled:
            self._config_entry.async_on_unload(
                async_track_state_change_event(
                    self.hass,
                    [self._load_entity_id],
                    self._async_load_changed,
                )
            )
            _LOGGER.debug(
                "Load observation listener registered for %s",
                self._load_entity_id,
            )
        elif self._load_entity_id is not None:
            _LOGGER.debug(
                "Load entity %s configured but load forecaster disabled; "
                "load observation listener not registered",
                self._load_entity_id,
            )

    @callback
    def _async_actual_rrp_changed(self, event: Event) -> None:
        """
        Handle a new realised-RRP reading: pair it with the buffered prediction
        for the same interval-start and post import + export calibration.

        The actual-RRP entity must report the realised wholesale RRP in $/MWh
        (AEMO TRADINGPRICE convention) — the same units as the PD7DAY raw RRP we
        buffered as the prediction.
        """
        new_state = event.data.get("new_state")
        actual_rrp = _state_to_float(new_state)
        if actual_rrp is None:
            return

        observed_at = datetime.now(timezone.utc)
        interval_start = _floor_to_interval(observed_at)

        # Debounce: one calibration post per interval-start (a noisy sensor may
        # tick many times within the same 30-min interval).
        if interval_start in self._calibrated_intervals:
            return

        predicted_rrp = self._predicted_rrp_by_interval.get(interval_start)
        if predicted_rrp is None:
            # No prediction buffered for this interval yet (e.g. the sidecar has
            # not been polled since this interval began).  Skip rather than pair
            # against a mismatched interval.
            _LOGGER.debug(
                "No buffered prediction for interval %s; skipping calibration",
                interval_start.isoformat(),
            )
            return

        hour_of_day_nem = interval_start.astimezone(_NEM_TIMEZONE).hour
        self._calibrated_intervals.add(interval_start)

        # Both calibrators learn from the same realised RRP.  Export feed-in is
        # the wholesale RRP under the Amber-style model, so the same (predicted,
        # actual) pair is the correct signal for both per-hour curves; the
        # sidecar keeps them in separate calibrators.
        self.hass.async_create_task(
            self._async_post_calibration_pair(
                predicted_rrp, actual_rrp, hour_of_day_nem, observed_at
            )
        )

    async def _async_post_calibration_pair(
        self,
        predicted_rrp_per_mwh: float,
        actual_rrp_per_mwh: float,
        hour_of_day_nem: int,
        observed_at: datetime,
    ) -> None:
        """Post the import + export calibration observations (best-effort)."""
        await self.async_add_import_calibration_observation(
            predicted_rrp_per_mwh, actual_rrp_per_mwh, hour_of_day_nem, observed_at
        )
        await self.async_add_export_calibration_observation(
            predicted_rrp_per_mwh, actual_rrp_per_mwh, hour_of_day_nem, observed_at
        )
        _LOGGER.debug(
            "Posted calibration observation: predicted=%.3f actual=%.3f $/MWh "
            "hour_nem=%d",
            predicted_rrp_per_mwh,
            actual_rrp_per_mwh,
            hour_of_day_nem,
        )

    @callback
    def _async_load_changed(self, event: Event) -> None:
        """
        Handle a new house-load reading: sample it (rate-limit + significant
        delta) and post it to the sidecar as a 30-min interval observation.

        load_watts is the instantaneous reading; the sidecar buckets/averages
        per interval-start, so over-sending within an interval is harmless but
        wasteful — hence the sampling gate.
        """
        new_state = event.data.get("new_state")
        load_watts = _state_to_float(new_state)
        if load_watts is None:
            return

        now_utc = datetime.now(timezone.utc)
        if not self._should_sample_load(now_utc, load_watts):
            return

        self._last_load_sample_utc = now_utc
        self._last_load_sample_watts = load_watts

        interval_start = _floor_to_interval(now_utc)
        self.hass.async_create_task(
            self.async_add_load_observation(interval_start, load_watts)
        )

    def _should_sample_load(self, now_utc: datetime, load_watts: float) -> bool:
        """
        Rate-limit load sampling: accept at most one reading per
        LOAD_SAMPLE_MIN_INTERVAL_SECONDS, UNLESS the value moved by at least
        LOAD_SAMPLE_SIGNIFICANT_DELTA_WATTS since the last accepted sample.
        """
        if self._last_load_sample_utc is None:
            return True
        elapsed_seconds = (now_utc - self._last_load_sample_utc).total_seconds()
        if elapsed_seconds >= LOAD_SAMPLE_MIN_INTERVAL_SECONDS:
            return True
        if self._last_load_sample_watts is not None:
            if abs(load_watts - self._last_load_sample_watts) >= (
                LOAD_SAMPLE_SIGNIFICANT_DELTA_WATTS
            ):
                return True
        return False

    def _buffer_predictions(
        self, forecast_slots: list[ForecastSlot], now_utc: datetime
    ) -> None:
        """
        Record each forecast slot's PD7DAY raw RRP ($/MWh) keyed by its
        interval-start, then evict stale entries by TTL and overall size.

        The buffered value is the prediction the integration last emitted for
        that interval; it is paired with the realised actual when it arrives.
        """
        for forecast_slot in forecast_slots:
            self._predicted_rrp_by_interval[forecast_slot.interval_start_utc] = (
                forecast_slot.raw_rrp_per_mwh
            )

        ttl_cutoff = now_utc - timedelta(hours=PREDICTION_BUFFER_TTL_HOURS)
        stale_keys = [
            interval_start
            for interval_start in self._predicted_rrp_by_interval
            if interval_start < ttl_cutoff
        ]
        for interval_start in stale_keys:
            self._predicted_rrp_by_interval.pop(interval_start, None)
            self._calibrated_intervals.discard(interval_start)

        # Hard size cap (defensive — drop the oldest interval-starts first).
        if len(self._predicted_rrp_by_interval) > PREDICTION_BUFFER_MAX_ENTRIES:
            for interval_start in sorted(self._predicted_rrp_by_interval)[
                : len(self._predicted_rrp_by_interval) - PREDICTION_BUFFER_MAX_ENTRIES
            ]:
                self._predicted_rrp_by_interval.pop(interval_start, None)
                self._calibrated_intervals.discard(interval_start)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def async_close(self) -> None:
        """Close the HTTP session. Called on config entry unload."""
        await self._sidecar_client.async_close()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_price_slots(raw_list: list[dict[str, Any]]) -> list[ForecastSlot]:
    """Parse the raw_forecast[] from /price_forecast into ForecastSlot objects."""
    slots: list[ForecastSlot] = []
    for raw_item in raw_list:
        try:
            interval_start = _parse_iso_or_now(raw_item.get("interval_start"))
            # interval_end is the PERIOD-ENDING (published) stamp.  Fall back to
            # start + 30 min (the native PD7DAY period) for older sidecars that
            # don't yet emit interval_end.
            interval_end_raw = raw_item.get("interval_end")
            if interval_end_raw is not None:
                interval_end = _parse_iso_or_now(interval_end_raw)
            else:
                interval_end = interval_start + timedelta(minutes=30)
            slots.append(
                ForecastSlot(
                    interval_start_utc=interval_start,
                    interval_end_utc=interval_end,
                    raw_rrp_per_mwh=float(raw_item.get("raw_rrp_per_mwh", 0.0)),
                    calibrated_wholesale_kwh=float(
                        raw_item.get("calibrated_wholesale_kwh", 0.0)
                    ),
                    import_price_kwh=float(raw_item.get("import_price_kwh", 0.0)),
                    export_price_kwh=float(raw_item.get("export_price_kwh", 0.0)),
                    network_tou_rate_kwh=float(raw_item.get("network_tou_rate_kwh", 0.0)),
                )
            )
        except (KeyError, ValueError, TypeError) as parse_error:
            _LOGGER.debug("Skipping malformed price slot: %s", parse_error)
    return slots


def _parse_load_slots(raw_list: list[dict[str, Any]]) -> list[LoadForecastSlot]:
    """Parse the raw_forecast[] from /load_forecast into LoadForecastSlot objects."""
    slots: list[LoadForecastSlot] = []
    for raw_item in raw_list:
        try:
            interval_start = _parse_iso_or_now(raw_item.get("interval_start"))
            slots.append(
                LoadForecastSlot(
                    interval_start_utc=interval_start,
                    load_watts=float(raw_item.get("load_watts", 0.0)),
                )
            )
        except (KeyError, ValueError, TypeError) as parse_error:
            _LOGGER.debug("Skipping malformed load slot: %s", parse_error)
    return slots


def _parse_iso_or_now(iso_string: Optional[str]) -> datetime:
    if iso_string is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


def _state_to_float(state: Any) -> Optional[float]:
    """
    Coerce an HA State object's value to a finite float, or None.

    Returns None for missing/unknown/unavailable states or non-numeric values so
    the caller can simply skip them.
    """
    if state is None:
        return None
    raw_value = getattr(state, "state", None)
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip().lower() in _UNUSABLE_STATES:
        return None
    try:
        numeric = float(raw_value)
    except (ValueError, TypeError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return numeric


def _floor_to_interval(moment_utc: datetime) -> datetime:
    """
    Snap a UTC datetime DOWN to the start of its NEM dispatch interval.

    PD7DAY intervals are period-beginning and 30-min wide.  Flooring in UTC is
    correct because the NEM offset (UTC+10, no DST) is a whole number of hours,
    so 30-min boundaries align in both zones.
    """
    discard_minutes = moment_utc.minute % _NEM_INTERVAL_MINUTES
    return moment_utc.replace(second=0, microsecond=0) - timedelta(
        minutes=discard_minutes
    )
