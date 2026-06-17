"""
NEM Price Forecaster — Home Assistant custom integration (sidecar mode).

The integration is now a thin HTTP client for the NEM forecaster sidecar.
All ML compute (PD7DAY fetch, isotonic/Darts calibration, tariff calculation,
load forecasting) runs in the sidecar container.

Python 3.14 compatible: no darts, no sklearn, no scipy imported here.
Only numpy + homeassistant builtins + aiohttp (bundled with HA).

Sidecar:
  See sidecar/ directory for the Dockerfile and docker-compose.yml.
  Default URL: http://localhost:8765

Sensors published:
  - sensor.nem_price_forecaster_{region}_import_price  ($/kWh, GST-inclusive)
  - sensor.nem_price_forecaster_{region}_export_price  ($/kWh, GST-excluded)
  - sensor.nem_price_forecaster_{region}_load_forecast (W, optional)

See README.md for full setup instructions.
"""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import (
    CONF_CALIBRATOR,
    CONF_PRICE_MODEL,
    CONF_SIDECAR_URL,
    DEFAULT_CALIBRATOR,
    DEFAULT_PRICE_MODEL,
    DEFAULT_SIDECAR_URL,
    DOMAIN,
    PLATFORMS,
)
from .coordinator import NemPriceForecastCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Set up NEM Price Forecaster from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    coordinator = NemPriceForecastCoordinator(hass, config_entry)

    # Perform an initial data fetch from the sidecar.
    # Raises ConfigEntryNotReady if the sidecar is unreachable so HA retries.
    await coordinator.async_config_entry_first_refresh()

    hass.data[DOMAIN][config_entry.entry_id] = coordinator

    # Wire the optional observation listeners (issue #9): realised-RRP ->
    # calibration, live house load -> load forecaster.  No-ops when the
    # corresponding source entity is not configured, so existing installs are
    # unaffected.  Teardown is registered via config_entry.async_on_unload().
    coordinator.async_setup_observation_listeners()

    # Apply the stored price_model / calibrator to the sidecar so the runtime
    # config stays consistent with the integration's choice.  This runs on every
    # setup, which also covers options-flow changes (the update listener reloads
    # the entry, re-running setup).  Best-effort — a transiently-down sidecar
    # does not block setup.
    await _async_apply_model_config(config_entry)

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    # Register update listener so options-flow changes trigger a reload
    config_entry.async_on_unload(
        config_entry.add_update_listener(_async_update_listener)
    )

    return True


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry — close the HTTP session."""
    coordinator: NemPriceForecastCoordinator = hass.data[DOMAIN].get(
        config_entry.entry_id
    )
    if coordinator is not None:
        await coordinator.async_close()

    unload_ok = await hass.config_entries.async_unload_platforms(
        config_entry, PLATFORMS
    )
    if unload_ok:
        hass.data[DOMAIN].pop(config_entry.entry_id)
    return unload_ok


async def _async_update_listener(
    hass: HomeAssistant, config_entry: ConfigEntry
) -> None:
    """Handle options update: reload the entry so the new tariff takes effect."""
    await hass.config_entries.async_reload(config_entry.entry_id)


async def _async_apply_model_config(config_entry: ConfigEntry) -> None:
    """
    Push the entry's stored price_model + calibrator to the sidecar (best-effort).

    Options take precedence over the initial install data.  Never raises — a
    sidecar that is briefly unreachable must not break entry setup.
    """
    merged = {**config_entry.data, **config_entry.options}
    price_model = merged.get(CONF_PRICE_MODEL, DEFAULT_PRICE_MODEL)
    calibrator = merged.get(CONF_CALIBRATOR, DEFAULT_CALIBRATOR)
    sidecar_url = merged.get(CONF_SIDECAR_URL, DEFAULT_SIDECAR_URL)

    try:
        from .sidecar_client import SidecarClient, SidecarUnavailable

        client = SidecarClient(sidecar_url)
        try:
            await client.async_post_config(
                price_model=price_model, calibrator=calibrator
            )
            _LOGGER.debug(
                "Applied stored model config to sidecar: price_model=%s calibrator=%s",
                price_model,
                calibrator,
            )
        except SidecarUnavailable as apply_error:
            _LOGGER.debug(
                "Could not apply model config to sidecar (non-fatal): %s",
                apply_error,
            )
        finally:
            await client.async_close()
    except Exception as unexpected:  # pragma: no cover - never block setup
        _LOGGER.debug("Sidecar model-config apply skipped: %s", unexpected)
