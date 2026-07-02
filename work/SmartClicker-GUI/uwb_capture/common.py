"""Shared models and small helpers for UWB capture."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

MAX_TABLE_ROWS = 400
ALERT_PROMPT_COOLDOWN_SECONDS = 15
INSTABILITY_WINDOW_SIZE = 12
DEFAULT_DB_NAME = "uwb_measurements.sqlite"


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def export_timestamp(now: datetime | None = None) -> str:
    stamp = now or datetime.now().astimezone()
    return stamp.strftime("%Y%m%d_%H%M%S")


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "session"


@dataclass
class ParsedRecord:
    kind: str
    anchor_id: str | None = None
    peer_anchor_id: str | None = None
    clicker_id: str | None = None
    sample_index: int | None = None
    distance_m: float | None = None
    mean_distance_m: float | None = None
    good_count: int | None = None
    rx_power_dbm: float | None = None
    fp_power_dbm: float | None = None
    cir_power: float | None = None
    cir_raw: str | None = None
    status: str | None = None
    error_code: str | None = None
    source: str | None = None
    raw_line: str | None = None
    # ML clicker diagnostic fields.
    event_seq: int | None = None
    scheduled_sample_count: int | None = None
    quality: int | None = None
    firmware_timestamp_ms: int | None = None
    phy_config_id: int | None = None
    burst_id: int | None = None
    tlv_json: str | None = None
    exchange_stride_us: int | None = None
    burst_duration_ms: int | None = None
    diag_status_flags: int | None = None
    diag_bytes_captured: int | None = None
    diag_bytes_transmitted: int | None = None
    report_fragment_count: int | None = None
    uwb_clock_offset_raw: int | None = None
    uwb_carrier_integrator: int | None = None
    clicker_diag_bytes: str | None = None
    cir_first_path_index: int | None = None
    cir_start_index: int | None = None
    diag_source: int | None = None
