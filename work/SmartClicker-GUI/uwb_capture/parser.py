"""Serial line parsing for IMEC UWB firmware output."""

from __future__ import annotations

import re

from .common import ParsedRecord, safe_float, safe_int

def _float_at(parts: list[str], index: int) -> float | None:
    return safe_float(parts[index]) if len(parts) > index else None


def _int_at(parts: list[str], index: int) -> int | None:
    return safe_int(parts[index]) if len(parts) > index else None


def parse_serial_line(line: str) -> list[ParsedRecord]:
    raw_line = line.rstrip("\r\n")
    stripped = raw_line.strip()
    if not stripped:
        return []

    parts = [part.strip() for part in stripped.split(",")]
    tag = parts[0].upper() if parts else ""
    records: list[ParsedRecord] = []

    try:
        if tag == "S" and len(parts) >= 4:
            records.append(
                ParsedRecord(
                    kind="sample",
                    anchor_id=parts[1],
                    sample_index=_int_at(parts, 2),
                    distance_m=_float_at(parts, 3),
                    status="ok",
                    source=parts[4] if len(parts) > 4 and parts[4] else "serial_csv",
                    raw_line=raw_line,
                )
            )
        elif tag == "F" and len(parts) >= 3:
            records.append(
                ParsedRecord(
                    kind="failure",
                    anchor_id=parts[1],
                    sample_index=_int_at(parts, 2),
                    status="failure",
                    error_code=parts[3] if len(parts) > 3 else "",
                    source="serial_csv",
                    raw_line=raw_line,
                )
            )
        elif tag == "M" and len(parts) >= 4:
            records.append(
                ParsedRecord(
                    kind="summary",
                    anchor_id=parts[1],
                    mean_distance_m=_float_at(parts, 2),
                    good_count=_int_at(parts, 3),
                    distance_m=_float_at(parts, 6),
                    status="summary",
                    source="serial_csv",
                    raw_line=raw_line,
                )
            )
        elif tag == "Q" and len(parts) >= 3:
            status = "quality_invalid" if len(parts) > 2 and parts[2].lower() == "invalid" else "quality"
            records.append(
                ParsedRecord(
                    kind="quality",
                    anchor_id=parts[1] if len(parts) > 1 else None,
                    rx_power_dbm=_float_at(parts, 2) if status == "quality" else None,
                    fp_power_dbm=_float_at(parts, 3) if status == "quality" else None,
                    cir_power=_float_at(parts, 11) if status == "quality" else None,
                    status=status,
                    source="serial_quality",
                    raw_line=raw_line,
                )
            )
        elif tag == "CIR" and len(parts) >= 5:
            records.append(
                ParsedRecord(
                    kind="cir",
                    anchor_id=parts[1],
                    cir_raw=raw_line,
                    status="cir",
                    source="serial_cir",
                    raw_line=raw_line,
                )
            )
        elif tag == "NLOS" and len(parts) >= 3:
            status = "nlos_invalid" if parts[2].lower() == "invalid" else "nlos"
            records.append(
                ParsedRecord(
                    kind="nlos",
                    anchor_id=parts[1],
                    status=status,
                    error_code=parts[3] if status == "nlos_invalid" and len(parts) > 3 else None,
                    source="serial_nlos",
                    raw_line=raw_line,
                )
            )
        elif tag == "PHY" and len(parts) >= 9:
            records.append(
                ParsedRecord(
                    kind="phy",
                    status="phy",
                    source="serial_phy",
                    raw_line=raw_line,
                )
            )
        elif tag == "FAIL" and len(parts) >= 2:
            records.append(
                ParsedRecord(
                    kind="failure",
                    anchor_id=parts[1],
                    status="failure",
                    error_code="FAIL",
                    source="serial_csv",
                    raw_line=raw_line,
                )
            )
        else:
            # A small compatibility parser for older text logs.
            distance_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*m", stripped, re.I)
            anchor_match = re.search(r"anchor\s*[:#-]?\s*([A-Za-z0-9_.:-]+)", stripped, re.I)
            if distance_match and anchor_match:
                records.append(
                    ParsedRecord(
                        kind="sample",
                        anchor_id=anchor_match.group(1),
                        distance_m=float(distance_match.group(1)),
                        status="ok",
                        source="serial_text",
                        raw_line=raw_line,
                    )
                )
    except Exception:
        records.append(
            ParsedRecord(
                kind="failure",
                status="parse_error",
                error_code="parse_error",
                source="parser",
                raw_line=raw_line,
            )
        )

    return records
