"""IMEC ML Clicker BLE packet codec and report conversion helpers.

The ML clicker exposes a byte-stream BLE GATT characteristic.  Each protocol
frame is COBS encoded and terminated with a single 0x00 byte.  This module
deals with the decoded ``proto_packet`` (header + TLV payload + CRC); the COBS
framing itself is owned by :class:`ImecPacketStream` and the BLE transport.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable

from .common import ParsedRecord

MAGIC = 0xC1
VERSION = 0x01
HEADER_SIZE = 32
CRC_SIZE = 2
MAX_PAYLOAD_SIZE = 255
BROADCAST_ID = 0
DEFAULT_TTL = 1

FLAG_DIAGNOSTIC = 0x10
FLAG_ERROR = 0x40

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


class ProtocolError(ValueError):
    """Raised when a packet cannot be encoded or decoded safely."""


class MessageType(IntEnum):
    CLICK_REPORT = 0x20
    COMMAND = 0x40
    COMMAND_RESULT = 0x41
    SURVEY_REACH_REPORT = 0x51
    MSG_ERROR = 0x7F


class CommandId(IntEnum):
    GET_STATUS = 0x0002
    START_HEARTBEAT = 0x0009
    STOP_HEARTBEAT = 0x000A
    SURVEY_REACHABILITY = 0x0100
    SURVEY_PREPARE_PAIR = 0x0101
    SURVEY_START_PAIR = 0x0102
    SURVEY_ABORT = 0x0103
    ML_START_COLLECTION = 0x8000
    ML_START_FAST_RANGING = 0x8001
    ML_START_ANCHOR_PAIR_SURVEY = 0x8002
    ML_START_LIVE_TRACKING = 0x8003
    ML_LIVE_TRACKING_HEARTBEAT = 0x8004
    ML_STOP_LIVE_TRACKING = 0x8005


class CommandStatus(IntEnum):
    COMMAND_OK = 0
    COMMAND_UNSUPPORTED = 1
    COMMAND_MALFORMED = 2
    COMMAND_BUSY = 3
    COMMAND_DENIED = 4
    COMMAND_TIMEOUT = 5
    COMMAND_RADIO_ERROR = 6
    COMMAND_INVALID_STATE = 7
    COMMAND_INTERNAL_ERROR = 8


class RangeStatus(IntEnum):
    RANGE_OK = 0
    RANGE_RX_TIMEOUT = 1
    RANGE_RX_ERROR = 2
    RANGE_BAD_FRAME = 3
    RANGE_WRONG_TARGET = 4
    RANGE_STS_QUALITY_FAIL = 5
    RANGE_DELAYED_TX_MISSED = 6
    RANGE_INTERNAL_ERROR = 7
    RANGE_TIMING_INVALID = 8


class TlvId(IntEnum):
    EVENT_SEQ = 0x06
    TIMESTAMP_MS = 0x07
    ANCHOR_ID = 0x0A
    CLICKER_ID = 0x0B
    DISTANCE_MM = 0x0C
    QUALITY = 0x0D
    SAMPLE_INDEX = 0x0E
    SAMPLE_COUNT = 0x0F
    COMMAND_ID = 0x10
    COMMAND_STATUS = 0x11
    SURVEY_ID = 0x15
    DURATION_MS = 0x1A
    REASON = 0x1E
    INITIATOR_ID = 0x1F
    RESPONDER_ID = 0x20
    RANGE_STATUS = 0x21
    UWB_RSL_DBM = 0x24
    DISTANCE_SAMPLES_MM = 0x25
    UWB_CIR_SAMPLE = 0x26
    RANGE_ROUND_INDICES = 0x28
    SEQUENCE_START_TIMESTAMPS_MS = 0x29
    DIAG_STATUS_FLAGS = 0x33
    BURST_ID = 0x34
    EXCHANGE_STRIDE_US = 0x35
    BURST_DURATION_MS = 0x36
    DIAG_RX_FRAME_INFO = 0x37
    DIAG_TX_FRAME_INFO = 0x38
    DIAG_BYTES_CAPTURED = 0x39
    DIAG_BYTES_TRANSMITTED = 0x3A
    DIAG_BYTES_TRUNCATED = 0x3B
    DIAG_FRAMES_DROPPED = 0x3C
    REPORT_FRAGMENT_COUNT = 0x3D
    CLICKER_DIAG_BYTES = 0x40
    ANCHOR_DIAG_BYTES = 0x41
    PHY_CONFIG_ID = 0x42
    DISCOVERY_SLOT_COUNT = 0x4C
    UWB_CLOCK_OFFSET_RAW = 0x4D
    UWB_CARRIER_INTEGRATOR = 0x4E
    UWB_CIR_FULL_CHUNK = 0x4F
    UWB_CIR_BYTE_OFFSET = 0x50
    UWB_CIR_TOTAL_BYTES = 0x51
    UWB_CIR_FIRST_PATH_INDEX = 0x52
    UWB_RAW_TIMESTAMPS = 0x53
    UWB_RX_DIAG_BYTES = 0x54
    DIAG_FRAGMENT_INDEX = 0x55
    DIAG_FRAGMENT_COUNT = 0x56
    DIAG_SOURCE = 0x57
    UWB_CIR_START_INDEX = 0x58


@dataclass(frozen=True)
class ImecPacket:
    msg_type: int
    flags: int
    source_id: int
    destination_id: int
    session_id: int
    sequence: int
    ttl: int
    message_age_ms: int
    payload: bytes


class ImecPacketStream:
    """Accumulates BLE notification fragments and extracts COBS-delimited frames.

    The firmware treats the BLE characteristic as a byte stream: a complete
    protocol frame is COBS encoded and terminated with a zero byte.  This class
    buffers incoming bytes until it sees ``0x00``, then decodes one COBS frame
    into the raw ``proto_packet`` bytes (header + TLV payload + CRC).
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        packets: list[bytes] = []
        while True:
            zero = self._buffer.find(0x00)
            if zero < 0:
                return packets
            frame = bytes(self._buffer[:zero])
            del self._buffer[: zero + 1]
            if not frame:
                continue
            try:
                decoded = cobs_decode(frame)
            except ProtocolError:
                continue
            packets.append(decoded)


def cobs_encode(data: bytes) -> bytes:
    """COBS-encode ``data`` (no trailing zero byte)."""
    output = bytearray()
    output.append(1)
    code_index = 0
    code = 1
    for byte in data:
        if byte:
            output.append(byte)
            code += 1
            if code == 0xFF:
                output[code_index] = code
                code = 1
                code_index = len(output)
                output.append(1)
        else:
            output[code_index] = code
            code = 1
            code_index = len(output)
            output.append(1)
    output[code_index] = code
    return bytes(output)


def cobs_decode(data: bytes) -> bytes:
    if not data:
        raise ProtocolError("Empty COBS frame")
    output = bytearray()
    index = 0
    while index < len(data):
        code = data[index]
        if code == 0 or index + code > len(data):
            raise ProtocolError("Invalid COBS frame")
        index += 1
        chunk = data[index : index + code - 1]
        output.extend(chunk)
        index += code - 1
        if code < 0xFF and index < len(data):
            output.append(0)
    return bytes(output)


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_tlv(tlv_id: int | TlvId, value: bytes) -> bytes:
    if len(value) > 255:
        raise ProtocolError(f"TLV {int(tlv_id):#x} is longer than 255 bytes")
    return bytes([int(tlv_id), len(value)]) + value


def encode_tlvs(items: Iterable[tuple[int | TlvId, bytes]]) -> bytes:
    return b"".join(encode_tlv(tlv_id, value) for tlv_id, value in items)


def decode_tlvs(payload: bytes) -> dict[int, list[bytes]]:
    tlvs: dict[int, list[bytes]] = {}
    index = 0
    while index < len(payload):
        if index + 2 > len(payload):
            raise ProtocolError("Truncated TLV header")
        tlv_id = payload[index]
        length = payload[index + 1]
        index += 2
        if index + length > len(payload):
            raise ProtocolError(f"Truncated TLV {tlv_id:#x}")
        tlvs.setdefault(tlv_id, []).append(payload[index : index + length])
        index += length
    return tlvs


def encode_packet(packet: ImecPacket) -> bytes:
    if len(packet.payload) > MAX_PAYLOAD_SIZE:
        raise ProtocolError("Packet payload is longer than 255 bytes")
    header = struct.pack(
        "<BBBBQQIHBBI",
        MAGIC,
        VERSION,
        int(packet.msg_type) & 0xFF,
        int(packet.flags) & 0xFF,
        packet.source_id & MASK64,
        packet.destination_id & MASK64,
        packet.session_id & MASK32,
        packet.sequence & 0xFFFF,
        packet.ttl & 0xFF,
        len(packet.payload),
        packet.message_age_ms & MASK32,
    )
    body = header + packet.payload
    return body + struct.pack("<H", crc16_ccitt_false(body))


def decode_packet(data: bytes) -> ImecPacket:
    if len(data) < HEADER_SIZE + CRC_SIZE:
        raise ProtocolError("Packet is too short")
    if data[0] != MAGIC:
        raise ProtocolError("Bad packet magic")
    if data[1] != VERSION:
        raise ProtocolError(f"Unsupported packet version: {data[1]}")
    payload_len = data[27]
    expected_len = HEADER_SIZE + payload_len + CRC_SIZE
    if len(data) != expected_len:
        raise ProtocolError(f"Packet length mismatch: expected {expected_len}, got {len(data)}")
    actual_crc = struct.unpack_from("<H", data, expected_len - CRC_SIZE)[0]
    expected_crc = crc16_ccitt_false(data[: expected_len - CRC_SIZE])
    if actual_crc != expected_crc:
        raise ProtocolError("Packet CRC mismatch")
    fields = struct.unpack("<BBBBQQIHBBI", data[:HEADER_SIZE])
    return ImecPacket(
        msg_type=fields[2],
        flags=fields[3],
        source_id=fields[4],
        destination_id=fields[5],
        session_id=fields[6],
        sequence=fields[7],
        ttl=fields[8],
        message_age_ms=fields[10],
        payload=data[HEADER_SIZE : HEADER_SIZE + payload_len],
    )


def u8(value: int) -> bytes:
    return struct.pack("<B", value & 0xFF)


def u16(value: int) -> bytes:
    return struct.pack("<H", value & 0xFFFF)


def u32(value: int) -> bytes:
    return struct.pack("<I", value & MASK32)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value & MASK64)


def i32(value: int) -> bytes:
    return struct.pack("<i", int(value))


def first_tlv(tlvs: dict[int, list[bytes]], tlv_id: int | TlvId) -> bytes | None:
    values = tlvs.get(int(tlv_id), [])
    return values[0] if values else None


def read_uint(data: bytes | None) -> int | None:
    if data is None:
        return None
    if len(data) == 1:
        return struct.unpack("<B", data)[0]
    if len(data) == 2:
        return struct.unpack("<H", data)[0]
    if len(data) == 4:
        return struct.unpack("<I", data)[0]
    if len(data) == 8:
        return struct.unpack("<Q", data)[0]
    return int.from_bytes(data, "little", signed=False)


def read_int(data: bytes | None) -> int | None:
    if data is None:
        return None
    if len(data) == 1:
        return struct.unpack("<b", data)[0]
    if len(data) == 2:
        return struct.unpack("<h", data)[0]
    if len(data) == 4:
        return struct.unpack("<i", data)[0]
    if len(data) == 8:
        return struct.unpack("<q", data)[0]
    return int.from_bytes(data, "little", signed=True)


def parse_device_id(value: str, *, default: int | None = None, bits: int = 64) -> int:
    text = str(value or "").strip()
    if not text:
        if default is not None:
            return default
        raise ProtocolError("Missing device ID")
    cleaned = text.replace("_", "").replace(":", "").replace("-", "")
    try:
        if cleaned.lower().startswith("0x"):
            number = int(cleaned, 16)
        elif any(char in "abcdefABCDEF" for char in cleaned):
            number = int(cleaned, 16)
        else:
            number = int(cleaned, 10)
    except ValueError as exc:
        raise ProtocolError(f"Invalid device ID: {value}") from exc
    max_value = (1 << bits) - 1
    if number < 0 or number > max_value:
        raise ProtocolError(f"Device ID is outside {bits}-bit range: {value}")
    return number


def format_device_id(value: int | None) -> str:
    if value is None:
        return ""
    return f"0x{int(value) & MASK64:016X}"


def next_sequence(sequence: int) -> int:
    return 1 if sequence >= 0xFFFF else sequence + 1


def build_command_packet(
    *,
    command_id: int | CommandId,
    source_id: int,
    destination_id: int,
    session_id: int,
    sequence: int,
    extra_tlvs: Iterable[tuple[int | TlvId, bytes]] = (),
    ttl: int = DEFAULT_TTL,
    message_age_ms: int = 0,
) -> bytes:
    payload = encode_tlvs(
        [(TlvId.COMMAND_ID, u16(int(command_id))), *list(extra_tlvs)]
    )
    packet = ImecPacket(
        msg_type=MessageType.COMMAND,
        flags=0,
        source_id=source_id,
        destination_id=destination_id,
        session_id=session_id,
        sequence=sequence,
        ttl=ttl,
        message_age_ms=message_age_ms,
        payload=payload,
    )
    return encode_packet(packet)


def _build_ml_start_packet(
    *,
    command_id: CommandId,
    source_id: int,
    destination_id: int,
    session_id: int,
    sequence: int,
    sample_count: int | None = None,
    discovery_slot_count: int | None = None,
) -> bytes:
    extra: list[tuple[int | TlvId, bytes]] = []
    if sample_count is not None:
        if not (1 <= sample_count <= 100):
            raise ProtocolError("Sample count must be from 1 to 100.")
        extra.append((TlvId.SAMPLE_COUNT, u8(sample_count)))
    if discovery_slot_count is not None:
        if not (1 <= discovery_slot_count <= 8):
            raise ProtocolError("Discovery slot count must be from 1 to 8.")
        extra.append((TlvId.DISCOVERY_SLOT_COUNT, u8(discovery_slot_count)))
    return build_command_packet(
        command_id=command_id,
        source_id=source_id,
        destination_id=destination_id,
        session_id=session_id,
        sequence=sequence,
        extra_tlvs=extra,
    )


def build_ml_start_collection_packet(
    *,
    source_id: int,
    destination_id: int,
    session_id: int,
    sequence: int,
    sample_count: int | None = None,
    discovery_slot_count: int | None = None,
) -> bytes:
    """Build a ``CMD_ML_START_COLLECTION`` command proto_packet."""
    return _build_ml_start_packet(
        command_id=CommandId.ML_START_COLLECTION,
        source_id=source_id,
        destination_id=destination_id,
        session_id=session_id,
        sequence=sequence,
        sample_count=sample_count,
        discovery_slot_count=discovery_slot_count,
    )


def build_ml_start_fast_ranging_packet(
    *,
    source_id: int,
    destination_id: int,
    session_id: int,
    sequence: int,
    sample_count: int | None = None,
    discovery_slot_count: int | None = None,
) -> bytes:
    """Build a ``CMD_ML_START_FAST_RANGING`` command proto_packet."""
    return _build_ml_start_packet(
        command_id=CommandId.ML_START_FAST_RANGING,
        source_id=source_id,
        destination_id=destination_id,
        session_id=session_id,
        sequence=sequence,
        sample_count=sample_count,
        discovery_slot_count=discovery_slot_count,
    )


def build_ml_start_live_tracking_packet(
    *,
    source_id: int,
    destination_id: int,
    session_id: int,
    sequence: int,
    sample_count: int | None = None,
    discovery_slot_count: int | None = None,
    duration_ms: int | None = None,
) -> bytes:
    """Build a ``CMD_ML_START_LIVE_TRACKING`` command proto_packet."""

    extra: list[tuple[int | TlvId, bytes]] = []
    if sample_count is not None:
        if not (1 <= sample_count <= 100):
            raise ProtocolError("Live tracking sample count must be from 1 to 100.")
        extra.append((TlvId.SAMPLE_COUNT, u8(sample_count)))
    if discovery_slot_count is not None:
        if not (1 <= discovery_slot_count <= 8):
            raise ProtocolError("Live tracking discovery slots must be from 1 to 8.")
        extra.append((TlvId.DISCOVERY_SLOT_COUNT, u8(discovery_slot_count)))
    if duration_ms is not None:
        if not (500 <= duration_ms <= 10000):
            raise ProtocolError("Live tracking watchdog duration must be from 500 to 10000 ms.")
        extra.append((TlvId.DURATION_MS, u32(duration_ms)))
    return build_command_packet(
        command_id=CommandId.ML_START_LIVE_TRACKING,
        source_id=source_id,
        destination_id=destination_id,
        session_id=session_id,
        sequence=sequence,
        extra_tlvs=extra,
    )


def build_ml_live_tracking_heartbeat_packet(
    *,
    source_id: int,
    destination_id: int,
    session_id: int,
    sequence: int,
) -> bytes:
    """Build a ``CMD_ML_LIVE_TRACKING_HEARTBEAT`` command proto_packet."""

    return build_command_packet(
        command_id=CommandId.ML_LIVE_TRACKING_HEARTBEAT,
        source_id=source_id,
        destination_id=destination_id,
        session_id=session_id,
        sequence=sequence,
    )


def build_ml_stop_live_tracking_packet(
    *,
    source_id: int,
    destination_id: int,
    session_id: int,
    sequence: int,
) -> bytes:
    """Build a ``CMD_ML_STOP_LIVE_TRACKING`` command proto_packet."""

    return build_command_packet(
        command_id=CommandId.ML_STOP_LIVE_TRACKING,
        source_id=source_id,
        destination_id=destination_id,
        session_id=session_id,
        sequence=sequence,
    )


def build_ml_start_anchor_pair_survey_packet(
    *,
    source_id: int,
    destination_id: int,
    session_id: int,
    sequence: int,
    discovery_slot_count: int | None = None,
) -> bytes:
    """Build a ``CMD_ML_START_ANCHOR_PAIR_SURVEY`` command proto_packet."""

    extra: list[tuple[int | TlvId, bytes]] = []
    if discovery_slot_count is not None:
        if not (2 <= discovery_slot_count <= 8):
            raise ProtocolError("Anchor pair survey discovery slots must be from 2 to 8.")
        extra.append((TlvId.DISCOVERY_SLOT_COUNT, u8(discovery_slot_count)))
    return build_command_packet(
        command_id=CommandId.ML_START_ANCHOR_PAIR_SURVEY,
        source_id=source_id,
        destination_id=destination_id,
        session_id=session_id,
        sequence=sequence,
        extra_tlvs=extra,
    )


def packet_summary(packet: ImecPacket) -> str:
    try:
        name = MessageType(packet.msg_type).name
    except ValueError:
        name = f"0x{packet.msg_type:02X}"
    flags = f" flags=0x{packet.flags:02X}" if packet.flags else ""
    return (
        f"{name} src={format_device_id(packet.source_id)} "
        f"dst={format_device_id(packet.destination_id)} "
        f"session={packet.session_id} seq={packet.sequence}{flags}"
    )


def command_result_summary(packet: ImecPacket) -> str:
    tlvs = decode_tlvs(packet.payload)
    command_id = read_uint(first_tlv(tlvs, TlvId.COMMAND_ID))
    status = read_uint(first_tlv(tlvs, TlvId.COMMAND_STATUS))
    reason = read_uint(first_tlv(tlvs, TlvId.REASON))
    event_seq = read_uint(first_tlv(tlvs, TlvId.EVENT_SEQ))
    sample_count = read_uint(first_tlv(tlvs, TlvId.SAMPLE_COUNT))
    command_name = _enum_name(CommandId, command_id, "CMD")
    status_name = _enum_name(CommandStatus, status, "COMMAND_STATUS")
    parts = [f"{command_name}: {status_name}"]
    if reason is not None:
        parts.append(f"reason={reason}")
    if event_seq is not None:
        parts.append(f"event_seq={event_seq}")
    if sample_count is not None:
        parts.append(f"samples={sample_count}")
    if packet.flags & FLAG_ERROR:
        parts.append("FLAG_ERROR")
    return ", ".join(parts)


def command_status_from_packet(packet: ImecPacket) -> int | None:
    if packet.msg_type != MessageType.COMMAND_RESULT:
        return None
    tlvs = decode_tlvs(packet.payload)
    return read_uint(first_tlv(tlvs, TlvId.COMMAND_STATUS))


def command_sample_count_from_packet(packet: ImecPacket) -> int | None:
    """Number of sample notifications the clicker reports for a command result."""
    if packet.msg_type != MessageType.COMMAND_RESULT:
        return None
    tlvs = decode_tlvs(packet.payload)
    return read_uint(first_tlv(tlvs, TlvId.SAMPLE_COUNT))


def records_from_packet(packet: ImecPacket) -> list[ParsedRecord]:
    if packet.msg_type == MessageType.CLICK_REPORT:
        return _ml_sample_records(packet)
    if packet.msg_type == MessageType.SURVEY_REACH_REPORT:
        return _anchor_pair_survey_records(packet, decode_tlvs(packet.payload))
    return []


def _anchor_pair_survey_records(
    packet: ImecPacket,
    tlvs: dict[int, list[bytes]],
) -> list[ParsedRecord]:
    initiator_id = read_uint(first_tlv(tlvs, TlvId.INITIATOR_ID))
    responder_id = read_uint(first_tlv(tlvs, TlvId.RESPONDER_ID))
    if initiator_id is None or responder_id is None:
        return []

    distance_mm = read_int(first_tlv(tlvs, TlvId.DISTANCE_MM))
    range_status = read_uint(first_tlv(tlvs, TlvId.RANGE_STATUS))
    clicker_id = read_uint(first_tlv(tlvs, TlvId.CLICKER_ID))
    survey_id = read_uint(first_tlv(tlvs, TlvId.SURVEY_ID))
    sample_index = read_uint(first_tlv(tlvs, TlvId.SAMPLE_INDEX))
    sample_count = read_uint(first_tlv(tlvs, TlvId.SAMPLE_COUNT))
    event_seq = read_uint(first_tlv(tlvs, TlvId.EVENT_SEQ))
    timestamp_ms = read_uint(first_tlv(tlvs, TlvId.TIMESTAMP_MS))
    quality = read_uint(first_tlv(tlvs, TlvId.QUALITY))
    rx_power = read_int(first_tlv(tlvs, TlvId.UWB_RSL_DBM))

    if distance_mm is None:
        return []
    ok = range_status in (None, 0)
    distance_m = distance_mm / 1000.0 if distance_mm >= 0 and ok else None
    raw_summary = packet_summary(packet)
    if survey_id is not None:
        raw_summary = f"{raw_summary} survey={survey_id}"

    return [
        ParsedRecord(
            kind="survey_pair" if ok else "survey_pair_failure",
            anchor_id=format_device_id(initiator_id),
            peer_anchor_id=format_device_id(responder_id),
            clicker_id=format_device_id(clicker_id) if clicker_id is not None else None,
            sample_index=sample_index,
            distance_m=distance_m,
            scheduled_sample_count=sample_count,
            event_seq=event_seq,
            firmware_timestamp_ms=timestamp_ms,
            quality=quality,
            rx_power_dbm=float(rx_power) if rx_power is not None else None,
            status=_range_status_text(range_status),
            error_code=None if ok else str(range_status),
            source="anchor_pair_survey",
            raw_line=raw_summary,
            tlv_json=_tlvs_to_json(tlvs),
        )
    ]


def _ml_sample_records(packet: ImecPacket) -> list[ParsedRecord]:
    tlvs = decode_tlvs(packet.payload)
    anchor_pair_records = _anchor_pair_survey_records(packet, tlvs)
    if anchor_pair_records:
        return anchor_pair_records

    anchor_id = read_uint(first_tlv(tlvs, TlvId.ANCHOR_ID))
    clicker_id = read_uint(first_tlv(tlvs, TlvId.CLICKER_ID))
    sample_index_raw = first_tlv(tlvs, TlvId.SAMPLE_INDEX)
    sample_index = read_uint(sample_index_raw)
    scheduled_count = read_uint(first_tlv(tlvs, TlvId.SAMPLE_COUNT))
    event_seq = read_uint(first_tlv(tlvs, TlvId.EVENT_SEQ))
    timestamp_ms = read_uint(first_tlv(tlvs, TlvId.TIMESTAMP_MS))
    quality = read_uint(first_tlv(tlvs, TlvId.QUALITY))
    range_status = read_uint(first_tlv(tlvs, TlvId.RANGE_STATUS))
    rx_power = read_int(first_tlv(tlvs, TlvId.UWB_RSL_DBM))
    phy_config_id = read_uint(first_tlv(tlvs, TlvId.PHY_CONFIG_ID))
    burst_id = read_uint(first_tlv(tlvs, TlvId.BURST_ID))
    cir = first_tlv(tlvs, TlvId.UWB_CIR_SAMPLE)
    exchange_stride_us = read_uint(first_tlv(tlvs, TlvId.EXCHANGE_STRIDE_US))
    burst_duration_ms = read_uint(first_tlv(tlvs, TlvId.BURST_DURATION_MS))
    diag_status_flags = read_uint(first_tlv(tlvs, TlvId.DIAG_STATUS_FLAGS))
    diag_bytes_captured = read_uint(first_tlv(tlvs, TlvId.DIAG_BYTES_CAPTURED))
    diag_bytes_transmitted = read_uint(first_tlv(tlvs, TlvId.DIAG_BYTES_TRANSMITTED))
    report_fragment_count = read_uint(first_tlv(tlvs, TlvId.REPORT_FRAGMENT_COUNT))
    uwb_clock_offset_raw = read_int(first_tlv(tlvs, TlvId.UWB_CLOCK_OFFSET_RAW))
    uwb_carrier_integrator = read_int(first_tlv(tlvs, TlvId.UWB_CARRIER_INTEGRATOR))
    clicker_diag = first_tlv(tlvs, TlvId.CLICKER_DIAG_BYTES)
    cir_first_path_index = read_uint(first_tlv(tlvs, TlvId.UWB_CIR_FIRST_PATH_INDEX))
    cir_start_index = read_uint(first_tlv(tlvs, TlvId.UWB_CIR_START_INDEX))
    diag_source = read_uint(first_tlv(tlvs, TlvId.DIAG_SOURCE))

    # Diagnostic-only fragments: packets without SAMPLE_INDEX and
    # DISTANCE_SAMPLES_MM carry CIR chunks, tune info, clock offset, etc.
    # for the last real sample. Return them as diagnostic_fragment records
    # so the GUI can merge their data into the corresponding sample.
    is_diagnostic_fragment = sample_index is None and first_tlv(tlvs, TlvId.DISTANCE_SAMPLES_MM) is None
    if is_diagnostic_fragment:
        return [
            ParsedRecord(
                kind="diagnostic_fragment",
                anchor_id=format_device_id(anchor_id) if anchor_id is not None else format_device_id(packet.source_id),
                clicker_id=format_device_id(clicker_id) if clicker_id is not None else None,
                event_seq=event_seq,
                firmware_timestamp_ms=timestamp_ms,
                quality=quality,
                phy_config_id=phy_config_id,
                burst_id=burst_id,
                rx_power_dbm=float(rx_power) if rx_power is not None else None,
                cir_raw=cir.hex() if cir else None,
                clicker_diag_bytes=clicker_diag.hex() if clicker_diag else None,
                status=_range_status_text(range_status),
                source="ml_diagnostic_fragment",
                raw_line=packet_summary(packet),
                tlv_json=_tlvs_to_json(tlvs),
                exchange_stride_us=exchange_stride_us,
                burst_duration_ms=burst_duration_ms,
                diag_status_flags=diag_status_flags,
                diag_bytes_captured=diag_bytes_captured,
                diag_bytes_transmitted=diag_bytes_transmitted,
                report_fragment_count=report_fragment_count,
                uwb_clock_offset_raw=uwb_clock_offset_raw,
                uwb_carrier_integrator=uwb_carrier_integrator,
                cir_first_path_index=cir_first_path_index,
                cir_start_index=cir_start_index,
                diag_source=diag_source,
            )
        ]

    distance_mm: int | None = None
    sample_array = first_tlv(tlvs, TlvId.DISTANCE_SAMPLES_MM)
    if sample_array and len(sample_array) >= 4:
        distance_mm = struct.unpack_from("<i", sample_array, 0)[0]
    else:
        distance_mm = read_int(first_tlv(tlvs, TlvId.DISTANCE_MM))

    ok = range_status in (None, 0)
    if distance_mm is None or distance_mm < 0 or not ok:
        distance_m = None
    else:
        distance_m = distance_mm / 1000.0

    anchor_text = format_device_id(anchor_id) if anchor_id is not None else format_device_id(packet.source_id)

    return [
        ParsedRecord(
            kind="sample" if ok else "failure",
            anchor_id=anchor_text,
            clicker_id=format_device_id(clicker_id) if clicker_id is not None else None,
            sample_index=sample_index or 0,
            scheduled_sample_count=scheduled_count,
            event_seq=event_seq,
            firmware_timestamp_ms=timestamp_ms,
            quality=quality,
            phy_config_id=phy_config_id,
            burst_id=burst_id,
            distance_m=distance_m,
            rx_power_dbm=float(rx_power) if rx_power is not None else None,
            cir_raw=cir.hex() if cir else None,
            clicker_diag_bytes=clicker_diag.hex() if clicker_diag else None,
            status=_range_status_text(range_status),
            error_code=None if ok else str(range_status),
            source="ml_click_report",
            raw_line=packet_summary(packet),
            tlv_json=_tlvs_to_json(tlvs),
            exchange_stride_us=exchange_stride_us,
            burst_duration_ms=burst_duration_ms,
            diag_status_flags=diag_status_flags,
            diag_bytes_captured=diag_bytes_captured,
            diag_bytes_transmitted=diag_bytes_transmitted,
            report_fragment_count=report_fragment_count,
            uwb_clock_offset_raw=uwb_clock_offset_raw,
            uwb_carrier_integrator=uwb_carrier_integrator,
            cir_first_path_index=cir_first_path_index,
            cir_start_index=cir_start_index,
            diag_source=diag_source,
        )
    ]


def _tlvs_to_json(tlvs: dict[int, list[bytes]]) -> str:
    obj: dict[str, list[str]] = {}
    for tlv_id, values in tlvs.items():
        obj[_tlv_name(tlv_id)] = [value.hex() for value in values]
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _tlv_name(tlv_id: int) -> str:
    try:
        return TlvId(tlv_id).name
    except ValueError:
        return f"0x{tlv_id:02X}"


def _range_status_text(status: int | None) -> str:
    if status is None or status == 0:
        return "ok"
    return _enum_name(RangeStatus, status, "range_status").lower()


def _enum_name(enum: Any, value: int | None, prefix: str) -> str:
    if value is None:
        return f"{prefix}_UNKNOWN"
    try:
        return enum(value).name
    except ValueError:
        return f"{prefix}_{value}"
