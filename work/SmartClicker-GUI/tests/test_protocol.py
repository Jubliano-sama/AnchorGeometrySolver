import unittest

from uwb_capture.protocol import (
    FLAG_DIAGNOSTIC,
    CommandId,
    CommandStatus,
    ImecPacket,
    ImecPacketStream,
    MessageType,
    ProtocolError,
    TlvId,
    build_ml_start_collection_packet,
    build_ml_start_fast_ranging_packet,
    build_ml_start_live_tracking_packet,
    build_ml_live_tracking_heartbeat_packet,
    build_ml_start_anchor_pair_survey_packet,
    build_ml_stop_live_tracking_packet,
    cobs_decode,
    cobs_encode,
    command_result_summary,
    command_sample_count_from_packet,
    command_status_from_packet,
    decode_packet,
    encode_packet,
    encode_tlvs,
    i32,
    records_from_packet,
    u8,
    u16,
    u32,
    u64,
)


class CobsTests(unittest.TestCase):
    def test_cobs_round_trip_preserves_zero_bytes(self) -> None:
        payload = bytes([0x11, 0x00, 0x22, 0x00, 0x00, 0x33, 0x00])
        self.assertEqual(cobs_decode(cobs_encode(payload)), payload)

    def test_cobs_round_trip_random(self) -> None:
        payload = bytes(range(256)) * 3
        self.assertEqual(cobs_decode(cobs_encode(payload)), payload)

    def test_cobs_encode_never_emits_interior_zero(self) -> None:
        payload = bytes([0x00, 0x01, 0x02, 0x00, 0x03])
        encoded = cobs_encode(payload)
        self.assertNotIn(0x00, encoded)


class HeaderTests(unittest.TestCase):
    def test_header_includes_message_age(self) -> None:
        packet = ImecPacket(
            msg_type=MessageType.COMMAND,
            flags=0,
            source_id=0x100,
            destination_id=0x200,
            session_id=1234,
            sequence=7,
            ttl=1,
            message_age_ms=0x1234,
            payload=b"",
        )
        decoded = decode_packet(encode_packet(packet))
        self.assertEqual(decoded.message_age_ms, 0x1234)
        # Message age occupies bytes 28..31 of the decoded packet.
        raw = encode_packet(packet)
        import struct

        self.assertEqual(struct.unpack_from("<I", raw, 28)[0], 0x1234)


class CommandTests(unittest.TestCase):
    def test_build_ml_start_collection_packet(self) -> None:
        raw = build_ml_start_collection_packet(
            source_id=0x100,
            destination_id=0,
            session_id=1234,
            sequence=7,
            sample_count=12,
            discovery_slot_count=8,
        )
        packet = decode_packet(raw)

        self.assertEqual(packet.msg_type, MessageType.COMMAND)
        self.assertEqual(packet.source_id, 0x100)
        self.assertEqual(packet.destination_id, 0)
        self.assertEqual(packet.session_id, 1234)
        self.assertEqual(packet.sequence, 7)
        self.assertEqual(packet.ttl, 1)
        self.assertEqual(packet.message_age_ms, 0)
        # Required COMMAND_ID TLV.
        self.assertIn(bytes([TlvId.COMMAND_ID, 2, 0x00, 0x80]), packet.payload)
        # Sample count and discovery slot count TLVs.
        self.assertIn(bytes([TlvId.SAMPLE_COUNT, 1, 12]), packet.payload)
        self.assertIn(bytes([TlvId.DISCOVERY_SLOT_COUNT, 1, 8]), packet.payload)

    def test_ml_start_collection_packet_accepts_current_limits(self) -> None:
        packet = decode_packet(
            build_ml_start_collection_packet(
                source_id=0x100,
                destination_id=0,
                session_id=1234,
                sequence=7,
                sample_count=100,
                discovery_slot_count=8,
            )
        )

        self.assertIn(bytes([TlvId.SAMPLE_COUNT, 1, 100]), packet.payload)
        self.assertIn(bytes([TlvId.DISCOVERY_SLOT_COUNT, 1, 8]), packet.payload)

    def test_build_ml_start_fast_ranging_packet_uses_same_tlvs(self) -> None:
        raw = build_ml_start_fast_ranging_packet(
            source_id=0x100,
            destination_id=0,
            session_id=1234,
            sequence=7,
            sample_count=12,
            discovery_slot_count=8,
        )
        packet = decode_packet(raw)

        self.assertEqual(packet.msg_type, MessageType.COMMAND)
        self.assertIn(bytes([TlvId.COMMAND_ID, 2, 0x01, 0x80]), packet.payload)
        self.assertIn(bytes([TlvId.SAMPLE_COUNT, 1, 12]), packet.payload)
        self.assertIn(bytes([TlvId.DISCOVERY_SLOT_COUNT, 1, 8]), packet.payload)

    def test_build_ml_start_live_tracking_packet(self) -> None:
        raw = build_ml_start_live_tracking_packet(
            source_id=0x100,
            destination_id=0x200,
            session_id=1234,
            sequence=7,
            sample_count=100,
            discovery_slot_count=8,
            duration_ms=3000,
        )
        packet = decode_packet(raw)

        self.assertEqual(packet.msg_type, MessageType.COMMAND)
        self.assertEqual(packet.source_id, 0x100)
        self.assertEqual(packet.destination_id, 0x200)
        self.assertEqual(packet.session_id, 1234)
        self.assertEqual(packet.sequence, 7)
        self.assertIn(bytes([TlvId.COMMAND_ID, 2, 0x03, 0x80]), packet.payload)
        self.assertIn(bytes([TlvId.SAMPLE_COUNT, 1, 100]), packet.payload)
        self.assertIn(bytes([TlvId.DISCOVERY_SLOT_COUNT, 1, 8]), packet.payload)
        self.assertIn(bytes([TlvId.DURATION_MS, 4, 0xB8, 0x0B, 0x00, 0x00]), packet.payload)

    def test_live_tracking_heartbeat_and_stop_packets_only_include_command_id(self) -> None:
        heartbeat = decode_packet(
            build_ml_live_tracking_heartbeat_packet(
                source_id=0x100,
                destination_id=0x200,
                session_id=1234,
                sequence=8,
            )
        )
        stop = decode_packet(
            build_ml_stop_live_tracking_packet(
                source_id=0x100,
                destination_id=0x200,
                session_id=1234,
                sequence=9,
            )
        )

        self.assertEqual(heartbeat.payload, bytes([TlvId.COMMAND_ID, 2, 0x04, 0x80]))
        self.assertEqual(stop.payload, bytes([TlvId.COMMAND_ID, 2, 0x05, 0x80]))

    def test_live_tracking_packet_rejects_invalid_limits(self) -> None:
        with self.assertRaises(ProtocolError):
            build_ml_start_live_tracking_packet(
                source_id=0x100,
                destination_id=0x200,
                session_id=1234,
                sequence=7,
                sample_count=101,
            )
        with self.assertRaises(ProtocolError):
            build_ml_start_live_tracking_packet(
                source_id=0x100,
                destination_id=0x200,
                session_id=1234,
                sequence=7,
                discovery_slot_count=9,
            )
        with self.assertRaises(ProtocolError):
            build_ml_start_live_tracking_packet(
                source_id=0x100,
                destination_id=0x200,
                session_id=1234,
                sequence=7,
                duration_ms=499,
            )

    def test_ml_start_collection_packet_rejects_out_of_range_limits(self) -> None:
        with self.assertRaises(ProtocolError):
            build_ml_start_collection_packet(
                source_id=0x100,
                destination_id=0,
                session_id=1234,
                sequence=7,
                sample_count=101,
            )
        with self.assertRaises(ProtocolError):
            build_ml_start_collection_packet(
                source_id=0x100,
                destination_id=0,
                session_id=1234,
                sequence=7,
                discovery_slot_count=9,
            )

    def test_ml_command_uses_cobs_on_the_wire(self) -> None:
        proto_packet = build_ml_start_collection_packet(
            source_id=1,
            destination_id=0,
            session_id=42,
            sequence=1,
        )
        frame = cobs_encode(proto_packet) + b"\x00"
        self.assertFalse(0x00 in frame[:-1])
        self.assertEqual(frame[-1], 0x00)
        self.assertEqual(cobs_decode(frame[:-1]), proto_packet)

    def test_build_anchor_pair_survey_packet(self) -> None:
        raw = build_ml_start_anchor_pair_survey_packet(
            source_id=0x100,
            destination_id=0x200,
            session_id=44,
            sequence=9,
            discovery_slot_count=4,
        )
        packet = decode_packet(raw)

        self.assertEqual(packet.msg_type, MessageType.COMMAND)
        self.assertEqual(packet.source_id, 0x100)
        self.assertEqual(packet.destination_id, 0x200)
        self.assertEqual(packet.session_id, 44)
        self.assertEqual(packet.sequence, 9)
        self.assertIn(bytes([TlvId.COMMAND_ID, 2, 0x02, 0x80]), packet.payload)
        self.assertIn(bytes([TlvId.DISCOVERY_SLOT_COUNT, 1, 4]), packet.payload)
        self.assertNotIn(bytes([TlvId.SAMPLE_COUNT]), packet.payload)

    def test_anchor_pair_survey_rejects_invalid_discovery_slots(self) -> None:
        with self.assertRaises(ProtocolError):
            build_ml_start_anchor_pair_survey_packet(
                source_id=0x100,
                destination_id=0x200,
                session_id=44,
                sequence=9,
                discovery_slot_count=1,
            )


class StreamTests(unittest.TestCase):
    def test_stream_reassembles_cobs_frames_across_notifications(self) -> None:
        proto_packet = build_ml_start_collection_packet(
            source_id=1,
            destination_id=2,
            session_id=3,
            sequence=4,
        )
        frame = cobs_encode(proto_packet) + b"\x00"
        stream = ImecPacketStream()

        self.assertEqual(stream.feed(frame[:5]), [])
        self.assertEqual(stream.feed(frame[5:]), [proto_packet])

    def test_stream_decodes_multiple_frames_in_one_notification(self) -> None:
        proto_a = build_ml_start_collection_packet(
            source_id=1, destination_id=0, session_id=1, sequence=1
        )
        proto_b = build_ml_start_collection_packet(
            source_id=1, destination_id=0, session_id=1, sequence=2
        )
        blob = cobs_encode(proto_a) + b"\x00" + cobs_encode(proto_b) + b"\x00"
        self.assertEqual(ImecPacketStream().feed(blob), [proto_a, proto_b])


class MlSampleTests(unittest.TestCase):
    def _ml_sample_packet(
        self,
        *,
        range_status: int = 0,
        distance_samples_mm: bytes | None = None,
        distance_mm: int | None = None,
    ) -> ImecPacket:
        tlvs = [
            (TlvId.CLICKER_ID, u64(0x1111)),
            (TlvId.ANCHOR_ID, u64(0xAABBCCDD)),
            (TlvId.EVENT_SEQ, u32(99)),
            (TlvId.TIMESTAMP_MS, u64(1_000_000)),
            (TlvId.SAMPLE_INDEX, u16(5)),
            (TlvId.SAMPLE_COUNT, u16(8)),
            (TlvId.RANGE_STATUS, u8(range_status)),
            (TlvId.QUALITY, u8(87)),
            (TlvId.UWB_RSL_DBM, u8(-75 & 0xFF)),
            (TlvId.PHY_CONFIG_ID, u8(9)),
            (TlvId.BURST_ID, u32(0xBEEF)),
            (TlvId.UWB_CIR_SAMPLE, b"\x01\x02\x03\x04\x05\x06"),
            (TlvId.REPORT_FRAGMENT_COUNT, u16(1)),
        ]
        if distance_samples_mm is not None:
            tlvs.append((TlvId.DISTANCE_SAMPLES_MM, distance_samples_mm))
        if distance_mm is not None:
            tlvs.append((TlvId.DISTANCE_MM, i32(distance_mm)))
        return ImecPacket(
            msg_type=MessageType.CLICK_REPORT,
            flags=FLAG_DIAGNOSTIC,
            source_id=0xAABBCCDD,
            destination_id=0,
            session_id=99,
            sequence=1,
            ttl=1,
            message_age_ms=0,
            payload=encode_tlvs(tlvs),
        )

    def test_ml_sample_becomes_training_row(self) -> None:
        packet = self._ml_sample_packet(distance_samples_mm=i32(1250))
        records = records_from_packet(decode_packet(encode_packet(packet)))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.kind, "sample")
        self.assertEqual(record.anchor_id, "0x00000000AABBCCDD")
        self.assertEqual(record.clicker_id, "0x0000000000001111")
        self.assertEqual(record.sample_index, 5)
        self.assertEqual(record.scheduled_sample_count, 8)
        self.assertEqual(record.event_seq, 99)
        self.assertEqual(record.firmware_timestamp_ms, 1_000_000)
        self.assertEqual(record.quality, 87)
        self.assertEqual(record.phy_config_id, 9)
        self.assertEqual(record.burst_id, 0xBEEF)
        self.assertEqual(record.distance_m, 1.25)
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.cir_raw, "010203040506")
        self.assertIsNone(record.error_code)
        self.assertEqual(record.source, "ml_click_report")
        self.assertIsNotNone(record.tlv_json)

    def test_ml_sample_falls_back_to_distance_mm_tlv(self) -> None:
        packet = self._ml_sample_packet(distance_mm=2750)
        record = records_from_packet(decode_packet(encode_packet(packet)))[0]
        self.assertEqual(record.distance_m, 2.75)

    def test_failed_range_status_stored_as_failure_row(self) -> None:
        packet = self._ml_sample_packet(range_status=1, distance_samples_mm=i32(-1))
        record = records_from_packet(decode_packet(encode_packet(packet)))[0]
        self.assertEqual(record.kind, "failure")
        self.assertIsNone(record.distance_m)
        self.assertEqual(record.error_code, "1")
        self.assertEqual(record.status, "range_rx_timeout")

    def test_tlv_json_preserves_unknown_tlvs(self) -> None:
        packet = self._ml_sample_packet(distance_samples_mm=i32(1000))
        # Append an unknown TLV (id 0xAA) directly into the payload.
        packet = ImecPacket(
            msg_type=packet.msg_type,
            flags=packet.flags,
            source_id=packet.source_id,
            destination_id=packet.destination_id,
            session_id=packet.session_id,
            sequence=packet.sequence,
            ttl=packet.ttl,
            message_age_ms=packet.message_age_ms,
            payload=packet.payload + bytes([0xAA, 2, 0xDE, 0xAD]),
        )
        record = records_from_packet(decode_packet(encode_packet(packet)))[0]
        self.assertIn("0xAA", record.tlv_json)
        self.assertIn("dead", record.tlv_json)

    def test_post_burst_diagnostic_is_not_training_row(self) -> None:
        payload = encode_tlvs(
            [
                (TlvId.CLICKER_ID, u64(0x1111)),
                (TlvId.ANCHOR_ID, u64(0xAABBCCDD)),
                (TlvId.EVENT_SEQ, u32(99)),
                (TlvId.TIMESTAMP_MS, u64(1_000_100)),
                (TlvId.DISTANCE_MM, i32(1260)),
                (TlvId.QUALITY, u8(80)),
                (TlvId.RANGE_STATUS, u8(0)),
                (TlvId.BURST_ID, u32(0xBEEF)),
                (TlvId.DIAG_SOURCE, u8(3)),
                (TlvId.UWB_CIR_BYTE_OFFSET, u16(0)),
                (TlvId.UWB_CIR_TOTAL_BYTES, u16(6)),
                (TlvId.UWB_CIR_FIRST_PATH_INDEX, u16(700)),
                (TlvId.UWB_CIR_START_INDEX, u16(640)),
                (TlvId.UWB_CIR_FULL_CHUNK, b"\x01\x00\x00\x02\x00\x00"),
            ]
        )
        packet = ImecPacket(
            msg_type=MessageType.CLICK_REPORT,
            flags=FLAG_DIAGNOSTIC,
            source_id=0xAABBCCDD,
            destination_id=0,
            session_id=99,
            sequence=1,
            ttl=1,
            message_age_ms=0,
            payload=payload,
        )

        records = records_from_packet(decode_packet(encode_packet(packet)))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.kind, "diagnostic_fragment")
        self.assertEqual(record.anchor_id, "0x00000000AABBCCDD")
        self.assertEqual(record.event_seq, 99)
        self.assertEqual(record.burst_id, 0xBEEF)
        self.assertEqual(record.cir_first_path_index, 700)
        self.assertEqual(record.cir_start_index, 640)
        self.assertEqual(record.diag_source, 3)
        self.assertIn("UWB_CIR_FULL_CHUNK", record.tlv_json)


class SurveyPairResultTests(unittest.TestCase):
    def test_diagnostic_click_report_with_initiator_responder_becomes_survey_pair_record(self) -> None:
        packet = ImecPacket(
            msg_type=MessageType.CLICK_REPORT,
            flags=FLAG_DIAGNOSTIC,
            source_id=0xA1,
            destination_id=0,
            session_id=55,
            sequence=3,
            ttl=1,
            message_age_ms=0,
            payload=encode_tlvs(
                [
                    (TlvId.CLICKER_ID, u64(0x1111)),
                    (TlvId.SURVEY_ID, u32(88)),
                    (TlvId.EVENT_SEQ, u32(99)),
                    (TlvId.TIMESTAMP_MS, u64(123456)),
                    (TlvId.INITIATOR_ID, u64(0xA1)),
                    (TlvId.RESPONDER_ID, u64(0xB2)),
                    (TlvId.SAMPLE_INDEX, u16(2)),
                    (TlvId.SAMPLE_COUNT, u16(6)),
                    (TlvId.DISTANCE_MM, i32(3456)),
                    (TlvId.QUALITY, u8(93)),
                    (TlvId.RANGE_STATUS, u8(0)),
                    (TlvId.UWB_RSL_DBM, u8(-75 & 0xFF)),
                ]
            ),
        )

        records = records_from_packet(decode_packet(encode_packet(packet)))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.kind, "survey_pair")
        self.assertEqual(record.anchor_id, "0x00000000000000A1")
        self.assertEqual(record.peer_anchor_id, "0x00000000000000B2")
        self.assertEqual(record.clicker_id, "0x0000000000001111")
        self.assertEqual(record.sample_index, 2)
        self.assertEqual(record.distance_m, 3.456)
        self.assertEqual(record.scheduled_sample_count, 6)
        self.assertEqual(record.event_seq, 99)
        self.assertEqual(record.firmware_timestamp_ms, 123456)
        self.assertEqual(record.quality, 93)
        self.assertEqual(record.rx_power_dbm, -75.0)
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.source, "anchor_pair_survey")
        self.assertIn("SURVEY_ID", record.tlv_json)

    def test_survey_reach_report_with_initiator_responder_becomes_survey_pair_record(self) -> None:
        packet = ImecPacket(
            msg_type=MessageType.SURVEY_REACH_REPORT,
            flags=0,
            source_id=0xA1,
            destination_id=0,
            session_id=55,
            sequence=3,
            ttl=1,
            message_age_ms=0,
            payload=encode_tlvs(
                [
                    (TlvId.SURVEY_ID, u32(88)),
                    (TlvId.INITIATOR_ID, u64(0xA1)),
                    (TlvId.RESPONDER_ID, u64(0xB2)),
                    (TlvId.SAMPLE_INDEX, u16(2)),
                    (TlvId.SAMPLE_COUNT, u16(3)),
                    (TlvId.DISTANCE_MM, i32(3456)),
                    (TlvId.RANGE_STATUS, u8(0)),
                ]
            ),
        )

        records = records_from_packet(decode_packet(encode_packet(packet)))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.kind, "survey_pair")
        self.assertEqual(record.anchor_id, "0x00000000000000A1")
        self.assertEqual(record.peer_anchor_id, "0x00000000000000B2")
        self.assertEqual(record.distance_m, 3.456)
        self.assertEqual(record.scheduled_sample_count, 3)
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.source, "anchor_pair_survey")
        self.assertIn("SURVEY_REACH_REPORT", record.raw_line or "")

    def test_diagnostic_click_report_with_failed_pair_status_becomes_survey_pair_failure(self) -> None:
        packet = ImecPacket(
            msg_type=MessageType.CLICK_REPORT,
            flags=FLAG_DIAGNOSTIC,
            source_id=0xA1,
            destination_id=0,
            session_id=55,
            sequence=3,
            ttl=1,
            message_age_ms=0,
            payload=encode_tlvs(
                [
                    (TlvId.INITIATOR_ID, u64(0xA1)),
                    (TlvId.RESPONDER_ID, u64(0xB2)),
                    (TlvId.SAMPLE_INDEX, u16(1)),
                    (TlvId.SAMPLE_COUNT, u16(6)),
                    (TlvId.DISTANCE_MM, i32(-1)),
                    (TlvId.RANGE_STATUS, u8(1)),
                ]
            ),
        )

        record = records_from_packet(decode_packet(encode_packet(packet)))[0]

        self.assertEqual(record.kind, "survey_pair_failure")
        self.assertEqual(record.anchor_id, "0x00000000000000A1")
        self.assertEqual(record.peer_anchor_id, "0x00000000000000B2")
        self.assertIsNone(record.distance_m)
        self.assertEqual(record.error_code, "1")


class CommandResultTests(unittest.TestCase):
    def test_command_result_summary_includes_ml_status(self) -> None:
        payload = encode_tlvs(
            [
                (TlvId.COMMAND_ID, u16(CommandId.ML_START_COLLECTION)),
                (TlvId.COMMAND_STATUS, u8(CommandStatus.COMMAND_OK)),
                (TlvId.REASON, u8(0)),
                (TlvId.EVENT_SEQ, u32(7)),
                (TlvId.SAMPLE_COUNT, u16(32)),
            ]
        )
        packet = ImecPacket(
            msg_type=MessageType.COMMAND_RESULT,
            flags=FLAG_DIAGNOSTIC,
            source_id=0xAABBCCDD,
            destination_id=0,
            session_id=1234,
            sequence=7,
            ttl=1,
            message_age_ms=0,
            payload=payload,
        )
        summary = command_result_summary(packet)
        self.assertIn("ML_START_COLLECTION", summary)
        self.assertIn("COMMAND_OK", summary)
        self.assertIn("event_seq=7", summary)
        self.assertIn("samples=32", summary)

    def test_command_result_summary_flags_error(self) -> None:
        payload = encode_tlvs(
            [
                (TlvId.COMMAND_ID, u16(CommandId.ML_START_COLLECTION)),
                (TlvId.COMMAND_STATUS, u8(CommandStatus.COMMAND_BUSY)),
                (TlvId.REASON, u8(0)),
            ]
        )
        packet = ImecPacket(
            msg_type=MessageType.COMMAND_RESULT,
            flags=FLAG_DIAGNOSTIC | 0x40,
            source_id=0xAABBCCDD,
            destination_id=0,
            session_id=1234,
            sequence=7,
            ttl=1,
            message_age_ms=0,
            payload=payload,
        )
        self.assertIn("FLAG_ERROR", command_result_summary(packet))

    def test_fast_ranging_command_result_summary_includes_ml_status(self) -> None:
        payload = encode_tlvs(
            [
                (TlvId.COMMAND_ID, u16(CommandId.ML_START_FAST_RANGING)),
                (TlvId.COMMAND_STATUS, u8(CommandStatus.COMMAND_OK)),
                (TlvId.SAMPLE_COUNT, u16(32)),
            ]
        )
        packet = ImecPacket(
            msg_type=MessageType.COMMAND_RESULT,
            flags=FLAG_DIAGNOSTIC,
            source_id=0xAABBCCDD,
            destination_id=0,
            session_id=1234,
            sequence=7,
            ttl=1,
            message_age_ms=0,
            payload=payload,
        )
        summary = command_result_summary(packet)

        self.assertIn("ML_START_FAST_RANGING", summary)
        self.assertIn("COMMAND_OK", summary)
        self.assertIn("samples=32", summary)

    def test_timeout_result_carries_zero_sample_count(self) -> None:
        payload = encode_tlvs(
            [
                (TlvId.COMMAND_ID, u16(CommandId.ML_START_COLLECTION)),
                (TlvId.COMMAND_STATUS, u8(CommandStatus.COMMAND_TIMEOUT)),
                (TlvId.REASON, u8(0)),
                (TlvId.SAMPLE_COUNT, u16(0)),
            ]
        )
        packet = ImecPacket(
            msg_type=MessageType.COMMAND_RESULT,
            flags=FLAG_DIAGNOSTIC | 0x40,
            source_id=0xAABBCCDD,
            destination_id=0,
            session_id=1234,
            sequence=7,
            ttl=1,
            message_age_ms=0,
            payload=payload,
        )
        self.assertEqual(command_status_from_packet(packet), CommandStatus.COMMAND_TIMEOUT)
        self.assertEqual(command_sample_count_from_packet(packet), 0)


if __name__ == "__main__":
    unittest.main()
