import math
import time
import unittest
from unittest import mock

from uwb_capture.common import ParsedRecord
from uwb_capture.extended_gui import (
    ExtendedUwbCaptureApp,
    LIVE_TRACKING_MAX_RESTART_BACKOFF_MS,
    LIVE_TRACKING_MAX_RESTART_RETRIES,
)
from uwb_capture.localization import LocalizationReading, solve_position
from uwb_capture.protocol import (
    CommandId,
    CommandStatus,
    ImecPacket,
    MessageType,
    TlvId,
    decode_packet,
    encode_tlvs,
    u8,
    u16,
)


class FakeStatusVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class FakeVar:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class FakeCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def winfo_width(self) -> int:
        return self.width

    def winfo_height(self) -> int:
        return self.height


class FakeStopEvent:
    def __init__(self, stopped: bool) -> None:
        self.stopped = stopped

    def is_set(self) -> bool:
        return self.stopped


class FakeWorker:
    def __init__(self, stopped: bool = False) -> None:
        self.stop_event = FakeStopEvent(stopped)
        self.packets: list[bytes] = []

    def send_packet(self, packet: bytes) -> bool:
        if self.stop_event.is_set():
            return False
        self.packets.append(packet)
        return True


class FakeRejectingWorker(FakeWorker):
    def send_packet(self, packet: bytes) -> bool:
        return False


class LocalizationGuiTests(unittest.TestCase):
    def test_plot_bounds_use_room_size_not_estimated_position(self) -> None:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.sim_width_var = FakeVar("7")
        app.sim_height_var = FakeVar("7")
        result = solve_position(
            [
                LocalizationReading("A1", 0.0, 0.0, math.hypot(1.0, 1.0)),
                LocalizationReading("A2", 0.3, 0.8, math.hypot(0.7, 0.2)),
                LocalizationReading("A3", 1.5, 1.0, 0.5),
            ]
        )

        self.assertEqual(result.confidence, "High")
        self.assertEqual(
            app._localization_plot_bounds(result),
            (-0.28, 7.28, -0.28, 7.28),
        )

    def test_manual_plot_view_uses_requested_center_and_scale(self) -> None:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.sim_width_var = FakeVar("7")
        app.sim_height_var = FakeVar("7")
        app.localization_view_center_x_var = FakeVar("2")
        app.localization_view_center_y_var = FakeVar("3")
        app.localization_view_scale_var = FakeVar("50")
        result = solve_position(
            [
                LocalizationReading("A1", 0.0, 0.0, math.hypot(1.0, 1.0)),
                LocalizationReading("A2", 0.3, 0.8, math.hypot(0.7, 0.2)),
                LocalizationReading("A3", 1.5, 1.0, 0.5),
            ]
        )

        center_x, center_y, scale, min_x, max_x, min_y, max_y = app._localization_plot_view(
            result,
            width=500,
            height=300,
            margin=32,
        )

        self.assertEqual(center_x, 2.0)
        self.assertEqual(center_y, 3.0)
        self.assertEqual(scale, 50.0)
        self.assertAlmostEqual(min_x, -2.36)
        self.assertAlmostEqual(max_x, 6.36)
        self.assertAlmostEqual(min_y, 0.64)
        self.assertAlmostEqual(max_y, 5.36)

    def test_relative_plot_controls_pan_and_zoom_from_current_view(self) -> None:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.sim_width_var = FakeVar("7")
        app.sim_height_var = FakeVar("7")
        app.localization_view_center_x_var = FakeVar("")
        app.localization_view_center_y_var = FakeVar("")
        app.localization_view_scale_var = FakeVar("")
        app.localization_status_var = FakeStatusVar()
        app.localization_canvas = FakeCanvas(500, 300)
        app.redraw_count = 0
        app._redraw_localization_views = lambda: setattr(app, "redraw_count", app.redraw_count + 1)
        app.localization_result = solve_position(
            [
                LocalizationReading("A1", 0.0, 0.0, math.hypot(1.0, 1.0)),
                LocalizationReading("A2", 0.3, 0.8, math.hypot(0.7, 0.2)),
                LocalizationReading("A3", 1.5, 1.0, 0.5),
            ]
        )
        auto_center_x, auto_center_y, auto_scale, min_x, max_x, min_y, max_y = app._localization_plot_view(
            app.localization_result,
            width=500,
            height=300,
            margin=32,
        )

        app.pan_localization_view(1.0, -1.0)

        expected_center_x = auto_center_x + (max_x - min_x) * 0.15
        expected_center_y = auto_center_y - (max_y - min_y) * 0.15
        self.assertAlmostEqual(float(app.localization_view_center_x_var.get()), expected_center_x, places=5)
        self.assertAlmostEqual(float(app.localization_view_center_y_var.get()), expected_center_y, places=5)
        self.assertAlmostEqual(float(app.localization_view_scale_var.get()), auto_scale, delta=1e-4)
        self.assertEqual(app.redraw_count, 1)

        app.zoom_localization_view(1.25)

        self.assertAlmostEqual(float(app.localization_view_center_x_var.get()), expected_center_x, places=5)
        self.assertAlmostEqual(float(app.localization_view_center_y_var.get()), expected_center_y, places=5)
        self.assertAlmostEqual(float(app.localization_view_scale_var.get()), auto_scale * 1.25, delta=1e-4)
        self.assertEqual(app.redraw_count, 2)

    def test_solved_anchor_layout_populates_localization_coordinates(self) -> None:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.anchor_layout_positions = {
            "A2": (1.234, 5.678),
            "A1": (-0.5, 0.25),
        }
        app.localization_rows = {}
        app.localization_row_order = []
        app.localization_anchor_positions = {}
        app.localization_status_var = FakeStatusVar()
        app.redraw_count = 0

        def ensure_row(anchor_id: str) -> dict[str, FakeVar]:
            if anchor_id not in app.localization_rows:
                app.localization_rows[anchor_id] = {
                    "enabled_var": FakeVar(False),
                    "anchor_var": FakeVar(anchor_id),
                    "x_var": FakeVar(""),
                    "y_var": FakeVar(""),
                    "range_var": FakeVar(""),
                    "offset_var": FakeVar("0"),
                    "sigma_var": FakeVar("0.05"),
                }
                app.localization_row_order.append(anchor_id)
            return app.localization_rows[anchor_id]

        app._ensure_localization_row = ensure_row
        app._redraw_localization_views = lambda: setattr(app, "redraw_count", app.redraw_count + 1)

        self.assertTrue(app.populate_localization_from_anchor_layout())

        self.assertEqual(app.localization_row_order, ["A1", "A2"])
        self.assertTrue(app.localization_rows["A1"]["enabled_var"].get())
        self.assertEqual(app.localization_rows["A1"]["x_var"].get(), "-0.5")
        self.assertEqual(app.localization_rows["A1"]["y_var"].get(), "0.25")
        self.assertEqual(app.localization_rows["A2"]["x_var"].get(), "1.234")
        self.assertEqual(app.localization_rows["A2"]["y_var"].get(), "5.678")
        self.assertEqual(app.localization_anchor_positions["A2"], ("1.234", "5.678"))
        self.assertEqual(app.redraw_count, 1)
        self.assertIn("Loaded 2 anchor coordinate", app.localization_status_var.value)

    def make_live_tracking_app(self) -> ExtendedUwbCaptureApp:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.live_tracking_event_seq = None
        app.live_tracking_expected_sample_count = None
        app.live_tracking_received_sample_count = 0
        app.live_tracking_ranges_by_anchor = {}
        app.live_tracking_finish_pending = False
        app.live_tracking_finish_after_id = None
        app.live_tracking_run_finish_pending = False
        app.live_tracking_final_status = None
        app.live_tracking_active = True
        app.live_tracking_stopping = False
        app.live_tracking_source_id = None
        app.live_tracking_destination_id = None
        app.live_tracking_session_id = None
        app.live_tracking_start_sequence = None
        app.live_tracking_last_range_monotonic = None
        app.live_tracking_restart_retry_count = 0
        app.anchor_measured_distances = {}
        app.anchor_distance_history = {}
        app.live_tracking_status_var = FakeStatusVar()
        app.applied_ranges = {}
        app.solve_count = 0
        app.after_idle_callbacks = []
        app.after_idle = lambda callback: app.after_idle_callbacks.append(callback) or "idle1"
        app._complete_live_tracking_run = lambda status: setattr(app, "completed_status", status)
        app._update_localization_range = lambda anchor_id, distance_m: app.applied_ranges.__setitem__(
            anchor_id, distance_m
        )

        def solve_once() -> bool:
            app.solve_count += 1
            return True

        app._try_live_tracking_solve = solve_once
        return app

    def test_live_tracking_applies_each_range_row_immediately(self) -> None:
        app = self.make_live_tracking_app()
        records = [
            ParsedRecord(
                kind="sample",
                anchor_id="A1",
                distance_m=1.0,
                event_seq=10,
                scheduled_sample_count=3,
            ),
            ParsedRecord(
                kind="sample",
                anchor_id="A2",
                distance_m=2.0,
                event_seq=10,
                scheduled_sample_count=3,
            ),
            ParsedRecord(
                kind="sample",
                anchor_id="A3",
                distance_m=3.0,
                event_seq=10,
                scheduled_sample_count=3,
            ),
        ]

        self.assertTrue(app._record_live_tracking_sample(records[0]))
        self.assertEqual(app.applied_ranges, {"A1": 1.0})
        self.assertTrue(app._record_live_tracking_sample(records[1]))
        self.assertEqual(app.applied_ranges, {"A1": 1.0, "A2": 2.0})
        self.assertTrue(app._record_live_tracking_sample(records[2]))

        self.assertEqual(app.applied_ranges, {"A1": 1.0, "A2": 2.0, "A3": 3.0})
        self.assertEqual(app.solve_count, 3)
        self.assertEqual(app.live_tracking_received_sample_count, 3)

    def test_live_tracking_event_seq_change_starts_new_counter_without_waiting(self) -> None:
        app = self.make_live_tracking_app()
        records = [
            ParsedRecord(
                kind="sample",
                anchor_id="A1",
                distance_m=1.0,
                event_seq=10,
                scheduled_sample_count=8,
            ),
            ParsedRecord(
                kind="sample",
                anchor_id="A2",
                distance_m=2.0,
                event_seq=10,
                scheduled_sample_count=8,
            ),
            ParsedRecord(
                kind="summary",
                anchor_id="A3",
                mean_distance_m=3.0,
                event_seq=10,
                scheduled_sample_count=8,
            ),
            ParsedRecord(
                kind="sample",
                anchor_id="A1",
                distance_m=1.5,
                event_seq=11,
                scheduled_sample_count=8,
            ),
        ]

        for record in records:
            self.assertTrue(app._record_live_tracking_sample(record))

        self.assertEqual(app.applied_ranges, {"A1": 1.5, "A2": 2.0, "A3": 3.0})
        self.assertEqual(app.solve_count, 4)
        self.assertEqual(app.live_tracking_event_seq, 11)
        self.assertEqual(app.live_tracking_received_sample_count, 1)
        self.assertEqual(app.live_tracking_ranges_by_anchor, {})

    def test_live_tracking_pending_finish_waits_for_failure_row_from_result_packet(self) -> None:
        app = self.make_live_tracking_app()
        records = [
            ParsedRecord(
                kind="sample",
                anchor_id="A1",
                distance_m=1.0,
            ),
            ParsedRecord(
                kind="sample",
                anchor_id="A2",
                distance_m=2.0,
            ),
            ParsedRecord(
                kind="sample",
                anchor_id="A3",
                distance_m=3.0,
            ),
            ParsedRecord(
                kind="failure",
                anchor_id="A4",
            ),
        ]

        app._request_live_tracking_batch_finish()
        self.assertTrue(app.live_tracking_finish_pending)
        self.assertEqual(app.solve_count, 0)

        for record in records:
            app._record_live_tracking_sample(record)

        app.live_tracking_run_finish_pending = True
        app.live_tracking_final_status = CommandStatus.COMMAND_OK
        self.assertFalse(app._finish_pending_live_tracking_batch())

        self.assertEqual(app.applied_ranges, {"A1": 1.0, "A2": 2.0, "A3": 3.0})
        self.assertEqual(app.solve_count, 3)
        self.assertFalse(app.live_tracking_finish_pending)
        self.assertEqual(app.completed_status, "Live tracking stopped.")

    def make_live_tick_app(self) -> ExtendedUwbCaptureApp:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.live_tracking_active = True
        app.live_tracking_stopping = False
        app.live_tracking_after_id = None
        app.live_tracking_interval_var = FakeVar("1000")
        app.live_tracking_status_var = FakeStatusVar()
        app.ml_command_in_flight = False
        app.protocol_sequence = 10
        app.live_tracking_source_id = 1
        app.live_tracking_destination_id = 2
        app.live_tracking_session_id = 123
        app.live_tracking_start_sequence = 10
        app.live_tracking_last_range_monotonic = time.monotonic()
        app.live_tracking_restart_retry_count = 0
        app.bluetooth_worker = FakeWorker()
        app.clicker_id_var = FakeVar("2")
        app.log_lines = []
        app.log_raw = lambda text: app.log_lines.append(text)
        app.after_calls = []
        app.after = lambda interval_ms, callback: app.after_calls.append((interval_ms, callback)) or "after1"
        app._set_live_tracking_button_state = lambda: None
        app._reset_live_tracking_batch = lambda: None
        return app

    def make_live_command_app(self) -> ExtendedUwbCaptureApp:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.protocol_sequence = 10
        app.bluetooth_worker = FakeWorker()
        app.host_id_var = FakeVar("0x100")
        app.clicker_id_var = FakeVar("0x200")
        app.ml_session_id_var = FakeVar("1234")
        app.ml_discovery_slot_count_var = FakeVar("8")
        app.live_tracking_status_var = FakeStatusVar()
        app.ml_status_var = FakeStatusVar()
        app.ml_timeout_after_id = None
        app.ml_command_in_flight = False
        app.ml_pending_session = None
        app.ml_pending_seq = None
        app.ml_pending_mode = None
        app.ml_expected_sample_notifications = None
        app.ml_received_sample_notifications = 0
        app.live_tracking_active = False
        app.live_tracking_stopping = False
        app.live_tracking_after_id = None
        app.live_tracking_event_seq = None
        app.live_tracking_expected_sample_count = None
        app.live_tracking_received_sample_count = 0
        app.live_tracking_ranges_by_anchor = {}
        app.live_tracking_finish_pending = False
        app.live_tracking_finish_after_id = None
        app.live_tracking_run_finish_pending = False
        app.live_tracking_final_status = None
        app.live_tracking_source_id = None
        app.live_tracking_destination_id = None
        app.live_tracking_session_id = None
        app.live_tracking_start_sequence = None
        app.live_tracking_last_range_monotonic = None
        app.live_tracking_restart_retry_count = 0
        app.log_lines = []
        app.log_raw = lambda text: app.log_lines.append(text)
        app.after_calls = []
        app.after = lambda interval_ms, callback: app.after_calls.append((interval_ms, callback)) or "after1"
        app.after_cancel = lambda _after_id: None
        app._reset_trigger_collection_state = lambda: None
        app._set_ml_collect_state = lambda in_flight: setattr(app, "ml_collect_in_flight", in_flight)
        app._update_ml_progress_status = lambda: None
        app._flush_diagnostics_to_all_samples = lambda: None
        app._set_live_tracking_button_state = lambda: None
        return app

    def command_result_packet(
        self,
        *,
        session_id: int,
        sequence: int,
        command_id: CommandId,
        status: CommandStatus,
    ) -> ImecPacket:
        return ImecPacket(
            msg_type=MessageType.COMMAND_RESULT,
            flags=0,
            source_id=0x200,
            destination_id=0x100,
            session_id=session_id,
            sequence=sequence,
            ttl=1,
            message_age_ms=0,
            payload=encode_tlvs(
                [
                    (TlvId.COMMAND_ID, u16(command_id)),
                    (TlvId.COMMAND_STATUS, u8(status)),
                ]
            ),
        )

    def test_live_tracking_start_command_uses_watchdog_protocol(self) -> None:
        app = self.make_live_command_app()

        self.assertTrue(app._send_live_tracking_start_command())

        packet = decode_packet(app.bluetooth_worker.packets[-1])
        self.assertEqual(packet.msg_type, MessageType.COMMAND)
        self.assertEqual(packet.source_id, 0x100)
        self.assertEqual(packet.destination_id, 0x200)
        self.assertEqual(packet.session_id, 1234)
        self.assertEqual(packet.sequence, 11)
        self.assertIn(bytes([TlvId.COMMAND_ID, 2, 0x03, 0x80]), packet.payload)
        self.assertIn(bytes([TlvId.SAMPLE_COUNT, 1, 1]), packet.payload)
        self.assertIn(bytes([TlvId.DISCOVERY_SLOT_COUNT, 1, 8]), packet.payload)
        self.assertIn(bytes([TlvId.DURATION_MS, 4, 0xB8, 0x0B, 0x00, 0x00]), packet.payload)
        self.assertTrue(app.ml_command_in_flight)
        self.assertEqual(app.ml_pending_seq, 11)

    def test_live_tracking_stop_sends_stop_without_completing_start_result(self) -> None:
        app = self.make_live_command_app()
        self.assertTrue(app._send_live_tracking_start_command())
        app.live_tracking_active = True

        app.stop_live_tracking()

        self.assertTrue(app.live_tracking_stopping)
        self.assertTrue(app.ml_command_in_flight)
        packet = decode_packet(app.bluetooth_worker.packets[-1])
        self.assertEqual(packet.sequence, 12)
        self.assertEqual(packet.payload, bytes([TlvId.COMMAND_ID, 2, 0x05, 0x80]))
        self.assertIn("waiting for final", app.live_tracking_status_var.value)

    def test_live_tracking_control_results_do_not_finish_start_command(self) -> None:
        app = self.make_live_command_app()
        self.assertTrue(app._send_live_tracking_start_command())
        app.live_tracking_active = True
        app.restart_reasons = []
        app._request_live_tracking_restart = lambda reason: app.restart_reasons.append(reason)

        heartbeat_result = self.command_result_packet(
            session_id=1234,
            sequence=12,
            command_id=CommandId.ML_LIVE_TRACKING_HEARTBEAT,
            status=CommandStatus.COMMAND_OK,
        )
        stop_result = self.command_result_packet(
            session_id=1234,
            sequence=13,
            command_id=CommandId.ML_STOP_LIVE_TRACKING,
            status=CommandStatus.COMMAND_OK,
        )
        start_result = self.command_result_packet(
            session_id=1234,
            sequence=11,
            command_id=CommandId.ML_START_LIVE_TRACKING,
            status=CommandStatus.COMMAND_TIMEOUT,
        )

        app._handle_command_result(heartbeat_result)
        app._handle_command_result(stop_result)

        self.assertTrue(app.ml_command_in_flight)
        self.assertEqual(app.restart_reasons, [])

        app._handle_command_result(start_result)

        self.assertFalse(app.ml_command_in_flight)
        self.assertEqual(app.restart_reasons, ["Live tracking ended by firmware watchdog timeout."])

    def test_live_tracking_stopped_run_finishes_when_start_result_arrives(self) -> None:
        app = self.make_live_command_app()
        self.assertTrue(app._send_live_tracking_start_command())
        app.live_tracking_active = True
        app.live_tracking_stopping = True
        app.batch_finish_requested = False
        app._request_live_tracking_batch_finish = lambda: setattr(app, "batch_finish_requested", True)

        start_result = self.command_result_packet(
            session_id=1234,
            sequence=11,
            command_id=CommandId.ML_START_LIVE_TRACKING,
            status=CommandStatus.COMMAND_OK,
        )
        app._handle_command_result(start_result)

        self.assertTrue(app.live_tracking_run_finish_pending)
        self.assertTrue(app.batch_finish_requested)
        self.assertEqual(app.live_tracking_final_status, CommandStatus.COMMAND_OK)

    def test_live_tracking_retries_when_bluetooth_is_temporarily_missing(self) -> None:
        app = self.make_live_tick_app()
        app.bluetooth_worker = None

        app._live_tracking_tick()

        self.assertTrue(app.live_tracking_active)
        self.assertEqual(app.live_tracking_after_id, "after1")
        self.assertEqual(app.after_calls[0][0], 1000)
        self.assertIn("retrying", app.live_tracking_status_var.value)

    def test_live_tracking_retries_when_worker_is_stopping(self) -> None:
        app = self.make_live_tick_app()
        app.bluetooth_worker = FakeWorker(stopped=True)

        app._live_tracking_tick()

        self.assertTrue(app.live_tracking_active)
        self.assertEqual(app.live_tracking_after_id, "after1")
        self.assertIn("reconnecting", app.live_tracking_status_var.value)

    def test_live_tracking_sends_heartbeat_on_tick(self) -> None:
        app = self.make_live_tick_app()

        app._live_tracking_tick()

        self.assertTrue(app.live_tracking_active)
        self.assertEqual(app.live_tracking_after_id, "after1")
        self.assertIn("heartbeat sent", app.live_tracking_status_var.value)
        packet = decode_packet(app.bluetooth_worker.packets[-1])
        self.assertEqual(packet.msg_type, MessageType.COMMAND)
        self.assertEqual(packet.session_id, 123)
        self.assertEqual(packet.sequence, 11)
        self.assertEqual(packet.payload, bytes([TlvId.COMMAND_ID, 2, 0x04, 0x80]))

    def test_live_tracking_restarts_when_ranges_go_silent(self) -> None:
        app = self.make_live_tick_app()
        app.live_tracking_last_range_monotonic = time.monotonic() - 4.0
        app.restart_reasons = []
        app._request_live_tracking_restart = lambda reason: app.restart_reasons.append(reason)

        app._live_tracking_tick()

        self.assertEqual(app.restart_reasons, ["no live ranges received"])
        self.assertEqual(app.bluetooth_worker.packets, [])

    def test_live_tracking_restarts_when_heartbeat_send_fails(self) -> None:
        app = self.make_live_tick_app()
        app.bluetooth_worker = FakeRejectingWorker()
        app.restart_reasons = []
        app._request_live_tracking_restart = lambda reason: app.restart_reasons.append(reason)

        app._live_tracking_tick()

        self.assertEqual(app.restart_reasons, ["live heartbeat was not sent"])
        self.assertEqual(app.after_calls, [])

    def test_live_tracking_retries_when_heartbeat_is_not_sent(self) -> None:
        app = self.make_live_tick_app()
        app.bluetooth_worker = FakeWorker(stopped=True)

        app._live_tracking_tick()

        self.assertTrue(app.live_tracking_active)
        self.assertEqual(app.live_tracking_after_id, "after1")
        self.assertIn("reconnecting", app.live_tracking_status_var.value)

    def test_live_tracking_restart_uses_random_backoff_after_first_retry(self) -> None:
        app = self.make_live_tick_app()
        app.restart_now_reasons = []
        app._restart_live_tracking_now = lambda reason: app.restart_now_reasons.append(reason) or True

        app._request_live_tracking_restart("timeout")

        self.assertEqual(app.live_tracking_restart_retry_count, 1)
        self.assertEqual(app.restart_now_reasons, ["timeout"])
        self.assertEqual(app.after_calls, [])

        self.assertEqual(LIVE_TRACKING_MAX_RESTART_BACKOFF_MS, 1000)
        with mock.patch("uwb_capture.extended_gui.random.randint", return_value=742) as randint_mock:
            app._request_live_tracking_restart("radio error")

        self.assertEqual(app.live_tracking_restart_retry_count, 2)
        self.assertEqual(app.restart_now_reasons, ["timeout"])
        self.assertEqual(app.live_tracking_after_id, "after1")
        self.assertEqual(app.after_calls[-1][0], 742)
        randint_mock.assert_called_once_with(0, 1000)

        app.after_calls[-1][1]()

        self.assertEqual(app.restart_now_reasons, ["timeout", "radio error"])

    def test_live_tracking_restart_stops_after_fifty_retries(self) -> None:
        app = self.make_live_tick_app()
        self.assertEqual(LIVE_TRACKING_MAX_RESTART_RETRIES, 50)
        app.live_tracking_restart_retry_count = LIVE_TRACKING_MAX_RESTART_RETRIES
        app.completed_statuses = []
        app._complete_live_tracking_run = lambda status: app.completed_statuses.append(status)

        app._request_live_tracking_restart("still failing")

        self.assertEqual(
            app.completed_statuses,
            ["Live tracking stopped after 50 restart retries: still failing"],
        )

    def test_static_range_offset_is_added_to_each_anchor_offset(self) -> None:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.range_static_offset_var = FakeVar("0.25")
        app.localization_row_order = ["A1", "A2", "A3"]
        app.localization_rows = {
            anchor_id: {
                "enabled_var": FakeVar(True),
                "anchor_var": FakeVar(anchor_id),
                "x_var": FakeVar(str(index)),
                "y_var": FakeVar("0"),
                "range_var": FakeVar("3"),
                "offset_var": FakeVar("0.05"),
                "sigma_var": FakeVar("0.05"),
            }
            for index, anchor_id in enumerate(app.localization_row_order)
        }

        readings = app._read_localization_form()

        for reading in readings:
            self.assertAlmostEqual(reading.offset_m, 0.30)

    def test_static_range_offset_must_be_numeric(self) -> None:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.range_static_offset_var = FakeVar("bad")
        app.localization_row_order = []
        app.localization_rows = {}

        with self.assertRaisesRegex(ValueError, "Static range offset"):
            app._read_localization_form()


if __name__ == "__main__":
    unittest.main()
