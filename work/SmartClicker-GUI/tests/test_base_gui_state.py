import queue
import unittest

from uwb_capture import base_gui
from uwb_capture.extended_gui import ExtendedUwbCaptureApp


class FakeButton:
    def __init__(self) -> None:
        self.state = ""

    def configure(self, **kwargs: object) -> None:
        if "state" in kwargs:
            self.state = str(kwargs["state"])


class FakeStatusVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class FakeWorker:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def make_command_app() -> base_gui.UwbCaptureApp:
    app = base_gui.UwbCaptureApp.__new__(base_gui.UwbCaptureApp)
    app.ml_command_in_flight = True
    app.ml_pending_session = 10
    app.ml_pending_seq = 20
    app.ml_pending_mode = base_gui.ML_COLLECTION_MODE_FAST
    app.ml_timeout_after_id = "timeout1"
    app.ml_expected_sample_notifications = 4
    app.ml_received_sample_notifications = 2
    app.bluetooth_manual_disconnect_pending = False
    app.ml_collect_button = FakeButton()
    app.ml_status_var = FakeStatusVar()
    app.cancelled_after_ids = []
    app.log_lines = []
    app.collection_sample_rows = [1]
    app.collection_sample_rows_by_anchor = {"A1": [1]}
    app.collection_table_items = ["row1"]
    app.collection_table_items_by_anchor = {"A1": ["row1"]}
    app.collection_cir_records_by_anchor = {"A1": [object()]}
    app.collection_event_seq = 7
    app.pending_diagnostics = [object()]
    app.cir_reassembly_groups = {"A1": object()}
    app.after_cancel = lambda after_id: app.cancelled_after_ids.append(after_id)
    app.log_raw = lambda text: app.log_lines.append(text)
    app._flush_diagnostics_to_all_samples = lambda: None
    return app


class CommandStateTests(unittest.TestCase):
    def test_abort_ml_command_clears_in_flight_guard_and_timeout(self) -> None:
        app = make_command_app()

        app._abort_ml_command("Bluetooth disconnected")

        self.assertFalse(app.ml_command_in_flight)
        self.assertIsNone(app.ml_pending_session)
        self.assertIsNone(app.ml_pending_seq)
        self.assertIsNone(app.ml_pending_mode)
        self.assertIsNone(app.ml_timeout_after_id)
        self.assertIsNone(app.ml_expected_sample_notifications)
        self.assertEqual(app.ml_received_sample_notifications, 0)
        self.assertEqual(app.cancelled_after_ids, ["timeout1"])
        self.assertEqual(app.ml_collect_button.state, "normal")
        self.assertIn("Bluetooth disconnected", app.ml_status_var.value)
        self.assertIn("Bluetooth disconnected", app.log_lines[-1])
        self.assertEqual(app.collection_sample_rows, [])
        self.assertEqual(app.pending_diagnostics, [])

    def test_disconnect_transport_aborts_in_flight_command(self) -> None:
        app = make_command_app()
        worker = FakeWorker()
        app.bluetooth_worker = worker
        app.connected_device = "clicker"
        app.status_var = FakeStatusVar()

        app.disconnect_transport()

        self.assertTrue(worker.stopped)
        self.assertIsNone(app.bluetooth_worker)
        self.assertIsNone(app.connected_device)
        self.assertEqual(app.status_var.value, "Disconnected")
        self.assertFalse(app.ml_command_in_flight)

    def test_unexpected_disconnected_event_reconnects_immediately(self) -> None:
        app = make_command_app()
        app.events = queue.Queue()
        app.events.put(("disconnected", "clicker-address"))
        app.bluetooth_worker = FakeWorker()
        app.connected_device = "clicker-address"
        app.status_var = FakeStatusVar()
        app.after = lambda _ms, _callback: None
        app.reconnects = []
        app.selected_device = lambda: "clicker-address"
        app.connect_selected_device = lambda: app.reconnects.append("connect")

        app.process_events()

        self.assertIsNone(app.bluetooth_worker)
        self.assertIsNone(app.connected_device)
        self.assertEqual(app.reconnects, ["connect"])
        self.assertFalse(app.bluetooth_manual_disconnect_pending)
        self.assertIn("reconnecting immediately", app.log_lines[-1])

    def test_manual_disconnected_event_does_not_reconnect(self) -> None:
        app = make_command_app()
        worker = FakeWorker()
        app.bluetooth_worker = worker
        app.connected_device = "clicker-address"
        app.status_var = FakeStatusVar()
        app.events = queue.Queue()
        app.after = lambda _ms, _callback: None
        app.reconnects = []
        app.selected_device = lambda: "clicker-address"
        app.connect_selected_device = lambda: app.reconnects.append("connect")

        app.disconnect_transport()
        app.events.put(("disconnected", "clicker-address"))
        app.process_events()

        self.assertTrue(worker.stopped)
        self.assertEqual(app.reconnects, [])
        self.assertFalse(app.bluetooth_manual_disconnect_pending)

    def test_stop_live_tracking_completes_locally_without_bluetooth(self) -> None:
        app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
        app.live_tracking_active = True
        app.live_tracking_stopping = False
        app.live_tracking_after_id = "live1"
        app.live_tracking_event_seq = 1
        app.live_tracking_expected_sample_count = 4
        app.live_tracking_received_sample_count = 2
        app.live_tracking_ranges_by_anchor = {"A1": [1.0]}
        app.live_tracking_finish_pending = True
        app.live_tracking_finish_after_id = "idle1"
        app.live_tracking_run_finish_pending = False
        app.live_tracking_final_status = None
        app.live_tracking_source_id = None
        app.live_tracking_destination_id = None
        app.live_tracking_session_id = None
        app.live_tracking_start_sequence = None
        app.live_tracking_last_range_monotonic = None
        app.live_tracking_restart_retry_count = 0
        app.live_tracking_status_var = FakeStatusVar()
        app.bluetooth_worker = None
        app.ml_command_in_flight = True
        app.ml_pending_mode = base_gui.ML_COLLECTION_MODE_FAST
        app.ml_timeout_after_id = None
        app.aborted_reasons = []
        app.cancelled_live_after_ids = []
        app._abort_ml_command = lambda reason: app.aborted_reasons.append(reason)
        app.after_cancel = lambda after_id: app.cancelled_live_after_ids.append(after_id)
        app._set_live_tracking_button_state = lambda: None

        app.stop_live_tracking("Stopped by test.")

        self.assertFalse(app.live_tracking_active)
        self.assertEqual(app.aborted_reasons, [])
        self.assertEqual(app.cancelled_live_after_ids, ["live1"])
        self.assertEqual(
            app.live_tracking_status_var.value,
            "Live tracking stopped locally; Bluetooth is disconnected.",
        )
        self.assertEqual(app.live_tracking_ranges_by_anchor, {})
        self.assertFalse(app.live_tracking_finish_pending)


if __name__ == "__main__":
    unittest.main()
