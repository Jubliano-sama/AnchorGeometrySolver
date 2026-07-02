import unittest

from uwb_capture.common import ParsedRecord
from uwb_capture.extended_gui import ExtendedUwbCaptureApp


class FakeStatusVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class FakeVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class FakeTree:
    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, ...]] = {}
        self._order: list[str] = []
        self._next = 0

    def get_children(self) -> tuple[str, ...]:
        return tuple(self._order)

    def delete(self, item: str) -> None:
        self._rows.pop(item, None)
        if item in self._order:
            self._order.remove(item)

    def insert(self, _parent: str, _index: str, *, values: tuple[str, ...]) -> str:
        item = f"I{self._next}"
        self._next += 1
        self._rows[item] = values
        self._order.append(item)
        return item

    @property
    def rows(self) -> list[tuple[str, ...]]:
        return [self._rows[item] for item in self._order]


def make_app() -> ExtendedUwbCaptureApp:
    app = ExtendedUwbCaptureApp.__new__(ExtendedUwbCaptureApp)
    app.anchor_pair_distances = {}
    app.anchor_pair_status = {}
    app.anchor_pair_failures = {}
    app.anchor_layout_positions = {}
    app.detected_anchor_ids = set()
    app.anchor_survey_in_flight = True
    app.anchor_survey_expected_pair_count = None
    app.anchor_survey_received_pair_count = 0
    app.anchor_survey_successful_pair_count = 0
    app.range_static_offset_var = FakeVar("0")
    app.anchor_geometry_status_var = FakeStatusVar()
    app.anchor_pair_table = FakeTree()
    app.log_raw = lambda _message: None
    app._register_detected_anchor = lambda anchor_id: app.detected_anchor_ids.add(str(anchor_id))
    app._update_anchor_pair_combos = lambda: None
    app._refresh_anchor_truth_table = lambda: None
    app._redraw_anchor_geometry_canvas = lambda: None
    return app


class AnchorPairGuiTests(unittest.TestCase):
    def test_failed_survey_pair_is_visible_but_not_solver_input(self) -> None:
        app = make_app()

        app._handle_anchor_pair_record(
            ParsedRecord(
                kind="survey_pair_failure",
                anchor_id="A",
                peer_anchor_id="B",
                scheduled_sample_count=3,
                status="range_rx_timeout",
                source="anchor_pair_survey",
            )
        )

        self.assertEqual(app.anchor_survey_received_pair_count, 1)
        self.assertEqual(app.anchor_survey_successful_pair_count, 0)
        self.assertEqual(app.anchor_pair_distances, {})
        self.assertEqual(
            app.anchor_pair_table.rows,
            [("A", "B", "", "range_rx_timeout", "survey")],
        )
        self.assertIn("1/3 pair row(s) received, 0 usable distance(s)", app.anchor_geometry_status_var.value)

    def test_successful_survey_pair_replaces_prior_failure_row(self) -> None:
        app = make_app()
        app._handle_anchor_pair_record(
            ParsedRecord(
                kind="survey_pair_failure",
                anchor_id="A",
                peer_anchor_id="B",
                scheduled_sample_count=3,
                status="range_rx_timeout",
            )
        )

        app._handle_anchor_pair_record(
            ParsedRecord(
                kind="survey_pair",
                anchor_id="B",
                peer_anchor_id="A",
                scheduled_sample_count=3,
                distance_m=1.234,
                status="ok",
            )
        )

        self.assertEqual(app.anchor_survey_received_pair_count, 2)
        self.assertEqual(app.anchor_survey_successful_pair_count, 1)
        self.assertEqual(app.anchor_pair_failures, {})
        self.assertEqual(app.anchor_pair_table.rows, [("A", "B", "1.234", "ok", "survey")])
        self.assertIn("2/3 pair row(s) received, 1 usable distance(s)", app.anchor_geometry_status_var.value)


if __name__ == "__main__":
    unittest.main()
