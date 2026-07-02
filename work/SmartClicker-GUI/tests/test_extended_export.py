import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from uwb_capture.common import ParsedRecord, now_local_iso
from uwb_capture.extended_gui import (
    RESPONDER_LABEL_TABLE,
    ensure_responder_label_schema,
    export_session_per_anchor,
    build_measurement_rows,
    measurement_list_default_filename,
)
from uwb_capture.store import MeasurementStore


class ExtendedExportTests(unittest.TestCase):
    def test_measurement_rows_use_saved_per_responder_los_nlos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "measurements.sqlite"
            store = MeasurementStore(db_path)
            session_id = store.create_session(
                {
                    "condition_los": True,
                    "condition_nlos": False,
                    "constellation_label": "bench",
                    "send_start_stop": False,
                }
            )
            self._insert_sample(store, session_id, "7", 1.25)
            self._insert_sample(store, session_id, "8", 2.50)
            ensure_responder_label_schema(store.conn)
            store.conn.executemany(
                f"""
                INSERT INTO {RESPONDER_LABEL_TABLE} (
                    session_id, anchor_id, ground_truth_los_nlos, updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (session_id, "7", "LOS", now_local_iso()),
                    (session_id, "8", "NLOS", now_local_iso()),
                ],
            )
            store.conn.commit()
            store.close()

            rows = build_measurement_rows(db_path)

        labels = {row["anchor_id"]: row["ground_truth_los_nlos"] for row in rows}
        hints = {row["anchor_id"]: row["condition_hint"] for row in rows}
        self.assertEqual(labels, {"7": "LOS", "8": "NLOS"})
        self.assertEqual(hints, {"7": "LOS", "8": "LOS"})

    def test_per_anchor_workbooks_use_saved_per_responder_los_nlos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MeasurementStore(Path(tmp) / "measurements.sqlite")
            session_id = store.create_session(
                {
                    "condition_los": True,
                    "condition_nlos": False,
                    "constellation_label": "bench",
                    "send_start_stop": False,
                }
            )
            self._insert_sample(store, session_id, "7", 1.25)
            self._insert_sample(store, session_id, "8", 2.50)
            ensure_responder_label_schema(store.conn)
            store.conn.executemany(
                f"""
                INSERT INTO {RESPONDER_LABEL_TABLE} (
                    session_id, anchor_id, ground_truth_los_nlos, updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (session_id, "7", "LOS", now_local_iso()),
                    (session_id, "8", "NLOS", now_local_iso()),
                ],
            )
            store.conn.commit()
            paths = export_session_per_anchor(store.conn, session_id, Path(tmp) / "exports")
            store.close()

            by_anchor = {self._anchor_from_filename(path): path for path in paths}
            self.assertEqual(set(by_anchor), {"7", "8"})
            self.assertIn("_LOS_", by_anchor["7"].name)
            self.assertIn("_NLOS_", by_anchor["8"].name)
            self.assertEqual(self._first_sample_los_nlos(by_anchor["7"]), "LOS")
            self.assertEqual(self._first_sample_los_nlos(by_anchor["8"]), "NLOS")

    def test_per_anchor_export_filenames_include_export_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MeasurementStore(Path(tmp) / "measurements.sqlite")
            session_id = store.create_session(
                {
                    "condition_los": True,
                    "condition_nlos": False,
                    "constellation_label": "bench",
                    "send_start_stop": False,
                }
            )
            self._insert_sample(store, session_id, "7", 1.25)
            paths = export_session_per_anchor(
                store.conn,
                session_id,
                Path(tmp) / "exports",
                timestamp="20260619_143015",
            )
            store.close()

        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].name.endswith("_20260619_143015.xlsx"))

    def test_measurement_rows_apply_static_range_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "measurements.sqlite"
            store = MeasurementStore(db_path)
            session_id = store.create_session({"constellation_label": "bench"})
            self._insert_sample(store, session_id, "7", 1.25)
            store.close()

            rows = build_measurement_rows(db_path, range_static_offset_m=0.25)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["uwb_mean_distance_m"], 1.0)

    def test_per_anchor_workbook_applies_static_range_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MeasurementStore(Path(tmp) / "measurements.sqlite")
            session_id = store.create_session({"constellation_label": "bench"})
            self._insert_sample(store, session_id, "7", 1.25)
            paths = export_session_per_anchor(
                store.conn,
                session_id,
                Path(tmp) / "exports",
                range_static_offset_m=0.25,
            )
            store.close()

            wb = load_workbook(paths[0], read_only=True)
            ws = wb["Samples"]
            headers = [cell.value for cell in ws[1]]
            measured = ws.cell(row=2, column=headers.index("measured_uwb_distance_m") + 1).value
            wb.close()

        self.assertAlmostEqual(measured, 1.0)

    def test_measurement_list_default_filename_includes_export_timestamp(self) -> None:
        self.assertEqual(
            measurement_list_default_filename(
                Path("My Capture.sqlite"),
                timestamp="20260619_143015",
            ),
            "My_Capture_ground_truth_measurement_list_20260619_143015.xlsx",
        )

    def _insert_sample(
        self,
        store: MeasurementStore,
        session_id: str,
        anchor_id: str,
        distance_m: float,
    ) -> None:
        store.insert_sample(
            session_id,
            ParsedRecord(
                kind="sample",
                anchor_id=anchor_id,
                sample_index=1,
                distance_m=distance_m,
                status="ok",
                source="test",
                raw_line=f"S,{anchor_id},1,{distance_m}",
            ),
        )

    def _anchor_from_filename(self, path: Path) -> str:
        parts = path.name.split("_")
        return parts[parts.index("ANCHOR") + 1]

    def _first_sample_los_nlos(self, path: Path) -> str:
        wb = load_workbook(path, read_only=True)
        ws = wb["Samples"]
        headers = [cell.value for cell in ws[1]]
        col = headers.index("los_nlos") + 1
        value = ws.cell(row=2, column=col).value
        wb.close()
        return str(value)


if __name__ == "__main__":
    unittest.main()
