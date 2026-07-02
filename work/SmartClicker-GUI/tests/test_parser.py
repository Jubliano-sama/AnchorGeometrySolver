import unittest

from uwb_capture.parser import parse_serial_line


class ParserTests(unittest.TestCase):
    def test_sample_row(self) -> None:
        record = parse_serial_line("S,7,3,4.1250")[0]

        self.assertEqual(record.kind, "sample")
        self.assertEqual(record.anchor_id, "7")
        self.assertEqual(record.sample_index, 3)
        self.assertEqual(record.distance_m, 4.125)
        self.assertEqual(record.status, "ok")

    def test_failure_row_with_error_code(self) -> None:
        record = parse_serial_line("F,7,3,-110")[0]

        self.assertEqual(record.kind, "failure")
        self.assertEqual(record.anchor_id, "7")
        self.assertEqual(record.sample_index, 3)
        self.assertEqual(record.error_code, "-110")

    def test_summary_row_uses_last_distance(self) -> None:
        record = parse_serial_line("M,7,4.0000,98,2,100,4.1250")[0]

        self.assertEqual(record.kind, "summary")
        self.assertEqual(record.good_count, 98)
        self.assertEqual(record.mean_distance_m, 4.0)
        self.assertEqual(record.distance_m, 4.125)

    def test_quality_nlos_phy_rows(self) -> None:
        quality = parse_serial_line(
            "Q,7,-70.00,-82.00,100,20,5,5,1,2,3,400,5,100,8,900"
        )[0]
        nlos = parse_serial_line("NLOS,7,30,4,7.2,1.1,1.2,1.3,20.0,3.0")[0]
        phy = parse_serial_line("PHY,5,64,9,1024,850,8,7,3,0,0x85858585")[0]

        self.assertEqual(quality.kind, "quality")
        self.assertEqual(quality.rx_power_dbm, -70.0)
        self.assertEqual(quality.cir_power, 400.0)
        self.assertEqual(nlos.kind, "nlos")
        self.assertEqual(nlos.anchor_id, "7")
        self.assertEqual(phy.kind, "phy")


if __name__ == "__main__":
    unittest.main()
