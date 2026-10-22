import unittest
from pathlib import Path

from src import dataco_tools
from src.dataco_country_coords import COUNTRY_COORDS
from src import dataco_map


class DataCoMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = dataco_tools._get_conn()

    def test_sensitive_contact_fields_are_not_in_agent_table(self):
        columns = {row[1] for row in self.conn.execute("DESCRIBE dataco_orders").fetchall()}
        for sensitive in ("customer_email", "customer_fname", "customer_lname", "customer_password",
                          "customer_street", "customer_zipcode", "product_image"):
            self.assertNotIn(sensitive, columns)
        self.assertIn("order_id", columns)
        self.assertIn("order_country", columns)

    def test_customer_and_destination_country_names_have_coordinates(self):
        names = set()
        for row in self.conn.execute(
            "SELECT DISTINCT TRIM(CAST(customer_country AS VARCHAR)) FROM dataco_orders "
            "UNION SELECT DISTINCT TRIM(CAST(order_country AS VARCHAR)) FROM dataco_orders"
        ).fetchall():
            if row[0]:
                names.add(row[0])
        missing = sorted(name for name in names if name not in COUNTRY_COORDS)
        self.assertEqual(missing, [])

    def test_map_filters_and_summary_read_real_dataset(self):
        filters = dataco_map.get_map_filters()
        summary = dataco_map.get_map_summary()
        self.assertTrue(filters["markets"])
        self.assertTrue(filters["shipping_modes"])
        self.assertGreater(summary["orders"], 0)
        self.assertGreater(summary["line_items"], 0)

    def test_country_flows_return_points_and_metrics(self):
        result = dataco_map.get_country_flows(limit=20, min_orders=20, metric="orders")
        self.assertNotIn("error", result)
        self.assertGreater(result["n_flows"], 0)
        flow = result["flows"][0]
        self.assertIn("lat", flow["origin"])
        self.assertIn("lng", flow["destination"])
        self.assertGreaterEqual(flow["orders"], 20)

    def test_supported_metric_set_and_unknown_coordinate(self):
        self.assertEqual(dataco_map._ALLOWED_METRICS, {"orders", "late_risk_pct", "avg_shipping_days", "sales"})
        self.assertIsNone(dataco_map._clean_coord("Unknown Location"))


if __name__ == "__main__":
    unittest.main()
