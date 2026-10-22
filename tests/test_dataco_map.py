import unittest

from src.dataco_country_coords import COUNTRY_COORDS
from src.dataco_map import _ALLOWED_METRICS, _clean_coord


class DataCoMapTests(unittest.TestCase):
    def test_supported_metric_set(self):
        self.assertEqual(
            _ALLOWED_METRICS,
            {"orders", "late_risk_pct", "avg_shipping_days", "sales"},
        )

    def test_dataset_customer_country_labels_have_coordinates(self):
        # The DataCo CSV uses Spanish labels, including the dataset-specific US alias.
        for country in ("EE. UU.", "Puerto Rico", "Estados Unidos", "España", "India", "Vietnam"):
            self.assertIn(country, COUNTRY_COORDS)
            self.assertIsNotNone(_clean_coord(country))

    def test_country_coordinates_are_valid_lat_lon(self):
        for country, (lat, lon) in COUNTRY_COORDS.items():
            with self.subTest(country=country):
                self.assertGreaterEqual(lat, -90)
                self.assertLessEqual(lat, 90)
                self.assertGreaterEqual(lon, -180)
                self.assertLessEqual(lon, 180)

    def test_unknown_country_is_not_given_a_fake_coordinate(self):
        self.assertIsNone(_clean_coord("Unknown Location"))


if __name__ == "__main__":
    unittest.main()
