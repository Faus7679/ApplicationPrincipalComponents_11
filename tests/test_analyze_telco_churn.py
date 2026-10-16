import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_telco_churn import analyze_dataset, load_dataset


class TelcoAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "customerID": [f"id-{index}" for index in range(12)],
                "tenure": [index * 3 for index in range(12)],
                "MonthlyCharges": [20 + index * 4 for index in range(12)],
                "TotalCharges": [0 if index == 0 else index * 50 for index in range(12)],
                "SeniorCitizen": [index % 2 for index in range(12)],
                "Contract": ["Month-to-month" if index % 2 else "One year" for index in range(12)],
                "Churn": ["Yes" if index % 3 == 0 else "No" for index in range(12)],
            }
        )

    def test_csv_input_and_analysis_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "telco.csv"
            self.data.to_csv(data_path, index=False)
            loaded = load_dataset(data_path)
            summary = analyze_dataset(loaded, root / "results", sample_size=12)

            self.assertEqual(summary["rows"], len(self.data))
            self.assertGreaterEqual(summary["components_for_threshold"], 1)
            self.assertTrue((root / "results" / "pca_score_plots.png").is_file())
            self.assertTrue((root / "results" / "numeric_correlations.csv").is_file())
            loadings = pd.read_csv(root / "results" / "pca_loadings.csv")
            self.assertEqual(
                len(loadings.columns) - 1, summary["components_for_threshold"]
            )

    def test_rejects_unknown_input_extension(self):
        with self.assertRaisesRegex(ValueError, "must be a"):
            load_dataset("data.json")

    def test_xlsx_input(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "telco.xlsx"
            self.data.to_excel(data_path, index=False)
            loaded = load_dataset(data_path)
            self.assertEqual(loaded.shape, self.data.shape)


if __name__ == "__main__":
    unittest.main()
