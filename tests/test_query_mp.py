import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
from src.mp_screening.query_mp import query_stable_li_materials

class TestQueryMP(unittest.TestCase):

    @patch('src.mp_screening.query_mp.MPRester')
    @patch('src.mp_screening.query_mp.os.environ.get')
    def test_query_stable_li_materials(self, mock_env_get, mock_mprester):
        # Mock environment variable
        mock_env_get.return_value = "fake_api_key"
        
        # Mock document returned by MPRester
        mock_doc = MagicMock()
        mock_doc.material_id = "mp-1234"
        mock_doc.formula_pretty = "Li2O"
        mock_doc.energy_above_hull = 0.01
        mock_doc.formation_energy_per_atom = -2.5
        mock_doc.elements = ["Li", "O"]
        
        # Setup the mock chain for MPRester Context Manager
        mock_instance = mock_mprester.return_value.__enter__.return_value
        mock_instance.summary.search.return_value = [mock_doc]
        
        # Call the function
        df = query_stable_li_materials(limit=1)
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        
        expected_columns = ["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom"]
        self.assertListEqual(list(df.columns), expected_columns)
        
        self.assertEqual(df.iloc[0]['formula_pretty'], "Li2O")
        self.assertEqual(df.iloc[0]['material_id'], "mp-1234")

if __name__ == "__main__":
    unittest.main()
