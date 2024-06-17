import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from iu_project import DataVisualiser

class TestDataVisualizer(unittest.TestCase):
    """
    Unit tests for the DataVisualiser class.
    """

    @patch('iu_project.db.create_engine')
    def setUp(self, mock_create_engine):
        """
        Set up the test environment by mocking the SQLAlchemy engine and connection.

        Parameters:
        mock_create_engine (MagicMock): Mock for the SQLAlchemy create_engine function.
        """
        self.mock_engine = MagicMock()
        self.mock_conn = MagicMock()
        mock_create_engine.return_value = self.mock_engine
        self.mock_engine.connect.return_value = self.mock_conn

        # Instantiate the DataVisualiser
        self.visualizer = DataVisualiser()

    def test_load_csv_to_db(self):
        """
        Test the load_csv_to_db method of DataVisualiser by mocking pandas read_csv and to_sql methods.
        """
        with patch('pandas.read_csv') as mock_read_csv, patch('pandas.DataFrame.to_sql') as mock_to_sql:
            # Create a mock DataFrame
            mock_df = pd.DataFrame({
                'x': [1, 2, 3],
                'y1': [4, 5, 6],
                'y2': [7, 8, 9],
                'y3': [10, 11, 12],
                'y4': [13, 14, 15]
            })
            mock_read_csv.return_value = mock_df

            # Call the method
            self.visualizer.load_csv_to_db('mock_path.csv', 'mock_table')

            # Assert that read_csv and to_sql were called
            mock_read_csv.assert_called_once_with('mock_path.csv')
            mock_to_sql.assert_called_once()

    def test_fetch_data(self):
        """
        Test the fetch_data method of DataVisualiser by mocking pandas read_sql_table method.
        """
        mock_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [4, 5, 6],
            'y2': [7, 8, 9],
            'y3': [10, 11, 12],
            'y4': [13, 14, 15]
        })
        with patch('pandas.read_sql_table', return_value=mock_df) as mock_read_sql_table:
            # Call the method
            result = self.visualizer.fetch_data('mock_table')

            # Assert that read_sql_table was called and returned the correct DataFrame
            mock_read_sql_table.assert_called_once_with('mock_table', con=self.visualizer.engine)
            pd.testing.assert_frame_equal(result, mock_df)

if __name__ == '__main__':
    """
    Run the unit tests.
    """
    unittest.main()