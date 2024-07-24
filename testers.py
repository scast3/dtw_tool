# Save this as test_calc_functions.py
import unittest
import pandas as pd
from calc_functions import profiles_comparison
from graph_setup import G
# python -m unittest test_calc_functions.py


class TestCalcFunctions(unittest.TestCase):

    def setUp(self):
        # Create sample dataframes for testing
        self.df1 = pd.DataFrame({'RES_DEEP': [100, 150, 200, 250, 300], 'DEPTH': [10, 20, 30, 40, 50]})
        self.df2 = pd.DataFrame({'RES_DEEP': [120, 170, 220, 270, 320], 'DEPTH': [10, 20, 30, 40, 50]})
        self.tops1 = pd.DataFrame({'Ref': [15, 35], 'Capa': ['A', 'B']})

    def test_profiles_comparison(self):
        # Test the profiles_comparison function
        profiles_comparison(self.df1, self.df2, self.tops1, name1='Sample1', name2='Sample2')
        # No assertions needed here as we are just testing if the function runs without error

    #def test_calc_tops(self):

if __name__ == '__main__':
    unittest.main()
