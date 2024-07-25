import pytest
import pandas as pd
from calc_functions import calc_tops, dtw_calc, predict, profiles_comparison, windows_test, optimal_window
from graph_setup import G

def test_calc_tops_1():
    n1 = "PVH-941"
    n2 = "PVH-937"
    calc_tops(G, n1, n2, [0, 1, 0, 1])
    assert "tops" in G.nodes[n2], "The key 'tops' does not exist in the node"
    assert not G.nodes[n2]["tops"].empty, "The DataFrame is empty"

if __name__ == "__main__":
    pytest.main()
