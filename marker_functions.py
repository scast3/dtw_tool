import pandas as pd
from graph_setup import G
import numpy as np

def mkr_calc_window(G, node_name1, node_name2):
    mkr_df = pd.DataFrame(["Label", f"Range {node_name1}", f"Range {node_name2}", "Error"])

    mark1 = G.nodes[node_name1]["markers"].copy()
    mark2 = G.nodes[node_name2]["markers"].copy()

    common_markers = mark1.merge(mark2, on="Mkr")
    common_markers = common_markers.rename(columns={"Depth_x": node_name1})
    common_markers = common_markers.rename(columns={"Depth_y": node_name2})

    marker_names = common_markers["Mkr"]
    mark_depths1 = common_markers[node_name1].tolist()
    mark_depths2 = common_markers[node_name2].tolist()

    zero1 = G.nodes[node_name1]["data"]["DEPTH"].iloc[0]
    inf1 = G.nodes[node_name1]["data"]["DEPTH"].iloc[-1]

    print(zero1, inf1)

    def create_subs(arr):
        result = [(5, arr[0])]
        
        # Add pairs of consecutive elements
        for i in range(len(arr) - 1):
            result.append([arr[i], arr[i + 1]])
        
        # Add the final pair with infinity
        if len(arr) > 0:
            result.append([arr[-1], 6000])
        
        return result

    sub1 = create_subs(mark_depths1)
    sub2 = create_subs(mark_depths2)

    print(sub1)
    print(sub2)


mkr_calc_window(G, "PVH-937", "PVH-941")