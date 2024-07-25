import pandas as pd
from graph_setup import G
import numpy as np
from calc_functions import optimal_window

def create_subs(arr):
    result = [(0, arr[0])]
    
    # Add pairs of consecutive elements
    for i in range(len(arr) - 1):
        result.append([arr[i], arr[i + 1]])
    
    # Add the final pair with infinity
    if len(arr) > 0:
        result.append([arr[-1], float('inf')])
    
    return result

def mkr_calc_window(G, node_name1, node_name2, show_plot = False):
    mkr_df = pd.DataFrame(columns=["Label", f"Range {node_name1}", f"Range {node_name2}", "Error"])

    mark1 = G.nodes[node_name1]["markers"].copy()
    mark2 = G.nodes[node_name2]["markers"].copy()

    # finding the markers that exist in both wells
    common_markers = mark1.merge(mark2, on="Mkr")
    common_markers = common_markers.rename(columns={"Depth_x": node_name1})
    common_markers = common_markers.rename(columns={"Depth_y": node_name2})

    marker_names = common_markers["Mkr"]
    mark_depths1 = common_markers[node_name1].tolist()
    mark_depths2 = common_markers[node_name2].tolist()

    # creating appropriate labels for the ranges
    labels = []
    prev_marker = "0"
    for marker in marker_names:
        labels.append(f"{prev_marker} - {marker}")
        prev_marker = marker
    labels.append(f"{prev_marker} - inf")

    # Add labels to the DataFrame
    mkr_df["Label"] = labels

    sub1 = create_subs(mark_depths1)
    sub2 = create_subs(mark_depths2)

    print(sub1)
    print(sub2)

    # run the optimization for the narrowed marker window
    m_tol = 5
    for i, label in enumerate(labels):
        print(f"Calculating windows for marker range {label}")
        bounds = [(sub1[i][0]-m_tol, sub1[i][0]+m_tol),
                (sub1[i][1]-m_tol, sub1[i][1]+m_tol),
                (sub2[i][0]-m_tol, sub2[i][0]+m_tol),
                (sub2[i][1]-m_tol, sub2[i][1]+m_tol)]
        

        opt, error = optimal_window(G, node_name1, node_name2, bounds)

        print(f"Optimal windows: {opt}")
        print(f"Minimum error: {error}")

    if show_plot:
        print("lol!")

    return mkr_df

marker_ranges = mkr_calc_window(G, "PVH-937", "PVH-941")
print(marker_ranges)