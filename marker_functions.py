"""
Script Name: marker_functions.py
Description: Funciones para crear ventanas usando marcadores

Author: Santiago Castillo
Date Created: 24/7/2024
Last Modified: 1/8/2024
Version: 1.0

Dependencies:
    - pandas (pip install pandas)
"""

import pandas as pd
from calc_functions import optimal_window

# helper method
def create_subs(arr):
    result = [(0, arr[0])]
    
    # Add pairs of consecutive elements
    for i in range(len(arr) - 1):
        result.append((arr[i], arr[i + 1]))
    
    # Add the final pair with infinity
    if len(arr) > 0:
        result.append((arr[-1], float('inf')))
    
    return result

def mkr_calc_window(G, node_name1, node_name2, show_ranges = False, show_error = False):
    """
    Calcula las ventanas óptimas de marcadores para dos nodos en un grafo, considerando el rango de profundidad de 
    los marcadores comunes entre los nodos.

    Parámetros:
    G : networkx.Graph
        El grafo que contiene los nodos con los datos de marcadores.
    node_name1 : str
        El nombre del nodo referencia en el grafo.
    node_name2 : str
        El nombre del segundo nodo en el grafo.
    show_ranges : bool, opcional
        Si es True, se mostrará el rango de marcadores (por defecto es False).
    show_error : bool, opcional
        Si es True, se mostrará el error asociado a las ventanas óptimas (por defecto es False).

    Retorna:
    pandas.DataFrame
        Un DataFrame con las siguientes columnas:
        - "Label": Etiquetas de los rangos de marcadores.
        - f"Range {node_name1}": Rango de profundidad del label para el primer nodo.
        - f"Range {node_name2}": Rango de profundidad del label para el segundo nodo.
        - f"Opt Win {node_name1}": Ventana óptima para el primer nodo.
        - f"Opt Win {node_name2}": Ventana óptima para el segundo nodo.
        - "Error": Error asociado a la ventana óptima.

    Ejemplo:
    mkr_calc_window(G, "Nodo1", "Nodo2", show_ranges=True, show_error=True)

    Notas:
    - Asegúrese de que ambos nodos tengan la clave "markers" que contenga los datos de marcadores y profundidades.
    - La función crea rangos de profundidad basados en los marcadores comunes entre los dos nodos y calcula 
      las ventanas óptimas y el error asociado.
    """

    mkr_df = pd.DataFrame(columns=["Label", f"Range {node_name1}", f"Range {node_name2}", 
                                   f"Opt Win {node_name1}", f"Opt Win {node_name2}", "Error"])

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

    # run the optimization for the narrowed marker window
    m_tol = 5 # +/- bounds based on marker depth
    for i, label in enumerate(labels):
        mkr_df.at[i, f"Range {node_name1}"] = sub1[i]
        mkr_df.at[i, f"Range {node_name2}"] = sub2[i]

        print(f"Calculating windows for marker range {label}")
        bounds = [(sub1[i][0]-m_tol, sub1[i][0]+m_tol),
                (sub1[i][1]-m_tol, sub1[i][1]+m_tol),
                (sub2[i][0]-m_tol, sub2[i][0]+m_tol),
                (sub2[i][1]-m_tol, sub2[i][1]+m_tol)]
        

        opt, error = optimal_window(G, node_name1, node_name2, bounds)

        print(f"Optimal windows: {opt}")
        print(f"Minimum error: {error}")

        mkr_df.at[i, "Error"] = error
        mkr_df.at[i, f"Opt Win {node_name1}"] = (opt[0],opt[1])
        mkr_df.at[i, f"Opt Win {node_name2}"] = (opt[2],opt[3])

    # need to implement later
    if show_ranges:
        print("show ranges")

    if show_error:
        print("show error")

    return mkr_df

