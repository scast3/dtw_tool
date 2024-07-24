"""
Script Name: solve_functions.py
Description: <Brief description of what the script does>

Author: Santiago Castillo
Date Created: 
Last Modified: 
Version: 1.0

Dependencies:
    - numpy (pip install numpy)
    - pandas (pip install pandas)
    - matplotlib (pip install matplotlib)
    - lasio (pip install lasio)
    - os (pip install os)
    - glob (pip install glob)
    - geodesic (pip install geodesic)
    - networkx (pip install networkx)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lasio
import os
import glob
from geopy.distance import geodesic
import networkx as nx


# directory on personal computer
ruta = os.getcwd()+"/Datos"
#os.chdir(ruta)

ruta_logs = ruta + "/Datos VH_III"
ruta_archivo = ruta_logs +"/Datos VH_III.xlsx"
markers_file = ruta_logs + "/Markers VH3.xlsx"

las_files = glob.glob(os.path.join(ruta_logs, '*.las'))
log_files = glob.glob(os.path.join(ruta_logs, '*.log'))


# Create a directed graph
G = nx.DiGraph()

# get data from excel
well_data = pd.read_excel(ruta_archivo)
markers = pd.read_excel(markers_file)

def las_to_node(G, filename, col_name):
    node_name = os.path.splitext(os.path.basename(filename))[0]
    las = lasio.read(filename)
    las_df = las.df().dropna()
    las_df = las_df.reset_index()
    new_las = pd.DataFrame(columns=["DEPTH",'RES_DEEP'])
    new_las["DEPTH"] = las_df["DEPTH"]
    new_las["RES_DEEP"] = las_df[col_name]

    coord_data = well_data[well_data["Nombre"]==node_name]
    x_coord = coord_data.iloc[0, 5]
    y_coord = coord_data.iloc[0, 6]

    G.add_node(node_name, data=new_las, pos=(x_coord, y_coord))

    # Add tops
    tops_data = pd.read_excel(ruta_archivo, sheet_name=node_name)
    tops_data = tops_data.drop(index=0)

    a = tops_data[["Capa", "Top"]].copy()
    a.rename(columns={"Top": "Ref"}, inplace=True)

    

    b=tops_data[["Capa", "Base"]].copy()
    b.rename(columns={"Base": "Ref"}, inplace=True)
    resultado = pd.concat([a, b], ignore_index=True)

    tops_vals = resultado.sort_values(by='Ref', ascending=True).reset_index().dropna()

    # print(f"Resultados {node_name}")
    # print(resultado.head(5))
    # print()


    G.nodes[node_name]["markers"] = markers[markers["Pozo"]==node_name]

    G.nodes[node_name]["known_tops"]=tops_vals.copy()

# TODO finish this, .log files currently do not work
def log_to_node(G, filename, col_name):
    log_data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Identify header and determine the indices of the columns of interest
    header = lines[0].strip().split()
    depth_idx = header.index("DEPTH")
    col_idx = header.index(col_name)

    new_log = pd.DataFrame(columns=["DEPTH",'RES_DEEP'])

    new_log["DEPTH"] = lines[depth_idx]
    new_log["RES_DEEP"] = lines[col_idx]

    print(lines[2])


las_to_node(G, las_files[0], "HT90")
las_to_node(G, las_files[1], "HT90")
las_to_node(G, las_files[2], "HDRS")

#log_to_node(G, log_files[0], "M2R9")
ruta_logs = ruta + "/Perfiles_3"
las_files = glob.glob(os.path.join(ruta_logs, '*.las'))

las_to_node(G, las_files[0], "RES_DEEP")
las_to_node(G, las_files[1], "RES_DEEP")
las_to_node(G, las_files[2], "RES_DEEP")


def load_markers(G, node_name):
    G.nodes[node_name]["markers"] = markers[markers["Pozo"]==node_name]

#%%
# Function to calculate geodesic distance between two points
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def euclidian_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

threshold_distance = 400  # Define a distance threshold

for node1, data1 in G.nodes(data=True):
    for node2, data2 in G.nodes(data=True):
        if node1 != node2:
            pos1 = data1['pos']
            pos2 = data2['pos']
            distance = euclidian_distance(pos1[0], pos1[1], pos2[0], pos2[1])
            if distance <= threshold_distance:
                G.add_edge(node1, node2, weight=distance)

# Visualize the graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title("Grafo Basado en Cordenadas x,y")
plt.show()