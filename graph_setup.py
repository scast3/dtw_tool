"""
Script Name: graph_setup.py
Description: Creando los nodos y enlaces del grafo G

Author: Santiago Castillo
Date Created: 24/7/2024
Last Modified: 1/8/2024
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


# directorio
ruta = os.getcwd()+"/Datos"

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
    """
    Convierte un archivo .las que contiene un perfil en un nodo del grafo G.

    Parámetros:
    G : networkx.Graph
        El grafo al que se añadirá el nodo.
    filename : str
        El nombre del archivo .las que contiene los datos del perfil.
    col_name : str
        El nombre de la columna que contiene los datos de resistividad.

    Retorna:
    None
        La función modifica el grafo G añadiendo un nuevo nodo con los datos del archivo .las.

    Ejemplo:
    las_to_node(G, perfil.las, "HT90")

    Notas:
    - Asegúrese de que el archivo .las contenga una columna de profundidad 'DEPTH' y una columna con los datos 
      de resistividad especificada por 'col_name'.
    - El archivo de tops debe estar en formato Excel y debe tener una hoja con el nombre del pozo/nodo.
    """

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


    # Assigning markers for each node
    G.nodes[node_name]["markers"] = markers[markers["Pozo"]==node_name]
    G.nodes[node_name]["markers"] = G.nodes[node_name]["markers"].drop("Pozo", axis=1)
    G.nodes[node_name]["markers"] = G.nodes[node_name]["markers"].rename(columns={"Depth (meters)": "Depth"})


    G.nodes[node_name]["known_tops"]=tops_vals.copy()

# lo mismo pero para .log files
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



# Cambiar el directorio para crear el grafo G

ruta_logs = ruta + "/Perfiles_3"
las_files = glob.glob(os.path.join(ruta_logs, '*.las'))

las_to_node(G, las_files[0], "RES_DEEP")
las_to_node(G, las_files[1], "RES_DEEP")
las_to_node(G, las_files[2], "RES_DEEP")


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

plt.title("Grafo de Pozos")
#plt.show()
