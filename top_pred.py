import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lasio
import os
import glob
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import networkx as nx
import scipy.interpolate
import time



# helper methods
def normalize_array(arr):
    reshaped_arr = arr.reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_arr = scaler.fit_transform(reshaped_arr)
    return normalized_arr

def custom_distance(p1, p2):
    return np.abs(p1 - p2) **(1/ 4.1)



# node 1 is the curernt node
# node 2 is the node you attempt to move to
# return: a dataframe with the predicted depths
def calc_tops(G, node1, node2):
    tolerance = 0.05
    df1 = G.nodes[node1]["data"] # RES_DEEP data for node1
    tops = G.nodes[node1]["tops"] # Tops data for node1

    df2 = G.nodes[node2]["data"] # RES_DEEP data for node2



    # mark rows within the tolerance range of any top
    for index_df1, row_df1 in df1.iterrows():
        depth_value = row_df1['DEPTH']
        matches = tops['Ref'].apply(lambda x: np.abs(x - depth_value) <= tolerance)
        if matches.any():
            df1.at[index_df1, 'Present'] = 1

    G.nodes[node1]["is_top"] = df1[df1['Present'] == 1] # store the filtered rows near the topes into the graph
    prof1 = df1[df1['Present'] == 1]["DEPTH"].reset_index()["DEPTH"][0] - 50
    prof2 = df1[df1['Present'] == 1]["DEPTH"].reset_index()["DEPTH"].iloc[-1] + 50
    df1 = df1[(df1['DEPTH'] > prof1) & (df1['DEPTH'] < prof2)].reset_index()
    df2 = df2[(df2['DEPTH'] > prof1) & (df2['DEPTH'] < prof2)].reset_index()


    # normalize res deep in df1 and df2
    w1 = np.array(df1['RES_DEEP'].dropna())
    w1_normalized = normalize_array(w1)
    df1['w1_normalized'] = w1_normalized

    w2 = np.array(df2['RES_DEEP'].dropna())
    w2_normalized = normalize_array(w2)


    # Calculate DTW Distance
    ref = df1[df1["Present"] == 1]
    distance, path = fastdtw(w1, w2, dist=custom_distance)
    correla = [tupla for tupla in path if tupla[0] in ref.index]
    correla2 = [tupla[1] for tupla in correla]

    # mark tops in df2
    df2["is_top"] = 0
    for elemento in correla2:
        if elemento in df2.index:
            df2.loc[elemento, "is_top"] = 1
    G.nodes[node2]["is_top"] = df2[df2["is_top"] == 1]
    G.nodes[node2]["Procesados"] = df2

    # create a list of j values for unique i values
    df_result = pd.DataFrame(columns=["Capa",'Ref'])
    j_values_dict = {}
    for i, j in correla:
        
        if i not in j_values_dict:
            j_values_dict[i] = []
        j_values_dict[i].append(j)

    rows_to_add = []
    # obtaining a 1 to 1 relationship between i and j by averaging 
    for i, j_values in j_values_dict.items():
        i_depth = df1["DEPTH"].iloc[i]
        j_depths = [df2["DEPTH"].iloc[j] for j in j_values]
        # averaging j values
        j_depth = np.mean(j_depths)
        #print(f'i = {i_depth}, j = {j_depth}')
        rows_to_add.append({'Ref': j_depth})
    capas = ["O4T", "O4T", "O5T", "O5T", "O5aT", "O5aT", "O5bT", "O5bT", "O7T", "O7T", ]
    df_result = pd.concat([df_result, pd.DataFrame(rows_to_add)], ignore_index=True)
    df_result["Capa"] = capas

    G.nodes[node2]["tops"]=df_result
    print(G.nodes[node2]["tops"])


def dtw_calc(G, node1, node2):

def profiles_comparison(G, node1, node2):
    for i, j in correla:
        # Plot the DTW path
        plt.plot([df1["DEPTH"].iloc[i], df2["DEPTH"].iloc[j]], [df1["RES_DEEP"].iloc[i], df2["RES_DEEP"].iloc[j] + offset], color='red', linestyle='-')

    plt.title(node2 + " " + node1)
    plt.xlabel('Depth (m)')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()