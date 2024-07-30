"""
Script Name: calc_functions.py
Description: <Brief description of what the script does>

Author: Santiago Castillo
Date Created: 24/7/2024
Last Modified: 
Version: 1.0

Dependencies:
    - numpy (pip install numpy)
    - warnings
    - pandas (pip install pandas)
    - matplotlib (pip install matplotlib)
    - fastdtw (pip install fastdtw)
    - scikit-learn (pip install scikit-learn)
    - hyperopt (pip install hyperopt)
    - networkx (pip install networkx)

"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from sklearn.preprocessing import MinMaxScaler
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import networkx as nx

# ignore warnings, they clutter the output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_rows', None)

# helper methods
def normalize_array(arr):
    reshaped_arr = arr.reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_arr = scaler.fit_transform(reshaped_arr)
    return normalized_arr

def custom_distance(p1, p2):
    return np.abs(p1 - p2) **(1/ 4.3)

def calc_tops(G, node1, node2, window = None, show_comparison = False):
    """
    Calculates the tops for the given graph nodes based on depth data and assigns the predicted 
    depths to the "tops" label in the destination node.

    Parameters:
    G : networkx.Graph
        The graph containing the nodes with their respective data.
    node1 : str
        The name of the current node (source node).
    node2 : str
        The name of the node you attempt to move to (destination node).
    window : list of int, optional
        A list of four integers defining the window range for filtering depth data in the format 
        [start1, end1, start2, end2]. Default is None.
    show_comparison : bool, optional
        If True, displays comparison plots of the profiles. Default is False.

    Returns:
    None
        The function modifies the graph in place by assigning the predicted depths to the "tops" 
        attribute of the destination node.

    Example:
    calc_tops(G, 'Node1', 'Node2', window=[1000, 2000, 1500, 2500], show_comparison=True)

    Notes:
    - Ensure that the graph G has nodes with 'data' and 'known_tops' attributes containing 
      the respective dataframes.
    - The function assumes that the 'known_tops' dataframe has a 'Capa' column.

    """

    # create a new dataframe that takes the values of node2 where the "capa" value is the same as in node1
    
    df1 = G.nodes[node1]["data"].copy() # RES_DEEP data for node1
    tops1 = G.nodes[node1]["known_tops"].copy() # Tops data for node1
    df2 = G.nodes[node2]["data"].copy() # RES_DEEP data for node2

    capas = G.nodes[node1]["known_tops"]["Capa"]

    if window:
        lower = np.min([window[0], window[2]])
        upper = np.max([window[1], window[3]])

        # verify correct window
        if len(window) != 4:
            print("invalid window")
            return
        
        start1 = window[0]
        end1 = window[1]
        start2 = window[2]
        end2 = window[3]

        # filter all the corresponding dataframes
        df1 = df1[(df1["DEPTH"]>=start1) & (df1["DEPTH"]<=end1)].copy()
        df2 = df2[(df2["DEPTH"]>=start2) & (df2["DEPTH"]<=end2)].copy()
        tops1 = tops1[(tops1["Ref"]>=start1) & (tops1["Ref"]<=end1)].reset_index(drop=True).copy()
        capas = tops1["Capa"].copy()
        
        if show_comparison:
            profiles_comparison(df1, df2, tops1, r=[lower, upper], name1=node1, name2=node2)

    df_result = predict(df1, df2, tops1)
    df_result["Capa"] = capas
    
    # assign new top values to the node
    G.nodes[node2]["tops"]=df_result

def dtw_calc(df1, df2, tops1):
    tolerance = 0.1 #original 0.05, was too small

    # mark rows within the tolerance range of any top
    
    for index_df1, row_df1 in df1.iterrows():
        depth_value = row_df1['DEPTH']
        matches = tops1['Ref'].apply(lambda x: np.abs(x - depth_value) <= tolerance)
        
        if matches.any():
            df1.loc[index_df1, 'Present'] = 1 # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead
            matched_index = matches.idxmax()
            tops1 = tops1.drop(index=matched_index) # if a match is found, delete the top to avoid duplicate matches
            
            # If tops1 is empty, break the loop
            if tops1.empty:
                break
            

    # store the filtered rows near the caps into the graph
    # Need to handle there being no matches present (for very small window), returntype must account for this
    prof1 = df1[df1['Present'] == 1]["DEPTH"].reset_index()["DEPTH"][0] - 50
    prof2 = df1[df1['Present'] == 1]["DEPTH"].reset_index()["DEPTH"].iloc[-1] + 50
    df1 = df1[(df1['DEPTH'] > prof1) & (df1['DEPTH'] < prof2)].reset_index()
    df2 = df2[(df2['DEPTH'] > prof1) & (df2['DEPTH'] < prof2)].reset_index()


    # normalize res deep in df1 and df2
    w1 = np.array(df1['RES_DEEP'].dropna())
    w2 = np.array(df2['RES_DEEP'].dropna())

    # use normalized values
    w1_normalized = normalize_array(w1)
    w2_normalized = normalize_array(w2)


    ref = df1[df1["Present"] == 1]
    distance, path = fastdtw(w1_normalized, w2_normalized, dist=custom_distance)
    correla = [tupla for tupla in path if tupla[0] in ref.index]

    return correla, df1, df2

def predict(df1, df2, tops1):
    correla, df1, df2 = dtw_calc(df1, df2, tops1)
    
    df_result = pd.DataFrame(columns=["Capa",'Ref'])

    # create a list of j values for unique i values
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
        j_depth = np.mean(j_depths) # averaging, may want to try a different method later
        #print(f'i = {i_depth}, j = {j_depths}') #print the j values for each corresponding i value
        rows_to_add.append({'Ref': j_depth})

    
    
    df_result = pd.concat([df_result, pd.DataFrame(rows_to_add)], ignore_index=True) # Future Warning

    return df_result

def profiles_comparison(df1, df2, tops1, r=None, vertical = False, name1 = None, name2 = None):

    correla, df1, df2 = dtw_calc(df1, df2, tops1)

    offset = 50
    plt.figure(figsize=(6, 10))

    # Plot df1
    plt.plot(df1["RES_DEEP"], df1["DEPTH"], label=name1, color='blue')

    # Plot df2 with offset
    plt.plot(df2["RES_DEEP"] + offset, df2["DEPTH"], label=name2, color='orange')


    for i, j in correla:
        # Plot the DTW path
        plt.plot([df1["RES_DEEP"].iloc[i], df2["RES_DEEP"].iloc[j] + offset], 
                 [df1["DEPTH"].iloc[i], df2["DEPTH"].iloc[j]], 
                 color='red', linestyle='-')
    if name1 is not None and name2 is not None:
        plt.title(f"Reference Node: {name1}, Prediction Node: {name2}")
    plt.xlabel('Valor')
    if r:
        plt.ylim(r[0], r[1])
    plt.ylabel('Depth')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()

def raw_data(G, names, r=None, offset = 50):

    colormap = plt.cm.viridis
    num_colors = len(names)
    colors = [colormap(i) for i in np.linspace(0, 1, num_colors)]

    plt.figure(figsize=(6, 10))
    index = 0
    for i, n in enumerate(names):
        df1 = G.nodes[n]["data"].copy()
        plt.plot(df1["RES_DEEP"] + offset*index, df1["DEPTH"], label=n, color=colors[i])
        index += 1

    plt.title(f"Raw Resistivity Data")
    plt.xlabel('Valor')

    if r:
        plt.ylim(r[0], r[1])
    plt.ylabel('Depth')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()

def windows_test(G, node_name1, node_name2, window, show_plots = False):
    start1 = window[0]
    end1 = window[1]
    start2 = window[2]
    end2 = window[3]

    data1 = G.nodes[node_name1]["data"].copy()
    data2 = G.nodes[node_name2]["data"].copy()
    tops1 = G.nodes[node_name1]["known_tops"].copy()
    tops2 = G.nodes[node_name2]["known_tops"].copy() # this is only used to verify

    f_data1 = data1[(data1["DEPTH"]>=start1) & (data1["DEPTH"]<=end1)]
    f_data2 = data2[(data2["DEPTH"]>=start2) & (data2["DEPTH"]<=end2)]
    f_tops1 = tops1[(tops1["Ref"]>=start1) & (tops1["Ref"]<=end1)].reset_index()
    f_tops2 = tops2[(tops2["Ref"]>=start2) & (tops2["Ref"]<=end2)].reset_index() # make sure that resetting index didn't mess up the algoritm

    capas = f_tops1["Capa"]
    capas2 = f_tops2["Capa"]
    

    result = predict(f_data1, f_data2, f_tops1)
    result["Capa"] = capas

    # note: there may be errors at end condition if window cuts off a top or base, will need to handle in the future

    error_df = result[result["Capa"].isin(capas2)].reset_index() 
    error_df.rename(columns={'Ref': 'pred'}, inplace=True)

    real = f_tops2[f_tops2["Capa"].isin(capas)].reset_index(drop=True)

    # handle edge case, clip real dataframe so that it has the same number of rows as error_df 
    # could encounter errors if dataframes are length 1
    if real.shape[0] != error_df.shape[0]:
        real_capa1 = real.iloc[1]['Capa']
        pred_capa1 = error_df.iloc[1]['Capa']

        # check whether to delete first or last row based on which capa was cut off
        if real_capa1 != pred_capa1:
            real = real.drop(0).reset_index(drop=True)
        else:
            real = real.drop(real.index[-1]).reset_index(drop=True) # deletes last row
        

    error_df["real"] = real["Ref"]
    error_df["abs error"] = np.abs(error_df['pred'] - error_df['real'])

    mae = np.mean(error_df["abs error"])
    mse = np.mean((error_df['pred'] - error_df['real'])**2)

    if show_plots:
        
        profiles_comparison(f_data1, f_data2, f_tops1, name1=node_name1, name2=node_name2)

        # error plot
        plt.figure(figsize=(12, 6))
        plt.plot(error_df['pred'], error_df['abs error'], label='Error Absoluto', color='purple')
        plt.xlabel('Depth (m)')
        plt.ylabel('Error Absoluto (m)')
        plt.title(f'Error Absoluto entre Predicción y Valores Verdaderos en {node_name2} con rango {start2}m - {end2}m')
        plt.legend()
        plt.show()
        
        # capas side by side plot
        plt.figure(figsize=(12, 6))
        plt.scatter(error_df['pred'], error_df['Capa'], color='blue', label='Predicción')
        plt.scatter(error_df['real'], error_df['Capa'], color='red', label='Verdadero')

        plt.xlabel('Depth (m)')
        plt.ylabel('Capa')
        plt.title(f'Evaluación de Capas {node_name2} con rango {start2}m - {end2}m')
        plt.legend()
        plt.show()
    return mae, mse

# optimization function to find the best window
# Source:
# https://towardsdatascience.com/hyperopt-demystified-3e14006eb6fa
def optimal_window(G, node_name1, node_name2, bounds, returnType = "mae", step_size = 0.5, num_iter = 50):    

    def objective(params):
        start1, end1, start2, end2 = params
        win = [start1, end1, start2, end2]
        if start1 >= end1 or start2 >= end2:
            return {'loss': float('inf'), 'status': STATUS_OK}  # Invalid window, return a large value

        try:
            mae, mse = windows_test(G, node_name1, node_name2, [start1, end1, start2, end2])
        except (ValueError, IndexError) as e:
            # Return a large value if an error occurs
            return {'loss': float('inf'), 'status': STATUS_OK}

        if returnType == "mse":
            return {'loss': mse, 'status': STATUS_OK}
        
        elif returnType == "mae":
            return {'loss': mae, 'status': STATUS_OK}
    
    # get first and last values of data
    zero1 = G.nodes[node_name1]["data"]["DEPTH"].iloc[0]
    inf1 = G.nodes[node_name1]["data"]["DEPTH"].iloc[-1]
    zero2 = G.nodes[node_name2]["data"]["DEPTH"].iloc[0]
    inf2 = G.nodes[node_name2]["data"]["DEPTH"].iloc[-1]


    # make sure bounds are reasonable, the bounds should not exceed the min and max data values
    bounds = [
        (max(bounds[0][0], zero1), max(bounds[0][1], zero1)),
        (min(bounds[1][0], inf1), min(bounds[1][1], inf1)),
        (max(bounds[2][0], zero2), max(bounds[2][1], zero2)),
        (min(bounds[3][0], inf2), min(bounds[3][1], inf2))
    ]

    # handling if a bounds tuple has the same value
    small_tol = 1
    for i in range(len(bounds)):
        if bounds[i][0] == bounds[i][1]:
            bounds[i] = (bounds[i][0]-small_tol, bounds[i][1]+small_tol)
            
    # bounds
    space = [
        hp.quniform('start1', bounds[0][0], bounds[0][1], step_size),
        hp.quniform('end1', bounds[1][0], bounds[1][1], step_size),
        hp.quniform('start2', bounds[2][0], bounds[2][1], step_size),
        hp.quniform('end2', bounds[3][0], bounds[3][1], step_size)
    ]
 
    trials = Trials()
    
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=num_iter,
        trials=trials,
        rstate=np.random.default_rng(0)
    )
    
    optimal_windows = [best['start1'], best['end1'], best['start2'], best['end2']]
    min_err = min(trial['result']['loss'] for trial in trials)

    return optimal_windows, min_err
    

def clear_tops(G, node_name):
    G.nodes[node_name]["tops"] = pd.DataFrame()

