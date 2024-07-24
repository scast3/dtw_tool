"""
Script Name: solve_functions.py
Description: Assuming a networkx graph G has the correct node values (resistivity profile) and edges (distance),
    these functions will traverse the graph to predict values through certain parameters and paths

Author: Santiago Castillo
Date Created: 24/7/2024
Last Modified: 
Version: 1.0

Dependencies:
    - networkx (pip install networkx)

"""

from calc_functions import optimal_window, calc_tops, windows_test
import networkx as nx

# This function will take several minutes to run
def train_window(G, train_nodes):
    for node in train_nodes:
        for neighbor in G.neighbors(node):
            if neighbor in train_nodes:
                
                edge_data = G.get_edge_data(node, neighbor)

                print(f"Calculating optimal window between {node} and {neighbor}...")
                bounds = [(900, 1200), (1800, 2000), (900, 1200), (1800, 2000)] # this may need to change based on the profile of the data
                opt, err = optimal_window(G, node, neighbor, bounds)
                
                G.add_edge(node, neighbor, data = opt)
                print(f"Edge: {node} - {neighbor}, dist: {edge_data['weight']}, optimal window: {edge_data['data']}, mean avg error: {err}")

def calc_tops_dijkstra(G, start, end, window):
    # Use Dijkstra's algorithm to find the shortest path
    try:
        shortest_path = nx.dijkstra_path(G, source=start, target=end)
        print(f"Shortest Path: {shortest_path}")
    except nx.NetworkXNoPath:
        print("No path exists between the start and end nodes.")
        return

    # Traverse the path and execute calc_tops between each pair of nodes
    for i in range(len(shortest_path) - 1):
        node1 = shortest_path[i]
        node2 = shortest_path[i + 1]
        calc_tops(G, node1, node2, window)

def calc_tops_shortest_jumps(G, start, end, window, show_plots = False):
    current_node = start
    visited = set()
    path = [start]

    while current_node != end:
        # Get neighbors of the current node
        neighbors = list(G.neighbors(current_node))

        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        
        if not unvisited_neighbors:
            print(f"No unvisited neighbors found for node {current_node}.")
            return
        
        # Find the neighbor with the shortest edge weight
        shortest_edge = min(unvisited_neighbors, key=lambda n: G[current_node][n]['weight'])
        
        visited.add(current_node)
        
        # Execute the calc_tops function
        calc_tops(G, current_node, shortest_edge, window=window)
        err, _ = windows_test(G, current_node, shortest_edge, window)
        print(f"executing calculation between {current_node} and {shortest_edge}. MAE: {err:.2f}")
        
        # Move to the next node
        current_node = shortest_edge
        path.append(current_node)

    if show_plots:
        windows_test(G, start, end, window, show_plots=True)

    return path

