# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:52:42 2016

@author: Group 1
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Function to read in two adjacency matrices in Q2
# If the 0-1 adjacency matrix contains more edges than the weighted adjacency matrix, 
# Add the 0-weight edges to the final graph
def graphAdjMatrix(nonweightedAdjMat, weightedAdjMat):
    
    G1 = nx.Graph(nonweightedAdjMat)
    G2 = nx.Graph(weightedAdjMat)
    
    G1_edge = G1.edges()
    G2_edge = G2.edges()
    
    if len(G1_edge) > len(G2_edge):
        for (src, dest) in G1_edge:
            if (src, dest) not in G2_edge:
                G2.add_edge(src, dest, weight = 0)
                
    return G2

DGraph = nx.read_weighted_edgelist("HW1_problem1.txt",create_using=nx.DiGraph())

# Graph vis
plt.figure(figsize = (8, 6))

pos = nx.circular_layout(DGraph)
edge_labels = dict([((u,v,), d['weight']) for u,v,d in DGraph.edges(data=True)])
nx.draw_networkx_edge_labels(DGraph, pos, edge_labels = edge_labels)
nx.draw_networkx_labels(DGraph, pos)
nx.draw(DGraph, pos, node_size = 500, node_color = 'lightblue')

# Qn 1a: Print Incidence Matrix
incidence_matrix = nx.incidence_matrix(DGraph, nodelist = sorted(DGraph.nodes()), oriented=True, weight = None)
print(pd.DataFrame(incidence_matrix.todense(), index = sorted(DGraph.nodes())))

# Qn 1b: Shortest Paths Matrix
shortestpaths = nx.floyd_warshall_numpy(DGraph, nodelist = sorted(DGraph.nodes()), weight = 'weight')
print(pd.DataFrame(shortestpaths, columns = sorted(DGraph.nodes()), index = sorted(DGraph.nodes())))

# Qn 1c: Diameter of the graph
shortestpaths[np.isfinite(shortestpaths)].max()

# Qn 1d: Degree Distribution
## Degree Distribution Table
degreeTable = pd.DataFrame.from_dict(nx.degree(DGraph), 'index').sort_index()
degreeTable.columns = ['degree']
print(degreeTable)

## Degree Distrubtion Plot
in_degrees = DGraph.in_degree()
in_values = sorted(set(in_degrees.values()))
in_degree_values = list(in_degrees.values())

out_degrees = DGraph.out_degree()
out_values = sorted(set(out_degrees.values()))
out_degree_values = list(out_degrees.values())

in_hist = [x + y for x, y in zip(in_degree_values, out_degree_values)]

plt.hist(in_hist, width = 0.60, bins = np.arange(6)-0.3)
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.xticks(range(max(in_hist)+1))
plt.yticks(range(Counter(in_hist).most_common(1)[0][1]+2))
plt.show()

# Qn 1e: Weakly Connected Graph
nx.is_weakly_connected(DGraph) # Weakly connected graph test
nx.is_strongly_connected(DGraph) # Strongly connected graph test

# Qn 2: Visualization Graph
adjMatrix = np.loadtxt("HW1_problem2.txt", dtype=int)

# Use graphAdjMatrix() function to read in both 0-1 and weighted adjacency matrices
# Ensure that no 0-weighted edges are missed out
adjShape = np.shape(adjMatrix)
graph = graphAdjMatrix(adjMatrix[0 : adjShape[1], ], adjMatrix[adjShape[1] : adjShape[0], ])
#graph = nx.from_numpy_matrix(adjMatrix[34:68,], parallel_edges=False, create_using=None)

# Circular Layout
plt.figure(figsize = (15, 12))

edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
pos = nx.circular_layout(graph)
nx.draw(graph, pos, node_color = 'black', edge_color = weights, 
        width=2.5, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(graph, pos, font_color="white", font_size=10)

# Graph without weights
plt.figure(figsize = (15, 12))

pos = nx.spring_layout(graph, k=0.25, iterations=100)
nx.draw(graph, pos, node_color = 'blue', width = 3.0)
nx.draw_networkx_labels(graph, pos, font_color = "white", font_size=10)

# Graph with weights denoted by darkness of edges
plt.figure(figsize = (15, 12))

pos = nx.spring_layout(graph,k=0.25,iterations=100)
nx.draw(graph, pos, node_color = 'black', edge_color = weights, 
        width = 5.0, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(graph, pos, font_color = "white", font_size=10)
nx.draw_networkx_edge_labels(graph, pos, edge_labels = nx.get_edge_attributes(graph, 'weight'), font_size = 8)





