# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:52:42 2016

@author: Anna Gopalan
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import collections

DGraph = nx.read_weighted_edgelist("HW1_problem1.txt",create_using=nx.DiGraph())

pos = nx.circular_layout(graph)
edge_labels = dict([((u,v,), d['weight']) for u,v,d in DGraph.edges(data=True)])
nx.draw_networkx_edge_labels(DGraph, pos, edge_labels = edge_labels)
nx.draw_networkx_labels(DGraph, pos)
nx.draw(DGraph, pos, node_size = 250)

# Qn 1a: Print Incidence Matrix
incidence_matrix = nx.incidence_matrix(DGraph, nodelist = sorted(DGraph.nodes()), oriented=True, weight = None)
print(incidence_matrix.todense())

# Qn 1b: Shortest Paths Matrix
shortestpaths = pd.DataFrame(nx.floyd_warshall_numpy(DGraph, nodelist = sorted(DGraph.nodes()), weight = 'weight'), 
                             columns = sorted(DGraph.nodes()), index = sorted(DGraph.nodes()))
print(shortestpaths)

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

# Qn 1e: Weakly Connected Graph
nx.is_weakly_connected(DGraph) # Weakly connected graph test
nx.is_strongly_connected(DGraph) # Strongly connected graph test

# Qn 2: Visualization Graph
adjMatrix = np.loadtxt("HW2_problem2.txt", dtype=int)
graph = nx.from_numpy_matrix(adjMatrix[34:68,], parallel_edges=False, create_using=None)

# Circular Layout
edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
pos = nx.circular_layout(graph)
nx.draw(graph, pos, node_color = 'black', edge_color = weights, 
        width=2.5, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(graph, pos, font_color="white", font_size=10)

# Graph without weights
pos = nx.spring_layout(graph, k=0.25, iterations=100)
nx.draw(graph, pos, node_color = 'blue', width = 3.0)
nx.draw_networkx_labels(graph, pos, font_color = "white", font_size=10)

# Graph with weights denoted by darkness of edges
pos = nx.spring_layout(graph,k=0.25,iterations=100)
nx.draw(graph, pos, node_color = 'black', edge_color = weights, 
        width = 5.0, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(graph, pos, font_color = "white", font_size=10)





