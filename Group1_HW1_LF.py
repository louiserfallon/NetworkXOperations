# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:47:19 2016

@author: louisefallon
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


graph = nx.read_weighted_edgelist("HW1_problem1.txt",create_using=nx.DiGraph())

incidence = nx.incidence_matrix(graph,
                                nodelist=['a','b','c','d','e'],
                                oriented=True)

print(incidence.todense())

## Edges go from -1 to 1. (e.g. the first column is telling us there is an
## edge from A to D)


shortestpath = nx.floyd_warshall_numpy(graph, nodelist=['a','b','c','d','e']) 
print(shortestpath)
##works for neg values.
##https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.shortest_paths.dense.floyd_warshall_numpy.html
##https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.shortest_paths.html


diameter = np.amax(shortestpath[shortestpath < np.Inf])
print(diameter)

#%%

import collections
degrees = graph.degree() # dictionary node:degree
degreeCount=collections.Counter(dict.values(degrees))
deg, cnt = zip(*degreeCount.items())

plt.bar(deg, cnt, width=0.1, color='b')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()

#%%

nx.is_weakly_connected(graph)
##The graph is weakly connected (you can get from
## any node to any other, ignoring direction)
nx.is_strongly_connected(graph)
##The graph is weakly connected (you can get from
## any node to any other, including direction)

#%%
graphmatrix = np.loadtxt("HW2_problem2.txt")
#%%
graph3 = nx.from_numpy_matrix(graphmatrix[34:68,0:34])
pos=nx.circular_layout(graph3)
nx.draw_networkx(graph3,pos)
labels = nx.get_edge_attributes(graph3,'weight')
nx.draw_networkx_edge_labels(graph3, pos,edge_labels=labels)
#%%
pos=nx.circular_layout(graph3)
nx.draw_networkx(graph3,pos)
#%%
pos=nx.spring_layout(graph3,k=0.25,iterations=100)
nx.draw_networkx(graph3,pos)
#%%
pos=nx.spring_layout(graph3,k=0.25,iterations=100)
edges,weights = zip(*nx.get_edge_attributes(graph3,'weight').items())
nx.draw(graph3, pos, node_color='b', edge_color=weights, width=5.0, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(graph3, pos,font_color="c", font_size=10)
##Weights shown by colour as there are too many lines for numbered weights
