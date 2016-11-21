# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections
from itertools import chain

## Individual Work
A = nx.read_weighted_edgelist("HW1_problem1.txt", delimiter = ' ', create_using = nx.DiGraph())

B = nx.adjacency_matrix(A, nodelist = ['a','b','c','d','e'])

B.todense()

## Group Work - part one
# Print out the node-arc incidence matrix
C = nx.incidence_matrix(A, nodelist=['a','b','c','d','e'], edgelist=None, oriented=True, weight=None)
C.todense()

# Print the shortest-path matrix
D = nx.floyd_warshall_numpy(A, nodelist=['a','b','c','d','e'], weight='weight')
print(D)

# Calculate the diameter of the graph
distances = nx.all_pairs_dijkstra_path_length(A)
alldistancevalues = chain(*(row.values() for row in distances.values()))
max(d for d in alldistancevalues if d != "inf")
 
# Plot the degree distrubition of a graph
degreeSeq = A.degree()
degreecount=collections.Counter(dict.values(degreeSeq))
deg, cnt = zip(*degreeCount.items())

plt.bar(deg, cnt, width=0.20, color='g')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()

# Check that it is a connected graph
nx.is_strongly_connected(A)
nx.is_weakly_connected(A)

## Group Work - part two
a = np.loadtxt("HW2_problem2.txt", dtype=int)

b = nx.from_numpy_matrix(a[:34,:34], parallel_edges=False, create_using=None)

# Visualization Graph
c = nx.from_numpy_matrix(a[34:68,], parallel_edges=False, create_using=None)
plt.figure(num=None, figsize=(10, 10), dpi=80)
plt.axis('off')
fig = plt.figure(1)
pos = nx.spring_layout(c)
nx.draw_networkx_nodes(c,pos)
nx.draw_networkx_edges(c,pos)
nx.draw_networkx_labels(c,pos)
