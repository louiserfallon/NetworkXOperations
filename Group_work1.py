# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:46:30 2016

@author: cheungcecilia
"""

import networkx as nx
import pandas as pd
import numpy as np

# read text file
data = pd.read_csv('/Users/cheungcecilia/Desktop/Network Analytics/Homework/HW1_problem1.txt', sep=" ", header = None)

# create empty directed graph
DG = nx.DiGraph()

# read dataframe into graph
for i in range(0, len(data.index)):    
    DG.add_edge(data.iloc[i,0], data.iloc[i,1], weight = data.iloc[i,2])
    

# turn directed graph into incidence matrix
A = nx.incidence_matrix(DG, nodelist = sorted(DG.nodes()))
print(A.toarray())

#shortest path matrix
F = nx.floyd_warshall_numpy(DG, nodelist = sorted(DG.nodes()))
print(F)

#diameter of the graph
from numpy import inf
F[F == inf] = 0
np.matrix.max(F)

#plot degree distribution in graph
import pylab as plt
in_degrees = DG.in_degree() # dictionary node:degree
in_values = sorted(set(in_degrees.values()))
in_degree_values = list(in_degrees.values())

out_degrees = DG.out_degree()
out_values = sorted(set(out_degrees.values()))
out_degree_values = list(out_degrees.values())

in_hist = [x + y for x, y in zip(in_degree_values, out_degree_values)]

plt.hist(in_hist, bins = np.arange(6)-0.5)
plt.xlabel('Degree')
plt.ylabel('Number of nodes')

#check connected graph
nx.is_strongly_connected(DG)


#part 2

#load adjacency matrix
a = np.loadtxt('/Users/cheungcecilia/Desktop/Network Analytics/Homework/HW2_problem2.txt')

graph = nx.from_numpy_matrix(a[34:68,0:34])
pos=nx.get_node_attributes(graph,'pos')

#circular looking graph

edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())

pos=nx.spring_layout(graph,k=0.6,iterations=100)
nx.draw_networkx(graph,pos, edge_color=weights, width=1.5, edge_cmap=plt.cm.Blues)

#shows that 32, 33, 0 and 2 have the most connections, intensities of lines represent the weight






#extra stuff

#weight labels
labels = nx.get_edge_attributes(graph,'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

#node labels
nx.draw_networkx_labels(graph, pos, font_size = 8, font_color = 'r')









