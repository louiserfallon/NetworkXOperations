# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:50:27 2016

@author: siowmeng
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""
Function to draw the weighted directed graph
Input: a DiGraph object containing all the nodes, edges and edge weights
Output: None Returned (the graph will be printed)
"""

# Read the edgelist from file directly and load into networkx graph
DG = nx.read_weighted_edgelist('HW1_problem1.txt', create_using = nx.DiGraph())

# Qn 1a: Print Incidence Matrix (weight value is printed)
incDF = pd.DataFrame(nx.incidence_matrix(DG, nodelist = sorted(DG.nodes()), 
                                         oriented = True, weight = 'weight').todense(), 
                     index = sorted(DG.nodes()))
print(incDF)

# Qn 1b: Shortest Paths Matrix
#paths = nx.johnson(DG, weight='weight') # Returns a dict of dict containing the shortest paths between all source-destination pairs
paths = nx.floyd_warshall_numpy(DG, nodelist = sorted(DG.nodes()), weight = 'weight') # Returns a matrix with shortest-path lengths between all source-destination pairs
print(paths)

# Qn 1c: Diameter of the graph
paths[np.isfinite(paths)].max()
#paths.max()
#nx.diameter(DG)

# Qn 1d: Degree Distribution
degreeTable = pd.DataFrame.from_dict(nx.degree(DG), 'index').sort_index()
degreeTable.columns = ['degree']
print(degreeTable)

# Distribution plot
plt.figure()
plt.plot(range(len(nx.degree_histogram(DG))), nx.degree_histogram(DG), 'bv-')
plt.xlabel('Degree')
plt.xticks(range(len(nx.degree_histogram(DG))))
plt.yticks(range(max(nx.degree_histogram(DG)) + 1))
plt.ylabel('Number of nodes')
plt.title('Degree Distribution of HW1 Problem 1')
plt.savefig('Problem1_DegreeDistribution.pdf')
plt.close()

# Distribution plot (separate for in-degree and out-degree)
plt.figure()
inDegree = DG.in_degree().values()
maxInDegree = max(DG.in_degree().values())
inDegreeCount = [list(inDegree).count(x) for x in (range(maxInDegree + 1))]
plt.plot(range(maxInDegree + 1), inDegreeCount, 'bv-')
outDegree = DG.out_degree().values()
maxOutDegree = max(DG.out_degree().values())
outDegreeCount = [list(outDegree).count(x) for x in (range(maxOutDegree + 1))]
plt.plot(range(maxOutDegree + 1), outDegreeCount, 'ro-')
plt.legend(['In-degree','Out-degree'])
plt.xlabel('Degree')
plt.xticks(range(max(maxInDegree, maxOutDegree) + 1))
plt.yticks(range(max(sorted(inDegreeCount, reverse = True)[0], 
                     sorted(outDegreeCount, reverse = True)[0]) + 1))
plt.ylabel('Number of nodes')
plt.title('Degree Distribution of HW1 Problem 1')
plt.savefig('Problem1_DegreeDistribution2.pdf')
plt.close()

# Histogram plot
plt.figure()
plt.bar(range(len(nx.degree_histogram(DG))), nx.degree_histogram(DG), width = 0.5, color='b')
plt.title("Degree Histogram of HW1 Problem 1")
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.xticks(np.arange(0.25, len(nx.degree_histogram(DG)) + 0.25), range(len(nx.degree_histogram(DG))))
plt.yticks(range(max(nx.degree_histogram(DG)) + 2))
plt.savefig('Problem1_DegreeHistogram.pdf')
plt.close()

# Histogram plot (separate for in-degree and out-degree)
plt.figure()
plt.bar(range(maxInDegree + 1), inDegreeCount, width = 0.5, color='b')
plt.bar(range(maxOutDegree + 1), outDegreeCount, width = 0.5, color='r', bottom = inDegreeCount)
plt.legend(['In-degree','Out-degree'])
plt.title("Degree Histogram of HW1 Problem 1")
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.xticks(np.arange(0.25, max(maxInDegree, maxOutDegree) + 1.25), range(max(maxInDegree, maxOutDegree) + 1))
plt.yticks(range(max(np.add(inDegreeCount, outDegreeCount)) + 2))
plt.savefig('Problem1_DegreeHistogram2.pdf')
plt.close()
    
# Qn 1e: Weakly Connected Graph
nx.is_weakly_connected(DG) # Weakly connected graph test
nx.is_strongly_connected(DG) # Strongly connected graph test

# Function to draw the graph
def drawWeightedGraph(weightedGraph, filename, figSize):
    
    plt.figure(figsize = figSize)
    
    pos = nx.spring_layout(weightedGraph)
    nx.draw_networkx_nodes(weightedGraph, pos = pos, node_size = 200, alpha = 0.8)
    nx.draw_networkx_labels(weightedGraph, pos = pos, font_size = 8)           

    nx.draw_networkx_edges(weightedGraph, pos = pos,
                           edge_color = [d['weight'] for (u, v, d) in weightedGraph.edges(data = True)])
    nx.draw_networkx_edge_labels(weightedGraph, pos = pos, 
                                 edge_labels = nx.get_edge_attributes(weightedGraph, 'weight'),
                                 font_size = 6, font_color = 'k')
    
#    nx.draw_networkx(weightedGraph, pos = pos)   
   
    plt.legend()
    plt.axis('off')
    plt.savefig(filename)

# Invoke the custom function to draw the directed graph (with weights)
drawWeightedGraph(DG, 'Problem1_DirectedGraph.pdf', (5, 5))

# Function to read in two adjacency matrices
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

# Qn 2
adjDF = pd.read_table('HW2_problem2.txt', sep = ' ', header = None, skipinitialspace = True)

UG = graphAdjMatrix(adjDF.iloc[0 : adjDF.shape[1], ].as_matrix(), 
               adjDF.iloc[adjDF.shape[1] : adjDF.shape[0], ].as_matrix())

drawWeightedGraph(UG, 'Problem2_UndirectedGraph.pdf', (15, 12))
