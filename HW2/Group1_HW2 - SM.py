# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:56:32 2016

@author: siowmeng
"""

import math
import pandas as pd
import networkx as nx
import numpy as np
import operator
from haversine import haversine

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import gurobipy

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

# Calculate Euclidean and Haversine distance given the coordinates matrices
def calcDistance(coordMat, coordMat_latlong):        
    # Initialize the Euclidean Distance Matrix
    distMat = np.zeros((coordMat.shape[0], coordMat.shape[0]))
    # Initialize the Haversine Distance Matrix
    havDistMat = np.zeros((coordMat.shape[0], coordMat.shape[0]))
    
    for i in range(coordMat.shape[0]):
        for j in range(i + 1, coordMat.shape[0]):
            # Update Euclidean distance between city i & j
            distMat[i][j] = math.sqrt((coordMat[j][0] - coordMat[i][0])**2 + (coordMat[j][1] - coordMat[i][1])**2)
            distMat[j][i] = distMat[i][j]
            # Update Haversine distance between city i & j
            havDistMat[i][j] = haversine((coordMat_latlong[i][0], coordMat_latlong[i][1]), (coordMat_latlong[j][0], coordMat_latlong[j][1]))
            havDistMat[j][i] = havDistMat[i][j]
            
    distDF = pd.DataFrame(distMat, index = coordDF.index, columns = coordDF.index)
    havDistDF = pd.DataFrame(havDistMat, index = coordDF.index, columns = coordDF.index)
    return distDF, havDistDF

# Given the selected path, find the subtour
def findSubTour(selectedPairs):    
    notVisited = np.unique(selectedPairs).tolist()
    neighbours = [notVisited[0]]
    visited = []

    while (len(neighbours) > 0):
        currCity = neighbours.pop(0)
        neighbours1 = [j for i, j in selectedPairs.select(currCity, '*') if j in notVisited and j not in neighbours]
        neighbours2 = [i for i, j in selectedPairs.select('*', currCity) if i in notVisited and i not in neighbours]
        notVisited.remove(currCity)
        visited.append(currCity)
        while len(neighbours1) > 0:
            neighbours.append(neighbours1.pop())
        while len(neighbours2) > 0:
            neighbours.append(neighbours2.pop())

    return visited, notVisited           

# Add lazy constraints to eliminate subtours
def elimSubTour(model, where):
    if where == gurobipy.GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._xVar)
        selectedPairs = gurobipy.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        visited, notVisited = findSubTour(selectedPairs)
        # If there is subtour, add constraint: the cut should have at least 2 edges
        if len(notVisited) > 0:
            model.cbLazy(gurobipy.quicksum(model._xVar[i, j] for j in notVisited for i in visited if i < j) + 
                                           gurobipy.quicksum(model._xVar[j, i] for j in notVisited for i in visited if i > j) 
                                           >= 2)

# Qn 2
adjMatrix = np.loadtxt("HW2_problem 2.txt", dtype=int)

# Use graphAdjMatrix() function to read in both 0-1 and weighted adjacency matrices
# Ensure that no 0-weighted edges are missed out
adjShape = np.shape(adjMatrix)
graph = graphAdjMatrix(adjMatrix[0 : adjShape[1], ], adjMatrix[adjShape[1] : adjShape[0], ])

# Degree Centrality
sorted(nx.degree_centrality(graph).items(), reverse = True, key = operator.itemgetter(1))
# Betweenness Centrality
sorted(nx.betweenness_centrality(graph, weight = 'weight').items(), reverse = True, key = operator.itemgetter(1))
# Current Flow Closeness Centrality
sorted(nx.current_flow_closeness_centrality(graph, weight = 'weight').items(), reverse = True, key = operator.itemgetter(1))
# Eigenvector Centrality
sorted(nx.eigenvector_centrality_numpy(graph, weight = 'weight').items(), reverse = True, key = operator.itemgetter(1))

# Read the coordinates from file
coordDF = pd.read_table('HW2_tsp.txt', sep = ' ', header = None, skiprows = 10, 
                        index_col = 0, names = ["x", "y"])

# Matrix containing (x, y) as Cartesian coordinates pair
coordMat = coordDF.as_matrix()
# Matrix containing (lat, long) as Latitude & Longitude pair
coordMat_latlong = coordMat / 1000

# Get Euclidean Distance and Haversine Distance
eucDistDF, havDistDF = calcDistance(coordMat, coordMat_latlong)
havDistMat = havDistDF.as_matrix()

# Initialize Gurobi Model
tspModel = gurobipy.Model('DjiboutiTSP')
# All distinct city pairs
tuplelist = [(x,y) for x in range(1, coordMat.shape[0] + 1) for y in range(1, coordMat.shape[0] + 1) if x < y]
# Distance between city pairs
distlist = [havDistMat[x - 1, y - 1] for x in range(1, coordMat.shape[0] + 1) for y in range(x + 1, coordMat.shape[0] + 1)]
# Declare Decision Variables (with Objective Coefficients)
xVar = tspModel.addVars(tuplelist, obj = distlist, vtype = gurobipy.GRB.BINARY, name = 'x')

# Constraints: Degree = 2
tspModel.addConstrs((sum(xVar.select(i, '*')) + sum(xVar.select('*', i))) == 2 for i in range(1, coordMat.shape[0] + 1))

# Enable Lazy Constraints
tspModel.setParam("LazyConstraints", 1)
# Store the decision variables as private variable (for access later)
tspModel._xVar = xVar
# Solve the TSP using elimSubTour as a sub-routine
tspModel.optimize(elimSubTour)

# Get the selected path
vals = tspModel.getAttr('x', xVar)
selected = gurobipy.tuplelist((i, j) for i, j in vals.keys() if vals[i,j] > 0.5)

# Plot Tour
coordMat_latlong.min(axis = 0)
coordMat_latlong.max(axis = 0)

# Plot the World map around Djibouti
mapDjibouti = Basemap(projection = 'mill', 
                      llcrnrlat = coordMat_latlong.min(axis = 0)[0] - 0.3, 
                      llcrnrlon = coordMat_latlong.min(axis = 0)[1] - 0.3, 
                      urcrnrlat = coordMat_latlong.max(axis = 0)[0] + 0.3, 
                      urcrnrlon = coordMat_latlong.max(axis = 0)[1] + 0.3, 
                      resolution = 'i')
# Draw coastline, country borders, and background image
mapDjibouti.drawcoastlines(linewidth = 1.0)
mapDjibouti.drawcountries(linewidth = 2.0, color = 'red')
mapDjibouti.bluemarble()

# Plot all the cities in Djibouti
xPlot, yPlot = mapDjibouti(coordMat_latlong[ : , 1], coordMat_latlong[ : , 0]) # x is longitude, y is latitude
mapDjibouti.plot(xPlot, yPlot, 'co', markersize = 5)

# Plot the optimal tour
for (i, j) in selected:
    x1, y1 = mapDjibouti(coordMat_latlong[i - 1, 1], coordMat_latlong[i - 1, 0])
    x2, y2 = mapDjibouti(coordMat_latlong[j - 1, 1], coordMat_latlong[j - 1, 0])
    xs = [x1, x2]
    ys = [y1, y2]
    mapDjibouti.plot(xs, ys, color = 'blue', linewidth = 1, label = 'Tour')

# Set the title and save to PDF
plt.title('Djibouti TSP')
plt.savefig('Djibouti.pdf')
plt.show()
