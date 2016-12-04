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

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

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

# Plot path
def plotPath(worldCoordinates, tour, filename):
    # Plot the World map around Djibouti
    mapDjibouti = Basemap(projection = 'mill', 
                          llcrnrlat = worldCoordinates.min(axis = 0)[0] - 0.3, 
                          llcrnrlon = worldCoordinates.min(axis = 0)[1] - 0.3, 
                          urcrnrlat = worldCoordinates.max(axis = 0)[0] + 0.3, 
                          urcrnrlon = worldCoordinates.max(axis = 0)[1] + 0.3, 
                          resolution = 'i')
    # Draw coastline, country borders, and background image
    mapDjibouti.drawcoastlines(linewidth = 1.0)
    mapDjibouti.drawcountries(linewidth = 2.0, color = 'red')
    mapDjibouti.bluemarble()
    
    # Plot all the cities in Djibouti
    xPlot, yPlot = mapDjibouti(worldCoordinates[ : , 1], worldCoordinates[ : , 0]) # x is longitude, y is latitude
    mapDjibouti.plot(xPlot, yPlot, 'co', markersize = 5)
    
    k = 1
    for (lat, long) in worldCoordinates:
        xlabel, ylabel = mapDjibouti(long, lat)
        plt.text(xlabel, ylabel, k, fontsize = 5)
        k += 1    
    
    xs = []
    ys = []
    
    # Plot the optimal tour
    for i in tour:    
        x, y = mapDjibouti(worldCoordinates[i - 1, 1], worldCoordinates[i - 1, 0])
        xs.append(x)
        ys.append(y)
    
    print('Route in', filename, ':', tour)
    mapDjibouti.plot(xs, ys, color = 'blue', linewidth = 1, label = 'Tour')
    
    # Set the title and save to PDF
    plt.title('Djibouti TSP')
    plt.savefig(filename)
    plt.show()
    plt.close()

''' Qn 2 '''
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

''' Qn 3b: Google OR Tools '''
# Distance callback
class CreateDistanceCallback(object):
    """Create callback to calculate distances between points."""
    def __init__(self, distMat):
        """Array of distances between points."""
        self.matrix = distMat

    def Distance(self, from_node, to_node):
        return self.matrix[from_node][to_node]

def ortools(distMat):
    
    tsp_size = distMat.shape[0]
    selected = []
    bestRoute = ''
    
    # Create routing model
    if tsp_size > 0:      
        
        bestObj = np.Inf
        
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()        
        # Setting first solution heuristic: the
        # method for finding a first solution to the problem.
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        # Create the distance callback, which takes two arguments (the from and to node indices)
        # and returns the distance between these nodes.        
        dist_between_nodes = CreateDistanceCallback(distMat)
        dist_callback = dist_between_nodes.Distance            
        
        for i in range(tsp_size):
            # TSP of size tsp_size
            # Second argument = 1 to build a single tour (it's a TSP).
            # Nodes are indexed from 0 to tsp_size - 1. By default the start of
            # the route is node 0.
            routing = pywrapcp.RoutingModel(tsp_size, 1, i)
            
            routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
            # Solve, returns a solution if any.
            assignment = routing.SolveWithParameters(search_parameters)
            if assignment:
                if assignment.ObjectiveValue() < bestObj:
                    bestObj = assignment.ObjectiveValue()
                    route_number = 0
                    index = routing.Start(route_number) # Index of the variable for the starting node.
                    bestRoute = ''
                    selected = []
                    while not routing.IsEnd(index):
                        # Convert variable indices to node indices in the displayed route.
                        bestRoute += str(routing.IndexToNode(index) + 1) + ' -> '
                        selected.append(routing.IndexToNode(index) + 1)
                        index = assignment.Value(routing.NextVar(index))
                        
                    bestRoute += str(routing.IndexToNode(index) + 1)
                    selected.append(routing.IndexToNode(index) + 1)
                    
            else:
                print('No solution found for starting node:', i)
                
        print("Total distance:", str(bestObj), " km\n")
        print("ORTools Route:\n\n", bestRoute)                    
            
    else:
        print('Specify an instance greater than 0.')
        
    return selected

ortoolsTour = ortools(havDistMat)

plotPath(coordMat_latlong, ortoolsTour, 'Djibouti-ortools.pdf')
    
''' Qn 3c: Gurobi Solver '''
# Given the selected path, find the subtour (for Qn 3c: Gurobi)
def findSubTour(selectedPairs):
    notVisited = np.unique(selectedPairs).tolist()
    neighbours = [notVisited[0]]
    visited = []

    while (len(neighbours) > 0):
        currCity = neighbours.pop()
        neighbours1 = [j for i, j in selectedPairs.select(currCity, '*') if j in notVisited and j not in neighbours]
        neighbours2 = [i for i, j in selectedPairs.select('*', currCity) if i in notVisited and i not in neighbours]
        notVisited.remove(currCity)
        visited.append(currCity)
        while len(neighbours1) > 0:
            neighbours.append(neighbours1.pop())
        while len(neighbours2) > 0:
            neighbours.append(neighbours2.pop())

    return visited, notVisited           

# Add lazy constraints to eliminate subtours (for Qn 3c: Gurobi)
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
gurobiSelected = gurobipy.tuplelist((i, j) for i, j in vals.keys() if vals[i,j] > 0.5)

gurobiTour = findSubTour(gurobiSelected)[0]
gurobiTour.append(gurobiTour[0]) # Return to the first city for complete tour

''' Qn 3d: Plot the Tour '''
plotPath(coordMat_latlong, gurobiTour, 'Djibouti-gurobi.pdf')

''' Print the total distance for ortools tour '''
sumDist = 0
for i in range(len(ortoolsTour) - 1):
    sumDist += havDistMat[ortoolsTour[i] - 1, ortoolsTour[i + 1] - 1]
print('ortools distance:', sumDist)

''' Print the total distance for gurobi tour '''
sumDist = 0
for i in range(len(gurobiTour) - 1):
    sumDist += havDistMat[gurobiTour[i] - 1, gurobiTour[i + 1] - 1]
print('gurobi distance:', sumDist)
