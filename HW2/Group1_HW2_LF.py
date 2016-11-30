# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:47:19 2016

@author: louisefallon
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% load hw2 data
graphmatrix = np.loadtxt("HW2_problem 2.txt")
#%% draw graph
graph = nx.from_numpy_matrix(graphmatrix[34:68,0:34])
pos=nx.circular_layout(graph)
nx.draw_networkx(graph,pos)
labels = nx.get_edge_attributes(graph,'weight')
nx.draw_networkx_edge_labels(graph, pos,edge_labels=labels)

#%%
#Top 2 nodes by betweenness centrality measure
betweenness = nx.betweenness_centrality(graph)
sorted(betweenness, key=betweenness.get, reverse=True)[:2]

## These two nodes are the two for which the highest sum of
## fractions of shortest paths for all pairs run through these
## nodes. Implying that they are "between" the most pairs, and
## so have a high importance within the network.

#%%
flowcloseness = nx.current_flow_closeness_centrality(graph)
sorted(flowcloseness, key=flowcloseness.get, reverse=True)[:2]

## Flow closeness centrality (similar to Information Centrality)
## takes into account not just the shortest paths but all possible
## paths, and identifies the nodes through which the highest amount
## of information (or current for electrical networks) is passed.
## In this case we get the two nodes 0 and 33, with the highest
## information flow centrality. This is similar to betweenness
## but node 0 has higher centrality when you only take into account
## shortest paths, and node 33 has higher centrality when you take
## into account all paths (which may be more robust).

#%%
eigenvector = nx.eigenvector_centrality(graph)
sorted(eigenvector, key=eigenvector.get, reverse=True)[:2]

## Eigenvector centrality takes into account the fact that the
## importance of a node's neighbours is an input into the importance
## of that node. We can see here that node 2 has a high eigenvector
## centrality because it has important neighbours (e.g. node 0, and 32)
## whereas node 0 has lower importance, most likely because it has
## lots of neighbours of lower importance.

#%% 
closeness = nx.closeness_centrality(graph)
sorted(closeness, key=closeness.get, reverse=True)[:2]

## These two nodes are the two that have the
## lowest sum distance to all other nodes. Implying that
## they are the "closest" to the most other nodes.
## The 0 node is included here, but the 33rd node is 3rd,
## implying that node 2 is "closer" to the rest of the 
## network as a whole than node 33.

#%% (or degree)
degree = nx.degree_centrality(graph)
sorted(degree, key=degree.get, reverse=True)[:2]

## These two nodes are the two that have the highest
## number of neighbours. This simple metric for centrality
## identifies nodes 33 and 0 as the highest "importance"
## for this network.

#%%

dat = pd.read_csv("HW2_tsp.txt", sep=" ",  header=None, skiprows=10,
                  names=["idx","lat","long"],index_col=0)

dat['lat'] = dat['lat']/1000
dat['long'] = dat['long']/1000

#%%
from mpl_toolkits.basemap import Basemap
         
#%%
themap = Basemap(projection='gall',
              llcrnrlon = 41.5,              # lower-left corner longitude
              llcrnrlat = 10.8,               # lower-left corner latitude
              urcrnrlon = 43.5,               # upper-right corner longitude
              urcrnrlat = 12.9,               # upper-right corner latitude
              resolution = 'l',
              area_thresh = 100000.0,
              )
              
themap.drawcoastlines()
themap.drawcountries()
themap.fillcontinents(color = '#A1D490')
themap.drawmapboundary(fill_color='lightblue')
x,y = themap(list(dat['long']), list(dat['lat']))
themap.plot(x, y, 
            '^',                    # marker shape
            color='darkblue',         # marker colour
            markersize=10            # marker size
            )

plt.show()
            
#%%
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from geopy.distance import vincenty

#%%
distancematrix = pd.DataFrame(np.zeros((38, 38)))

##populate the data frame with the distances between each pair
for i in list(range(0,38)):
    for k in list(range(0,38)):
        distancematrix.iloc[i][k] = (vincenty((dat.iloc[k][0],dat.iloc[k][1]),
                      (dat.iloc[i][0],dat.iloc[i][1])).kilometers)

#%%

class CreateDistanceCallback(object):
  """Create callback to calculate distances between points."""
  def __init__(self):
    """Array of distances between points."""

    self.matrix = distancematrix.as_matrix()
    
  def Distance(self, from_node, to_node):
    return self.matrix[from_node][to_node]

#%%

##code from Google's OR toolsdocumentation
tsp_size=38          
if tsp_size > 0:
 
    routing = pywrapcp.RoutingModel(tsp_size, 1,1)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    # Setting first solution heuristic: the
    # method for finding a first solution to the problem.
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Create the distance callback, which takes two arguments (the from and to node indices)
    # and returns the distance between these nodes.

    dist_between_nodes = CreateDistanceCallback()
    dist_callback = dist_between_nodes.Distance
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
#%%
city_names = range(0, 38)
  
# Solve, returns a solution if any.
assignment = routing.SolveWithParameters(search_parameters)
if assignment:
      # Solution cost.
    print ("Total distance: " + str(assignment.ObjectiveValue()) + " km\n")
      # Inspect solution.
      # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
    route_number = 0
    index = routing.Start(route_number) # Index of the variable for the starting node.
    route = ''
    while not routing.IsEnd(index):
        # Convert variable indices to node indices in the displayed route.
        route += str(city_names[routing.IndexToNode(index)]) + ' -> '
        index = assignment.Value(routing.NextVar(index))
    route += str(city_names[routing.IndexToNode(index)])
    print ("Route:\n\n" + route)
else:
    print ('No solution found.')
