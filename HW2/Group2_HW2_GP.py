# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:56:48 2016

Author: George Pastakas
Title: Homework 2
Subtitle: Group Part
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pyplot
from haversine import haversine
from mpl_toolkits.basemap import Basemap
from gurobipy import *
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# Change working directory
os.chdir('C:/Users/user/Dropbox (Personal)/Imperial College Business School/MSc Business Analytics/Autumn Term/Network Analytics/Homework 2/Group Part')

##############
### PART 2 ###
##############

# Load HW1_problem21.txt file
df = pd.read_csv('HW2_problem2.txt', sep = ' ', header = None).drop(0, 1)

# Create numpy matrices
n = df.shape[1]
adjMatrix = np.matrix(df.iloc[:n, :])
weights = np.matrix(df.iloc[n:, :], dtype = [('weight', float)])

# Create graphs
Gw = nx.from_numpy_matrix(weights)

# Calculate 4 centrality measures
# 1. Shortest-path betweenness centrality for nodes
bw = nx.betweenness_centrality(Gw, normalized = True, weight = 'weight')
# 2. Current-flow closeness centrality for nodes
cfc = nx.current_flow_closeness_centrality(Gw, weight = 'weight', solver = 'full')
# 3. Eigenvector centrality for the graph G
ev = nx.eigenvector_centrality(Gw, max_iter = 100, tol = 1e-06, weight = 'weight')
# 4. Current-flow betweenness centrality for nodes
cfbw = nx.current_flow_betweenness_centrality(Gw, normalized = True, weight = 'weight', solver='full')

# Set the 4 centrality measures as node attributes
nx.set_node_attributes(Gw, 'betweenness', bw)
nx.set_node_attributes(Gw, 'current_flow_closeness', cfc)
nx.set_node_attributes(Gw, 'eigenvector', ev)
nx.set_node_attributes(Gw, 'current_flow_betweenness', cfbw)
centrality_measures = ['betweenness', 'current_flow_closeness', \
                       'eigenvector', 'current_flow_betweenness']
                       
for cm in centrality_measures:
    # Visualise the graph
    plt.figure(figsize = (10, 10))
    plt.title(cm + ' centrality measure')
    pos = nx.spring_layout(Gw, k = .25, iterations = 2000)
    labs = nx.get_node_attributes(Gw, cm)
    for key, value in labs.items(): 
        labs[key] = round(value, 3)
    nx.draw(Gw, pos, node_color = 'lightblue', edge_color = [d['weight'] for (u, v, d) in Gw.edges(data = True)], 
            width = 3, edge_cmap=plt.cm.Blues, alpha = .4, 
            node_size = np.add(np.multiply(list(nx.get_node_attributes(Gw, cm).values()), 10000), 200))
    nx.draw_networkx_labels(Gw, pos, labels =labs, font_color = 'black', font_size = 8)
    #nx.draw_networkx_edge_labels(Gw, pos, edge_labels = nx.get_edge_attributes(Gw, 'weight'), font_size = 8)

##############
### PART 3 ###
##############

###  Q. a  ###

# Read file
f = open('HW2_tsp.txt', 'r')

# Create a dataframe with the required data
rows = f.read().split('\n')[10: -1]
nodes = []
for i in range(len(rows)):
    nodes.append(rows[i].split(' '))    
nodes_df = pd.DataFrame(nodes, columns=['id', 'latitude', 'longitude'])
nodes_df = nodes_df.set_index(nodes_df['id'].values)
nodes_df = nodes_df.drop('id', 1)
nodes_df['latitude'] = pd.to_numeric(nodes_df['latitude']) / 1000
nodes_df['longitude'] = pd.to_numeric(nodes_df['longitude']) / 1000

# Draw scatterplot
plt.figure(figsize = (6, 6))
plt.scatter(nodes_df['longitude'], nodes_df['latitude'], alpha = .5, color = 'b')
plt.title('Djibouti cities')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Draw map
plt.figure(figsize = (6, 6))
map = Basemap(width = 500000, height = 500000, projection = 'lcc', \
              resolution = 'l', lat_0 = 12, lon_0 = 43)
map.drawcoastlines()
#map.drawcounties()
xn, yn = map(nodes_df['longitude'].values, nodes_df['latitude'].values)
map.scatter(xn, yn, marker = 'o', alpha = .5, color = 'b')
map.bluemarble()
plt.title('Djibouti cities')
plt.show()

# Create a dataframe with the distances
distances = pd.DataFrame([[0] * nodes_df.shape[0]] * nodes_df.shape[0], index = nodes_df.index.values, columns = nodes_df.index.values)
for i in range(len(rows)):
    for j in range(len(rows)):
        distances.iloc[i, j] = haversine((nodes_df['latitude'][i], nodes_df['longitude'][i]), (nodes_df['latitude'][j], nodes_df['longitude'][j]))

###  Q. b  ###

# Distance callback
class CreateDistanceCallback(object):
    """Create callback to calculate distances between points."""
    def __init__(self):
        """Array of distances between points."""
        self.matrix = (np.array(distances).tolist())

    def Distance(self, from_node, to_node):
        return self.matrix[from_node][to_node]

def main():
    # Cities
    city_names = distances.index.tolist()
    tsp_size = len(city_names)

    # Create routing model
    if tsp_size > 0:
        # TSP of size tsp_size
        # Second argument = 1 to build a single tour (it's a TSP).
        # Nodes are indexed from 0 to tsp_size - 1. By default the start of
        # the route is node 0.
        routing = pywrapcp.RoutingModel(tsp_size, 1, 1)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

        # Setting first solution heuristic: the
        # method for finding a first solution to the problem.
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Create the distance callback, which takes two arguments (the from and to node indices)
        # and returns the distance between these nodes.
    
        dist_between_nodes = CreateDistanceCallback()
        dist_callback = dist_between_nodes.Distance
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
        # Solve, returns a solution if any.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            # Solution cost.
            print('Total distance: ' + str(assignment.ObjectiveValue()) + ' miles\n')
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
            print('Route:\n\n' + route)
        else:
            print('No solution found.')
    else:
        print('Specify an instance greater than 0.')
    return(route)
        
optimal_route = main().split(' -> ')
optimal_route = [int(x) for x in optimal_route]

# Draw scatterplot with optimal trip
plt.figure(figsize = (10, 10))
plt.scatter(nodes_df['longitude'], nodes_df['latitude'], alpha = .5, color = 'b')
plt.title('Djibouti cities')
plt.xlabel('Longitude')
for i in range(len(optimal_route[:-1])):
    x = [nodes_df.iloc[optimal_route[i]-1, 1], nodes_df.iloc[optimal_route[i+1]-1, 1]]
    y = [nodes_df.iloc[optimal_route[i]-1, 0], nodes_df.iloc[optimal_route[i+1]-1, 0]]
    plt.plot(x, y, color = 'b', alpha = .5)
pyplot.show()
    
###  Q. c  ###

# Traveling salesman problem using GUROBI
n = distances.shape[0]

# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
  if where == GRB.callback.MIPSOL:
    selected = []
    # make a list of edges selected in the solution
    for i in range(n):
      sol = model.cbGetSolution([model._vars[i,j] for j in range(n)])
      selected += [(i,j) for j in range(n) if sol[j] > 0.5]
    # find the shortest cycle in the selected edge list
    tour = subtour(selected)
    if len(tour) < n:
      # add a subtour elimination constraint
      expr = 0
      for i in range(len(tour)):
        for j in range(i+1, len(tour)):
          expr += model._vars[tour[i], tour[j]]
      model.cbLazy(expr <= len(tour)-1)

# Given a list of edges, finds the shortest subtour
def subtour(edges):
  visited = [False]*n
  cycles = []
  lengths = []
  selected = [[] for i in range(n)]
  for x,y in edges:
    selected[x].append(y)
  while True:
    current = visited.index(False)
    thiscycle = [current]
    while True:
      visited[current] = True
      neighbors = [x for x in selected[current] if not visited[x]]
      if len(neighbors) == 0:
        break
      current = neighbors[0]
      thiscycle.append(current)
    cycles.append(thiscycle)
    lengths.append(len(thiscycle))
    if sum(lengths) == n:
      break
  return cycles[lengths.index(min(lengths))]

# Create model and variables
m = Model()
vars = {}
for i in range(n):
    for j in range(i + 1):
        vars[i, j] = m.addVar(obj = distances.iloc[i, j], vtype = GRB.BINARY,
                              name = 'e' + str(i) + '_' + str(j))
        vars[j, i] = vars[i, j]
    m.update()

# Add degree-2 constraint, and forbid loops
for i in range(n):
    m.addConstr(quicksum(vars[i, j] for j in range(n)) == 2)
    vars[i, i].ub = 0
m.update()

# Optimize model
m._vars = vars
m.params.LazyConstraints = 1
m.optimize(subtourelim)

solution = m.getAttr('x', vars)
selected = [(i,j) for i in range(n) for j in range(n) if solution[i,j] > 0.5]
assert len(subtour(selected)) == n

# Draw scatterplot with optimal trip
plt.figure(figsize = (10, 10))
plt.scatter(nodes_df['longitude'], nodes_df['latitude'], alpha = .5, color = 'b')
plt.title('Djibouti cities')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
for i in range(len(selected)):
    x = [nodes_df.iloc[selected[i][0], 1], nodes_df.iloc[selected[i][1], 1]]
    y = [nodes_df.iloc[selected[i][0], 0], nodes_df.iloc[selected[i][1], 0]]
    plt.plot(x, y, color = 'b', alpha = .5)
    
pyplot.show()

# Draw map with optimal trip
plt.figure(figsize = (10, 10))
map = Basemap(width = 500000, height = 500000, projection = 'lcc', \
              resolution = 'l', lat_0 = 12, lon_0 = 43)
map.drawcoastlines()
#map.drawcounties()
xn, yn = map(nodes_df['longitude'].values, nodes_df['latitude'].values)
map.scatter(xn, yn, marker = 'o', alpha = .5, color = 'b')
for i in range(len(selected)):
    x = [nodes_df.iloc[selected[i][0], 1], nodes_df.iloc[selected[i][1], 1]]
    y = [nodes_df.iloc[selected[i][0], 0], nodes_df.iloc[selected[i][1], 0]]
    x, y = map(x, y)
    map.plot(x, y, color = 'b', alpha = .5)   
map.bluemarble()
plt.title('Djibouti cities')
plt.show()



