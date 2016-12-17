# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 15:32:52 2016

@author: George Pastakas 
"""

import os
import pandas as pd
import networkx as nx
from collections import OrderedDict
import json

# Change directory to your local directory to run code
os.chdir('C:/Users/user/Dropbox (Personal)/Imperial College Business School/MSc Business Analytics/Autumn Term/Network Analytics/Git_Repo/NetworkXOperations/HW3')

# Create an empty list for the villages characteristics
csd = []

# Load the info for the households of all villages
hh_data = pd.read_stata('Data_Group_HW/household_characteristics.dta')

# The villages we need
v_id = [1, 2, 3, 4, 6, 9, 12, 15, 19, 20, 21, 23, 24, 25, 29, 31, 32, 33, 36, \
        39, 42, 43, 45, 46, 47, 48, 50, 51, 52, 55, 57, 59, 60, 62, 64 ,65,  \
        67, 68, 70,71, 72, 73, 75]

for i in v_id:
    vil_dict = OrderedDict()

    # MF dummy variables of village i
    mf = pd.read_csv('Data_Group_HW/MF dummy/MF' + str(i) + '.csv', names = ['MF'])
    
    # Take the household data for village i
    v_data = hh_data[hh_data['village'] == i]
    v_data = v_data.reset_index()
    
    # Merge the two dataframes based on HH ids
    v_data = pd.concat([mf, v_data], axis = 1)
    
    # Read adjacency matrix for village i HH
    adj = pd.read_csv('Data_Group_HW/Adjacency Matrices/adj_allVillageRelationships_HH_vilno_' \
                      + str(i) + '.csv', header = None)

    # Create undirected graph based on adjacency matrix
    G = nx.from_numpy_matrix(adj.as_matrix())

    ### 1/ Number of village
    vil_dict['village'] = i

    ### 2/ MF take up rate
    vil_dict['mf'] = v_data[v_data['leader'] == 0]['MF'].mean()
    
    ### 3/ Average degree centrality of leaders
    v_data['degree'] = list(nx.degree(G).values())
    
    vil_dict['degree_leader'] = v_data[v_data['leader'] == 1][v_data['hhSurveyed'] == 1]['degree'].mean()

    ### 4/ Average eigenvector centrality of leaders
    v_data['eigenvector'] = list(nx.eigenvector_centrality(G, tol = 5e-06).values())
    
    vil_dict['eigenvector_centrality_leader'] = v_data[v_data['leader'] == 1]['eigenvector'].mean()

    ### 11/ Number of households
    vil_dict['numHH'] = v_data.shape[0]

    ### 12/ Fraction of leaders
    vil_dict['fractionLeaders'] = v_data[v_data['leader'] == 1].shape[0]/v_data.shape[0]

    ### extra/ Fraction of taking leaders (divided by the total number of leaders)
    vil_dict['fractionTakingLeaders_leaders'] = v_data[v_data['leader'] == 1][v_data['MF'] == 1].shape[0]/ \
                                                v_data[v_data['leader'] == 1].shape[0]

    ### extra/ Average eigenvector centrality of taking leaders
    vil_dict['eigenvector_centrality_taking_leader'] = v_data[v_data['leader'] == 1][v_data['MF'] == 1]['eigenvector'].mean()
    
    # Append dictionary with village i characteristics to the list
    csd.append(vil_dict)
    
# Convert list of dictionaries to json file and then to pandas
final_df = pd.read_json(json.dumps(csd, indent = 2))
final_df = final_df[['village', 'mf', 'degree_leader', 'eigenvector_centrality_leader', 'numHH', 'fractionLeaders', \
                     'fractionTakingLeaders_leaders', 'eigenvector_centrality_taking_leader']]

# Export dataframe to .csv file
final_df.to_csv('our_cross_sectional.csv', sep = ',', index = False) 
