#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:49:26 2020

@author: Zachary Miller
"""
#%% Import libraries
import numpy as np
import pandas as pd
import os

import networkx as nx 
import tensorflow as tf
import dgl
#%% Set paths to data and labels
data_path = "/home/zack/Desktop/Lab_Work/Data/neuron_morphologies/Zebrafish/aligned_040120/Zbrain_neurons_graphs"
lbl_path = "/home/zack/Desktop/Lab_Work/Data/neuron_morphologies/Zebrafish/aligned_040120/test_NBLAST_labels.csv"

#%% Read the data
graph_list = []
lbl_list = []
name_list = []

lbls_df = pd.read_csv(lbl_path, index_col=0)
lbls_df.index = list(map(lambda x : os.path.splitext(x)[0], lbls_df.index))

dir_obj = os.fsencode(data_path)
for file in os.listdir(dir_obj):
    filename = os.fsdecode(file)
    #filename = os.path.splitext(filename)[0]
    file_path = os.path.join(data_path, filename)
    
    if os.path.isdir(file_path) == False:
        # Load the graph as an nx_graph and get the node attributes for
        # conversion into a DGL graph
        nx_graph = nx.read_gml(file_path)
        nx_atbs = list(nx_graph.nodes.data())
        num_nodes = len(nx_atbs)
        node_atbs = np.zeros((num_nodes, 4))
        
        # Aggregate all the node attributes into one numpy array
        for idx, node in enumerate(nx_atbs):
            node_atbs[idx, 0] = node[1]['X']
            node_atbs[idx, 1] = node[1]['Y']
            node_atbs[idx, 2] = node[1]['Z']
            node_atbs[idx, 1] = node[1]['diam']
        
        
        # Create the DGL graph with the node attributes
        dgl_graph = dgl.DGLGraph()
        dgl_graph.from_networkx(nx_graph)
        dgl_graph.ndata['data'] = tf.convert_to_tensor(node_atbs, 
                                                       dtype=tf.float32)
        graph_list.append(dgl_graph)
        lbl_list.append(lbls_df.loc[filename,"nblast_cluster"])
        name_list.append(filename)
        
combined_list = list(zip(name_list, graph_list, lbl_list))
#%%