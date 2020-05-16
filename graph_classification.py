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
import random

import networkx as nx 
import tensorflow as tf
import dgl
from sklearn.preprocessing import OneHotEncoder

from classifier import GCN
#%% Set paths to data and labels
data_path = "/home/zack/Desktop/Lab_Work/Data/neuron_morphologies/Zebrafish/aligned_040120/Zbrain_neurons_graphs"
lbl_path = "/home/zack/Desktop/Lab_Work/Data/neuron_morphologies/Zebrafish/aligned_040120/test_NBLAST_labels.csv"

#%% Read in the data
graph_list = []
lbl_list = []
name_list = []

# Read in the labels and remove the file extension from the names
lbls_df = pd.read_csv(lbl_path, index_col=0)
lbls_df.index = list(map(lambda x : os.path.splitext(x)[0], lbls_df.index))

# 
dir_obj = os.fsencode(data_path)
for file in os.listdir(dir_obj):
    filename = os.fsdecode(file)
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
            node_atbs[idx, 3] = node[1]['diam']
            
        # Add "self" edges so each node will be included in its own convolution
        nx_graph.add_edges_from(zip(nx_graph.nodes(), nx_graph.nodes()))
        
        # Create the DGL graph with the node attributes
        dgl_graph = dgl.DGLGraph()
        dgl_graph.from_networkx(nx_graph)
        dgl_graph.ndata['data'] = tf.convert_to_tensor(node_atbs, 
                                                       dtype=tf.float32)
        
        # Add all the elements to lists
        graph_list.append(dgl_graph)
        lbl_list.append(lbls_df.loc[filename,"nblast_cluster"])
        name_list.append(filename)

#%% Format the data for training

# One-hot encode the labels
lbl_arr = np.asarray(lbl_list)[:,np.newaxis]
enc = OneHotEncoder(sparse=False)
lbl_arr = enc.fit_transform(lbl_arr)

# Create splits
combined_list = list(zip(name_list, graph_list, lbl_arr))
random.shuffle(combined_list)
split_1 = int(0.7*len(combined_list))
split_2 = int(0.15*len(combined_list))
train_data = combined_list[:split_1]
val_data = combined_list[split_1:split_1+split_2]
test_data = combined_list[split_1+split_2:]

#%% Train the model

model_path = '/home/zack/Documents/AM216/final_project/best_model/1/'

model = GCN(4, 32, 10)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
EPOCHS = 20
PATIENCE = 2

train_loss_list = []
val_loss_list = []

for epoch in range(EPOCHS):
    epoch_train_loss = 0
    epoch_val_loss = 0
    print("Training...")
    for (name, graph, lbl) in train_data:
        with tf.GradientTape() as tape:
            lbl = lbl.reshape(1,10)
            prediction = model(graph)
            loss = loss_func(lbl, prediction)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_train_loss += loss
            
    print('Epoch {}, total training loss {:.4f}'.format(epoch, epoch_train_loss))
    print("Calculating validation loss...")
    for (name, graph, lbl) in val_data:
        lbl = lbl.reshape(1,10)
        prediction = model(graph)
        loss = loss_func(lbl, prediction)
        epoch_val_loss += loss
    
    print('Epoch {}, total validation loss {:.4f}'.format(epoch, epoch_val_loss))
    val_loss_list.append(epoch_val_loss)
    if epoch >= PATIENCE:
        if max(val_loss_list[epoch-PATIENCE:])<epoch_val_loss:
            print('Training stopped on epoch {}'.format(epoch))
            break
    
    # Can't use standard ways of saving a tf model since this is a custom model.
    # Normally this means you have to set the input shape manually and then you
    # can save, but in this case the input shape is variable. Will have to figure
    # this out in the future...
    # tf.saved_model.save(model, model_path)
    
#%% Test the model

true_lbls = np.asarray([element[2] for element in test_data])
model_predictions = np.zeros((len(test_data),10))

test_loss = 0
for i, (name, graph, lbl) in enumerate(test_data):
    lbl = lbl.reshape(1,10)
    prediction = model(graph)
    loss = loss_func(lbl, prediction)
    test_loss += loss
    model_predictions[i,np.argmax(prediction)] = 1
    
results = true_lbls+model_predictions
num_correct = results[results==2].shape[0]
total_num = len(test_data)
acc = num_correct/total_num

print("Test loss was {:.4f} for {:.2f} percent accuracy".format(test_loss, 100*acc))
    

    


        
        
        



