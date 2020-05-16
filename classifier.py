#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:28:29 2020

@author: Zachary Miller
"""

import tensorflow as tf
import dgl
from tensorflow.keras import layers
from dgl.nn.tensorflow import GraphConv, AvgPooling

class GCN(tf.keras.Model):
    def __init__(self, in_dim, hidden_dim, n_classes):
        
        super(GCN, self).__init__()
        
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=tf.nn.leaky_relu)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=tf.nn.leaky_relu)
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=tf.nn.leaky_relu)
        self.dense = layers.Dense(n_classes, input_shape=(hidden_dim,))
        
    def call(self, g):
                
        # Convolve over the node attributes
        h = g.ndata['data']
        h = self.conv1(g,h)
        h = self.conv2(g,h)
        h = self.conv3(g,h)
        
        # Set the h attribute of all the nodes to the convolution results
        g.ndata['h'] = h
        
        # Average the h attributes over the whole graph
        hg = dgl.mean_nodes(g, 'h')
        
        # Put the average graph attribute tensor through the final dense layer
        hg = self.dense(hg)
        
        return hg



