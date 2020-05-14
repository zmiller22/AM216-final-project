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
    def __init__(self,
                    in_feats,
                    n_hidden,
                    n_classes,
                    n_layers,
                    activation,
                    dropout):
        
        super(GCN, self).__init__()
        
        self.conv1 = GraphConv(in_feats, n_hidden, activation=tf.nn.relu)
        self.conv2 = GraphConv(n_hidden, n_hidden, activation=tf.nn.relu)
        self.classify = layers.Dense(n_classes, input_shape=(n_hidden,))
        

    # Note to self, this is a predefined method in the keras Model class, and 
    # therefore the first argument is taken to be the input to the classifier 
    # implicitly
    def call(self, g):
        
        """Note to self, this g.ndata might be problematic since it is getting 
        a batched graph object. However, I think that the batch graph object
        has this method as long as I take care to make sure the node attributes
        are passed along properly (the tutorial for batching has no such attributes)"""
        
        # Convolve over the node attributes
        h = g.ndata
        h = self.conv1(g,h)
        h = self.conv2(g,h)
        
        # Set the h attribute of all the nodes to the convolution results
        g.ndata['h'] = h
        
        # Average the h attributes over the whole graph
        hg = dgl.mean_nodes(g, 'h')
        
        # Put the average graph attribute tensor through the final dense layer
        # to get the classification result
        output = self.classify(hg)
        
        return output



