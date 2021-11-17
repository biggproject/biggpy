#!/usr/bin/python3
# -*- coding: utf-8 -*-

#!pip install torch
import torch.nn as nn

###############################################################################################
###############################################################################################
# DenseNet: used to create a dense neural network used for prediction of disturbance variables
#
#
###############################################################################################
###############################################################################################


def DenseNet(layers, bias = True):
    '''
      :param layers: list of nodes in each layer. Eg: [1,10,1]
      :param bias: bool: if bias term should be included in the weights
      :return: A pytorch neural network module
      '''
    input_dim = layers[0]
    output_dim = layers[-1]
    hidden_dim = layers[1:-1]
    current_dim = input_dim
    Net = nn.ModuleList()
    for hdim in hidden_dim:
        Net.append(nn.Linear(current_dim, hdim, bias=bias))
        current_dim = hdim
    Net.append(nn.Linear(current_dim, output_dim))
    return Net