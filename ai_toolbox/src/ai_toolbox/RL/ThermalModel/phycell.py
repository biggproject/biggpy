#!/usr/bin/python3
# -*- coding: utf-8 -*-

#!pip install torch
import torch
import dynamics as dynamics
import json

###############################################################################################
###############################################################################################
# PhyCell: A sequential physics based unit that is used for the thermal modelling of a building
#   - implemented in pytorch
#   - Dynamics of the systems will be used here
#
#
#
###############################################################################################
###############################################################################################

class PhyCell(torch.nn.Module):
    # Initialize the class
    def __init__(self,paras_json_loc = "\default_house.json"):
        super(PhyCell, self).__init__()
        f = open(paras_json_loc, "r")
        parameter_dict = json.loads(f.read())

    def forward(self, Inputs, State):
        '''
        :param Inputs: [Batch Size X input features] - RAOdt
        :param State: [Batch Size X state features] - OIB
        :return: Outputs, next state
        '''

        raise NotImplementedError

    def param_loss(self):
        '''
        :return: the weight loss
        '''
        raise  NotImplementedError

    def get_param(self):
        '''
        :return: the current parameter dictionary
        '''

        raise NotImplementedError

    def set_param(self,pri = True, **kwargs):
        '''
        **kwargs: key value pairs for parameters and their values
        '''

        raise NotImplementedError

    def set_param_grad(self,pri = True, **kwargs):
        '''
        **kwargs: key value pairs for parameters and a bool
        '''

        raise NotImplementedError
