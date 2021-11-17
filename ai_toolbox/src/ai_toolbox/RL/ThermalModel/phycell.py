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
        self.phyparam = torch.nn.ParameterDict({})
        self.set_param(dict=parameter_dict)

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
        l = torch.tensor(0.0)
        loss = torch.stack([torch.add(l, torch.relu(-self.phyparam[p])) for p in self.phyparam], dim=0).sum(dim=0)
        return loss

    def get_param(self):
        '''
        :return: the current parameter dictionary
        '''
        para_dict = {key: float(self.phyparam[key].data.numpy()) for key in self.PhyCell.phyparam}
        return para_dict

    def set_param(self,pri = True, **kwargs):
        '''
        **kwargs: key value pairs for parameters and their values
        '''
        for key, value in kwargs.items():
            if pri: print("Parameters setting: %s == %s" % (key, value))
            if isinstance(value, dict):
                for (k, v) in value.items():
                    self.phyparam.update({k: torch.nn.Parameter(torch.tensor([v], requires_grad=True))})
            else:
                self.phyparam.update({key: torch.nn.Parameter(torch.tensor([value], requires_grad=True))})

    def set_param_grad(self,pri = True, **kwargs):
        '''
        **kwargs: key value pairs for parameters and a bool
        '''
        for key, value in kwargs.items():
            if pri: print("Parameters gradient setting: %s == %s" % (key, value))
            if isinstance(value, dict):
                for (k, v) in value.items():
                    self.phyparam[k].requires_grad = v
            else:
                self.phyparam[key].requires_grad = value
