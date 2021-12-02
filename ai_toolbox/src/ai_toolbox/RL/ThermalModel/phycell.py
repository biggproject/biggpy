#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import dynamics as dynamics
import json

###############################################################################################
###############################################################################################
# - Manu Lahariya, IDLab, 2/12/21
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
        :param Inputs: [Batch Size X input features]
        :param State: [Batch Size X state features]
        :return: Outputs, next state
        '''

        RoomT = Inputs[:, 0:1]
        BoilerOutletT = Inputs[:, 1:2]
        RoomSet_next = Inputs[:, 2:3]
        BoilerSetpoint_next = Inputs[:, 3:4]
        dt = Inputs[:, 4:5]
        AmbientT = Inputs[:, 5:6]

        BoilerInletT = State[:, 0:1].reshape(-1, 1)
        BuildingT = State[:, 1:2].reshape(-1, 1)

        BoilerOutletT_next = dynamics.BoilerOutletT_next(RoomT=RoomT,
                                                         BoilerOutletT=BoilerOutletT,
                                                         BoilerSet_next=BoilerSetpoint_next,
                                                         a0=self.phyparam['a0'],
                                                         a1=self.phyparam['a1'],
                                                         a2=self.phyparam['a2'],
                                                         dt=dt)

        RoomT_next = dynamics.RoomT_next(RoomT=RoomT,
                                         BuildingT=BuildingT,
                                         BoilerInletT=BoilerInletT,
                                         BoilerOutletT_next=BoilerOutletT_next,
                                         AmbientT=AmbientT,
                                         dt=dt,
                                         Ra=self.phyparam['Ra'],
                                         Ri=self.phyparam['Ri'],
                                         Rb=self.phyparam['Rb'],
                                         Ro=self.phyparam['Ro'],
                                         Cr=1 / self.phyparam['InvCr'])

        BuildingT_next = dynamics.BuildingT_next(RoomT=RoomT,
                                                 BuildingT=BuildingT,
                                                 dt=dt,
                                                 Cb=1 / self.phyparam['InvCb'],
                                                 Rb=self.phyparam['Rb'])

        BoilerInlet_next = dynamics.BoilerInletT_next(RoomT=RoomT,
                                                      BoilerInletT=BoilerInletT,
                                                      dt=dt,
                                                      Ci=1 / self.phyparam['InvCi'],
                                                      Ri=self.phyparam['Ri'])

        Gas_consumption = dynamics.Gas_modulation(BoilerInletT_next=BoilerInlet_next,
                                                  RoomT=RoomT,
                                                  RoomSet_next=RoomSet_next,
                                                  dt=dt,
                                                  BoilerOutletT_next=BoilerOutletT_next,
                                                  BoilerOutletT=BoilerOutletT,
                                                  BoilerSetpoint_next=BoilerSetpoint_next,
                                                  mdot=self.phyparam['mdot'],
                                                  b2=self.phyparam['b2'],
                                                  b1=self.phyparam['b1'],
                                                  cg=self.phyparam['cg'])

        Output = torch.cat([RoomT_next, Gas_consumption, BoilerOutletT_next], 1)
        State = torch.cat([BoilerInlet_next, BuildingT_next], 1)

        return Output, State

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
