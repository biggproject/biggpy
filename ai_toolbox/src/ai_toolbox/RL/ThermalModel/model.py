#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from .phycell import PhysicsCell
from .densenet import DenseNet

###############################################################################################
###############################################################################################
# - Manu Lahariya, IDLab, 2/12/21
# thermalmodel: thermal model of the house,
#   - Implemented using the phycell - the recurrent unit that implements the dynamics of space heating
#   - phycell uses the dynamics of house
#   - A dense network is used to approximate exogenous disturbances
#
#
###############################################################################################
###############################################################################################


torch.set_default_dtype(torch.float64)

def MSE_loss(y, y_target):
    return F.mse_loss(y, y_target)

class thermalmodel(pl.LightningModule):
    '''
    This is the thermal model for space heating
    - Implemented using the phycell - the recurrent unit that implements the dynamics of space heating
    - phycell uses the dynamics of house
    - A dense network is used to approximate exogenous disturbances

    '''

    def __init__(self,learning_rate=1e-3):
        super().__init__()
        self.sequence_length = 1
        self.PhyCell = PhysicsCell()
        self.AmbientTNet = DenseNet([1,32,32,1])
        self.learning_rate = learning_rate


    def forward(self, x):
        '''
        :param x: input
        :return: Output, state, exogenous
        '''
        batch_size, channels ,sequence_length, input_dim = x.shape
        self.sequence_length = sequence_length
        x = x.reshape([batch_size, sequence_length, input_dim])

        # Ambient Temperature Network: RSdtT
        TOD = x[:, :,5:6].reshape([-1,1])
        for layer in self.AmbientTNet[:-1]:
            TOD = F.relu(layer(TOD))
        TOD = (self.AmbientTNet[-1](TOD))
        TOD = TOD.reshape([batch_size, self.sequence_length, 1])

        # Update the X
        x = torch.concat([x[:, :,0:5].reshape([batch_size, self.sequence_length, 5]),
                          TOD ], 2)

        # Physics Module: RSdtA
        # initialization
        Tb_inital = x[:, 0, 0:1]
        Ti_initial = x[:, 0, 1:2]
        state = torch.cat((Ti_initial, Tb_inital), 1)

        # this is the recurrence
        for i in range(self.sequence_length):
            output, state = self.PhyCell.forward(Inputs = torch.reshape(x[:,i,:],[-1,input_dim]), State=state)

        return output, state, TOD

    def forward_pass(self, data):
        '''
        Forward pass of the model,
        - used by all steps: training, validation and testing steps
        '''
        x, y, h, Ta = data['x'], data['y'], data['h'], data['Ta']
        output, state, AmbientT = self.forward(x)
        losses = self.losses(y=y, y_hat=output,
                             h=h, h_hat=state, Ta=Ta, Ta_hat=AmbientT)
        return losses

    def training_step(self, train_batch, batch_idx):
        '''
        Torch training step
        - data should be fed as a batch
            - dictionary {x,y,h,Ta}
        '''
        losses = self.forward_pass(data=train_batch)
        self.mylog(losses, type="Training")
        return losses[4]

    def validation_step(self, val_batch, batch_idx):
        '''
        Torch validation step
        '''
        losses = self.forward_pass(data=val_batch)
        self.mylog(losses, type="Validation")
        return losses[4]

    def test_step(self, test_batch, batch_idx):
        '''
        Torch testing step
        '''
        losses = self.forward_pass(data=test_batch)
        self.mylog(losses, type="Tetsing")
        return losses[4]

    def losses(self, y, y_hat, h, h_hat, Ta, Ta_hat):
        '''
        Loss functions used to train the thermal model
        - y, y_hat: real and predicted values for output
        - h, h_hat: real and predicted values for the hidden variables
        - Ta, Ta_hat: real and predicted values for the exogenous variables
        '''
        h_hat = h_hat[:, 0:1].reshape([-1, 1, 1])
        h_loss = MSE_loss(h, h_hat) / 100
        weight_loss = self.PhyCell.param_loss()
        y_loss = MSE_loss(y.reshape(-1,3)[:,0:3], y_hat.reshape(-1,3)[:,0:3])
        Ta_hat = Ta_hat.reshape([-1, 1, self.sequence_length, 1])
        Ta_loss = MSE_loss(Ta, Ta_hat)
        total = Ta_loss/5 + y_loss/50 + weight_loss

        return [y_loss, h_loss, Ta_loss, weight_loss, total]

    def mylog(self, losses, type="Train"):
        '''
        This function logs the losses
        Used by all steps: training, validation and testing.
        '''
        self.log(str(type) + "_y_loss", losses[0], on_epoch=True, on_step=False)
        self.log(str(type) + "_h_loss", losses[1], on_epoch=True, on_step=False)
        self.log(str(type) + "_Ta_loss", losses[2], on_epoch=True, on_step=False)
        self.log(str(type) + "_weight_loss", losses[3], on_epoch=True, on_step=False)
        self.log(str(type) + "_loss", losses[4], on_epoch=True, on_step=False)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
