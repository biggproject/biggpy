"""
Neural Network Class

"""
import numpy as np
import torch
from torch import nn
from torch import optim
from tqdm.auto import tqdm


# -------------------------------------------------------------------------------------------------------------------- #


class NeuralNetwork(nn.Module):

    def __init__(self, params: dict = None):
        """

        Args:
            params:
        """
        super().__init__()

        if params is None:
            params = {
                'lr': 0.005, 'batch_size': 128, 'depth': 8,
                'boiler_network': {'input_size': 4,
                                   'fc': [16, ],
                                   'output_size': 1,
                                   'activation': 'tanh',
                                   'dropout_rate': 0.0},

                'house_network': {'input_size': 4,
                                  'fc': [16, ],
                                  'output_size': 1,
                                  'activation': 'tanh',
                                  'dropout_rate': 0.0},

                'aggregator': {'input_size': 8,
                               'fc': [16, ],
                               'output_size': 1,
                               'activation': 'tanh',
                               'dropout_rate': 0.0},
            }

        self.parameter_dict = params
        self.training_data_size = None

        self.boiler_network = nn.Sequential(*make_network(network_params=self.parameter_dict['boiler_network']))
        self.house_network = nn.Sequential(*make_network(network_params=self.parameter_dict['house_network']))
        self.aggregator_network = nn.Sequential(*make_network(network_params=self.parameter_dict['aggregator']))

        # Model Parameters
        self.lr = self.parameter_dict['lr']

        # Extra learning rates for fine-tuning
        self.boiler_lr = None
        self.house_lr = None
        self.aggregator_lr = None

        self.optimizer = self.configure_optimizers()

        self.batch_size = self.parameter_dict['batch_size']
        self.depth = self.parameter_dict['depth']
        self.max_epochs = self.parameter_dict['max_epochs']
        self.loss = None
        self.training_loss = {'Total Loss': []}

    def forward(self, x):
        x = np.array(x)
        x1 = torch.tensor(x, dtype=torch.float32)
        house_network_input = x1[:, 3: 3+1*self.depth]
        boiler_network_input = x1[:, 3+1*self.depth: 3+2*self.depth]

        observable_current_inputs = x1[:, (0, 1, -1)]  # [Time, Ta, Tr]

        boiler_latent_state = self.boiler_network(boiler_network_input)     # Encoding of past boiler modulations
        house_latent_state = self.house_network(house_network_input)        # Encoding of past room temperatures

        aggregator_input = torch.cat([observable_current_inputs, boiler_latent_state, house_latent_state], dim=1)

        prediction = self.aggregator_network(aggregator_input)

        return prediction

    @torch.no_grad()
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        prediction = self.forward(x)
        predicted_scaler = (prediction.data.numpy())

        return predicted_scaler

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), weight_decay=1e-8, lr=self.lr)

        return optimiser

    def training_step(self, batch):
        loss = self.dqn_mse_loss(batch)
        self.loss = loss.data

        return loss

    def dqn_mse_loss(self, batch):
        """Calculates the mse loss using a (mini) batch.

        Args:
            batch:

        Returns:
            loss
        """
        states, actions, target_q = batch
        act = actions.clone().detach()
        action_indices = act.type(dtype=torch.int64)
        state_action_values = torch.gather(self.forward(states), dim=1, index=action_indices.unsqueeze(-1)).squeeze(-1)

        return nn.MSELoss()(state_action_values, torch.flatten(target_q))

    def fit(self, train_dataloaders):

        pbar = tqdm(range(self.max_epochs), desc="Neural Network Epochs", leave=False)
        for current_epoch in pbar:
            loss_stack = []
            for batch in train_dataloaders:
                self.optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()
                loss_stack.append(loss.data.numpy())
            epoch_loss = np.mean(loss_stack)
            pbar.set_postfix({"train_loss": epoch_loss})
            self.training_loss['Total Loss'].append(epoch_loss)


# -------------------------------------------------------------------------------------------------------------------- #
# Helper Functions for Neural Network Class

# -------------------------------------------------------------------------------------------------------------------- #

def make_network(network_params: dict = None):
    """
    Function to create a neural network of type nn.Sequential based on the input network parameters

    Args:
        network_params:

    Returns:
        network: nn.Module
    """

    if len(network_params['fc']) == 0:
        network = [fc_module([network_params['input_size'], network_params['output_size']],
                             activation=network_params['activation'], dropout_rate=network_params['dropout_rate'])]
    else:
        network = [fc_module([network_params['input_size'], network_params['fc'][0]],
                             activation=network_params['activation'], dropout_rate=network_params['dropout_rate'])]
        for l_i in range(len(network_params['fc'][:-1])):
            network += [fc_module([network_params['fc'][l_i], network_params['fc'][l_i + 1]],
                                  activation=network_params['activation'],
                                  dropout_rate=network_params['dropout_rate'])]
        network += [fc_module([network_params['fc'][-1], network_params['output_size']],
                              activation='sigmoid', dropout_rate=network_params['dropout_rate'])]
    return network


class fc_module(nn.Module):
    """
    Fully connected module of a neural network
    """

    def __init__(self, layer_params: list = None,
                 activation: str = 'tanh',
                 dropout_rate: float = 0.0):
        """

        Args:
            layer_params:
            activation:
            dropout_rate:
        """

        super(fc_module, self).__init__()

        if activation == 'linear':
            self.fc_module = nn.Sequential(
                nn.Linear(layer_params[0], layer_params[1]),
                nn.Dropout(p=dropout_rate)
            )
        else:
            if activation == 'tanh':
                activation_function = nn.Tanh

            elif activation == 'relu':
                activation_function = nn.ReLU

            elif activation == 'sigmoid':
                activation_function = nn.Sigmoid

            else:
                raise ValueError(f"Unknown Activation function: {activation}")
            self.fc_module = nn.Sequential(
                nn.Linear(layer_params[0], layer_params[1]),
                activation_function(),
                nn.Dropout(p=dropout_rate)
            )

    def forward(self, x):
        return self.fc_module(x)

