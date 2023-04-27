"""
Neural Network support functions

"""
# -------------------------------------------------------------------------------------------------------------------- #

from torch import nn

# -------------------------------------------------------------------------------------------------------------------- #
# Network Class


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
