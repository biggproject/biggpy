"""
Utility functions for data related support

"""
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from collections import namedtuple, deque

from torch.utils.data.dataset import IterableDataset

# -------------------------------------------------------------------------------------------------------------------- #
# Data Handling Classes

Experience = namedtuple('Experience', field_names=['state', 'action', 'u_phys',
                                                   'done', 'next_state', 'next_action'])


class ExperienceBuffer:
    """
    Replay Buffer for storing experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        """
        Add experience to the buffer
        Args:
            experience: tuple (input_state, target_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int = 1000):
        """
        Sample a batch (sequential) from the buffer
        """

        # Check if sample size <= buffer size
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        indices = np.arange(0, batch_size)
        states, actions, u_phys, dones, next_states,  next_actions = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(u_phys, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states), np.array(next_actions))

    def clear(self):
        self.buffer.clear()


class BatchDataset(IterableDataset):
    """
    Iterable Dataset containing the Input-Output pairs for
    training the functional approximators
    """

    def __init__(self, states, actions, target_q, shuffle=False):
        self.states = states
        self.actions = actions
        self.target_q = target_q
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(0, len(self.actions))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in indices:
            yield self.states[i], self.actions[i], self.target_q[i]
