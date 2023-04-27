from dataclasses import dataclass
import numpy as np

from test.backend.action_coordination_algorithm.helper.DummyTransformationObj import \
    DummyTransformationObj


@dataclass
class DummyAgent:
    """
    This class is used to test the ActionCoordinationAlgorithm.
    It is a dummy Q function that always returns the same value.
    """

    value_on: float
    value_off: float

    transformation_obj = DummyTransformationObj()

    def meanQ(self, state: np.array) -> np.array:
        # Ignore the state, always return the same value
        return np.array([[self.value_on, self.value_off]])
