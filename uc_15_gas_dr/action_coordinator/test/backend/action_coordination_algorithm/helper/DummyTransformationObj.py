from dataclasses import dataclass
import numpy as np


@dataclass
class DummyTransformationObj:
    def transform_time(self, value):
        return value

    def transform_outside_temp(self, value):
        return value

    def transform_temp(self, value):
        return value

    def transform_boiler_modulation(self, value):
        return value
