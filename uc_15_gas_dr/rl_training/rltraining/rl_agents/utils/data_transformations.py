"""
Class definition for data transformation required for scaling/ rescaling data in required ranges

"""
import yaml


class DataTransformation:
    """CLass encapsulating transformation functions """

    def __init__(self, home_id: int = 2, config_path: str = None):
        """

        Args:
            home_id:
            config_path:
        """
        self.home_id = home_id
        self.config_path = config_path

        with open(self.config_path, 'r') as f_open:
            config_data = yaml.safe_load(f_open)

        self.global_values = config_data[self.home_id]

    def transform_temp(self, unscaled_temp):
        if type(unscaled_temp) == list:
            TypeError("Input Must be ndarray")

        scaled_temp = (unscaled_temp - self.global_values['TEMP_CONSTANT']) / self.global_values['MAX_TEMP']
        return scaled_temp

    def inverse_transform_temp(self, scaled_temp):
        if type(scaled_temp) == list:
            TypeError("Input Must be ndarray")

        unscaled_temp = (scaled_temp * self.global_values['MAX_TEMP']) + self.global_values['TEMP_CONSTANT']
        return unscaled_temp

    def transform_boiler_temperature(self, unscaled_boiler_temperature):
        if type(unscaled_boiler_temperature) == list:
            TypeError("Input Must be ndarray")

        scaled_boiler_temperature = (unscaled_boiler_temperature - self.global_values['BOILER_TEMPERATURE_CONSTANT']) / self.global_values['MAX_BOILER_TEMPERATURE']
        return scaled_boiler_temperature

    def inverse_transform_boiler_temperature(self, scaled_boiler_temperature):
        if type(scaled_boiler_temperature) == list:
            TypeError("Input Must be ndarray")

        unscaled_boiler_temperature = (scaled_boiler_temperature * self.global_values['MAX_BOILER_TEMPERATURE']) + self.global_values['BOILER_TEMPERATURE_CONSTANT']
        return unscaled_boiler_temperature

    def transform_boiler_modulation(self, unscaled_boiler_modulation):
        if type(unscaled_boiler_modulation) == list:
            TypeError("Input Must be ndarray")

        scaled_boiler_modulation = (unscaled_boiler_modulation - self.global_values['BOILER_MODULATION_CONSTANT']) / self.global_values['MAX_BOILER_MODULATION']
        return scaled_boiler_modulation

    def inverse_transform_boiler_modulation(self, scaled_boiler_modulation):
        if type(scaled_boiler_modulation) == list:
            TypeError("Input Must be ndarray")

        unscaled_boiler_modulation = (scaled_boiler_modulation * self.global_values['MAX_BOILER_MODULATION']) + self.global_values['BOILER_MODULATION_CONSTANT']
        return unscaled_boiler_modulation

    def transform_outside_temp(self, unscaled_temp):
        if type(unscaled_temp) == list:
            TypeError("Input Must be ndarray")

        scaled_temp = (unscaled_temp - self.global_values['OUTSIDE_AIR_TEMP_CONSTANT']) / self.global_values['MAX_OUTSIDE_AIR_TEMP']
        return scaled_temp

    def inverse_transform_outside_temp(self, scaled_temp):
        if type(scaled_temp) == list:
            TypeError("Input Must be ndarray")

        unscaled_temp = (scaled_temp * self.global_values['MAX_OUTSIDE_AIR_TEMP']) + self.global_values['OUTSIDE_AIR_TEMP_CONSTANT']
        return unscaled_temp

    def transform_time(self, unscaled_time):
        if isinstance(unscaled_time, list):
            TypeError("Input must be ndarray")

        scaled_time = (unscaled_time - self.global_values['TIME_CONSTANT']) / self.global_values['MAX_TIME']
        return scaled_time

    def inverse_transform_time(self, scaled_time):
        if isinstance(scaled_time, list):
            TypeError("Input must be ndarray")

        unscaled_time = (scaled_time * self.global_values['MAX_TIME']) + self.global_values['TIME_CONSTANT']
        return unscaled_time
