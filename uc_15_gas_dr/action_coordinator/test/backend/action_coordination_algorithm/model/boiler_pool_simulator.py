from dataclasses import dataclass


class BoilerPoolSimulator:
    last_response_level: int = 0

    def step(self, response_level):
        """
        Simulate the boiler pool. Takes the Response Level as input and returns the observed power.
        :param response_level:
        :return:
        """
        output = (2 * self.last_response_level)
        self.last_response_level = response_level
        return output
