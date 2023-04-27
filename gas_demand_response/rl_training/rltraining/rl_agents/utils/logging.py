"""
Basic Logging Class for saving training metrics


"""
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------------------------------------------------- #


class Logger:
    """
    Basic logger class for storing training information
    """

    def __init__(self,
                 home_id: int = 2,
                 logging_path: str = None,
                 total_iterations: int = 144,
                 ):

        """

        Args:
            home_id:
            logging_path:
        """

        self.home_id = home_id

        if logging_path is None:
            raise ValueError("Logging path not provided")
            # self.logging_path = None
        else:
            self.logging_path = logging_path

        self.iterations = total_iterations

        self.prediction_dict = dict()
        self.target_dict = dict()
        self.validation_data = dict()

    def log_iteration(self, predictions: np.ndarray, target: np.ndarray, index: int):
        """
        Log entries for a specific iteration in the appropriate dict
        Args:
            predictions:
            target:
            index:

        Returns: None
        """
        self.target_dict[f'Itr_{index}'] = target
        target_df = pd.DataFrame.from_dict(self.target_dict)
        target_df.to_csv(path_or_buf=f"{self.logging_path}/target_data.csv")

        self.prediction_dict[f'Itr_{index}'] = predictions
        prediction_df = pd.DataFrame.from_dict(self.prediction_dict)
        prediction_df.to_csv(path_or_buf=f"{self.logging_path}/prediction_data.csv")

    def log_validation_data(self, val_data_dict: dict):
        """

        Args:
            val_data_dict: Dictionary related to validation data used. Must include the following keys:
                - time:     time in minutes
                - t_r:      room temperature
                - t_r_set:  room temperature setpoint
                - t_out:    outside air temperature
                - action:   boiler setpoint

        Returns: None

        """

        self.validation_data = val_data_dict

        validation_df = pd.DataFrame.from_dict(self.validation_data)
        validation_df.to_csv(path_or_buf=f"{self.logging_path}/validation_data.csv")

        return None
