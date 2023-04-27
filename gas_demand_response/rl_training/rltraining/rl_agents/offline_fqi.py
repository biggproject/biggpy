"""
Offline FQI Agent

Implements a meanQ based Offline RL agent:
    - N individual Q-function approximators --> mean value as final value
    - Evaluates the existing policy
    - Logs training metrics in internal system and on weights and biases (WandB)

"""
from multiprocessing import Process
import os
from collections import deque

import numpy
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler

# Neural Networks
import torch
from torch.utils.data import DataLoader

# Data specific
from rltraining.rl_agents.modular_net import NeuralNetwork as NeuralNet
from rltraining.rl_agents.utils.data_support import Experience, ExperienceBuffer, BatchDataset
from rltraining.rl_agents.utils.logging import Logger


# -------------------------------------------------------------------------------------------------------------------- #

class OfflineAgent:

    def __init__(self,
                 home_id: int = 2,
                 save_dir: str = None,
                 agent_params: dict = None,
                 network_params: dict = None,
                 seed: int = 1,
                 mode: str = "PolicyEvaluation",
                 data_frequency: int = 5,
                 transformation_obj=None,
                 monitoring: bool = False,
                 wandb_logger=None):

        if save_dir is None:
            save_dir = self._set_default_params(param="save_dir")
        if agent_params is None:
            agent_params = self._set_default_params(param="agent_params")
        if network_params is None:
            network_params = self._set_default_params(param="network_params")

        if monitoring:
            if wandb_logger is None:
                raise ValueError("wandb_logger cannot be empty when monitoring is True")
            else:
                self.wandb_logger = wandb_logger
                self.wandb_table = wandb.Table(columns=["Validation Data Plots"])
                self.wandb_table_data = []

        self.seed = seed
        self.home_id = home_id
        self.save_dir = save_dir
        self.agent_params = agent_params
        self.network_params = network_params
        self.monitoring = monitoring
        self.mode = mode
        self.data_frequency = data_frequency
        self.transformation_obj = transformation_obj
        self.logger = Logger(home_id=home_id,
                             logging_path=save_dir,
                             total_iterations=(2 * self.agent_params['horizon']))

        # Q-function Ensemble
        self.Q_ensemble = dict()
        for i in range(self.agent_params['ensemble_size']):
            torch.manual_seed(seed=100 * seed + i)
            Q_hat = [NeuralNet(params=self.network_params) for _ in range(2 * self.agent_params['horizon'])]
            self.Q_ensemble[f'Q_hat_{i + 1}'] = Q_hat

        self.target_scaler = [MinMaxScaler(feature_range=(0, 1)) for _ in range(2 * self.agent_params['horizon'])]

        # Buffer
        self.buffer = ExperienceBuffer(capacity=self.agent_params['buffer_size'])
        self.val_buffer = ExperienceBuffer(capacity=self.agent_params['buffer_size'])

        self._room_temperature_trajectory = deque(maxlen=self.network_params['depth'])
        self._boiler_modulation_trajectory = deque(maxlen=self.network_params['depth'])

        for i in range(self.network_params['depth']):
            self._room_temperature_trajectory.append(self.transformation_obj.transform_temp(19.0))
            self._boiler_modulation_trajectory.append(self.transformation_obj.transform_boiler_modulation(0.0))

        # Misc.

        # File check
        if os.path.isdir(self.save_dir) is False:
            os.makedirs(self.save_dir)

        self.figure_dir = f"{self.save_dir}/Figures"
        if os.path.isdir(self.figure_dir) is False:
            os.makedirs(self.figure_dir)

    def get_action(self, state, index=0):

        if len(state) == 1:
            state = np.array(state).reshape(1, -1)
        else:
            state = np.array(state)

        q_values = self._calculateQ(state, index)
        action_index = np.argmin(q_values)
        return action_index

    def meanQ(self, state, index=0):
        """
        Calculates the Q-value for given state and index.
        Rescales it using the appropriate scale

        """
        scaled_mean_Q = np.clip(self._calculateQ(state=state, index=index), a_min=0, a_max=1)
        mean_Q = self.target_scaler[index].inverse_transform(scaled_mean_Q)

        return mean_Q

    def _calculateQ(self, state, index=0):
        """
        Calculates the meanQ value for the given state and index using the trained function approximators
        Returns the raw mean value without any rescaling, clipping
        """
        # Dimension check
        if state.ndim == 1:
            state = state.reshape(1, -1)
        pred_Q = [self.Q_ensemble[f"Q_hat_{i + 1}"][index].predict(state) for i in
                  range(self.agent_params['ensemble_size'])]
        mean_Q = sum(pred_Q) / self.agent_params['ensemble_size']
        return mean_Q

    def batch_train(self, train_batch_df: pd.DataFrame = None, val_batch_df: pd.DataFrame = None):

        if train_batch_df is None:
            raise ValueError("train_batch_df argument cannot be None")

        batch = self._populate_buffer(train_batch_df=train_batch_df)

        states, actions, u_phys, dones, next_states, next_actions = batch.sample(len(batch))

        pbar = tqdm(range(2 * self.agent_params['horizon']), desc=f"Iteration")
        for idx in pbar:
            itr_index = 2 * self.agent_params['horizon'] - 1 - idx
            rewards = u_phys

            if itr_index == (2 * self.agent_params['horizon'] - 1):  # Last iteration
                next_state_values = np.zeros_like(rewards)
            else:
                next_Q_value_prediction = self._calculateQ(next_states, index=(itr_index + 1))
                if self.mode == "PolicyEvaluation":
                    value_prediction = np.clip(np.take_along_axis(next_Q_value_prediction, next_actions.reshape(-1, 1),
                                                                  axis=1), a_min=0, a_max=1)
                    next_state_values = self.target_scaler[itr_index + 1].inverse_transform(value_prediction)
                else:
                    min_value = np.min(next_Q_value_prediction, axis=1).reshape(-1, 1)
                    next_state_values = self.target_scaler[itr_index + 1].inverse_transform(min_value)

            Q_target = rewards + next_state_values.flatten()
            scaled_Q_target = self.target_scaler[itr_index].fit_transform(Q_target.reshape(-1, 1))
            self._fit(states=states,
                      actions=actions,
                      target_q=scaled_Q_target,
                      index=itr_index)

            if val_batch_df is None:
                val_mse = -1
            else:
                val_mse = self._validation_batch(val_batch_df=val_batch_df, itr_index=itr_index)
            pbar.set_description(desc=f"Iteration: {idx + 1} Val Error: {val_mse}")

            if self.monitoring:
                self.wandb_logger.log({"val_error": val_mse, "Itr_index": itr_index})

        # Monotonicity check
        _, monotonicity_vector = self._check_monotonicity(val_batch_df=val_batch_df, type="Val")

        print(f" Q-function Monotonicity = {monotonicity_vector.sum()}/{monotonicity_vector.size}")

        return None

    def _fit(self, states, actions, target_q, index):

        train_dataset = BatchDataset(states=torch.tensor(states, dtype=torch.float32),
                                     actions=torch.tensor(actions, dtype=torch.int64),
                                     target_q=torch.tensor(target_q, dtype=torch.float32),
                                     shuffle=False
                                     )
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.network_params['batch_size'])

        processes = []

        for i in range(self.agent_params['ensemble_size']):
            p = Process(target=self.Q_ensemble[f"Q_hat_{i + 1}"][index].fit, args=(train_dataloader,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            p.terminate()

        return None

    def _validation_batch(self, val_batch_df: pd.DataFrame = None, itr_index: int = 0):

        if val_batch_df is None:
            raise ValueError("Validation batch cannot be None")

        batch = self._populate_buffer(train_batch_df=None, val_batch_df=val_batch_df)

        states, actions, u_phys, dones, next_states, next_actions = batch.sample(len(batch))

        rewards = u_phys

        if itr_index == (2 * self.agent_params['horizon'] - 1):  # Last iteration
            next_state_values = np.zeros_like(rewards)
        else:
            Q_value_prediction = self._calculateQ(next_states, index=(itr_index + 1))
            if self.mode == "PolicyEvaluation":
                value_prediction = np.clip(np.take_along_axis(Q_value_prediction, next_actions.reshape(-1, 1),
                                                              axis=1), a_min=0, a_max=1)
                next_state_values = self.target_scaler[itr_index + 1].inverse_transform(value_prediction)
            else:
                min_value = np.min(Q_value_prediction, axis=1).reshape(-1, 1)
                next_state_values = self.target_scaler[itr_index + 1].inverse_transform(min_value)

        Q_target = rewards + next_state_values.flatten()
        scaled_Q_target = self.target_scaler[itr_index].transform(Q_target.reshape(-1, 1))

        scaled_Q_value_prediction = np.take_along_axis(self._calculateQ(states, index=itr_index),
                                                       actions.reshape(-1, 1),
                                                       axis=1)
        Q_value_prediction = self.target_scaler[itr_index].inverse_transform(scaled_Q_value_prediction)

        val_mse = np.mean((scaled_Q_target - scaled_Q_value_prediction.flatten()) ** 2)

        # Log values
        if itr_index == (2 * self.agent_params['horizon'] - 1):
            val_data_dict = dict()

            val_data_dict['time'] = self.transformation_obj.inverse_transform_time(states[:, 0])
            val_data_dict['t_r'] = self.transformation_obj.inverse_transform_temp(states[:, -1])
            val_data_dict['t_r_set'] = self.transformation_obj.inverse_transform_temp(states[:, 2])
            val_data_dict['t_out'] = self.transformation_obj.inverse_transform_outside_temp(states[:, 1])
            val_data_dict['action'] = actions

            self.logger.log_validation_data(val_data_dict=val_data_dict)

        self.logger.log_iteration(predictions=Q_value_prediction.flatten(),
                                  target=Q_target.flatten(),
                                  index=itr_index)

        _ = self._zero_action_Q_function(states=states, actions=actions, predicted_q=Q_value_prediction,
                                         index=itr_index, type="Val", )

        return val_mse

    def _populate_buffer(self, train_batch_df: pd.DataFrame = None, val_batch_df: pd.DataFrame = None):
        """

        Args:
            train_batch_df: Preprocessed dataframe

        Returns: buffer

        """
        if train_batch_df is not None:
            buffer = self.buffer
            buffer.clear()
            df = train_batch_df
        else:
            if val_batch_df is not None:
                buffer = self.val_buffer
                df = val_batch_df

            else:
                raise ValueError(f"Both train and val_df cannot be none")

        buffer.clear()
        for i in range(len(df) - 1):
            action_index = self._action_filter(action=df.loc[i, "t_set"])

            state = np.array([self.transformation_obj.transform_time(df.loc[i, "time"]),
                              self.transformation_obj.transform_outside_temp(df.loc[i, "t_out"]),
                              self.transformation_obj.transform_temp(df.loc[i, "t_r_set"]),
                              *self._room_temperature_trajectory,
                              *self._boiler_modulation_trajectory,
                              self.transformation_obj.transform_temp(df.loc[i, "t_r"]),
                              ])
            action = action_index

            u_phys = df.loc[i, "blr_mod_lvl"].sum() / 100

            # Update trajectories
            self._room_temperature_trajectory.append(self.transformation_obj.transform_temp(df.loc[i, "t_r"]))
            self._boiler_modulation_trajectory.append(
                self.transformation_obj.transform_boiler_modulation(df.loc[i, "blr_mod_lvl"]))

            next_state = np.array([self.transformation_obj.transform_time(df.loc[i + 1, "time"]),
                                   self.transformation_obj.transform_outside_temp(df.loc[i + 1, "t_out"]),
                                   self.transformation_obj.transform_temp(df.loc[i + 1, "t_r_set"]),
                                   *self._room_temperature_trajectory,
                                   *self._boiler_modulation_trajectory,
                                   self.transformation_obj.transform_temp(df.loc[i + 1, "t_r"]),
                                   # df.loc[i + 1, "heat"],
                                   # df.loc[i + 1, "day"] / 7 - 0.35
                                   ])
            next_action_index = self._action_filter(df.loc[i + 1, "t_set"])

            exp = Experience(state=state, action=action, u_phys=u_phys, done=False,
                             next_state=next_state, next_action=next_action_index)
            buffer.append(exp)

        return buffer

    @staticmethod
    def _action_filter(action):
        if action < 20:
            action_index = 0
        else:
            action_index = 1
        return action_index

    @staticmethod
    def _calculate_rollout_reward(u_phys: numpy.ndarray, rollout_steps: int = 1):
        rollout_reward = np.array(u_phys, np.ones(rollout_steps))[rollout_steps-1:]

        return rollout_reward

    def _check_monotonicity(self, val_batch_df: pd.DataFrame = None, save_fig: bool = True, **kwargs):
        """
        Function to check if the trained Q-function is monotonically decreasing with time.

        Args:
            states:
            actions:
            **kwargs:

        Returns:
            fig: plotly figure visualizing the Q-functions with a shrinking horizon
            monotonicity_vector: booleans indicating where the function was monotonous and where it wasn't

        """
        if val_batch_df is None:
            raise ValueError("Val batch cannot be None")

        gas_consumption_actual = self._calculate_shrinking_window_consumption(val_batch_df=val_batch_df)
        val_batch = self._populate_buffer(train_batch_df=None, val_batch_df=val_batch_df)
        states, actions, _, dones, _, _ = val_batch.sample(len(val_batch))

        monotonicity_vector = np.zeros_like(actions, dtype=float)
        time_indices = np.zeros_like(actions, dtype=float)
        Q_function = np.zeros_like(gas_consumption_actual, dtype=float)

        previous_Q_value = 1000000
        day_index = 0

        for i, current_state in enumerate(states):

            index = int((self.transformation_obj.inverse_transform_time(current_state[0])) / self.data_frequency) % (
                    2 * self.agent_params['horizon'])
            scaled_Q_prediction = self._calculateQ(current_state, index=index)
            Q_value = self.target_scaler[index].inverse_transform(scaled_Q_prediction)[0, actions[i]]

            if (i > 0) and (int(self.transformation_obj.inverse_transform_time(current_state[0]) == 0)):
                is_decreasing = int(True)
                previous_Q_value = 1000000
                day_index += 1
            else:
                is_decreasing = int(Q_value <= previous_Q_value)
                previous_Q_value = Q_value

            Q_function[i] = Q_value
            monotonicity_vector[i] = is_decreasing
            time_indices[i] = (self.transformation_obj.inverse_transform_time(current_state[0]) / 60) + (
                    day_index * 24.0)

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Room Temperature", "Q-function"],
                            vertical_spacing=0.1,
                            specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
                            )
        fig.add_trace(
            go.Scatter(x=time_indices,
                       y=self.transformation_obj.inverse_transform_temp(states[:, -1]),
                       name="Room Temperature",
                       opacity=0.75,
                       mode="markers+lines",
                       marker=dict(color='rgb(70,73,255)', size=1),
                       line=dict(color='rgb(70,73,255)'),
                       ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_indices,
                       y=self.transformation_obj.inverse_transform_temp(states[:, 2]),
                       name="Room Temperature Setpoint",
                       opacity=0.75,
                       mode="markers+lines",
                       marker=dict(color='rgb(70, 184, 255)', size=1),
                       line=dict(color='rgb(70, 184, 255)', dash='dot'),
                       ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_indices,
                       y=Q_function,
                       name="Predicted Q-function",
                       opacity=0.75,
                       mode="markers+lines",
                       marker=dict(color='red', size=1),
                       line=dict(color='red'),
                       ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_indices,
                       y=gas_consumption_actual,
                       name="Actual Q-function",
                       mode="markers+lines",
                       marker=dict(color='blue', size=1),
                       line=dict(color='blue'),
                       ),
            row=2, col=1,
        )

        fig.add_trace(go.Scatter(x=time_indices, y=actions, mode="lines", name="Actions",
                                 line=dict(color="black", dash='dash')),
                      row=2, col=1,
                      secondary_y=True
                      )

        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Temperature", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Q-function", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Action", row=2, col=1, range=[0, 2], secondary_y=True)

        if "type" in kwargs.keys():
            data_type = kwargs['type']
        else:
            data_type = "Training"

        if "title" in kwargs.keys():
            title = kwargs['title']
        else:
            title = f"Q-function over time"

        fig.update_layout(
            title=f"{title}",
            template="plotly_white",
        )

        gas_consumption_data_dict = {"target_consumption": gas_consumption_actual,
                                     "predicted_consumption": Q_function}
        gas_consumption_df = pd.DataFrame.from_dict(gas_consumption_data_dict)
        gas_consumption_df.to_csv(path_or_buf=f"{self.save_dir}/gas_consumption_data.csv")

        if save_fig:
            file_path = f"{self.figure_dir}/Q_function_over_time_{data_type}_data.html"
            fig.write_html(file_path)

            if self.monitoring:
                table = wandb.Table(columns=["Q-function Comparison"])
                table.add_data(wandb.Html(file_path))
                self.wandb_logger.log({"Q-function Comparison": table})

        return fig, monotonicity_vector

    def _calculate_shrinking_window_consumption(self, val_batch_df: pd.DataFrame = None):
        """
        Calculates the actual gas consumption from current time till the end of the day
        Works in a shrinking horizon setting

        Args:
            val_batch_df:

        Returns:
            gas_consumption_list:

        """
        if val_batch_df is None:
            raise ValueError("Val batch cannot be None")

        scaled_blr_mod_lvl = (val_batch_df.loc[:, "blr_mod_lvl"])
        number_of_steps_in_day = 24 * 60 // self.data_frequency
        gas_consumption_list = np.zeros(len(val_batch_df))

        for i in range(len(val_batch_df) - 1):
            index_modulo_steps = i % number_of_steps_in_day
            gas_consumption = scaled_blr_mod_lvl.values[i:i + (number_of_steps_in_day - index_modulo_steps - 1)].sum()
            gas_consumption_list[i] = gas_consumption / 100

        return gas_consumption_list

    def _zero_action_Q_function(self, states, actions, predicted_q, index=0, save_fig: bool = True, **kwargs):
        """
        Function visualizing the Q-function when each action is set to 0

        """

        scaled_zero_action_q = np.take_along_axis(self._calculateQ(states, index=index),
                                                  np.zeros_like(actions.reshape(-1, 1)), axis=1)
        zero_action_q = self.target_scaler[index].inverse_transform(scaled_zero_action_q).flatten()

        scaled_one_action_q = np.take_along_axis(self._calculateQ(states, index=index),
                                                 np.ones_like(actions.reshape(-1, 1)), axis=1)
        one_action_q = self.target_scaler[index].inverse_transform(scaled_one_action_q).flatten()

        fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                            specs=[[{"secondary_y": True}]],
                            subplot_titles=["Predictions"], vertical_spacing=0.1
                            )

        x_index = np.arange(0, len(predicted_q), 1)

        fig.add_trace(
            go.Scatter(x=x_index, y=predicted_q.flatten(), mode="lines+markers", name="Predicted Q",
                       marker=dict(color="#242423", size=1), line=dict(color="#242423")),
            row=1, col=1,
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=x_index, y=zero_action_q.flatten(), mode="lines+markers", name="Zero-action Q",
                       marker=dict(color="#e71d36", size=1), line=dict(color="#e71d36")),
            row=1, col=1,
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=x_index, y=one_action_q.flatten(), mode="lines+markers", name="One-action Q",
                       marker=dict(color="#5390d9", size=1), line=dict(color="#5390d9")),
            row=1, col=1,
            secondary_y=False
        )

        fig.add_trace(go.Scatter(x=x_index, y=actions.flatten(), mode="lines", name="Actions",
                                 line=dict(color="black", dash='dash')),
                      row=1, col=1,
                      secondary_y=True
                      )

        if "type" in kwargs.keys():
            data_type = kwargs['type']
        else:
            data_type = "Training"

        if "title" in kwargs.keys():
            title = kwargs['title']
        else:
            title = f"Sequential Visualization of Q-function for Hour {index * self.data_frequency / 60}"

        fig.update_yaxes(title_text="Q-value", row=1, col=1, secondary_y=False)

        fig.update_yaxes(title_text="Action", row=1, col=1, range=[0, 2], secondary_y=True)
        fig.update_layout(
            title=f"{title}",
            template="plotly_white",
            autosize=True,
        )

        if save_fig:
            fig_dir = f"{self.figure_dir}/Zero_action_Sequential_Q_function_dir_{data_type}_data"
            if os.path.isdir(fig_dir) is False:
                os.mkdir(fig_dir)

            file_path = f"{fig_dir}/Q_function_iteration_{index}.html"
            fig.write_html(file_path)

            if self.monitoring:
                table = wandb.Table(columns=["Zero action Q-function"])
                table.add_data(wandb.Html(file_path))
                self.wandb_logger.log({"Zero-action Q-function": table})

        return fig

    # -------- Misc Methods --------

    @staticmethod
    def _set_default_params(param: str = None):
        """

        Args:
            param:  Parameter for which default parameters should be set

        Returns:
            default value of the parameter

        """
        if param == "save_dir":
            default_value = f"./data/Default"
        elif param == "network_params":
            default_value = {
                'lr': 0.01,
                'batch_size': 1024,
                'depth': 4,
                'max_epochs': 100,
                'boiler_network': {'input_size': 2 * 4,  # [x_t-1, x_t-2]
                                   'fc': [32] * 1,
                                   'output_size': 1,  # [T_m]
                                   'activation': 'relu',
                                   'dropout_rate': 0.0},
                'house_network': {'input_size': 2 * 4,  # [x_t-1, x_t-2]
                                  'fc': [32] * 1,
                                  'output_size': 1,  # [T_m]
                                  'activation': 'relu',
                                  'dropout_rate': 0.0},
                'aggregator': {'input_size': 7,  # [t, Ta, Tr_set, Tm, Tbm, bm, Tr]
                               'fc': [32, ],
                               'output_size': 2,  # [Next bm (boiler_modulation)]
                               'activation': 'relu',
                               'dropout_rate': 0.0},
            }
        elif param == "agent_params":
            default_value = {
                'horizon': 24,
                'ensemble_size': 1,
                'action_space': [0, 1],
                'buffer_size': 10000,
            }
        else:
            default_value = -1
            raise ValueError("Invalid Parameter")

        return default_value
