import os
from datetime import datetime
from os import path
import pickle
import pandas as pd

from test.backend.action_coordination_algorithm.helper.DummyAgent import DummyAgent

root = path.join(path.dirname(__file__), "..", "..", "..", "..")
date_str = '{dt.day}_{dt.month}_{dt.year}'.format(dt=datetime.now())

def clear_action_coordinator_logs():
    with open(path.join(root, 'data', 'action-coord', 'coordinator_actions.txt'), 'w') as f:
        f.write("")


def create_simple_Q_agent(device_id: str, value_on, value_off):
    # Generate date string in format DD_MM_YYYY
    folder_name = f'{device_id}_{date_str}'
    folder_path = path.join(root, 'data', 'training', folder_name)
    # Create folder if it does not exist
    if not path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = path.join(folder_path, 'models.pkl')
    q_fn = DummyAgent(value_on, value_off)
    pickle.dump(q_fn, open(file_path, 'wb'))


def create_measurement_file(device_id: str, power: float, action: float, trajectory_length: int, time_interval: float):
    folder_name = f'{device_id}_{date_str}'
    file_dir = path.join(root, 'data', 'training', folder_name, device_id + '_measurements.csv')
    df = pd.DataFrame(data={"time": [0], 't_out': [0], 't_r_set': [0], 't_r': [0], "blr_mod_lvl": [power], "t_set": [action]})

    for i in range(1, trajectory_length):
        action_mapped = 65 if action == 1 else 10
        df = pd.concat([df, pd.DataFrame([[i * time_interval, 0, 0, 0, power, action_mapped]],
                                         columns=['time', 't_out', 't_r_set', 't_r', 'blr_mod_lvl', 't_set'])],
                       ignore_index=True)
    df.to_csv(file_dir, index=False)


def read_latest_coordination_action() -> tuple[float, int, dict]:
    with open(path.join(root, 'data', 'action-coord', 'coordinator_actions.txt'), 'r') as f:
        lines = f.readlines()
        # Get last line
        if len(lines) > 0:
            line = lines[-1]
        else:
            assert False, "No lines in coordinator_actions.txt"
        # Get action dictionary
        timestamp, response_level, action_dict = line.split("; ")
        action_dict = eval(action_dict)
        return timestamp, response_level, action_dict


def add_measurement_to_device(device_id, power, action, timestamp):
    folder_name = f'{device_id}_{date_str}'
    file_dir = path.join(root, 'data', 'training', folder_name, device_id + '_measurements.csv')
    df = pd.read_csv(file_dir)
    df = pd.concat([df, pd.DataFrame([[timestamp, 0, 0, 0, power, action]],
                                     columns=['time', 't_out', 't_r_set', 't_r', 'blr_mod_lvl', 't_set'])],
                   ignore_index=True)
    df.to_csv(file_dir, index=False)
