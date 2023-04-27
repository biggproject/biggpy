import os
from os import path
import numpy as np
from src.backend.action_coordination_algorithm.model.State import State
from src.backend.controller.storageacces import Storage
from src.backend.action_coordination_algorithm.helper import load_config
from src.misc.logger import log, log_test


def test_q_functions():
    log_tag = 'test_q_functions'
    root = path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    config = load_config()

    path_training = os.path.join(root, 'data', 'training')
    assert os.path.isdir(path_training), 'Training directory does not exist'

    house_id = 'House_13'
    storage = Storage()
    agent = storage.load_agent(house_id)
    state = _generate_state(config['agent']['trajectory_length'])
    state = state.transform(agent)
    log_test(f'Start inference for house {house_id}', log_tag)
    mean_qs = agent.meanQ(state)
    action_space = config['agent']['action_space']
    u = action_space[np.argmax(mean_qs)]
    log_test(f'Mean Qs for {house_id} is {mean_qs}, resulting in action {u}', log_tag)


def _generate_state(trajectory_length) -> State:
    """
    Generates a state, which can be used as input of the meanQ() function.
    """
    time = int(9 * (60 / 5))  # 9:00 in the morning with 5 min resolution
    t_out = 10.0  # 10 degrees outside
    t_r_setpoint = 20.0  # 20 degrees setpoint room
    t_r_trajectory = np.linspace(15, 19, trajectory_length)
    b_m_trajectory = np.zeros(trajectory_length)
    t_r = 19.1

    # Transform data
    state = State(time, t_out, t_r_setpoint, t_r_trajectory, b_m_trajectory, t_r)
    return state


