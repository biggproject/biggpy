import os
from os import path

from src.misc.logger import log_test
from src.backend.controller.storageacces import Storage
from test.backend.action_coordination_algorithm.tests_unit.q_function_inference import \
    _generate_state
from src.backend.action_coordination_algorithm.helper import load_config
from src.backend.action_coordination_algorithm.model.A_Function import A_Function


def test_a_functions():
    root = path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    config = load_config()

    path_training = os.path.join(root, 'data', 'training')
    assert os.path.isdir(path_training), 'Training directory does not exist'

    house_id = 'House_13'
    storage = Storage()
    agent = storage.load_agent(house_id)
    state = _generate_state(config['agent']['trajectory_length'])

    a_function = A_Function(agent, 0, state)
    log_tag = 'test_a_functions'
    log_test(f'A function object: {a_function}', log_tag)
    a_values, u = a_function(state)
    log_test(f'A values: {a_values} | chosen action: {u}', log_tag)
    assert u == 1
