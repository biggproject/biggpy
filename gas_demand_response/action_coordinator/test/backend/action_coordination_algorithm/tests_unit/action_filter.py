import numpy as np
import pandas as pd

from src.backend.action_coordination_algorithm.action_coordinator import ActionCoordinator
from src.misc.logger import bcolors, log, log_test


def test_filter_unreachable_actions_scenario_1():
    log_test(f"Test filter unreachable device-action pairs (scenario 1)", 'test_filter_unreachable_actions_scenario_1')
    action_coordinator = ActionCoordinator(None, None, None, None)

    # Create dataframe
    ranking_table = [
        [1, 1, 2, 2, 1, 2],
        [2, 3, 2, 3, 1, 1],
        [11, 12, 15, 16, 16, 20]
    ]
    ranking_table = pd.DataFrame(np.transpose(ranking_table), columns=['device_id', 'action', 'cost'])
    filtered = action_coordinator._filter_unreachable_actions_upwards(ranking_table)
    assert len(filtered) == 4

def test_filter_unreachable_actions_scenario_2():
    log_test(f"Test filter unreachable device-action pairs (scenario 2)", 'test_filter_unreachable_actions_scenario_2')
    action_coordinator = ActionCoordinator(None, None, None, None)

    # Create dataframe
    ranking_table = [
        [1, 1, 1, 2, 2, 2],
        [1, 2, 3, 1, 2, 3],
        [11, 12, 15, 16, 17, 20]
    ]
    ranking_table = pd.DataFrame(np.transpose(ranking_table), columns=['device_id', 'action', 'cost'])
    filtered = action_coordinator._filter_unreachable_actions_upwards(ranking_table)
    assert len(filtered) == 6


def test_filter_unreachable_actions_scenario_3():
    log_test(f"Test filter unreachable device-action pairs (scenario 3)", 'test_filter_unreachable_actions_scenario_3')
    action_coordinator = ActionCoordinator(None, None, None, None)

    # Create dataframe
    ranking_table = [
        [1, 1, 1, 2, 2, 2],
        [1, 2, 3, 1, 2, 3],
        [11, 12, 15, 16, 17, 20]
    ]
    ranking_table = pd.DataFrame(np.transpose(ranking_table), columns=['device_id', 'action', 'cost'])
    filtered = action_coordinator._filter_unreachable_actions_downwards(ranking_table)
    assert len(filtered) == 2

def test_filter_unreachable_actions_scenario_4():
    log_test(f"Test filter unreachable device-action pairs (scenario 4)", 'test_filter_unreachable_actions_scenario_4')
    action_coordinator = ActionCoordinator(None, None, None, None)

    # Create dataframe
    ranking_table = [
        [1, 1, 1, 2, 2, 2],
        [3, 2, 1, 3, 2, 1],
        [11, 12, 15, 16, 17, 20]
    ]
    ranking_table = pd.DataFrame(np.transpose(ranking_table), columns=['device_id', 'action', 'cost'])
    filtered = action_coordinator._filter_unreachable_actions_downwards(ranking_table)
    assert len(filtered) == 6

