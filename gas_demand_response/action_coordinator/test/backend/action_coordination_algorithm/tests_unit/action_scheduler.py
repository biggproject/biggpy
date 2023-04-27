import pandas as pd
from test.backend.action_coordination_algorithm.helper.house_names import HOUSE_ID_1, HOUSE_ID_2
from src.backend.action_coordination_algorithm.action_coordinator import ActionCoordinator
from src.backend.model.DR_Event import DrEvent
from src.misc.logger import bcolors, log, log_test


# %% =========================  Test upper response levels  =========================

def test_increase_actions_scenario_1():
    log_test(f"Start Elevate Actions scenario 1", 'test_increase_actions_scenario_1')

    # Test the elevation function
    baseline_actions = {HOUSE_ID_1: 0, HOUSE_ID_2: 0}
    action_coordinator = ActionCoordinator(
        dr_event=DrEvent(10, 1),  # just an upwards DR event
        action_space=[0, 1, 2],
        control_interval_s=60,
        participants=[HOUSE_ID_1, HOUSE_ID_2]
    )

    # Manually set the ranking table
    ranking_table = pd.DataFrame(
        data={
            'device_id': [HOUSE_ID_1, HOUSE_ID_1, HOUSE_ID_2, HOUSE_ID_2],
            'action': [1, 2, 1, 2],  # 1 = lower power heating, 2 = higher power heating
            'cost': [14, 16, 12, 18],
        }
    )
    ranking_table['rank'] = ranking_table['cost'].rank(method='dense', ascending=False)
    action_coordinator._ranking_table_up = ranking_table
    action_coordinator._baseline_actions = baseline_actions
    action_coordinator._calculate_actions_per_response_level()

    # When starting the action coordinator, the current actions should be empty
    assert action_coordinator._current_response_level == 0
    actions = action_coordinator._actions_by_response_level[0]
    assert len(action_coordinator._actions_by_response_level[0]) == 2
    assert actions[HOUSE_ID_1] == 0
    assert actions[HOUSE_ID_2] == 0

    # Test the first elevation of actions
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 1
    actions = action_coordinator._actions_by_response_level[1]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 0
    assert actions[HOUSE_ID_2] == 1

    # Test the second elevation of actions
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 2
    actions = action_coordinator._actions_by_response_level[2]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 1
    assert actions[HOUSE_ID_2] == 1

    # Test the third elevation of actions
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 3
    actions = action_coordinator._actions_by_response_level[3]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 2
    assert actions[HOUSE_ID_2] == 1

    # Test the fourth elevation of actions
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 4
    actions = action_coordinator._actions_by_response_level[4]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 2
    assert actions[HOUSE_ID_2] == 2

    # Test the fifth elevation of actions, which should throw an exception because there are no more actions to elevate
    try:
        action_coordinator._increase_power_actions()
        assert False, "Exception should have been thrown"
    except Exception as e:
        assert True


def test_increase_actions_scenario_2():
    log_test(f"Start Elevate Actions scenario 2", 'test_increase_actions_scenario_2')

    # Test the elevation function
    baseline_actions = {HOUSE_ID_1: 0, HOUSE_ID_2: 0}
    action_coordinator = ActionCoordinator(
        dr_event=DrEvent(10, 1),  # just an upwards DR event
        action_space=[0, 1, 2],
        participants=[HOUSE_ID_1, HOUSE_ID_2],
        control_interval_s=60,
    )

    # Manually set the ranking table
    ranking_table = pd.DataFrame(
        data={
            'device_id': [HOUSE_ID_1, HOUSE_ID_1, HOUSE_ID_2, HOUSE_ID_2],
            'action': [1, 2, 1, 2],  # 1 = lower power heating, 2 = higher power heating
            'cost': [14, 16, 12, 13]
        }
    )
    ranking_table['rank'] = ranking_table['cost'].rank(method='dense', ascending=False)
    action_coordinator._ranking_table_up = ranking_table
    action_coordinator._baseline_actions = baseline_actions
    action_coordinator._calculate_actions_per_response_level()

    # When starting the action coordinator, the current actions should be empty
    assert action_coordinator._current_response_level == 0
    assert len(action_coordinator._actions_by_response_level[0]) == 2
    actions = action_coordinator._actions_by_response_level[0]
    assert actions[HOUSE_ID_1] == 0
    assert actions[HOUSE_ID_2] == 0

    # Test the first elevation of actions
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 1
    actions = action_coordinator._actions_by_response_level[1]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 0
    assert actions[HOUSE_ID_2] == 1

    # Test the second elevation of actions
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 2
    actions = action_coordinator._actions_by_response_level[2]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 0
    assert actions[HOUSE_ID_2] == 2

    # Test the third elevation of actions
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 3
    actions = action_coordinator._actions_by_response_level[3]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 1
    assert actions[HOUSE_ID_2] == 2

    # Test the fourth elevation of actions
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 4
    actions = action_coordinator._actions_by_response_level[4]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 2
    assert actions[HOUSE_ID_2] == 2

    # Test the fifth elevation of actions, which should throw an exception because there are no more actions to elevate
    try:
        action_coordinator._increase_power_actions()
        assert False, "Exception should have been thrown"
    except Exception as e:
        assert True


# %% =========================  Test lower response levels  =========================
def test_decrease_actions_scenario():
    log_test(f"Start Decrease Actions scenario", 'test_decrease_actions_scenario')

    # Test the decrease function
    baseline_actions = {HOUSE_ID_1: 2, HOUSE_ID_2: 2}
    action_coordinator = ActionCoordinator(
        dr_event=DrEvent(10, -1),  # just a downwards DR event
        action_space=[0, 1, 2],
        control_interval_s=60,
        participants=[HOUSE_ID_1, HOUSE_ID_2]
    )

    # Manually set the ranking table
    ranking_table_down = pd.DataFrame(
        data={
            'device_id': [HOUSE_ID_1, HOUSE_ID_1, HOUSE_ID_2, HOUSE_ID_2],
            'action': [0, 1, 0, 1],  # 0 = off, 1 = heating
            'cost': [19, 16, 20, 18],
        }
    )
    ranking_table_down['rank'] = ranking_table_down['cost'].rank(method='dense', ascending=False)
    action_coordinator._ranking_table_down = ranking_table_down
    action_coordinator._baseline_actions = baseline_actions
    action_coordinator._calculate_actions_per_response_level()

    # When starting the action coordinator, the current actions should be the baseline actions
    assert action_coordinator._current_response_level == 0
    actions = action_coordinator._actions_by_response_level[0]
    assert len(action_coordinator._actions_by_response_level[0]) == 2
    assert actions[HOUSE_ID_1] == 2
    assert actions[HOUSE_ID_2] == 2

    # Test an increase is not possible
    action_coordinator._increase_power_actions()
    assert action_coordinator._current_response_level == 0

    # Test the first decrease of actions
    action_coordinator._decrease_power_actions()
    assert action_coordinator._current_response_level == -1
    actions = action_coordinator._actions_by_response_level[-1]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 1
    assert actions[HOUSE_ID_2] == 2

    # Test the second decrease of actions
    action_coordinator._decrease_power_actions()
    assert action_coordinator._current_response_level == -2
    actions = action_coordinator._actions_by_response_level[-2]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 1
    assert actions[HOUSE_ID_2] == 1

    # Test the third decrease of actions
    action_coordinator._decrease_power_actions()
    assert action_coordinator._current_response_level == -3
    actions = action_coordinator._actions_by_response_level[-3]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 0
    assert actions[HOUSE_ID_2] == 1

    # Test the fourth decrease of actions
    action_coordinator._decrease_power_actions()
    assert action_coordinator._current_response_level == -4
    actions = action_coordinator._actions_by_response_level[-4]
    assert len(actions) == 2
    assert actions[HOUSE_ID_1] == 0
    assert actions[HOUSE_ID_2] == 0

    # Test the fifth decrease of actions, which shouldn't work
    action_coordinator._decrease_power_actions()
    assert action_coordinator._current_response_level == -4
