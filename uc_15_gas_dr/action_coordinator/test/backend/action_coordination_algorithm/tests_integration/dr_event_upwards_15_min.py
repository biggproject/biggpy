import time

from src.backend.action_coordination_algorithm.action_coordinator import ActionCoordinator
from src.backend.action_coordination_algorithm.helper import load_config
from src.backend.model.DR_Event import DrEvent
from src.misc.logger import log_error, log_test
from test.backend.action_coordination_algorithm.helper.file_helper import clear_action_coordinator_logs, \
    create_simple_Q_agent, create_measurement_file, read_latest_coordination_action, add_measurement_to_device
from test.backend.action_coordination_algorithm.helper.house_names import *


# %% =========================  Setup  =========================
def setup_integration_upwards_dr(trajectory_length: int, time_interval: float):
    clear_action_coordinator_logs()

    # Create Q agents
    create_simple_Q_agent(HOUSE_ID_1, 16, 15)
    create_simple_Q_agent(HOUSE_ID_2, 12, 10)
    create_simple_Q_agent(HOUSE_ID_3, 8, 5)
    create_simple_Q_agent(HOUSE_ID_4, 4, 2)
    create_simple_Q_agent(HOUSE_ID_5, 2, 1)

    # Create initial measurements
    create_measurement_file(HOUSE_ID_1, 4.0, 1, trajectory_length, time_interval)
    create_measurement_file(HOUSE_ID_2, 0.0, 0, trajectory_length, time_interval)
    create_measurement_file(HOUSE_ID_3, 0.0, 0, trajectory_length, time_interval)
    create_measurement_file(HOUSE_ID_4, 4.0, 1, trajectory_length, time_interval)
    create_measurement_file(HOUSE_ID_5, 0.0, 0, trajectory_length, time_interval)

# %% =========================  Tests  =========================
def test_integration_upwards_dr_15_min():
    config = load_config()
    trajectory_length = config['agent']['trajectory_length']
    action_space = [0, 1]  # Just the 'on' or 'off' action
    speedup_factor = 10
    interval = 10 / speedup_factor  # Normally 10 second, but we speed up the test by 10x
    setup_integration_upwards_dr(trajectory_length, interval)
    log_test(f"Integration Tests for Upwards DR", 'test_integration_upwards_dr_15_min')

    # %%  1. Create a DR Event
    dr_event = DrEvent(
        duration_sec=int(900/speedup_factor),  # 900 / 300 = 15 minutes real time = 3 seconds test
        power_alternation=15.0,
    )

    # %% 2. Create and start the action coordinator in a separate thread
    action_coordinator = ActionCoordinator(
        dr_event,
        action_space,
        participants=[HOUSE_ID_1, HOUSE_ID_2, HOUSE_ID_3, HOUSE_ID_4, HOUSE_ID_5],
        control_interval_s=interval,
        kp=0.01,
        ki=0.001,
    )
    action_coordinator.start()  # Start the action coordinator in a separate thread

    # %% 3. Append fake observations, just to emulate feedback from the environment
    test_device_time = trajectory_length * interval
    test_device_time_dr = interval
    while test_device_time_dr <= dr_event.duration_sec:
        while action_coordinator.get_cur_time() < test_device_time_dr:
            time.sleep(interval / 10)  # Waiting for coordinator to catch up
        if (action_coordinator.get_cur_time() - test_device_time_dr) > interval:
            offset = action_coordinator.get_cur_time() - test_device_time_dr
            log_error(
                f"Coordinator time has an offset of {offset} seconds,"
                f" consider increasing the control interval such that this test can keep up with the coordinator.",
                "test_integration_upwards_dr_15_min"
            )

        # For the observed power we just use some multiple of the action
        timestamp, response_level, action_dict = read_latest_coordination_action()
        assert len(action_dict) == 5, "Expected 3 actions"
        assert HOUSE_ID_1 in action_dict, "Expected action for house 1"
        assert HOUSE_ID_2 in action_dict, "Expected action for house 2"
        assert HOUSE_ID_3 in action_dict, "Expected action for house 3"
        assert HOUSE_ID_4 in action_dict, "Expected action for house 4"
        assert HOUSE_ID_5 in action_dict, "Expected action for house 5"

        # Get actions and create fake observations
        add_measurement_to_device(HOUSE_ID_1, (action_dict[HOUSE_ID_1]) * 4.0, -1, test_device_time)
        add_measurement_to_device(HOUSE_ID_2, (action_dict[HOUSE_ID_2]) * 24.0, -1, test_device_time)
        add_measurement_to_device(HOUSE_ID_3, (action_dict[HOUSE_ID_3]) * 9, -1, test_device_time)
        add_measurement_to_device(HOUSE_ID_4, (action_dict[HOUSE_ID_4]) * 4, -1, test_device_time)
        add_measurement_to_device(HOUSE_ID_5, (action_dict[HOUSE_ID_5]) * 5.5, -1, test_device_time)

        test_device_time += interval
        test_device_time_dr += interval
    action_coordinator.join()  # Wait for the action coordinator to finish
    assert action_coordinator.finished, "DR event not finished"

    # %% 4. Plot results
    action_coordinator.plot_dr_response()
