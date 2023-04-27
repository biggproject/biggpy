import time

from src.backend.action_coordination_algorithm.helper import load_config
from test.backend.action_coordination_algorithm.helper.house_names import HOUSE_ID_1, HOUSE_ID_2, HOUSE_ID_3
from test.backend.action_coordination_algorithm.helper.file_helper import clear_action_coordinator_logs, \
    create_simple_Q_agent, create_measurement_file, read_latest_coordination_action, add_measurement_to_device
from src.backend.action_coordination_algorithm.action_coordinator import ActionCoordinator
from src.backend.model.DR_Event import DrEvent
from src.misc.logger import log_error, log, bcolors, log_test


# %% =========================  Setup  =========================

def setup_integration_downwards_dr(trajectory_length: int, time_interval: float):
    clear_action_coordinator_logs()

    # Create A functions
    create_simple_Q_agent(HOUSE_ID_1, 16, 15)
    create_simple_Q_agent(HOUSE_ID_2, 12, 10)
    create_simple_Q_agent(HOUSE_ID_3, 8, 5)

    # Create measurement files
    create_measurement_file(HOUSE_ID_1, 2.25, 1, trajectory_length, time_interval)
    create_measurement_file(HOUSE_ID_2, 2.25, 1, trajectory_length, time_interval)
    create_measurement_file(HOUSE_ID_3, 2.25, 1, trajectory_length, time_interval)


# %% =========================  Tests  =========================
def test_integration_downwards_dr_1h():
    config = load_config()
    trajectory_length = config['agent']['trajectory_length']
    action_space = [0, 1]  # Just the 'on' or 'off' action
    speedup_factor = 10
    interval = 10 / speedup_factor  # Normally 10 second, but we speed up the test by 10x

    setup_integration_downwards_dr(trajectory_length, interval)
    log_test(f"[TEST] Start Test Scenario 2", 'test_integration_downwards_dr_1h')
    # %%  1. Create a DR Event
    dr_event = DrEvent(
        duration_sec=int(3600/speedup_factor),  # 3600 / 360 = 10 seconds
        power_alternation=-2.0,
    )

    # %% 2. Create and start the action coordinator in a separate thread
    action_coordinator = ActionCoordinator(
        dr_event,
        action_space,
        participants=[HOUSE_ID_1, HOUSE_ID_2, HOUSE_ID_3],
        control_interval_s=interval,
    )
    action_coordinator.start()  # Start the action coordinator in a separate thread

    # %% 3. Append fake observations, just to emulate feedback from the environment
    test_device_time = trajectory_length * interval
    test_device_time_dr = interval  # Skip the first observation, because we don't have the action coordinator's logs yet.
    while test_device_time_dr <= dr_event.duration_sec:
        while action_coordinator.get_cur_time() < test_device_time_dr:
            time.sleep(interval / 10)  # Waiting for coordinator to catch up
        if (action_coordinator.get_cur_time() - test_device_time_dr) > interval:
            offset = action_coordinator.get_cur_time() - test_device_time_dr
            log_error(
                f"Coordinator time has an offset of {offset} seconds,"
                f" consider increasing the control interval such that this test can keep up with the coordinator.",
                "test_integration_downwards_dr_1h"
            )

        # For the observed power we just use some multiple of the action
        timestamp, response_level, action_dict = read_latest_coordination_action()
        assert len(action_dict) == 3, "Expected 3 actions"
        assert HOUSE_ID_1 in action_dict, "Expected action for house 1"
        assert HOUSE_ID_2 in action_dict, "Expected action for house 2"
        assert HOUSE_ID_3 in action_dict, "Expected action for house 3"
        # Get actions and create fake observations
        add_measurement_to_device(HOUSE_ID_1, (action_dict[HOUSE_ID_1] + 0.43) * 1.5, -1, test_device_time)
        add_measurement_to_device(HOUSE_ID_2, (action_dict[HOUSE_ID_2] + 0.33) * 2.0, -1, test_device_time)
        add_measurement_to_device(HOUSE_ID_3, (action_dict[HOUSE_ID_3] + 0.43) * 3.5, -1, test_device_time)
        test_device_time_dr += interval
    action_coordinator.join()  # Wait for the action coordinator to finish
    assert action_coordinator.finished, "DR event not finished"

    # %% 4. Plot results
    action_coordinator.plot_dr_response()
