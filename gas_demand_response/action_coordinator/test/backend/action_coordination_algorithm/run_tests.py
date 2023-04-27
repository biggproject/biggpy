from test.backend.action_coordination_algorithm.tests_integration.trial_scenario_1 import test_trial_scenario_1
from test.backend.action_coordination_algorithm.tests_integration.dr_event_upwards import \
    test_trigger_real_dr_event
from test.backend.action_coordination_algorithm.tests_unit.q_function_inference import test_q_functions

from test.backend.action_coordination_algorithm.tests_unit.a_function_inference import test_a_functions
from test.backend.action_coordination_algorithm.tests_unit.pi_controller_test import test_pi_controller
from test.backend.action_coordination_algorithm.tests_integration.dr_event_upwards_15_min import \
    test_integration_upwards_dr_15_min
from test.backend.action_coordination_algorithm.tests_unit.action_scheduler import test_increase_actions_scenario_1, \
    test_increase_actions_scenario_2, test_decrease_actions_scenario
from test.backend.action_coordination_algorithm.tests_integration.dr_event_upwards_1h import \
    test_integration_upwards_dr_1h
from test.backend.action_coordination_algorithm.tests_integration.dr_event_downwards_1h import \
    test_integration_downwards_dr_1h
from test.backend.action_coordination_algorithm.tests_unit.action_filter import \
    test_filter_unreachable_actions_scenario_2, test_filter_unreachable_actions_scenario_1, \
    test_filter_unreachable_actions_scenario_3, test_filter_unreachable_actions_scenario_4

# %%  Unit tests
# Filter
test_filter_unreachable_actions_scenario_1()
test_filter_unreachable_actions_scenario_2()
test_filter_unreachable_actions_scenario_3()
test_filter_unreachable_actions_scenario_4()

# Action scheduler
test_increase_actions_scenario_1()
test_increase_actions_scenario_2()
test_decrease_actions_scenario()

# Pi controller
test_pi_controller()

# Q-loader
test_q_functions()
test_a_functions()

# %%  Integration tests
test_integration_upwards_dr_1h()  # 3 houses
test_integration_downwards_dr_1h()  # 3 houses
test_integration_upwards_dr_15_min()  # 5 houses
test_trial_scenario_1()  # 'realistic' test scenario
test_trigger_real_dr_event()  # CAUTION: May trigger real DR responses!
