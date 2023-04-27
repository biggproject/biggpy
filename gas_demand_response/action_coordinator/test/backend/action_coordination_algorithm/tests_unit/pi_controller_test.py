from src.backend.action_coordination_algorithm.pi_controller import PIController
from test.backend.action_coordination_algorithm.model.boiler_pool_simulator import BoilerPoolSimulator
from matplotlib import pyplot as plt


def test_pi_controller():
    boiler_pool = BoilerPoolSimulator()

    # Test the PI controller
    pi_controller = PIController(0.1, 0.01)
    p_target = 5.0  # DR [kW]
    p_observed = [0.0]  # Pool
    duration = 120

    for i in range(duration):
        response_level = pi_controller.step(p_target, p_observed[i])
        p_observed.append(boiler_pool.step(response_level))

    # Create a line plot
    plt.plot(list(range(duration + 1)), p_observed)
    plt.plot(list(range(duration + 1)), [p_target] * (duration + 1))
    plt.show()
