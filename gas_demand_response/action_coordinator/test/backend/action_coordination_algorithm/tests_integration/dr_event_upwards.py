import time
from threading import Event

from backend.business_as_usual_controller.BusinessAsUsualController import BusinessAsUsualController
from backend.main import senddrevent
from src.backend.service.action_service import ActionService
from src.misc.logger import log_test


def test_trigger_real_dr_event():
    """ Triggers a DR event"""
    log_test(f"Starting REAL WORLD Integration Tests")


    # ======
    dr_event_signal = Event()
    bau_controller = BusinessAsUsualController(dr_event_signal)
    bau_controller.start()
    # ======
    # Wait for 60 seconds
    time.sleep(15)
    # ======
    # Signal BaU controller to stop
    dr_event_signal.set()
    while bau_controller.stopped:
        # wait for BaU controller to stop
        time.sleep(0.1)
    log_test("BaU controller stopped, now starting DR-event")
    # Start Action Coordinator controller
    energy = 10
    seconds = 120
    action = ActionService()
    dr_status = action.post_dr_event(seconds, energy, dr_event_signal)

