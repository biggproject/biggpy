from dataclasses import dataclass


@dataclass
class DevicedataResponse:
    """ Response for api when asking for device data contains boilder and boiler aggregator data
    """
    boiler: str
    boiler_aggregator: str

