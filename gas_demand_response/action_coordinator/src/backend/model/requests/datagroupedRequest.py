from dataclasses import dataclass


@dataclass
class DataGrouped:
    """ the format to request data from obelisk through an API call
    """
    metrics_boiler_aggregator: list[str]
    metrics_boiler: list[str]
    start: int
    end: int
    house_id: str
    user_id_obelisk : str = '' # default value ms