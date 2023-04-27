from src.backend.model.responses.boileraggregatorresponse import BoilerAggregatorResponse
import re


def map_json_boiler(data) -> BoilerAggregatorResponse:
    try: 
        d = BoilerAggregatorResponse(**data)
        return d

    except TypeError as e:
        result = re.split(r'\'', str(e))
        data.pop(result[1])
        return map_json_boiler(data)

