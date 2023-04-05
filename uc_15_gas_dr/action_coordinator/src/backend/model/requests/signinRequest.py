from dataclasses import dataclass


@dataclass
class SigninRequest:
    """ Data fields needed to sign in in the boiler aggregator api
    """
    email: str
    password: str

