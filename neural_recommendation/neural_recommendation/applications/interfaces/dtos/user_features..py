from dataclasses import dataclass


@dataclass
class UserFeaturesDto:
    user_id: int
    timestamp: float
    user_age: float
    gender: int
