from pydantic import BaseModel


class UserFeaturesDto(BaseModel):
    user_id: int
    timestamp: float
    user_age: float
    gender: int
