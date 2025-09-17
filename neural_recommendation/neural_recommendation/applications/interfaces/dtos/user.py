from pydantic import BaseModel, ConfigDict


class UserSchema(BaseModel):
    name: str
    age: int
    gender: str
    occupation: int


class Message(BaseModel):
    message: str


class UserPublic(BaseModel):
    id: int
    name: str
    age: int
    gender: str
    occupation: int
    model_config = ConfigDict(from_attributes=True)


class UserList(BaseModel):
    users: list[UserPublic]
