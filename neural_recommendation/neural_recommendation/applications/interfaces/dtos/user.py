from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# TODO: move all of them to DTO folder



class UserSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[int] = None


class Message(BaseModel):
    message: str


class UserPublic(BaseModel):
    id: int
    username: str
    email: EmailStr
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class UserList(BaseModel):
    users: list[UserPublic]
