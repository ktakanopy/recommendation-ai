from pydantic import BaseModel, ConfigDict, EmailStr, Field

# presentation layer dtos


class FilterPage(BaseModel):
    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, gt=0, le=1000, description="Maximum number of items to return")


class Token(BaseModel):
    access_token: str
    token_type: str


class UserSchema(BaseModel):
    username: str
    email: EmailStr
    password: str


class Message(BaseModel):
    message: str


class UserPublic(BaseModel):
    id: int
    username: str
    email: EmailStr
    model_config = ConfigDict(from_attributes=True)


class UserDB(UserSchema):
    id: int


class UserList(BaseModel):
    users: list[UserPublic]
