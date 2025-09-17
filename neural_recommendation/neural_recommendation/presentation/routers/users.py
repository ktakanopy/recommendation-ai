from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from neural_recommendation.applications.interfaces.dtos.filter_page import FilterPage
from neural_recommendation.applications.interfaces.dtos.message import Message
from neural_recommendation.applications.interfaces.dtos.user import (
    UserList,
    UserPublic,
    UserSchema,
)
from neural_recommendation.applications.use_cases.user.create_user import CreateUserUseCase
from neural_recommendation.applications.use_cases.user.delete_all_users import DeleteAllUsersUseCase
from neural_recommendation.applications.use_cases.user.delete_user import DeleteUserUseCase
from neural_recommendation.applications.use_cases.user.get_users import GetUsersUseCase
from neural_recommendation.applications.use_cases.user.update_user import UpdateUserUseCase
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.infrastructure.config.dependencies import get_user_repository

router = APIRouter(prefix="/users", tags=["users"])

UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]


@router.post("/", status_code=HTTPStatus.CREATED, response_model=UserPublic)
async def create_user(user: UserSchema, user_repository: UserRepositoryDep):
    try:
        use_case = CreateUserUseCase(user_repository)
        return await use_case.execute(user)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=str(e))


@router.get("/", response_model=UserList)
async def read_users(filter_users: Annotated[FilterPage, Query()], user_repository: UserRepositoryDep):
    use_case = GetUsersUseCase(user_repository)
    return await use_case.execute(filter_users)


@router.put("/{user_id}", response_model=UserPublic)
async def update_user(user_id: int, user: UserSchema, user_repository: UserRepositoryDep):
    try:
        use_case = UpdateUserUseCase(user_repository)
        return await use_case.execute(user_id, user)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
        else:
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=str(e))


@router.delete("/{user_id}", response_model=Message)
async def delete_user(user_id: int, user_repository: UserRepositoryDep):
    try:
        use_case = DeleteUserUseCase(user_repository)
        return await use_case.execute(user_id)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))


@router.delete("/", response_model=Message)
async def delete_all_users(user_repository: UserRepositoryDep):
    use_case = DeleteAllUsersUseCase(user_repository)
    return await use_case.execute()
