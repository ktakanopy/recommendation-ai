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
from neural_recommendation.applications.use_cases.user.delete_user import DeleteUserUseCase
from neural_recommendation.applications.use_cases.user.get_users import GetUsersUseCase
from neural_recommendation.applications.use_cases.user.update_user import UpdateUserUseCase
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.domain.ports.services.auth_service import AuthService
from neural_recommendation.infrastructure.config.dependencies import (
    get_auth_service,
    get_current_user,
    get_user_repository,
)

router = APIRouter(prefix="/users", tags=["users"])

UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]


@router.post("/", status_code=HTTPStatus.CREATED, response_model=UserPublic)
async def create_user(user: UserSchema, user_repository: UserRepositoryDep, auth_service: AuthServiceDep):
    try:
        use_case = CreateUserUseCase(user_repository, auth_service)
        return await use_case.execute(user)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=str(e))


@router.get("/me", response_model=UserPublic)
async def read_current_user(current_user: CurrentUserDep):
    if current_user.id is None:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="User ID is not set")
    return UserPublic(id=current_user.id, username=current_user.username, email=current_user.email)


@router.get("/", response_model=UserList)
async def read_users(filter_users: Annotated[FilterPage, Query()], user_repository: UserRepositoryDep):
    use_case = GetUsersUseCase(user_repository)
    return await use_case.execute(filter_users)


@router.put("/{user_id}", response_model=UserPublic)
async def update_user(
    user_id: int,
    user: UserSchema,
    current_user: CurrentUserDep,
    user_repository: UserRepositoryDep,
    auth_service: AuthServiceDep,
):
    try:
        use_case = UpdateUserUseCase(user_repository, auth_service)
        return await use_case.execute(user_id, user, current_user)
    except PermissionError:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Not enough permissions")
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
        else:
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=str(e))


@router.delete("/{user_id}", response_model=Message)
async def delete_user(user_id: int, current_user: CurrentUserDep, user_repository: UserRepositoryDep):
    try:
        use_case = DeleteUserUseCase(user_repository)
        return await use_case.execute(user_id, current_user)
    except PermissionError:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Not enough permissions")
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
