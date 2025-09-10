from http import HTTPStatus
from typing import Annotated, List

from fastapi import APIRouter, Depends

from neural_recommendation.applications.interfaces.schemas import RatingPublic, RatingSchema
from neural_recommendation.applications.use_cases.rating.create_rating import CreateRatingUseCase
from neural_recommendation.applications.use_cases.rating.list_user_ratings import ListUserRatingsUseCase
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.rating_repository import RatingRepository
from neural_recommendation.infrastructure.config.dependencies import get_current_user, get_rating_repository

router = APIRouter(prefix="/ratings", tags=["ratings"])

RatingRepositoryDep = Annotated[RatingRepository, Depends(get_rating_repository)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]


@router.post("/", status_code=HTTPStatus.CREATED, response_model=List[RatingPublic])
async def create_rating(payload: List[RatingSchema], rating_repository: RatingRepositoryDep):
    use_case = CreateRatingUseCase(rating_repository)
    return await use_case.execute(payload)

# TODO: this listing should receive an id from the database not from the logged user
# @router.get("/me", response_model=List[RatingPublic])
# async def list_my_ratings(current_user: CurrentUserDep, rating_repository: RatingRepositoryDep):
#     use_case = ListUserRatingsUseCase(rating_repository)
#     return await use_case.execute(current_user.id)


