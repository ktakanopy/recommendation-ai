from typing import List

from neural_recommendation.applications.interfaces.schemas import RatingPublic

from neural_recommendation.domain.ports.repositories.rating_repository import RatingRepository


class ListUserRatingsUseCase:
    def __init__(self, rating_repository: RatingRepository):
        self.rating_repository = rating_repository

    async def execute(self, user_id: int) -> List[RatingPublic]:
        ratings = await self.rating_repository.get_by_user_id(user_id)
        return [
            RatingPublic(
                id=r.id,
                user_id=r.user_id,
                movie_id=r.movie_id,
                rating=r.rating,
                timestamp=r.timestamp,
            )
            for r in ratings
        ]
