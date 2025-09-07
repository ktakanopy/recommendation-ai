import uuid
from datetime import datetime

from neural_recommendation.applications.interfaces.schemas import RatingPublic, RatingSchema
from neural_recommendation.domain.models.rating import Rating as DomainRating
from neural_recommendation.domain.ports.repositories.rating_repository import RatingRepository


class CreateRatingUseCase:
    def __init__(self, rating_repository: RatingRepository):
        self.rating_repository = rating_repository

    async def execute(self, user_id: int, rating: RatingSchema) -> RatingPublic:
        domain = DomainRating(
            id=uuid.uuid4(),
            user_id=user_id,
            movie_id=rating.movie_id,
            rating=rating.rating,
            timestamp=rating.timestamp or datetime.now(),
        )
        await self.rating_repository.bulk_create(user_id, [domain])
        return RatingPublic(
            id=domain.id,
            user_id=user_id,
            movie_id=domain.movie_id,
            rating=domain.rating,
            timestamp=domain.timestamp,
        )


