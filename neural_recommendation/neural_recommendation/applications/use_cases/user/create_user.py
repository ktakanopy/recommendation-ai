import uuid
from datetime import datetime

from neural_recommendation.applications.interfaces.schemas import UserPublic, UserSchema
from neural_recommendation.domain.models.rating import Rating
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.domain.ports.services.auth_service import AuthService
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class CreateUserUseCase:
    def __init__(self, user_repository: UserRepository, auth_service: AuthService):
        self.user_repository = user_repository
        self.auth_service = auth_service

    async def execute(self, user_data: UserSchema, include_details: bool = False) -> UserPublic:
        """Create a new user with optional ratings and demographics"""
        logger.info(f"Creating user: {user_data.username}")

        # Check if user already exists
        existing_user = await self.user_repository.get_by_username_or_email(user_data.username, user_data.email)

        if existing_user:
            if existing_user.username == user_data.username:
                raise ValueError("Username already exists")
            if existing_user.email == user_data.email:
                raise ValueError("Email already exists")

        # Hash password
        hashed_password = self.auth_service.hash_password(user_data.password)

        # Convert ratings schema to domain models if provided
        ratings = []
        if user_data.ratings:
            logger.info(f"Processing {len(user_data.ratings)} ratings for user")
            for rating_data in user_data.ratings:
                try:
                    movie_uuid = (
                        uuid.UUID(rating_data.movie_id)
                        if not isinstance(rating_data.movie_id, uuid.UUID)
                        else rating_data.movie_id
                    )
                except ValueError:
                    # If movie_id is not a valid UUID, generate one or handle error
                    logger.warning(f"Invalid movie_id format: {rating_data.movie_id}, generating new UUID")
                    movie_uuid = uuid.uuid4()

                rating = Rating(
                    id=uuid.uuid4(),
                    user_id=uuid.uuid4(),  # Temporary, will be updated after user creation
                    movie_id=movie_uuid,
                    rating=rating_data.rating,
                    timestamp=rating_data.timestamp or datetime.now(),
                )
                ratings.append(rating)

        # Create user domain model
        user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password,
            age=user_data.age,
            gender=user_data.gender,
            occupation=user_data.occupation,
            ratings=ratings,
        )

        # Save user
        created_user = await self.user_repository.create(user)

        # Return appropriate DTO
        if created_user.id is None:
            raise RuntimeError("User creation failed - no ID assigned")

        logger.info(f"User created successfully: {created_user.username} with {len(ratings)} ratings")

        # Return UserPublic with optional details
        return UserPublic(
            id=created_user.id,
            username=created_user.username,
            email=created_user.email,
            age=created_user.age if include_details else None,
            gender=created_user.gender if include_details else None,
            occupation=created_user.occupation if include_details else None,
            ratings_count=len(created_user.ratings) if include_details and created_user.ratings else None,
        )
