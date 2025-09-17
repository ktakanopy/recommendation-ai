from neural_recommendation.applications.interfaces.dtos.user import UserPublic, UserSchema
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class CreateUserUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def execute(self, user_data: UserSchema, include_details: bool = False) -> UserPublic:
        logger.info(f"Creating user: {user_data.name}")

        user = User(
            name=user_data.name,
            age=user_data.age,
            gender=user_data.gender,
            occupation=user_data.occupation,
        )

        created_user = await self.user_repository.create(user)

        if created_user.id is None:
            raise RuntimeError("User creation failed - no ID assigned")

        logger.info(f"User created successfully: {created_user.name}")

        return UserPublic(
            id=created_user.id,
            name=created_user.name,
            age=created_user.age,
            gender=created_user.gender,
            occupation=created_user.occupation,
        )
