from neural_recommendation.applications.interfaces.dtos.user import UserPublic, UserSchema
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
        logger.info(f"Creating user: {user_data.username}")

        existing_user = await self.user_repository.get_by_username_or_email(user_data.username, user_data.email)

        if existing_user:
            if existing_user.username == user_data.username:
                raise ValueError("Username already exists")
            if existing_user.email == user_data.email:
                raise ValueError("Email already exists")

        hashed_password = self.auth_service.hash_password(user_data.password)

        user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password,
            age=user_data.age,
            gender=user_data.gender,
            occupation=user_data.occupation,
        )

        created_user = await self.user_repository.create(user)

        if created_user.id is None:
            raise RuntimeError("User creation failed - no ID assigned")

        logger.info(f"User created successfully: {created_user.username}")

        return UserPublic(
            id=created_user.id,
            username=created_user.username,
            email=created_user.email,
            age=created_user.age,
            gender=created_user.gender,
            occupation=created_user.occupation,
        )
