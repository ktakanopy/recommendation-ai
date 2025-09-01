from neural_recommendation.applications.interfaces.schemas import UserPublic, UserSchema
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.domain.ports.services.auth_service import AuthService


class CreateUserUseCase:
    def __init__(self, user_repository: UserRepository, auth_service: AuthService):
        self.user_repository = user_repository
        self.auth_service = auth_service

    async def execute(self, user_data: UserSchema) -> UserPublic:
        # Check if user already exists
        existing_user = await self.user_repository.get_by_username_or_email(user_data.username, user_data.email)

        if existing_user:
            if existing_user.username == user_data.username:
                raise ValueError("Username already exists")
            if existing_user.email == user_data.email:
                raise ValueError("Email already exists")

        # Hash password and create user
        hashed_password = self.auth_service.hash_password(user_data.password)
        user = User(username=user_data.username, email=user_data.email, password_hash=hashed_password)

        # Save user
        created_user = await self.user_repository.create(user)

        # Return as DTO
        if created_user.id is None:
            raise RuntimeError("User creation failed - no ID assigned")

        return UserPublic(id=created_user.id, username=created_user.username, email=created_user.email)
