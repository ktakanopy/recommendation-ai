from two_towers.applications.interfaces.schemas import UserPublic, UserSchema
from two_towers.domain.models.user import User
from two_towers.domain.ports.repositories.user_repository import UserRepository
from two_towers.domain.ports.services.auth_service import AuthService


class UpdateUserUseCase:
    def __init__(self, user_repository: UserRepository, auth_service: AuthService):
        self.user_repository = user_repository
        self.auth_service = auth_service

    async def execute(self, user_id: int, user_data: UserSchema, current_user: User) -> UserPublic:
        # Check permissions
        if current_user.id != user_id:
            raise PermissionError("Not enough permissions")

        # Check if user exists
        existing_user = await self.user_repository.get_by_id(user_id)
        if not existing_user:
            raise ValueError("User not found")

        # Hash new password and update user
        hashed_password = self.auth_service.hash_password(user_data.password)
        updated_user = User(
            id=user_id,
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password,
            created_at=existing_user.created_at,
        )

        try:
            saved_user = await self.user_repository.update(updated_user)

            if saved_user.id is None:
                raise RuntimeError("User update failed - no ID assigned")

            return UserPublic(id=saved_user.id, username=saved_user.username, email=saved_user.email)
        except Exception:
            raise ValueError("Username or Email already exists")
