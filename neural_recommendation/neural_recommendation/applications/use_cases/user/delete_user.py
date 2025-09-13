from neural_recommendation.applications.interfaces.dtos.message import Message
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository


class DeleteUserUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def execute(self, user_id: int, current_user: User) -> Message:
        # Check permissions
        if current_user.id != user_id:
            raise PermissionError("Not enough permissions")

        # Check if user exists
        existing_user = await self.user_repository.get_by_id(user_id)
        if not existing_user:
            raise ValueError("User not found")

        # Delete user
        success = await self.user_repository.delete(user_id)
        if not success:
            raise RuntimeError("Failed to delete user")

        return Message(message="User deleted")
