from neural_recommendation.applications.interfaces.dtos.message import Message
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository


class DeleteUserUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def execute(self, user_id: int) -> Message:
        existing_user = await self.user_repository.get_by_id(user_id)
        if not existing_user:
            raise ValueError("User not found")

        success = await self.user_repository.delete(user_id)
        if not success:
            raise RuntimeError("Failed to delete user")

        return Message(message="User deleted")
