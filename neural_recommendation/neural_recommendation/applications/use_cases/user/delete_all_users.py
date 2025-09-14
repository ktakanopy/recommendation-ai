from neural_recommendation.applications.interfaces.dtos.message import Message
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository


class DeleteAllUsersUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def execute(self) -> Message:
        deleted = await self.user_repository.delete_all()
        return Message(message=f"Deleted {deleted} users")

