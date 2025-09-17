from neural_recommendation.applications.interfaces.dtos.user import UserPublic, UserSchema
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository


class UpdateUserUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def execute(self, user_id: int, user_data: UserSchema) -> UserPublic:
        existing_user = await self.user_repository.get_by_id(user_id)
        if not existing_user:
            raise ValueError("User not found")

        updated_user = User(
            id=user_id,
            name=user_data.name,
            age=user_data.age,
            gender=user_data.gender,
            occupation=user_data.occupation,
            created_at=existing_user.created_at,
        )

        saved_user = await self.user_repository.update(updated_user)
        if saved_user.id is None:
            raise RuntimeError("User update failed - no ID assigned")

        return UserPublic(
            id=saved_user.id,
            name=saved_user.name,
            age=saved_user.age,
            gender=saved_user.gender,
            occupation=saved_user.occupation,
        )
