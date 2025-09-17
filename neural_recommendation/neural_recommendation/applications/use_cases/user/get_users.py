from neural_recommendation.applications.interfaces.dtos.filter_page import FilterPage
from neural_recommendation.applications.interfaces.dtos.user import UserList, UserPublic
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository


class GetUsersUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def execute(self, filter_page: FilterPage) -> UserList:
        users = await self.user_repository.get_all(offset=filter_page.offset, limit=filter_page.limit)

        user_publics = []
        for user in users:
            if user.id is not None:
                user_publics.append(
                    UserPublic(
                        id=user.id,
                        name=user.name,
                        age=user.age,
                        gender=user.gender,
                        occupation=user.occupation,
                    )
                )

        return UserList(users=user_publics)
