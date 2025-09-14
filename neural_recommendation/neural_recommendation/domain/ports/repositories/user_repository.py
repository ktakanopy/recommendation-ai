from abc import ABC, abstractmethod
from typing import List, Optional

from neural_recommendation.domain.models.rating import Rating
from neural_recommendation.domain.models.user import User


class UserRepository(ABC):
    @abstractmethod
    async def create(self, user: User) -> User:
        pass

    @abstractmethod
    async def get_by_id(self, user_id: int) -> Optional[User]:
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        pass

    @abstractmethod
    async def get_user_ratings(self, user_id: int) -> List[Rating]:
        pass

    @abstractmethod
    async def get_by_username_or_email(self, username: str, email: str) -> Optional[User]:
        pass

    @abstractmethod
    async def get_all(self, offset: int = 0, limit: int = 100) -> List[User]:
        pass

    @abstractmethod
    async def update(self, user: User) -> User:
        pass

    @abstractmethod
    async def delete(self, user_id: int) -> bool:
        pass

    @abstractmethod
    async def delete_all(self) -> int:
        pass
