from abc import ABC, abstractmethod
from typing import Optional

from two_towers.domain.models.user import User


class AuthService(ABC):
    @abstractmethod
    def hash_password(self, password: str) -> str:
        pass

    @abstractmethod
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        pass

    @abstractmethod
    def create_access_token(self, user: User) -> str:
        pass

    @abstractmethod
    async def get_current_user(self, token: str) -> Optional[User]:
        pass
