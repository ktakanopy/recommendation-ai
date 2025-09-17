from datetime import datetime
from typing import Optional

from pwdlib import PasswordHash

from neural_recommendation.domain.models.user import User as DomainUser
from neural_recommendation.infrastructure.persistence.models import User as SQLUser


class UserFactory:
    """Factory for creating test users"""

    def __init__(self):
        self.pwd_context = PasswordHash.recommended()

    def create_domain_user(
        self,
        *,
        id: Optional[int] = None,
        name: str = "testuser",
        age: int = 20,
        gender: int = 1,
        occupation: int = 1,
        created_at: Optional[datetime] = None,
    ) -> DomainUser:
        """Create a domain user for testing"""
        if created_at is None:
            created_at = datetime(2024, 1, 1)

        return DomainUser(
            id=id,
            name=name,
            age=age,
            gender=gender,
            occupation=occupation,
            created_at=created_at,
        )

    def create_sql_user(
        self,
        *,
        id: Optional[int] = None,
        name: str = "testuser",
        age: int = 20,
        gender: int = 1,
        occupation: int = 1,
        created_at: Optional[datetime] = None,
    ) -> SQLUser:
        """Create a SQL user for testing"""
        user = SQLUser(
            name=name,
            age=age,
            gender=gender,
            occupation=occupation,
        )
        if id is not None:
            user.id = id
        return user

    def create_user_data(
        self, *, name: str = "testuser", age: int = 20, gender: int = 1, occupation: int = 1
    ) -> dict:
        """Create user data dictionary for API tests"""
        return {"name": name, "age": age, "gender": gender, "occupation": occupation}


# Global factory instance
user_factory = UserFactory()
