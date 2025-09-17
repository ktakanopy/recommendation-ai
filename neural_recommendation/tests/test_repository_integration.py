import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.user import User as DomainUser
from neural_recommendation.infrastructure.adapters.repositories.sqlalchemy_user_repository import (
    SQLAlchemyUserRepository,
)

from .conftest import BaseIntegrationTest


class TestSQLAlchemyUserRepository(BaseIntegrationTest):
    """Integration tests for SQLAlchemy user repository"""

    @pytest_asyncio.fixture
    async def postgres_session(self, postgres_engine):
        """Create a test session"""
        async with AsyncSession(postgres_engine, expire_on_commit=False) as session:
            yield session

    @pytest.fixture
    def user_repository(self, postgres_session):
        """Create repository instance"""
        return SQLAlchemyUserRepository(postgres_session)

    @pytest.mark.asyncio
    async def test_create_user(self, user_repository):
        """Test creating a user through repository"""
        domain_user = DomainUser(name="testuser", age=20, gender=1, occupation=1)

        created_user = await user_repository.create(domain_user)

        assert created_user.id is not None
        assert created_user.name == "testuser"
        assert created_user.age == 20
        assert created_user.gender == 1
        assert created_user.occupation == 1

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, user_repository):
        """Test retrieving user by ID"""
        # Create user first
        domain_user = DomainUser(name="getuser", age=20, gender=1, occupation=1)
        created_user = await user_repository.create(domain_user)

        # Retrieve by ID
        found_user = await user_repository.get_by_id(created_user.id)

        assert found_user is not None
        assert found_user.id == created_user.id
        assert found_user.name == "getuser"

    @pytest.mark.asyncio
    async def test_get_all_users(self, user_repository):
        """Test retrieving all users"""
        # Create multiple users
        users_data = [
            ("user1"),
            ("user2"),
            ("user3"),
        ]

        for name in users_data:
            domain_user = DomainUser(name=name, age=20, gender=1, occupation=1)
            await user_repository.create(domain_user)

        all_users = await user_repository.get_all()

        assert len(all_users) == 3
        names = [user.name for user in all_users]
        assert "user1" in names
        assert "user2" in names
        assert "user3" in names

    @pytest.mark.asyncio
    async def test_update_user(self, user_repository):
        """Test updating a user"""
        domain_user = DomainUser(name="updateuser", age=20, gender=1, occupation=1)
        created_user = await user_repository.create(domain_user)

        # Update user
        created_user.name = "updateduser"
        updated_user = await user_repository.update(created_user)

        assert updated_user.name == "updateduser"
        assert updated_user.id == created_user.id

    @pytest.mark.asyncio
    async def test_delete_user(self, user_repository):
        """Test deleting a user"""
        domain_user = DomainUser(name="deleteuser", age=20, gender=1, occupation=1)
        created_user = await user_repository.create(domain_user)
        user_id = created_user.id

        # Delete user
        await user_repository.delete(user_id)

        # Verify user is deleted
        deleted_user = await user_repository.get_by_id(user_id)
        assert deleted_user is None

    @pytest.mark.asyncio
    async def test_user_not_found(self, user_repository):
        """Test handling of non-existent users"""
        user = await user_repository.get_by_id(999)
        assert user is None
