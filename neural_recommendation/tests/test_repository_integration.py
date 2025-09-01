import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.user import User as DomainUser
from neural_recommendation.infrastructure.adapters.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository

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
        domain_user = DomainUser(username="testuser", email="test@example.com", password_hash="hashed_password")

        created_user = await user_repository.create(domain_user)

        assert created_user.id is not None
        assert created_user.username == "testuser"
        assert created_user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, user_repository):
        """Test retrieving user by ID"""
        # Create user first
        domain_user = DomainUser(username="getuser", email="get@example.com", password_hash="password")
        created_user = await user_repository.create(domain_user)

        # Retrieve by ID
        found_user = await user_repository.get_by_id(created_user.id)

        assert found_user is not None
        assert found_user.id == created_user.id
        assert found_user.username == "getuser"

    @pytest.mark.asyncio
    async def test_get_user_by_username(self, user_repository):
        """Test retrieving user by username"""
        domain_user = DomainUser(username="uniqueuser", email="unique@example.com", password_hash="password")
        await user_repository.create(domain_user)

        found_user = await user_repository.get_by_username("uniqueuser")

        assert found_user is not None
        assert found_user.username == "uniqueuser"

    @pytest.mark.asyncio
    async def test_get_user_by_email(self, user_repository):
        """Test retrieving user by email"""
        domain_user = DomainUser(username="emailuser", email="email@example.com", password_hash="password")
        await user_repository.create(domain_user)

        found_user = await user_repository.get_by_email("email@example.com")

        assert found_user is not None
        assert found_user.email == "email@example.com"

    @pytest.mark.asyncio
    async def test_get_all_users(self, user_repository):
        """Test retrieving all users"""
        # Create multiple users
        users_data = [
            ("user1", "user1@example.com"),
            ("user2", "user2@example.com"),
            ("user3", "user3@example.com"),
        ]

        for username, email in users_data:
            domain_user = DomainUser(username=username, email=email, password_hash="password")
            await user_repository.create(domain_user)

        all_users = await user_repository.get_all()

        assert len(all_users) == 3
        usernames = [user.username for user in all_users]
        assert "user1" in usernames
        assert "user2" in usernames
        assert "user3" in usernames

    @pytest.mark.asyncio
    async def test_update_user(self, user_repository):
        """Test updating a user"""
        domain_user = DomainUser(username="updateuser", email="update@example.com", password_hash="password")
        created_user = await user_repository.create(domain_user)

        # Update user
        created_user.email = "updated@example.com"
        updated_user = await user_repository.update(created_user)

        assert updated_user.email == "updated@example.com"
        assert updated_user.id == created_user.id

    @pytest.mark.asyncio
    async def test_delete_user(self, user_repository):
        """Test deleting a user"""
        domain_user = DomainUser(username="deleteuser", email="delete@example.com", password_hash="password")
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

        user = await user_repository.get_by_username("nonexistent")
        assert user is None

        user = await user_repository.get_by_email("nonexistent@example.com")
        assert user is None
