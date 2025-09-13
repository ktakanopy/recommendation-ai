from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from neural_recommendation.applications.interfaces.dtos.filter_page import FilterPage
from neural_recommendation.applications.interfaces.dtos.user import UserList
from neural_recommendation.applications.use_cases.user.get_users import GetUsersUseCase
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository


class TestGetUsersUseCase:
    @pytest.fixture
    def filter_page(self):
        """Default filter page parameters"""
        return FilterPage(offset=0, limit=10)

    @pytest.fixture
    def sample_users(self):
        """Sample list of domain users"""
        return [
            User(
                id=1,
                username="alice",
                email="alice@example.com",
                password_hash="hash1",
                created_at=datetime(2024, 1, 1),
            ),
            User(id=2, username="bob", email="bob@example.com", password_hash="hash2", created_at=datetime(2024, 1, 2)),
            User(
                id=3,
                username="charlie",
                email="charlie@example.com",
                password_hash="hash3",
                created_at=datetime(2024, 1, 3),
            ),
        ]

    @pytest.fixture
    def get_users_use_case(self, mock_user_repository):
        """Get users use case with mocked repository"""
        return GetUsersUseCase(mock_user_repository)

    @pytest.mark.asyncio
    async def test_get_users_success(self, get_users_use_case, mock_user_repository, filter_page, sample_users):
        """Test successful retrieval of users"""
        # Arrange
        mock_user_repository.get_all.return_value = sample_users

        # Act
        result = await get_users_use_case.execute(filter_page)

        # Assert
        assert isinstance(result, UserList)
        assert len(result.users) == 3

        # Verify user data is correctly mapped to UserPublic
        assert result.users[0].id == 1
        assert result.users[0].username == "alice"
        assert result.users[0].email == "alice@example.com"

        assert result.users[1].id == 2
        assert result.users[1].username == "bob"
        assert result.users[1].email == "bob@example.com"

        # Verify repository was called with correct parameters
        mock_user_repository.get_all.assert_called_once_with(offset=0, limit=10)

    @pytest.mark.asyncio
    async def test_get_users_empty_result(self, get_users_use_case, mock_user_repository, filter_page):
        """Test retrieval when no users exist"""
        # Arrange
        mock_user_repository.get_all.return_value = []

        # Act
        result = await get_users_use_case.execute(filter_page)

        # Assert
        assert isinstance(result, UserList)
        assert len(result.users) == 0
        assert result.users == []

        # Verify repository was called
        mock_user_repository.get_all.assert_called_once_with(offset=0, limit=10)

    @pytest.mark.asyncio
    async def test_get_users_with_pagination(self, get_users_use_case, mock_user_repository, sample_users):
        """Test retrieval with custom pagination parameters"""
        # Arrange
        custom_filter = FilterPage(offset=5, limit=20)
        mock_user_repository.get_all.return_value = sample_users

        # Act
        result = await get_users_use_case.execute(custom_filter)

        # Assert
        assert isinstance(result, UserList)
        assert len(result.users) == 3

        # Verify repository was called with custom pagination
        mock_user_repository.get_all.assert_called_once_with(offset=5, limit=20)

    @pytest.mark.asyncio
    async def test_get_users_filters_none_ids(self, get_users_use_case, mock_user_repository, filter_page):
        """Test that users with None IDs are filtered out"""
        # Arrange
        users_with_none_id = [
            User(
                id=1,
                username="alice",
                email="alice@example.com",
                password_hash="hash1",
                created_at=datetime(2024, 1, 1),
            ),
            User(
                id=None,  # This should be filtered out
                username="bob",
                email="bob@example.com",
                password_hash="hash2",
                created_at=datetime(2024, 1, 2),
            ),
            User(
                id=3,
                username="charlie",
                email="charlie@example.com",
                password_hash="hash3",
                created_at=datetime(2024, 1, 3),
            ),
        ]
        mock_user_repository.get_all.return_value = users_with_none_id

        # Act
        result = await get_users_use_case.execute(filter_page)

        # Assert
        assert isinstance(result, UserList)
        assert len(result.users) == 2  # Only users with valid IDs

        # Verify that only users with IDs are included
        user_ids = [user.id for user in result.users]
        assert 1 in user_ids
        assert 3 in user_ids
        assert None not in user_ids

    @pytest.mark.asyncio
    async def test_get_users_repository_exception(self, get_users_use_case, mock_user_repository, filter_page):
        """Test handling of repository exceptions"""
        # Arrange
        mock_user_repository.get_all.side_effect = Exception("Database connection error")

        # Act & Assert
        with pytest.raises(Exception, match="Database connection error"):
            await get_users_use_case.execute(filter_page)

    @pytest.mark.asyncio
    async def test_get_users_use_case_isolation(self, mock_user_repository, sample_users):
        """Test that use case only depends on repository abstraction"""
        # Arrange
        filter_page = FilterPage(offset=0, limit=5)
        use_case = GetUsersUseCase(mock_user_repository)
        mock_user_repository.get_all.return_value = sample_users

        # Act
        result = await use_case.execute(filter_page)

        # Assert - Use case should work with any repository implementation
        assert isinstance(result, UserList)
        assert len(result.users) == 3

        # Verify the use case only calls repository methods
        mock_user_repository.get_all.assert_called_once_with(offset=0, limit=5)

    @pytest.mark.asyncio
    async def test_get_users_different_page_sizes(self, get_users_use_case, mock_user_repository):
        """Test use case with different page sizes"""
        test_cases = [
            (FilterPage(offset=0, limit=1), "Single item page"),
            (FilterPage(offset=10, limit=50), "Large page"),
            (FilterPage(offset=0, limit=100), "Maximum page size"),
        ]

        for filter_page, description in test_cases:
            # Arrange
            mock_user_repository.reset_mock()
            mock_user_repository.get_all.return_value = []

            # Act
            result = await get_users_use_case.execute(filter_page)

            # Assert
            assert isinstance(result, UserList), f"Failed for {description}"
            (
                mock_user_repository.get_all.assert_called_once_with(
                    offset=filter_page.offset, limit=filter_page.limit
                ),
                f"Failed for {description}",
            )


class TestGetUsersUseCaseIntegration:
    """Integration-style tests that verify the complete flow"""

    @pytest.mark.asyncio
    async def test_complete_user_list_mapping(self):
        """Test complete mapping from domain users to UserPublic DTOs"""
        # Arrange
        mock_repository = AsyncMock(spec=UserRepository)
        use_case = GetUsersUseCase(mock_repository)

        domain_users = [
            User(
                id=100,
                username="testuser",
                email="test@domain.com",
                password_hash="sensitive_hash_should_not_be_exposed",
                created_at=datetime(2024, 3, 15, 10, 30),
            )
        ]
        mock_repository.get_all.return_value = domain_users

        # Act
        result = await use_case.execute(FilterPage(offset=0, limit=10))

        # Assert
        assert len(result.users) == 1
        user_public = result.users[0]

        # Verify UserPublic only contains safe fields
        assert hasattr(user_public, "id")
        assert hasattr(user_public, "username")
        assert hasattr(user_public, "email")
        assert not hasattr(user_public, "password_hash")  # Should not expose password
        assert not hasattr(user_public, "created_at")  # Not included in UserPublic

        # Verify values are correctly mapped
        assert user_public.id == 100
        assert user_public.username == "testuser"
        assert user_public.email == "test@domain.com"
