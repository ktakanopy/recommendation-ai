import pytest

from neural_recommendation.applications.interfaces.schemas import UserPublic, UserSchema
from neural_recommendation.applications.use_cases.user.create_user import CreateUserUseCase

from .factories import user_factory


class TestCreateUserUseCase:
    @pytest.fixture
    def user_schema(self):
        """Sample user input data"""
        return UserSchema(**user_factory.create_user_data())

    @pytest.fixture
    def domain_user(self):
        """Sample domain user"""
        return user_factory.create_domain_user(id=1)

    @pytest.fixture
    def create_user_use_case(self, mock_user_repository, mock_auth_service):
        """Create user use case with mocked dependencies"""
        return CreateUserUseCase(mock_user_repository, mock_auth_service)

    @pytest.mark.asyncio
    async def test_create_user_success(
        self, create_user_use_case, mock_user_repository, mock_auth_service, user_schema, domain_user
    ):
        """Test successful user creation"""
        # Arrange
        mock_user_repository.get_by_username_or_email.return_value = None  # No existing user
        mock_auth_service.hash_password.return_value = "hashed_password"
        mock_user_repository.create.return_value = domain_user

        # Act
        result = await create_user_use_case.execute(user_schema)

        # Assert
        assert isinstance(result, UserPublic)
        assert result.id == domain_user.id
        assert result.username == domain_user.username
        assert result.email == domain_user.email

        # Verify interactions
        mock_user_repository.get_by_username_or_email.assert_called_once_with(domain_user.username, domain_user.email)
        mock_auth_service.hash_password.assert_called_once_with(user_schema.password)
        mock_user_repository.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_username_already_exists(
        self, create_user_use_case, mock_user_repository, mock_auth_service, user_schema
    ):
        """Test user creation fails when username already exists"""
        # Arrange
        existing_user = user_factory.create_domain_user(id=1, email="different@example.com")
        mock_user_repository.get_by_username_or_email.return_value = existing_user

        # Act & Assert
        with pytest.raises(ValueError, match="Username already exists"):
            await create_user_use_case.execute(user_schema)

        # Verify no password hashing or user creation occurred
        mock_auth_service.hash_password.assert_not_called()
        mock_user_repository.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_user_email_already_exists(
        self, create_user_use_case, mock_user_repository, mock_auth_service, user_schema
    ):
        """Test user creation fails when email already exists"""
        # Arrange
        existing_user = user_factory.create_domain_user(id=1, username="differentuser")
        mock_user_repository.get_by_username_or_email.return_value = existing_user

        # Act & Assert
        with pytest.raises(ValueError, match="Email already exists"):
            await create_user_use_case.execute(user_schema)

        # Verify no password hashing or user creation occurred
        mock_auth_service.hash_password.assert_not_called()
        mock_user_repository.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_user_repository_failure(
        self, create_user_use_case, mock_user_repository, mock_auth_service, user_schema
    ):
        """Test handling of repository creation failure"""
        # Arrange
        mock_user_repository.get_by_username_or_email.return_value = None
        mock_auth_service.hash_password.return_value = "hashed_password"

        # Create a user without an ID to simulate repository failure
        user_without_id = user_factory.create_domain_user(id=None)
        mock_user_repository.create.return_value = user_without_id

        # Act & Assert
        with pytest.raises(RuntimeError, match="User creation failed - no ID assigned"):
            await create_user_use_case.execute(user_schema)

    @pytest.mark.asyncio
    async def test_create_user_with_different_input_data(
        self, create_user_use_case, mock_user_repository, mock_auth_service
    ):
        """Test user creation with different input data"""
        # Arrange
        user_data = user_factory.create_user_data(username="alice", email="alice@company.com", password="securepass456")
        user_schema = UserSchema(**user_data)

        created_user = user_factory.create_domain_user(
            id=42, username="alice", email="alice@company.com", password="securepass456"
        )

        mock_user_repository.get_by_username_or_email.return_value = None
        mock_auth_service.hash_password.return_value = "hashed_securepass456"
        mock_user_repository.create.return_value = created_user

        # Act
        result = await create_user_use_case.execute(user_schema)

        # Assert
        assert result.id == created_user.id
        assert result.username == created_user.username
        assert result.email == created_user.email

        # Verify password was hashed correctly
        mock_auth_service.hash_password.assert_called_once_with(user_schema.password)

    @pytest.mark.asyncio
    async def test_use_case_isolation(self, mock_user_repository, mock_auth_service):
        """Test that use case doesn't depend on external infrastructure"""
        # This test verifies that the use case only depends on abstractions (ports)
        # and not on concrete implementations

        # Arrange
        user_schema = UserSchema(**user_factory.create_user_data())
        domain_user = user_factory.create_domain_user(id=1)
        use_case = CreateUserUseCase(mock_user_repository, mock_auth_service)

        mock_user_repository.get_by_username_or_email.return_value = None
        mock_auth_service.hash_password.return_value = "hashed_password"
        mock_user_repository.create.return_value = domain_user

        # Act
        result = await use_case.execute(user_schema)

        # Assert - Use case should work with any implementation of the ports
        assert isinstance(result, UserPublic)
        assert result.username == user_schema.username
        assert result.email == user_schema.email

        # Verify that the use case called the abstract methods
        assert mock_user_repository.get_by_username_or_email.called
        assert mock_auth_service.hash_password.called
        assert mock_user_repository.create.called


class TestCreateUserUseCaseEdgeCases:
    """Additional edge case tests for CreateUserUseCase"""

    @pytest.mark.asyncio
    async def test_empty_username_validation(self):
        """Test behavior with edge case inputs - should be handled by Pydantic"""
        # Note: In a real application, input validation would be handled by Pydantic
        # at the schema level, but we can test the use case behavior
        pass

    @pytest.mark.asyncio
    async def test_concurrent_user_creation_scenario(self, mock_user_repository, mock_auth_service):
        """Test scenario where user is created between check and creation"""
        # This simulates a race condition scenario
        user_schema = UserSchema(**user_factory.create_user_data())
        use_case = CreateUserUseCase(mock_user_repository, mock_auth_service)

        # First call returns None (no user exists)
        # Second call (during create) could throw an exception due to unique constraint
        mock_user_repository.get_by_username_or_email.return_value = None
        mock_auth_service.hash_password.return_value = "hashed_password"
        mock_user_repository.create.side_effect = Exception("Unique constraint violation")

        # Act & Assert
        with pytest.raises(Exception, match="Unique constraint violation"):
            await use_case.execute(user_schema)
