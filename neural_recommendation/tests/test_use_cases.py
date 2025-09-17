import pytest

from neural_recommendation.applications.interfaces.dtos.user import UserPublic, UserSchema
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
    def create_user_use_case(self, mock_user_repository):
        """Create user use case with mocked dependencies"""
        return CreateUserUseCase(mock_user_repository)

    @pytest.mark.asyncio
    async def test_create_user_success(
        self, create_user_use_case, mock_user_repository, user_schema, domain_user
    ):
        """Test successful user creation"""
        # Arrange
        mock_user_repository.create.return_value = domain_user

        # Act
        result = await create_user_use_case.execute(user_schema)

        # Assert
        assert isinstance(result, UserPublic)
        assert result.id == domain_user.id
        assert result.name == domain_user.name
        assert result.age == domain_user.age
        assert result.gender == domain_user.gender
        assert result.occupation == domain_user.occupation

        # Verify interactions
        mock_user_repository.create.assert_called_once()


    @pytest.mark.asyncio
    async def test_create_user_repository_failure(
        self, create_user_use_case, mock_user_repository, user_schema
    ):
        """Test handling of repository creation failure"""
        # Arrange

        # Create a user without an ID to simulate repository failure
        user_without_id = user_factory.create_domain_user(id=None)
        mock_user_repository.create.return_value = user_without_id

        # Act & Assert
        with pytest.raises(RuntimeError, match="User creation failed - no ID assigned"):
            await create_user_use_case.execute(user_schema)


    @pytest.mark.asyncio
    async def test_use_case_isolation(self, mock_user_repository):
        """Test that use case doesn't depend on external infrastructure"""
        # This test verifies that the use case only depends on abstractions (ports)
        # and not on concrete implementations

        # Arrange
        user_schema = UserSchema(**user_factory.create_user_data())
        domain_user = user_factory.create_domain_user(id=1)
        use_case = CreateUserUseCase(mock_user_repository)

        mock_user_repository.create.return_value = domain_user

        result = await use_case.execute(user_schema)

        # Assert - Use case should work with any implementation of the ports
        assert isinstance(result, UserPublic)
        assert result.name == user_schema.name
        assert result.age == user_schema.age
        assert result.gender == user_schema.gender
        assert result.occupation == user_schema.occupation

        # Verify that the use case called the abstract methods
        assert mock_user_repository.create.called