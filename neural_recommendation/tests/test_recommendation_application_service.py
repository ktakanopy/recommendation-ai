import uuid
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from neural_recommendation.applications.services.recommendation_application_service import (
    RecommendationApplicationService,
)
from neural_recommendation.domain.models.deep_learning.recommendation import Recommendation, RecommendationResult
from neural_recommendation.domain.models.rating import Rating
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository
from neural_recommendation.domain.ports.repositories.rating_repository import RatingRepository
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.domain.services.recommendation_service import RecommendationService
from neural_recommendation.infrastructure.config.settings import MLModelSettings


class TestRecommendationApplicationService:
    """Unit tests for RecommendationApplicationService"""

    @pytest.fixture
    def mock_model_repository(self):
        """Create mock model inference repository"""
        repository = Mock(spec=ModelInferenceRepository)

        mock_model = Mock()
        mock_model.parameters = Mock(return_value=iter([1]))
        mock_model.eval = Mock()
        mock_model.predict_batch = Mock()

        mock_feature_info = Mock()
        mock_feature_info.sentence_embeddings = Mock()
        mock_feature_info.sentence_embeddings.title_to_idx = {"Movie A": 1, "Movie B": 2}

        repository.load_model_and_features = Mock(return_value=(mock_model, mock_feature_info))
        return repository

    @pytest.fixture
    def mock_user_repository(self):
        """Create mock user repository"""
        repository = AsyncMock(spec=UserRepository)
        return repository

    @pytest.fixture
    def mock_rating_repository(self):
        repository = AsyncMock(spec=RatingRepository)
        repository.get_by_user_id.return_value = []
        return repository

    @pytest.fixture
    def ml_settings(self):
        return Mock(spec=MLModelSettings)

    @pytest.fixture
    def mock_domain_service(self):
        """Create mock domain service"""
        service = Mock(spec=RecommendationService)

        service.generate_recommendations_cold_start.return_value = RecommendationResult(
            user_id="cold_start_user",
            recommendations=[
                Recommendation(movie_id=3, title="Cold Start Movie", similarity_score=0.7, genres="Drama"),
            ],
            total_available_movies=50,
        )

        return service

    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing"""
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            age=25,
            gender="M",
            occupation=1,
            ratings=[
                Rating(
                    id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=uuid.uuid4(), rating=4.5, timestamp=datetime.now()
                )
            ],
        )

    @pytest.fixture
    def app_service(self, ml_settings, mock_model_repository, mock_user_repository, mock_rating_repository):
        """Create RecommendationApplicationService instance"""
        return RecommendationApplicationService(ml_settings, mock_model_repository, mock_user_repository, mock_rating_repository)

    @pytest.mark.asyncio
    async def test_generate_recommendations_cold_start_success(
        self, app_service, mock_user_repository, mock_domain_service, sample_user
    ):
        """Test successful cold start recommendation generation"""
        # Setup
        app_service._domain_service = mock_domain_service
        user_id = 1
        num_recommendations = 3

        mock_user_repository.get_by_id.return_value = sample_user

        # Execute
        result = await app_service.generate_recommendations_cold_start(
            user_id=user_id, num_recommendations=num_recommendations
        )

        # Assert
        assert isinstance(result, RecommendationResult)
        mock_user_repository.get_by_id.assert_called_once_with(user_id)
        mock_domain_service.generate_recommendations_cold_start.assert_called_once_with(
            user=sample_user, num_recommendations=num_recommendations
        )

    @pytest.mark.asyncio
    async def test_generate_recommendations_cold_start_user_not_found(self, app_service, mock_user_repository):
        """Test cold start recommendation when user is not found"""
        # Setup
        user_id = 999
        mock_user_repository.get_by_id.return_value = None

        # Execute & Assert
        with pytest.raises(ValueError, match="User with ID 999 not found"):
            await app_service.generate_recommendations_cold_start(user_id=user_id)

        mock_user_repository.get_by_id.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_generate_recommendations_cold_start_default_parameters(
        self, app_service, mock_user_repository, mock_domain_service, sample_user
    ):
        """Test cold start recommendations with default parameters"""
        # Setup
        app_service._domain_service = mock_domain_service
        user_id = 1
        mock_user_repository.get_by_id.return_value = sample_user

        # Execute
        result = await app_service.generate_recommendations_cold_start(user_id=user_id)

        # Assert
        mock_domain_service.generate_recommendations_cold_start.assert_called_once_with(
            user=sample_user,
            num_recommendations=10,  # Default value
        )


    @pytest.mark.asyncio
    async def test_cold_start_with_user_without_ratings(self, app_service, mock_user_repository, mock_domain_service):
        """Test cold start recommendations for user without ratings"""
        # Setup
        user_without_ratings = User(
            id=2,
            username="noratinguser",
            email="norating@example.com",
            password_hash="hashed",
            age=22,
            gender="F",
            occupation=2,
            ratings=None,
        )

        app_service._domain_service = mock_domain_service
        mock_user_repository.get_by_id.return_value = user_without_ratings

        # Execute
        result = await app_service.generate_recommendations_cold_start(user_id=2)

        # Assert
        assert isinstance(result, RecommendationResult)
        mock_domain_service.generate_recommendations_cold_start.assert_called_once_with(
            user=user_without_ratings, num_recommendations=10
        )


# Import asyncio at the end to avoid circular import issues
