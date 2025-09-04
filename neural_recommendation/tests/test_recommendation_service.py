import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import torch

from neural_recommendation.applications.use_cases.deep_learning.candidate_generator import CandidateGenerator
from neural_recommendation.applications.use_cases.deep_learning.cold_start_recommender import ColdStartRecommender
from neural_recommendation.applications.use_cases.deep_learning.ncf_feature_processor import NCFFeatureProcessor
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.domain.models.deep_learning.recommendation import Recommendation, RecommendationResult
from neural_recommendation.domain.models.rating import Rating
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.services.recommendation_service import RecommendationService


class TestRecommendationService:
    """Unit tests for RecommendationService"""

    @pytest.fixture
    def mock_ncf_model(self):
        """Create a mock NCF model"""
        model = Mock(spec=NCFModel)
        model.parameters.return_value = iter([torch.tensor([1.0])])  # For device detection
        model.eval = Mock()
        model.predict_batch = Mock(return_value=torch.tensor([0.7, 0.8, 0.6]))
        return model

    @pytest.fixture
    def mock_feature_processor(self):
        """Create a mock feature processor"""
        processor = Mock(spec=NCFFeatureProcessor)
        processor.process_user_demographics = Mock(return_value=torch.tensor([0.1, 0.2, 0.3]))
        processor.get_movie_features = Mock(return_value=torch.tensor([0.4, 0.5, 0.6]))
        processor.user_feature_dim = 10
        return processor

    @pytest.fixture
    def mock_candidate_generator(self):
        """Create a mock candidate generator"""
        generator = Mock(spec=CandidateGenerator)
        generator.generate_hybrid_candidates = Mock(return_value=[1, 2, 3])
        generator.user_interacted_items = {1: [101, 102], 2: [103, 104]}
        generator.movie_to_genres = {1: ["Action"], 2: ["Comedy"], 3: ["Drama"]}
        generator.get_genres_from_movies = Mock(return_value={"Action": 2, "Comedy": 1})
        return generator

    @pytest.fixture
    def mock_cold_start_recommender(self):
        """Create a mock cold start recommender"""
        recommender = Mock(spec=ColdStartRecommender)
        recommender.recommend_for_new_user = Mock(
            return_value=[(1, "Movie A", 0.9), (2, "Movie B", 0.8), (3, "Movie C", 0.7)]
        )
        recommender.get_onboarding_movies = Mock(
            return_value=[
                (1, "Popular Movie 1", "Action|Adventure"),
                (2, "Popular Movie 2", "Comedy"),
                (3, "Popular Movie 3", "Drama"),
            ]
        )
        return recommender

    @pytest.fixture
    def movie_mappings(self):
        """Create sample movie mappings"""
        return {
            "title_to_idx": {"Movie A": 1, "Movie B": 2, "Movie C": 3},
            "idx_to_title": {1: "Movie A", 2: "Movie B", 3: "Movie C"},
            "all_movie_titles": ["Movie A", "Movie B", "Movie C"],
        }

    @pytest.fixture
    def recommendation_service(self, mock_ncf_model, mock_feature_processor, movie_mappings):
        """Create RecommendationService instance with mocked dependencies"""
        with (
            patch("neural_recommendation.domain.services.recommendation_service.CandidateGenerator") as mock_cg_class,
            patch(
                "neural_recommendation.domain.services.recommendation_service.ColdStartRecommender"
            ) as mock_csr_class,
        ):
            # Configure the mocked classes to return our mock instances
            mock_cg_class.return_value = Mock(spec=CandidateGenerator)
            mock_csr_class.return_value = Mock(spec=ColdStartRecommender)

            service = RecommendationService(mock_ncf_model, mock_feature_processor, movie_mappings)
            return service

    def test_init(self, mock_ncf_model, mock_feature_processor, movie_mappings):
        """Test RecommendationService initialization"""
        with (
            patch("neural_recommendation.domain.services.recommendation_service.CandidateGenerator") as mock_cg_class,
            patch(
                "neural_recommendation.domain.services.recommendation_service.ColdStartRecommender"
            ) as mock_csr_class,
        ):
            service = RecommendationService(mock_ncf_model, mock_feature_processor, movie_mappings)

            assert service.model == mock_ncf_model
            assert service.feature_service == mock_feature_processor
            assert service.title_to_idx == movie_mappings["title_to_idx"]
            assert service.idx_to_title == movie_mappings["idx_to_title"]
            assert service.all_movie_titles == movie_mappings["all_movie_titles"]
            mock_ncf_model.eval.assert_called_once()

    def test_create_top_recommendations(self, recommendation_service):
        """Test creation of top recommendations from probabilities"""
        # Setup
        movie_titles = ["Movie A", "Movie B", "Movie C"]
        movie_ids = [1, 2, 3]
        probabilities = [0.9, 0.7, 0.8]
        num_recommendations = 2

        # Execute
        recommendations = recommendation_service._create_top_recommendations(
            movie_titles, movie_ids, probabilities, num_recommendations
        )

        # Assert
        assert len(recommendations) == num_recommendations
        assert all(isinstance(rec, Recommendation) for rec in recommendations)

        # Check that recommendations are sorted by score (descending)
        assert recommendations[0].similarity_score >= recommendations[1].similarity_score
        assert recommendations[0].title == "Movie A"  # Highest score (0.9)
        assert recommendations[1].title == "Movie C"  # Second highest (0.8)

    def test_generate_recommendations_cold_start_with_user_ratings(
        self, recommendation_service, mock_cold_start_recommender
    ):
        """Test cold start recommendations for user with ratings"""
        # Setup
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            password_hash="hashed",
            age=25,
            gender="M",
            occupation=1,
            ratings=[
                Rating(
                    id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=uuid.uuid4(), rating=4.5, timestamp=datetime.now()
                )
            ],
        )

        # Mock the cold start recommender
        recommendation_service.cold_start_recommender = mock_cold_start_recommender

        # Execute
        result = recommendation_service.generate_recommendations_cold_start(user, num_recommendations=3)

        # Assert
        assert isinstance(result, RecommendationResult)
        assert result.user_id == "1"
        assert len(result.recommendations) == 3

        # Verify cold start recommender was called with correct parameters
        mock_cold_start_recommender.recommend_for_new_user.assert_called_once()
        call_args = mock_cold_start_recommender.recommend_for_new_user.call_args
        assert call_args[1]["user_demographics"]["gender"] == "M"
        assert call_args[1]["user_demographics"]["age"] == 25
        assert call_args[1]["user_ratings"] is not None
        assert len(call_args[1]["user_ratings"]) == 1

    def test_generate_recommendations_cold_start_without_ratings(
        self, recommendation_service, mock_cold_start_recommender
    ):
        """Test cold start recommendations for user without ratings"""
        # Setup
        user = User(
            id=2,
            username="newuser",
            email="new@example.com",
            password_hash="hashed",
            age=30,
            gender="F",
            occupation=2,
            ratings=None,
        )

        # Mock the cold start recommender
        recommendation_service.cold_start_recommender = mock_cold_start_recommender

        # Execute
        result = recommendation_service.generate_recommendations_cold_start(user, num_recommendations=5)

        # Assert
        assert isinstance(result, RecommendationResult)
        assert result.user_id == "2"

        # Verify cold start recommender was called with no ratings
        mock_cold_start_recommender.recommend_for_new_user.assert_called_once()
        call_args = mock_cold_start_recommender.recommend_for_new_user.call_args
        assert call_args[1]["user_demographics"]["gender"] == "F"
        assert call_args[1]["user_demographics"]["age"] == 30
        assert call_args[1]["user_ratings"] is None
