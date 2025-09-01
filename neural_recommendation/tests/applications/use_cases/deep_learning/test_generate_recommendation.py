from unittest.mock import Mock

import pytest

from neural_recommendation.applications.use_cases.deep_learning.generate_recommendation import RecommendationGenerator
from neural_recommendation.domain.models.deep_learning.recommendation import Recommendation, RecommendationResult
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository


@pytest.fixture
def mock_repository():
    return Mock(spec=ModelInferenceRepository)


@pytest.fixture
def mock_recommendation_result():
    recommendations = [
        Recommendation(
            movie_id=1,
            title="Test Movie 1",
            genres="Action|Comedy",
            similarity_score=0.95
        ),
        Recommendation(
            movie_id=2,
            title="Test Movie 2",
            genres="Drama",
            similarity_score=0.88
        )
    ]
    return RecommendationResult(
        user_id="test_user",
        recommendations=recommendations,
        total_available_movies=100
    )


class TestRecommendationGenerator:

    def test_init(self, mock_repository):
        generator = RecommendationGenerator(mock_repository)
        assert generator.model_repository == mock_repository
        assert generator._recommendation_service is None

    def test_get_recommendation_service_lazy_loading(self, mock_repository):
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_feature_info = Mock()
        mock_feature_info.age_mean = 25.0
        mock_feature_info.age_std = 5.0
        mock_feature_info.timestamp_stats = {'mean': 1000, 'std': 100}
        mock_feature_info.unique_user_ids = ['user1', 'user2']
        mock_feature_info.unique_movie_ids = ['movie1', 'movie2']
        mock_sentence_embeddings = Mock()
        mock_sentence_embeddings.title_to_idx = {'movie1': 0, 'movie2': 1}
        mock_feature_info.sentence_embeddings = mock_sentence_embeddings
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # First call should create the service
        service1 = generator._get_recommendation_service()
        assert service1 is not None
        assert generator._recommendation_service is service1
        mock_repository.load_model_and_features.assert_called_once()

        # Second call should return the same service (lazy loading)
        service2 = generator._get_recommendation_service()
        assert service2 is service1
        assert mock_repository.load_model_and_features.call_count == 1

    def test_generate_recommendations_for_existing_user_default_params(self, mock_repository, mock_recommendation_result):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service
        mock_service = Mock()
        mock_service.generate_recommendations_for_user.return_value = mock_recommendation_result
        generator._recommendation_service = mock_service

        result = generator.generate_recommendations_for_existing_user("user123")

        mock_service.generate_recommendations_for_user.assert_called_once_with(
            user_id="user123",
            user_age=25.0,
            gender="M",
            excluded_movie_titles=set(),
            num_recommendations=10
        )
        assert result == mock_recommendation_result

    def test_generate_recommendations_for_existing_user_custom_params(self, mock_repository, mock_recommendation_result):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service
        mock_service = Mock()
        mock_service.generate_recommendations_for_user.return_value = mock_recommendation_result
        generator._recommendation_service = mock_service

        watched_movies = {"Movie A", "Movie B"}

        result = generator.generate_recommendations_for_existing_user(
            user_id="user456",
            user_age=30.0,
            gender="F",
            watched_movie_titles=watched_movies,
            num_recommendations=5
        )

        mock_service.generate_recommendations_for_user.assert_called_once_with(
            user_id="user456",
            user_age=30.0,
            gender="F",
            excluded_movie_titles=watched_movies,
            num_recommendations=5
        )
        assert result == mock_recommendation_result

    def test_generate_recommendations_for_existing_user_none_watched_movies(self, mock_repository, mock_recommendation_result):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service
        mock_service = Mock()
        mock_service.generate_recommendations_for_user.return_value = mock_recommendation_result
        generator._recommendation_service = mock_service

        result = generator.generate_recommendations_for_existing_user(
            user_id="user789",
            watched_movie_titles=None
        )

        mock_service.generate_recommendations_for_user.assert_called_once_with(
            user_id="user789",
            user_age=25.0,
            gender="M",
            excluded_movie_titles=set(),
            num_recommendations=10
        )
        assert result == mock_recommendation_result

    def test_generate_recommendations_for_new_user_default_params(self, mock_repository, mock_recommendation_result):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service
        mock_service = Mock()
        mock_service.generate_recommendations_for_user.return_value = mock_recommendation_result
        generator._recommendation_service = mock_service

        result = generator.generate_recommendations_for_new_user(
            user_age=25.0,
            gender="M"
        )

        mock_service.generate_recommendations_for_user.assert_called_once_with(
            user_id="new_user",
            user_age=25.0,
            gender="M",
            excluded_movie_titles=set(),
            num_recommendations=10
        )
        assert result == mock_recommendation_result

    def test_generate_recommendations_for_new_user_custom_params(self, mock_repository, mock_recommendation_result):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service
        mock_service = Mock()
        mock_service.generate_recommendations_for_user.return_value = mock_recommendation_result
        generator._recommendation_service = mock_service

        result = generator.generate_recommendations_for_new_user(
            user_age=35.0,
            gender="F",
            preferred_genres=["Action", "Comedy"],
            num_recommendations=15
        )

        mock_service.generate_recommendations_for_user.assert_called_once_with(
            user_id="new_user",
            user_age=35.0,
            gender="F",
            excluded_movie_titles=set(),
            num_recommendations=15
        )
        assert result == mock_recommendation_result

    def test_explain_recommendation_movie_found(self, mock_repository):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service
        target_movie = Recommendation(
            movie_id=1,
            title="Target Movie",
            genres="Action|Drama",
            similarity_score=0.92
        )

        mock_result = RecommendationResult(
            user_id="user123",
            recommendations=[target_movie],
            total_available_movies=100
        )

        mock_service = Mock()
        mock_service.generate_recommendations_for_user.return_value = mock_result
        generator._recommendation_service = mock_service

        explanation = generator.explain_recommendation(
            user_id="user123",
            movie_title="Target Movie"
        )

        expected_explanation = {
            "movie_title": "Target Movie",
            "similarity_score": 0.92,
            "similarity_percentage": 92.0,
            "explanation": "This movie has a 92.0% similarity match with your preferences based on your viewing history and demographic profile.",
            "genres": "Action|Drama"
        }

        assert explanation == expected_explanation

    def test_explain_recommendation_movie_not_found(self, mock_repository):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service to return empty result
        mock_result = RecommendationResult(
            user_id="user123",
            recommendations=[],
            total_available_movies=100
        )

        mock_service = Mock()
        mock_service.generate_recommendations_for_user.return_value = mock_result
        generator._recommendation_service = mock_service

        explanation = generator.explain_recommendation(
            user_id="user123",
            movie_title="Non-existent Movie"
        )

        expected_explanation = {
            "movie_title": "Non-existent Movie",
            "explanation": "Movie not found in recommendation candidates",
            "similarity_score": 0.0
        }

        assert explanation == expected_explanation

    def test_explain_recommendation_default_params(self, mock_repository, mock_recommendation_result):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service
        mock_service = Mock()
        mock_service.generate_recommendations_for_user.return_value = mock_recommendation_result
        generator._recommendation_service = mock_service

        generator.explain_recommendation(
            user_id="user123",
            movie_title="Some Movie"
        )

        mock_service.generate_recommendations_for_user.assert_called_once_with(
            user_id="user123",
            user_age=25.0,
            gender="M",
            excluded_movie_titles=set(),
            num_recommendations=1000
        )

    def test_integration_lazy_loading_with_actual_service_call(self, mock_repository, mock_recommendation_result):
        mock_model = Mock()
        mock_feature_info = Mock()
        mock_repository.load_model_and_features.return_value = (mock_model, mock_feature_info)

        generator = RecommendationGenerator(mock_repository)

        # Mock the recommendation service at the class level
        with pytest.MonkeyPatch().context() as m:
            mock_service_class = Mock()
            mock_service = Mock()
            mock_service.generate_recommendations_for_user.return_value = mock_recommendation_result
            mock_service_class.return_value = mock_service

            m.setattr('neural_recommendation.applications.use_cases.deep_learning.generate_recommendation.RecommendationService', mock_service_class)

            result = generator.generate_recommendations_for_existing_user("user123")

            # Verify lazy loading worked
            assert generator._recommendation_service is not None
            mock_repository.load_model_and_features.assert_called_once()
            mock_service_class.assert_called_once_with(mock_model, mock_feature_info)
            assert result == mock_recommendation_result
