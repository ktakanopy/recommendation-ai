import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import torch

from neural_recommendation.applications.services.recommendation_application_service import (
    RecommendationApplicationService,
)
from neural_recommendation.domain.models.rating import Rating as DomainRating
from neural_recommendation.domain.models.user import User as DomainUser
from neural_recommendation.infrastructure.adapters.repositories.sqlalchemy_movie_repository import (
    SQLAlchemyMovieRepository,
)
from neural_recommendation.infrastructure.adapters.repositories.sqlalchemy_user_repository import (
    SQLAlchemyUserRepository,
)
from neural_recommendation.infrastructure.adapters.repositories.sqlalchemy_rating_repository import (
    SQLAlchemyRatingRepository,
)
from neural_recommendation.infrastructure.config.settings import MLModelSettings
from neural_recommendation.infrastructure.persistence.models import Movie as SQLMovie
from neural_recommendation.infrastructure.persistence.models import User as SQLUser
from tests.conftest import BaseIntegrationTest


class TestColdStartIntegration(BaseIntegrationTest):
    """Integration test for cold start recommendations with real database operations"""

    @pytest.fixture
    def ml_settings(self):
        """Mock ML settings for testing"""
        settings = Mock(spec=MLModelSettings)
        settings.data_dir = "/tmp/test_data"
        settings.model_path = "/tmp/test_model.pth"
        return settings

    @pytest.fixture
    def mock_model_repository(self):
        """Mock model inference repository"""
        repo = Mock()

        # Mock NCF model
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.eval = Mock()
        mock_model.predict_batch = Mock(return_value=torch.tensor([0.8, 0.7, 0.6]))

        # Mock feature info with sentence embeddings
        mock_feature_info = Mock()
        mock_feature_info.sentence_embeddings = Mock()
        mock_feature_info.sentence_embeddings.title_to_idx = {
            "Toy Story (1995)": 1,
            "Jumanji (1995)": 2,
            "Grumpier Old Men (1995)": 3,
            "Waiting to Exhale (1995)": 4,
            "Father of the Bride Part II (1995)": 5,
        }

        repo.load_model_and_features.return_value = (mock_model, mock_feature_info)
        return repo

    @pytest.fixture
    def mock_feature_processor(self):
        """Mock NCF feature processor"""
        processor = Mock()
        processor.process_user_demographics = Mock(return_value=torch.tensor([0.1, 0.2, 0.3]))
        processor.get_movie_features = Mock(return_value=torch.tensor([0.4, 0.5, 0.6]))
        processor.user_feature_dim = 10
        return processor

    @pytest.fixture
    def mock_cold_start_recommender(self):
        """Mock cold start recommender"""
        recommender = Mock()
        recommender.recommend_for_new_user = Mock(
            return_value=[
                (1, "Toy Story (1995)", 0.95),
                (2, "Jumanji (1995)", 0.87),
                (3, "Grumpier Old Men (1995)", 0.82),
            ]
        )
        return recommender

    @pytest.fixture
    async def test_movies(self, test_session):
        """Create test movies in the database"""
        movies = [
            SQLMovie(
                title="Toy Story (1995)",
                genres="Animation|Children's|Comedy",
            ),
            SQLMovie(
                title="Jumanji (1995)",
                genres="Adventure|Children's|Fantasy",
            ),
            SQLMovie(
                title="Grumpier Old Men (1995)",
                genres="Comedy|Romance",
            ),
            SQLMovie(
                title="Waiting to Exhale (1995)",
                genres="Comedy|Drama",
            ),
            SQLMovie(
                title="Father of the Bride Part II (1995)",
                genres="Comedy",
            ),
        ]

        for movie in movies:
            test_session.add(movie)

        await test_session.commit()

        for movie in movies:
            await test_session.refresh(movie)

        return movies

    @pytest.fixture
    async def test_user_with_ratings(self, test_session, test_movies, rating_repository):
        """Create a test user with ratings for some movies"""
        movies = await test_movies

        user = SQLUser(
            username="testuser",
            email="test@example.com",
            password="hashed_password",
            age=25,
            gender="M",
            occupation="Engineer",
        )
        test_session.add(user)
        await test_session.commit()
        await test_session.refresh(user)

        domain_ratings = [
            DomainRating(id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=movies[0].id, rating=4.5, timestamp=datetime.now()),
            DomainRating(id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=movies[1].id, rating=3.0, timestamp=datetime.now()),
            DomainRating(id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=movies[2].id, rating=4.0, timestamp=datetime.now()),
        ]
        await rating_repository.bulk_create(user.id, domain_ratings)

        return user, domain_ratings

    @pytest.fixture
    def user_repository(self, test_session):
        """Create user repository with test session"""
        return SQLAlchemyUserRepository(test_session)

    @pytest.fixture
    def movie_repository(self, test_session):
        """Create movie repository with test session"""
        return SQLAlchemyMovieRepository(test_session)

    @pytest.fixture
    def rating_repository(self, test_session):
        return SQLAlchemyRatingRepository(test_session)

    def create_recommendation_service(self, ml_settings, mock_model_repository, user_repository, rating_repository):
        """Create recommendation application service with mocked dependencies"""
        return RecommendationApplicationService(ml_settings, mock_model_repository, user_repository, rating_repository)

    @pytest.mark.asyncio
    async def test_cold_start_recommendations_with_database_user(
        self,
        ml_settings,
        mock_model_repository,
        user_repository,
        test_user_with_ratings,
        mock_feature_processor,
        mock_cold_start_recommender,
        rating_repository,
    ):
        """Test cold start recommendations using a real user from database with ratings"""
        user, ratings = await test_user_with_ratings
        recommendation_service = self.create_recommendation_service(ml_settings, mock_model_repository, user_repository, rating_repository)

        # Mock the domain service components
        with (
            patch.object(recommendation_service._domain_service, "cold_start_recommender", mock_cold_start_recommender),
            patch.object(recommendation_service._domain_service, "feature_service", mock_feature_processor),
        ):
            # Execute cold start recommendations
            result = await recommendation_service.generate_recommendations_cold_start(
                user_id=user.id, num_recommendations=3
            )

            # Assertions
            assert result is not None
            assert result.user_id == str(user.id)
            assert len(result.recommendations) == 3
            assert result.total_available_movies > 0

            # Verify recommendation structure
            for rec in result.recommendations:
                assert hasattr(rec, "movie_id")
                assert hasattr(rec, "title")
                assert hasattr(rec, "similarity_score")
                assert isinstance(rec.similarity_score, float)
                assert 0.0 <= rec.similarity_score <= 1.0

            # Verify the cold start recommender was called with correct user data
            mock_cold_start_recommender.recommend_for_new_user.assert_called_once()
            call_args = mock_cold_start_recommender.recommend_for_new_user.call_args

            # Check user demographics were passed correctly
            user_demographics = call_args[1]["user_demographics"]
            assert user_demographics["gender"] == "M"
            assert user_demographics["age"] == 25
            assert user_demographics["occupation"] == "Engineer"

            # Check user ratings were passed correctly
            user_ratings_arg = call_args[1]["user_ratings"]
            assert user_ratings_arg is not None
            assert len(user_ratings_arg) == 3  # Should have 3 ratings

    @pytest.mark.asyncio
    async def test_cold_start_recommendations_new_user_no_ratings(
        self,
        test_session,
        ml_settings,
        mock_model_repository,
        user_repository,
        mock_feature_processor,
        mock_cold_start_recommender,
        rating_repository,
    ):
        """Test cold start recommendations for a new user without any ratings"""
        recommendation_service = self.create_recommendation_service(ml_settings, mock_model_repository, user_repository, rating_repository)
        # Create a new user without ratings
        user = SQLUser(
            username="newuser",
            email="newuser@example.com",
            password="hashed_password",
            age=30,
            gender="F",
            occupation="Teacher",
        )
        test_session.add(user)
        await test_session.commit()
        await test_session.refresh(user)

        # Mock the domain service components
        with (
            patch.object(recommendation_service._domain_service, "cold_start_recommender", mock_cold_start_recommender),
            patch.object(recommendation_service._domain_service, "feature_service", mock_feature_processor),
        ):
            # Execute cold start recommendations
            result = await recommendation_service.generate_recommendations_cold_start(
                user_id=user.id, num_recommendations=5
            )

            # Assertions
            assert result is not None
            assert result.user_id == str(user.id)
            assert len(result.recommendations) == 3  # Mocked to return 3

            # Verify the cold start recommender was called with correct user data
            mock_cold_start_recommender.recommend_for_new_user.assert_called_once()
            call_args = mock_cold_start_recommender.recommend_for_new_user.call_args

            # Check user demographics were passed correctly
            user_demographics = call_args[1]["user_demographics"]
            assert user_demographics["gender"] == "F"
            assert user_demographics["age"] == 30
            assert user_demographics["occupation"] == "Teacher"

            # Check user ratings should be None for new user
            user_ratings_arg = call_args[1]["user_ratings"]
            assert user_ratings_arg is None

    @pytest.mark.asyncio
    async def test_cold_start_recommendations_user_not_found(self, ml_settings, mock_model_repository, user_repository, rating_repository):
        """Test cold start recommendations when user doesn't exist"""
        recommendation_service = self.create_recommendation_service(ml_settings, mock_model_repository, user_repository, rating_repository)
        non_existent_user_id = 99999

        # Execute and expect ValueError
        with pytest.raises(ValueError, match=f"User with ID {non_existent_user_id} not found"):
            await recommendation_service.generate_recommendations_cold_start(user_id=non_existent_user_id)

    @pytest.mark.asyncio
    async def test_full_integration_create_user_and_recommend(
        self,
        test_session,
        user_repository,
        movie_repository,
        test_movies,
        ml_settings,
        mock_model_repository,
        mock_feature_processor,
        mock_cold_start_recommender,
        rating_repository,
    ):
        """Full integration test: create user, add ratings, get recommendations"""
        movies = await test_movies

        domain_user = DomainUser(
            username="integrationuser",
            email="integration@example.com",
            password_hash="hashed_password",
            age=28,
            gender="M",
            occupation="Developer",
        )

        created_user = await user_repository.create(domain_user)
        assert created_user.id is not None

        ratings = [
            DomainRating(id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=movies[0].id, rating=4.5, timestamp=datetime.now()),
            DomainRating(id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=movies[1].id, rating=2.0, timestamp=datetime.now()),
        ]

        await rating_repository.bulk_create(created_user.id, ratings)

        # Step 3: Create recommendation service and get recommendations
        recommendation_service = self.create_recommendation_service(ml_settings, mock_model_repository, user_repository, rating_repository)

        # Mock the domain service components
        with (
            patch.object(recommendation_service._domain_service, "cold_start_recommender", mock_cold_start_recommender),
            patch.object(recommendation_service._domain_service, "feature_service", mock_feature_processor),
        ):
            # Execute cold start recommendations
            result = await recommendation_service.generate_recommendations_cold_start(
                user_id=created_user.id, num_recommendations=3
            )

            # Step 4: Verify the complete flow worked
            assert result is not None
            assert result.user_id == str(created_user.id)
            assert len(result.recommendations) == 3

            # Verify that the user's ratings were considered
            mock_cold_start_recommender.recommend_for_new_user.assert_called_once()
            call_args = mock_cold_start_recommender.recommend_for_new_user.call_args

            user_ratings_arg = call_args[1]["user_ratings"]
            assert user_ratings_arg is not None
            assert len(user_ratings_arg) == 2  # Should have 2 ratings we added

            # Verify user demographics
            user_demographics = call_args[1]["user_demographics"]
            assert user_demographics["gender"] == "M"
            assert user_demographics["age"] == 28
            assert user_demographics["occupation"] == "Developer"

    def test_recommendation_service_initialization(self, ml_settings, mock_model_repository, user_repository, rating_repository):
        """Test that the recommendation service initializes correctly"""
        service = self.create_recommendation_service(ml_settings, mock_model_repository, user_repository, rating_repository)

        assert service.ml_settings == ml_settings
        assert service._model_repository == mock_model_repository
        assert service.user_repository == user_repository
        assert service._domain_service is not None

        # Verify that the model and features were loaded
        mock_model_repository.load_model_and_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_domain_user_conversion_with_ratings(self, test_session, user_repository, test_movies, rating_repository):
        """Test that user ratings are properly converted from SQL to domain models"""
        movies = await test_movies
        sql_user = SQLUser(
            username="ratinguser",
            email="rating@example.com",
            password="hashed_password",
            age=35,
            gender="F",
            occupation="Designer",
        )
        test_session.add(sql_user)
        await test_session.commit()
        await test_session.refresh(sql_user)

        domain_ratings = [
            DomainRating(id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=movies[0].id, rating=5.0, timestamp=datetime.now()),
            DomainRating(id=uuid.uuid4(), user_id=uuid.uuid4(), movie_id=movies[2].id, rating=3.5, timestamp=datetime.now()),
        ]
        await rating_repository.bulk_create(sql_user.id, domain_ratings)
