import uuid
from datetime import datetime
from unittest.mock import Mock, patch
import pytest
import torch
import pandas as pd

from neural_recommendation.applications.use_cases.deep_learning.cold_start_recommender import ColdStartRecommender
from neural_recommendation.applications.use_cases.deep_learning.candidate_generator import CandidateGenerator
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel


class TestColdStartFunctionality:
    """Integration tests for cold start recommendation functionality"""

    @pytest.fixture
    def sample_movies_df(self):
        """Create sample movies DataFrame"""
        return pd.DataFrame({
            'movie_id': [1, 2, 3, 4, 5],
            'title': ['Action Movie', 'Comedy Movie', 'Drama Movie', 'Horror Movie', 'Romance Movie'],
            'genres': ['Action|Adventure', 'Comedy', 'Drama', 'Horror|Thriller', 'Romance|Comedy']
        })

    @pytest.fixture
    def sample_train_ratings(self):
        """Create sample training ratings DataFrame"""
        return pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4],
            'movie_id': [1, 2, 1, 3, 2, 4, 5],
            'rating': [4.5, 3.0, 5.0, 2.0, 4.0, 3.5, 4.5]
        })

    @pytest.fixture
    def cold_start_recommender(self, mock_ncf_model, mock_feature_processor, sample_movies_df):
        """Create ColdStartRecommender instance with mocked dependencies"""
        # Create candidate generator with sample data
        candidate_generator = Mock(spec=CandidateGenerator)
        candidate_generator.user_interacted_items = {
            1: [1, 2],
            2: [1, 3], 
            3: [2, 4],
            4: [5]
        }
        candidate_generator.movie_to_genres = {
            1: ['Action', 'Adventure'],
            2: ['Comedy'],
            3: ['Drama'],
            4: ['Horror', 'Thriller'],
            5: ['Romance', 'Comedy']
        }
        candidate_generator.generate_hybrid_candidates = Mock(return_value=[1, 2, 3])
        candidate_generator.generate_popularity_candidates = Mock(return_value=[1, 2, 5])
        candidate_generator.generate_content_candidates = Mock(return_value=[1, 3, 4])
        candidate_generator.get_genres_from_movies = Mock(return_value={'Action': 2, 'Comedy': 1})
        
        return ColdStartRecommender(
            trained_model=mock_ncf_model,
            feature_processor=mock_feature_processor,
            candidate_generator=candidate_generator,
            movies_df=sample_movies_df,
            liked_threshold=4.0
        )

    def test_cold_start_recommender_initialization(self, mock_ncf_model, mock_feature_processor, sample_movies_df):
        """Test ColdStartRecommender initialization"""
        candidate_generator = Mock()
        
        recommender = ColdStartRecommender(
            trained_model=mock_ncf_model,
            feature_processor=mock_feature_processor,
            candidate_generator=candidate_generator,
            movies_df=sample_movies_df,
            liked_threshold=3.5
        )
        
        assert recommender.model == mock_ncf_model
        assert recommender.feature_processor == mock_feature_processor
        assert recommender.candidate_generator == candidate_generator
        assert recommender.movies_df is sample_movies_df
        assert recommender.liked_threshold == 3.5

    def test_get_similar_users_neural_embedding(self, cold_start_recommender, mock_feature_processor, mock_ncf_model):
        """Test finding similar users using neural embeddings"""
        # Setup
        user_demographics = {'gender': 'M', 'age': 25, 'occupation': 1}
        
        # Mock feature processor
        mock_feature_processor.process_user_demographics.return_value = torch.tensor([0.1, 0.2, 0.3])
        mock_feature_processor.user_features_cache = {
            1: torch.tensor([0.1, 0.2]),
            2: torch.tensor([0.3, 0.4]),
            3: torch.tensor([0.2, 0.1])
        }
        
        # Mock model layers for similarity computation
        mock_ncf_model.user_fc1.return_value = torch.tensor([[0.5, 0.6, 0.7]])
        mock_ncf_model.user_bn1.return_value = torch.tensor([[0.5, 0.6, 0.7]])
        mock_ncf_model.user_fc2.return_value = torch.tensor([[0.8, 0.9]])
        mock_ncf_model.user_bn2.return_value = torch.tensor([[0.8, 0.9]])
        mock_ncf_model.dropout.return_value = torch.tensor([[0.5, 0.6, 0.7]])
        
        # Execute
        similar_users = cold_start_recommender.get_similar_users_neural_embedding(user_demographics, top_k=2)
        
        # Assert
        assert isinstance(similar_users, list)
        assert len(similar_users) <= 2
        mock_feature_processor.process_user_demographics.assert_called_once_with(user_demographics)

    def test_get_similar_user_candidates(self, cold_start_recommender):
        """Test getting candidate movies from similar users"""
        # Setup
        user_demographics = {'gender': 'F', 'age': 30, 'occupation': 2}
        
        with patch.object(cold_start_recommender, 'get_similar_users_neural_embedding', return_value=[1, 2]):
            # Execute
            candidates = cold_start_recommender.get_similar_user_candidates(
                user_demographics, top_k=2, num_candidates=5
            )
            
            # Assert
            assert isinstance(candidates, list)
            assert len(candidates) <= 5

    def test_generate_cold_start_candidates_no_ratings(self, cold_start_recommender):
        """Test candidate generation for pure cold start (no ratings)"""
        # Setup
        user_demographics = {'gender': 'M', 'age': 25, 'occupation': 1}
        
        with patch.object(cold_start_recommender, 'get_similar_users_neural_embedding', return_value=[1, 2]):
            # Execute
            candidates = cold_start_recommender.generate_cold_start_candidates(
                user_demographics, user_ratings=None, num_candidates=10
            )
            
            # Assert
            assert isinstance(candidates, list)
            assert len(candidates) <= 10
            # Should call hybrid candidates and similar user methods
            cold_start_recommender.candidate_generator.generate_hybrid_candidates.assert_called()

    def test_generate_cold_start_candidates_with_ratings(self, cold_start_recommender):
        """Test candidate generation for warm start (with ratings)"""
        # Setup
        user_demographics = {'gender': 'F', 'age': 28, 'occupation': 3}
        user_ratings = [(1, 4.5), (2, 3.0), (3, 4.0)]  # Some liked, some not
        
        with patch.object(cold_start_recommender, 'get_similar_user_candidates', return_value=[4, 5]):
            # Execute
            candidates = cold_start_recommender.generate_cold_start_candidates(
                user_demographics, user_ratings=user_ratings, num_candidates=15
            )
            
            # Assert
            assert isinstance(candidates, list)
            assert len(candidates) <= 15
            # Should call various candidate generation methods
            cold_start_recommender.candidate_generator.generate_popularity_candidates.assert_called()
            cold_start_recommender.candidate_generator.generate_content_candidates.assert_called()

    def test_recommend_for_new_user_no_ratings(self, cold_start_recommender, mock_feature_processor, mock_ncf_model):
        """Test recommendations for new user without ratings"""
        # Setup
        user_demographics = {'gender': 'M', 'age': 22, 'occupation': 1}
        
        # Mock feature processing and model prediction
        mock_feature_processor.process_user_demographics.return_value = torch.tensor([0.1, 0.2, 0.3])
        mock_feature_processor.get_movie_features.return_value = torch.tensor([0.4, 0.5, 0.6])
        mock_ncf_model.return_value = torch.tensor([0.85])  # High prediction score
        
        with patch.object(cold_start_recommender, 'generate_cold_start_candidates', return_value=[1, 2, 3]):
            # Execute
            recommendations = cold_start_recommender.recommend_for_new_user(
                user_demographics, user_ratings=None, num_recommendations=3
            )
            
            # Assert
            assert isinstance(recommendations, list)
            assert len(recommendations) <= 3
            assert all(len(rec) == 3 for rec in recommendations)  # (movie_id, title, score)
            assert all(isinstance(rec[0], int) for rec in recommendations)  # movie_id
            assert all(isinstance(rec[1], str) for rec in recommendations)  # title
            assert all(isinstance(rec[2], float) for rec in recommendations)  # score

    def test_recommend_for_new_user_with_ratings(self, cold_start_recommender, mock_feature_processor, mock_ncf_model):
        """Test recommendations for new user with initial ratings"""
        # Setup
        user_demographics = {'gender': 'F', 'age': 35, 'occupation': 4}
        user_ratings = [(1, 5.0), (2, 2.0)]  # Loved movie 1, disliked movie 2
        
        # Mock feature processing and model prediction
        mock_feature_processor.process_user_demographics.return_value = torch.tensor([0.2, 0.3, 0.4])
        mock_feature_processor.get_movie_features.return_value = torch.tensor([0.5, 0.6, 0.7])
        mock_ncf_model.return_value = torch.tensor([0.75])
        
        with patch.object(cold_start_recommender, 'generate_cold_start_candidates', return_value=[3, 4, 5]):
            # Execute
            recommendations = cold_start_recommender.recommend_for_new_user(
                user_demographics, user_ratings=user_ratings, num_recommendations=2
            )
            
            # Assert
            assert isinstance(recommendations, list)
            assert len(recommendations) <= 2
            # Should filter out already rated movies
            recommended_movie_ids = [rec[0] for rec in recommendations]
            assert 1 not in recommended_movie_ids  # Already rated
            assert 2 not in recommended_movie_ids  # Already rated

    def test_get_onboarding_movies(self, cold_start_recommender):
        """Test getting diverse movies for onboarding"""
        # Setup
        with patch.object(cold_start_recommender.candidate_generator, 'generate_popularity_candidates', 
                         return_value=[1, 2, 3, 4, 5]):
            
            # Execute
            onboarding_movies = cold_start_recommender.get_onboarding_movies(num_movies=3)
            
            # Assert
            assert isinstance(onboarding_movies, list)
            assert len(onboarding_movies) <= 3
            assert all(len(movie) == 3 for movie in onboarding_movies)  # (movie_id, title, genres)
            
            # Check for genre diversity (should try to get different genres)
            movie_ids = [movie[0] for movie in onboarding_movies]
            assert len(set(movie_ids)) == len(movie_ids)  # No duplicates

    def test_onboarding_movies_genre_diversity(self, cold_start_recommender):
        """Test that onboarding movies provide genre diversity"""
        # Setup - ensure different genres are available
        cold_start_recommender.candidate_generator.movie_to_genres = {
            1: ['Action'],
            2: ['Comedy'],
            3: ['Drama'],
            4: ['Horror'],
            5: ['Romance']
        }
        
        with patch.object(cold_start_recommender.candidate_generator, 'generate_popularity_candidates', 
                         return_value=[1, 2, 3, 4, 5]):
            
            # Execute
            onboarding_movies = cold_start_recommender.get_onboarding_movies(num_movies=5)
            
            # Assert
            assert len(onboarding_movies) == 5
            # Should have different genres represented
            genres_represented = []
            for movie_id, title, genres_str in onboarding_movies:
                genres_represented.extend(genres_str.split('|'))
            
            # Should have some genre diversity
            assert len(set(genres_represented)) > 1

    def test_error_handling_in_recommendations(self, cold_start_recommender, mock_feature_processor):
        """Test error handling in recommendation generation"""
        # Setup - make feature processing fail
        mock_feature_processor.process_user_demographics.side_effect = Exception("Feature processing failed")
        
        # Execute
        recommendations = cold_start_recommender.recommend_for_new_user(
            {'gender': 'M', 'age': 25, 'occupation': 1}
        )
        
        # Assert - should return empty list instead of crashing
        assert isinstance(recommendations, list)
        assert len(recommendations) == 0

    def test_liked_threshold_filtering(self, cold_start_recommender):
        """Test that liked threshold is correctly applied"""
        # Setup
        user_demographics = {'gender': 'M', 'age': 30, 'occupation': 2}
        user_ratings = [
            (1, 4.5),  # Above threshold (should be considered liked)
            (2, 3.0),  # Below threshold (should not be considered liked)
            (3, 4.0),  # At threshold (should be considered liked)
            (4, 2.5)   # Below threshold (should not be considered liked)
        ]
        
        with patch.object(cold_start_recommender, 'get_similar_user_candidates', return_value=[5]):
            # Execute
            candidates = cold_start_recommender.generate_cold_start_candidates(
                user_demographics, user_ratings=user_ratings, num_candidates=10
            )
            
            # Assert
            assert isinstance(candidates, list)
            # The method should have identified movies 1 and 3 as liked (>= 4.0 threshold)
            cold_start_recommender.candidate_generator.get_genres_from_movies.assert_called()

    def test_recommendation_scoring_and_sorting(self, cold_start_recommender, mock_feature_processor, mock_ncf_model):
        """Test that recommendations are properly scored and sorted"""
        # Setup
        user_demographics = {'gender': 'F', 'age': 25, 'occupation': 1}
        
        # Mock different scores for different movies
        scores = [0.9, 0.7, 0.8, 0.6]  # Intentionally not sorted
        mock_ncf_model.side_effect = [torch.tensor([score]) for score in scores]
        
        mock_feature_processor.process_user_demographics.return_value = torch.tensor([0.1, 0.2, 0.3])
        mock_feature_processor.get_movie_features.return_value = torch.tensor([0.4, 0.5, 0.6])
        
        with patch.object(cold_start_recommender, 'generate_cold_start_candidates', return_value=[1, 2, 3, 4]):
            # Execute
            recommendations = cold_start_recommender.recommend_for_new_user(
                user_demographics, num_recommendations=4
            )
            
            # Assert
            assert len(recommendations) == 4
            # Should be sorted by score in descending order
            rec_scores = [rec[2] for rec in recommendations]
            assert rec_scores == sorted(rec_scores, reverse=True)
            assert rec_scores[0] == 0.9  # Highest score first
