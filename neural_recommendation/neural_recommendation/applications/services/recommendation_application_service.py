from typing import Any, Dict, List

from neural_recommendation.applications.use_cases.deep_learning.ncf_feature_processor import NCFFeatureProcessor
from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository
from neural_recommendation.domain.ports.services.recommendation_service_port import RecommendationServicePort
from neural_recommendation.domain.services.recommendation_service import RecommendationService
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class RecommendationApplicationService(RecommendationServicePort):
    """Application service for NCF-based recommendations"""
    
    def __init__(self, model_repository: ModelInferenceRepository):
        self._model_repository = model_repository
        self._domain_service = None

    def _get_domain_service(self) -> RecommendationService:
        """Initialize the domain service with NCF model and feature processor"""
        if self._domain_service is None:
            logger.info("Initializing NCF-based recommendation service")
            
            # Load NCF model
            model, feature_info = self._model_repository.load_model_and_features()
            
            # Initialize NCF feature processor
            feature_service = NCFFeatureProcessor()
            
            # Create movie mappings - for NCF we'll use a simplified approach
            # In a real implementation, these would come from the training data
            title_to_idx = {}
            if hasattr(feature_info, 'sentence_embeddings') and hasattr(feature_info.sentence_embeddings, 'title_to_idx'):
                title_to_idx = feature_info.sentence_embeddings.title_to_idx
            else:
                # Create dummy mappings for testing if not available
                logger.warning("No movie mappings found, creating dummy mappings for testing")
                title_to_idx = {f"Movie_{i}": i for i in range(100)}
            
            idx_to_title = {idx: title for title, idx in title_to_idx.items()}
            all_movie_titles = list(title_to_idx.keys())
            
            movie_mappings = {
                "title_to_idx": title_to_idx,
                "idx_to_title": idx_to_title,
                "all_movie_titles": all_movie_titles,
            }
            
            # Create domain service
            self._domain_service = RecommendationService(
                model=model, 
                feature_service=feature_service, 
                movie_mappings=movie_mappings
            )
            
            logger.info(f"NCF recommendation service initialized with {len(all_movie_titles)} movies")
            
        return self._domain_service

    def generate_recommendations_for_existing_user(
        self,
        user_id: str,
        user_age: float = 25.0,
        gender: str = "M",
        num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for an existing user using NCF model"""
        logger.info(f"Generating recommendations for existing user {user_id}")
        
        domain_service = self._get_domain_service()
        return domain_service.generate_recommendations_for_user(
            user_id=user_id,
            user_age=user_age,
            gender=gender,
            num_recommendations=num_recommendations,
        )

    def generate_recommendations_for_new_user(
        self,
        user_age: float,
        gender: str,
        preferred_genres: list[str] = None,
        occupation: int = 1,
        num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for a new user using NCF cold start approach"""
        logger.info(f"Generating cold start recommendations for new user: age={user_age}, gender={gender}")
        
        domain_service = self._get_domain_service()
        # Use the specialized cold start method for new users
        return domain_service.generate_recommendations_for_new_user(
            user_age=user_age,
            gender=gender,
            occupation=occupation,
            num_recommendations=num_recommendations,
        )

    def explain_recommendation(
        self,
        user_id: str,
        movie_title: str,
        user_age: float = 25.0,
        gender: str = "M"
    ) -> Dict[str, Any]:
        """Explain why a specific movie was recommended using NCF model"""
        logger.info(f"Explaining recommendation for user {user_id} and movie {movie_title}")
        
        domain_service = self._get_domain_service()
        return domain_service.explain_recommendation(
            user_id=user_id,
            movie_title=movie_title,
            user_age=user_age,
            gender=gender
        )

    def get_onboarding_movies(self, num_movies: int = 10) -> List[Dict[str, Any]]:
        """Get diverse movies for new user onboarding"""
        logger.info(f"Getting {num_movies} onboarding movies for new user")
        
        domain_service = self._get_domain_service()
        return domain_service.get_onboarding_movies(num_movies)