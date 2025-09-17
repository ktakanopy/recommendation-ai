from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.applications.services.recommendation_application_service import (
    RecommendationApplicationService,
)
from neural_recommendation.domain.models.deep_learning.model_config import ModelConfig
from neural_recommendation.domain.ports.repositories.feature_encoder_repository import (
    FeatureEncoderRepository,
)
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository
from neural_recommendation.domain.ports.repositories.movie_features_repository import MovieFeaturesRepository
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository
from neural_recommendation.domain.ports.repositories.rating_repository import RatingRepository
from neural_recommendation.domain.ports.repositories.user_features_repository import UserFeaturesRepository
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.domain.ports.services.logger import LoggerPort
from neural_recommendation.domain.ports.services.recommendation_application_service_port import (
    RecommendationApplicationServicePort,
)
from neural_recommendation.infrastructure.adapters.repositories.annoy_movie_features_repository import (
    AnnoyMovieFeaturesRepository,
)
from neural_recommendation.infrastructure.adapters.repositories.annoy_user_features_repository import (
    AnnoyUserFeaturesRepository,
)
from neural_recommendation.infrastructure.adapters.repositories.model_inference_manager_adapter import (
    ModelInferenceManagerAdapter,
)
from neural_recommendation.infrastructure.adapters.repositories.pickle_feature_encoder_repository import (
    PickleFeatureEncoderRepository,
)
from neural_recommendation.infrastructure.adapters.repositories.sqlalchemy_movie_repository import (
    SQLAlchemyMovieRepository,
)
from neural_recommendation.infrastructure.adapters.repositories.sqlalchemy_rating_repository import (
    SQLAlchemyRatingRepository,
)
from neural_recommendation.infrastructure.adapters.repositories.sqlalchemy_user_repository import (
    SQLAlchemyUserRepository,
)
from neural_recommendation.infrastructure.config.settings import MLModelSettings, Settings
from neural_recommendation.infrastructure.logging.std_logger_adapter import StdLoggerAdapter
from neural_recommendation.infrastructure.persistence.database import get_session


def get_logger() -> LoggerPort:
    return StdLoggerAdapter(__name__)


def get_settings() -> Settings:
    return Settings()


def get_ml_settings() -> MLModelSettings:
    return MLModelSettings()


def get_model_config(ml_settings: Annotated[MLModelSettings, Depends(get_ml_settings)]) -> ModelConfig:
    return ModelConfig.from_settings(ml_settings)


def get_user_repository(session: Annotated[AsyncSession, Depends(get_session)]) -> UserRepository:
    return SQLAlchemyUserRepository(session)


def get_movie_repository(session: Annotated[AsyncSession, Depends(get_session)]) -> MovieRepository:
    return SQLAlchemyMovieRepository(session)


def get_rating_repository(session: Annotated[AsyncSession, Depends(get_session)]) -> RatingRepository:
    return SQLAlchemyRatingRepository(session)


def get_model_inference_repository(
    config: Annotated[ModelConfig, Depends(get_model_config)],
    ml_settings: Annotated[MLModelSettings, Depends(get_ml_settings)],
    logger_port: Annotated[LoggerPort, Depends(get_logger)],
) -> ModelInferenceRepository:
    """Get model inference repository with injected config"""
    return ModelInferenceManagerAdapter(
        models_dir=config.models_dir,
        device=config.device,
        data_dir=ml_settings.data_dir,
        processed_data_dir=ml_settings.processed_data_dir,
        logger_port=logger_port,
    )


def get_movie_features_repository(
    ml_settings: Annotated[MLModelSettings, Depends(get_ml_settings)],
) -> MovieFeaturesRepository:
    return AnnoyMovieFeaturesRepository(
        data_path=ml_settings.processed_data_dir,
        index_path=ml_settings.movie_features_index_path,
        top_popular_movies_path=ml_settings.top_popular_movies_path,
        movie_features_cache_path=ml_settings.movie_features_cache_path,
        top_popular_movies_by_genres_path=ml_settings.top_popular_movies_by_genres_path,
    )


def get_user_features_repository(
    ml_settings: Annotated[MLModelSettings, Depends(get_ml_settings)],
) -> UserFeaturesRepository:
    return AnnoyUserFeaturesRepository(
        data_path=ml_settings.processed_data_dir, index_path=ml_settings.user_features_index_path
    )


def get_feature_encoder_repository(
    ml_settings: Annotated[MLModelSettings, Depends(get_ml_settings)],
) -> FeatureEncoderRepository:
    return PickleFeatureEncoderRepository(
        data_path=ml_settings.processed_data_dir, encoder_path=ml_settings.feature_encoder_index_path
    )


def get_recommendation_service(
    model_repository: Annotated[ModelInferenceRepository, Depends(get_model_inference_repository)],
    user_repository: Annotated[UserRepository, Depends(get_user_repository)],
    ml_settings: Annotated[MLModelSettings, Depends(get_ml_settings)],
    rating_repository: Annotated[RatingRepository, Depends(get_rating_repository)],
    movie_features_repository: Annotated[MovieFeaturesRepository, Depends(get_movie_features_repository)],
    user_features_repository: Annotated[UserFeaturesRepository, Depends(get_user_features_repository)],
    movie_repository: Annotated[MovieRepository, Depends(get_movie_repository)],
    feature_encoder_repository: Annotated[FeatureEncoderRepository, Depends(get_feature_encoder_repository)],
    logger: Annotated[LoggerPort, Depends(get_logger)],
) -> RecommendationApplicationServicePort:
    return RecommendationApplicationService(
        ml_settings=ml_settings,
        model_repository=model_repository,
        movie_repository=movie_repository,
        user_repository=user_repository,
        rating_repository=rating_repository,
        feature_encoder_repository=feature_encoder_repository,
        movie_features_repository=movie_features_repository,
        user_features_repository=user_features_repository,
        logger=logger,
    )
