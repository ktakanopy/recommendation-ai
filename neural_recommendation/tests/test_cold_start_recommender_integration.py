from neural_recommendation.applications.services.candidate_generator_service import (
    CandidateGeneratorService,
)
from neural_recommendation.applications.services.ncf_feature_service import (
    NCFFeatureService,
)
from neural_recommendation.applications.use_cases.deep_learning.cold_start_recommender import (
    ColdStartRecommender,
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
from neural_recommendation.infrastructure.config.settings import MLModelSettings


def build_recommender(num_candidates: int = 50):
    settings = MLModelSettings()
    movie_repo = AnnoyMovieFeaturesRepository(
        data_path=settings.processed_data_dir,
        index_path=settings.movie_features_index_path,
        top_popular_movies_path=settings.top_popular_movies_path,
        movie_features_cache_path=settings.movie_features_cache_path,
        top_popular_movies_by_genres_path=settings.top_popular_movies_by_genres_path,
    )
    user_repo = AnnoyUserFeaturesRepository(
        data_path=settings.processed_data_dir, index_path=settings.user_features_index_path
    )
    encoder_repo = PickleFeatureEncoderRepository(
        data_path=settings.processed_data_dir, encoder_path=settings.feature_encoder_index_path
    )
    feature_service = NCFFeatureService(feature_encoder_repository=encoder_repo)
    candidate_gen = CandidateGeneratorService(movie_repo, user_repo, feature_service)
    model_repo = ModelInferenceManagerAdapter(
        models_dir=settings.models_dir,
        device=settings.device,
        data_dir=settings.data_dir,
        processed_data_dir=settings.processed_data_dir,
    )
    model = model_repo.load_model()
    recommender = ColdStartRecommender(
        trained_model=model,
        movie_features_repository=movie_repo,
        feature_service=feature_service,
        candidate_generator=candidate_gen,
        liked_threshold=4.0,
        num_candidates=num_candidates,
    )
    return recommender


def test_recommend_for_new_user_integration_filters_and_sorts():
    recommender = build_recommender(num_candidates=100)
    user_demographics = {"gender": "M", "age": 25, "occupation": 1}
    user_ratings = []
    results = recommender.recommend_for_new_user(user_demographics, user_ratings, num_recommendations=5)
    assert isinstance(results, list)
    assert 1 <= len(results) <= 5
    ids = [mid for mid, _ in results]
    assert all(isinstance(x, int) for x in ids)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_get_onboarding_movies_integration():
    recommender = build_recommender(num_candidates=100)
    res = recommender.get_onboarding_movies(num_movies=3)
    assert isinstance(res, dict)
    assert len(res.keys()) >= 1
    assert all(len(v) == 3 for v in res.values())
