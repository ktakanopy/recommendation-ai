from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from neural_recommendation.applications.interfaces.dtos.recommendation import (
    NewUserRecommendationRequest,
    OnboardingMoviesResultResponse,
    RecommendationResultResponse,
)
from neural_recommendation.domain.ports.services.recommendation_application_service_port import (
    RecommendationApplicationServicePort,
)
from neural_recommendation.infrastructure.config.dependencies import get_recommendation_service
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/cold-start", response_model=RecommendationResultResponse)
async def get_recommendations_cold_start(
    request: NewUserRecommendationRequest,
    recommendation_service: Annotated[RecommendationApplicationServicePort, Depends(get_recommendation_service)],
):
    """Generate recommendations for a new user (cold-start)"""
    try:
        result = await recommendation_service.generate_recommendations_cold_start(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
        )

        return RecommendationResultResponse(**result.model_dump())

    except ValueError as e:
        logger.warning(f"Bad request in cold-start recommendations: {e}")
        raise HTTPException(status_code=400, detail="Invalid request for recommendations")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error generating cold start recommendations")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/onboarding-movies", response_model=OnboardingMoviesResultResponse)
async def get_onboarding_movies(
    recommendation_service: Annotated[RecommendationApplicationServicePort, Depends(get_recommendation_service)],
    num_movies: int = Query(10, ge=1, le=100),
):
    """Get onboarding movies for new user"""
    try:
        result = await recommendation_service.get_onboarding_movies(num_movies=num_movies)
        return OnboardingMoviesResultResponse(**result.model_dump())
    except ValueError as e:
        logger.warning(f"Bad request in onboarding movies: {e}")
        raise HTTPException(status_code=400, detail="Invalid request for onboarding movies")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error getting onboarding movies")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def recommendation_health_check():
    """Health check endpoint for recommendation service"""
    return {"status": "healthy", "service": "recommendation-engine", "message": "Recommendation service is operational"}
