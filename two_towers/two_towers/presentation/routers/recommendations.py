import uuid
from datetime import datetime
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from two_towers.applications.services.recommendation_dto_mapper import RecommendationDtoMapper
from two_towers.domain.ports.services.recommendation_service_port import RecommendationServicePort
from two_towers.infrastructure.config.dependencies import get_recommendation_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RatingRequest(BaseModel):
    """Request schema for rating data"""
    id: uuid.UUID
    user_id: uuid.UUID
    movie_id: uuid.UUID
    timestamp: datetime
    rating: float


class RecommendationRequest(BaseModel):
    """Request schema for existing user recommendations"""
    user_id: str
    user_age: float = 25.0
    gender: str = "M"
    num_recommendations: int = 10


class NewUserRecommendationRequest(BaseModel):
    """Request schema for new user recommendations"""
    user_age: float
    gender: str
    preferred_genres: Optional[List[str]] = None
    num_recommendations: int = 10


class RecommendationResponse(BaseModel):
    """Response schema for recommendations"""
    movie_id: int
    title: str
    genres: str
    similarity_score: float
    similarity_percentage: float


class RecommendationResultResponse(BaseModel):
    """Response schema for recommendation results"""
    user_id: str
    recommendations: List[RecommendationResponse]
    total_available_movies: int
    recommendation_count: int


class ExplanationResponse(BaseModel):
    """Response schema for recommendation explanations"""
    movie_title: str
    similarity_score: float
    similarity_percentage: float
    explanation: str
    genres: str


@router.post("/user", response_model=RecommendationResultResponse)
async def get_recommendations_for_user(
    request: RecommendationRequest,
    recommendation_service: Annotated[RecommendationServicePort, Depends(get_recommendation_service)]
):
    """Generate recommendations for an existing user"""

    try:
        # Convert RatingRequest objects to domain Rating objects
        ratings = RecommendationDtoMapper.rating_requests_to_domain(request.ratings)

        result = recommendation_service.generate_recommendations_for_existing_user(
            user_id=request.user_id,
            user_age=request.user_age,
            gender=request.gender,
            num_recommendations=request.num_recommendations
        )

        # Convert domain model to response model using mapper
        response_dict = RecommendationDtoMapper.to_recommendation_result_response_dict(result)

        return RecommendationResultResponse(**response_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@router.post("/new-user", response_model=RecommendationResultResponse)
async def get_recommendations_for_new_user(
    request: NewUserRecommendationRequest,
    recommendation_service: Annotated[RecommendationServicePort, Depends(get_recommendation_service)]
):
    """Generate recommendations for a new user based on demographics"""

    try:
        result = recommendation_service.generate_recommendations_for_new_user(
            user_age=request.user_age,
            gender=request.gender,
            preferred_genres=request.preferred_genres,
            num_recommendations=request.num_recommendations
        )

        # Convert domain model to response model using mapper
        response_dict = RecommendationDtoMapper.to_recommendation_result_response_dict(result)

        return RecommendationResultResponse(**response_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@router.get("/explain")
async def explain_recommendation(
    user_id: str = Query(..., description="User ID"),
    movie_title: str = Query(..., description="Movie title to explain"),
    user_age: float = Query(25.0, description="User's age"),
    gender: str = Query("M", description="User's gender (M/F)"),
    recommendation_service: Annotated[RecommendationServicePort, Depends(get_recommendation_service)] = None
) -> ExplanationResponse:
    """Explain why a specific movie was recommended for a user"""

    try:
        explanation = recommendation_service.explain_recommendation(
            user_id=user_id,
            movie_title=movie_title,
            user_age=user_age,
            gender=gender
        )

        # Convert using mapper
        response_dict = RecommendationDtoMapper.to_explanation_response_dict(explanation)

        return ExplanationResponse(**response_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining recommendation: {str(e)}")


@router.get("/health")
async def recommendation_health_check():
    """Health check endpoint for recommendation service"""
    return {
        "status": "healthy",
        "service": "recommendation-engine",
        "message": "Recommendation service is operational"
    }
