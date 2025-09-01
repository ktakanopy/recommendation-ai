import uuid
from datetime import datetime
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from neural_recommendation.applications.services.recommendation_dto_mapper import RecommendationDtoMapper
from neural_recommendation.domain.ports.services.recommendation_application_service_port import (
    RecommendationApplicationServicePort,
)
from neural_recommendation.infrastructure.config.dependencies import get_recommendation_service

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

    user_id: int  # ID of the created user
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


@router.post("/training-user", response_model=RecommendationResultResponse)
async def get_recommendations_for_training_user(
    request: RecommendationRequest,
    recommendation_service: Annotated[RecommendationApplicationServicePort, Depends(get_recommendation_service)],
):
    """Generate recommendations for an existing user from the training dataaset"""

    try:
        result = recommendation_service.generate_recommendations_for_training_user(
            user_id=request.user_id,
            user_age=request.user_age,
            gender=request.gender,
            num_recommendations=request.num_recommendations,
        )

        # Convert domain model to response model using mapper
        response_dict = RecommendationDtoMapper.to_recommendation_result_response_dict(result)

        return RecommendationResultResponse(**response_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@router.post("/cold-start", response_model=RecommendationResultResponse)
async def get_recommendations_cold_start(
    request: NewUserRecommendationRequest,
    recommendation_service: Annotated[RecommendationApplicationServicePort, Depends(get_recommendation_service)],
):
    """Generate recommendations for a new user (cold-start)"""
    try:
        result = recommendation_service.generate_recommendations_cold_start(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
        )

        # Convert domain model to response model using mapper
        response_dict = RecommendationDtoMapper.to_recommendation_result_response_dict(result)

        return RecommendationResultResponse(**response_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@router.get("/health")
async def recommendation_health_check():
    """Health check endpoint for recommendation service"""
    return {"status": "healthy", "service": "recommendation-engine", "message": "Recommendation service is operational"}
