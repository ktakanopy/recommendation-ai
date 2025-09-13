import uuid
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from neural_recommendation.applications.interfaces.dtos.filter_page import FilterPage
from neural_recommendation.applications.interfaces.dtos.message import Message
from neural_recommendation.applications.interfaces.dtos.movie import (
    MovieList,
    MoviePublic,
    MovieSchema,
)
from neural_recommendation.applications.use_cases.movie.create_movie import CreateMovieUseCase
from neural_recommendation.applications.use_cases.movie.delete_movie import DeleteMovieUseCase
from neural_recommendation.applications.use_cases.movie.get_movie import GetMovieUseCase
from neural_recommendation.applications.use_cases.movie.get_movies import GetMoviesUseCase
from neural_recommendation.applications.use_cases.movie.update_movie import UpdateMovieUseCase
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository
from neural_recommendation.infrastructure.config.dependencies import get_movie_repository

router = APIRouter(prefix="/movies", tags=["movies"])

MovieRepositoryDep = Annotated[MovieRepository, Depends(get_movie_repository)]


@router.post("/", status_code=HTTPStatus.CREATED, response_model=MoviePublic)
async def create_movie(movie: MovieSchema, movie_repository: MovieRepositoryDep):
    try:
        use_case = CreateMovieUseCase(movie_repository)
        return await use_case.execute(movie)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=str(e))


@router.get("/{movie_id}", response_model=MoviePublic)
async def read_movie(movie_id: uuid.UUID, movie_repository: MovieRepositoryDep):
    try:
        use_case = GetMovieUseCase(movie_repository)
        return await use_case.execute(movie_id)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))


@router.get("/", response_model=MovieList)
async def read_movies(filter_movies: Annotated[FilterPage, Query()], movie_repository: MovieRepositoryDep):
    use_case = GetMoviesUseCase(movie_repository)
    return await use_case.execute(filter_movies)


@router.put("/{movie_id}", response_model=MoviePublic)
async def update_movie(movie_id: uuid.UUID, movie: MovieSchema, movie_repository: MovieRepositoryDep):
    try:
        use_case = UpdateMovieUseCase(movie_repository)
        return await use_case.execute(movie_id, movie)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
        else:
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=str(e))


@router.delete("/{movie_id}", response_model=Message)
async def delete_movie(movie_id: uuid.UUID, movie_repository: MovieRepositoryDep):
    try:
        use_case = DeleteMovieUseCase(movie_repository)
        return await use_case.execute(movie_id)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(e))
