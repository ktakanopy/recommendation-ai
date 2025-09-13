from http import HTTPStatus

from fastapi import FastAPI

from neural_recommendation.applications.interfaces.dtos.message import Message
from neural_recommendation.infrastructure.logging.logger import setup_logging
from neural_recommendation.presentation.routers import auth, recommendations, users, ratings, movies

setup_logging()

app = FastAPI()

app.include_router(users.router)
app.include_router(auth.router)
app.include_router(recommendations.router)
app.include_router(ratings.router)
app.include_router(movies.router)


@app.get("/", status_code=HTTPStatus.OK, response_model=Message)
def read_root():
    return {"message": "Olá Mundo!"}
