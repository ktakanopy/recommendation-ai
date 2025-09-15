from http import HTTPStatus

from contextlib import asynccontextmanager
from fastapi import FastAPI

from neural_recommendation.applications.interfaces.dtos.message import Message
from neural_recommendation.infrastructure.logging.logger import setup_logging
from neural_recommendation.infrastructure.persistence.database import dispose_engine, get_engine, set_engine
from neural_recommendation.presentation.routers import auth, movies, ratings, recommendations, users

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    set_engine(engine)
    try:
        yield
    finally:
        await dispose_engine()


app = FastAPI(lifespan=lifespan)

app.include_router(users.router)
app.include_router(auth.router)
app.include_router(recommendations.router)
app.include_router(ratings.router)
app.include_router(movies.router)


@app.get("/", status_code=HTTPStatus.OK, response_model=Message)
def read_root():
    return {"message": "Olá Mundo!"}
