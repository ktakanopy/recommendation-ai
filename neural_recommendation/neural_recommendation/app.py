from http import HTTPStatus

from fastapi import FastAPI

from neural_recommendation.applications.interfaces.schemas import Message
from neural_recommendation.infrastructure.logging.logger import setup_logging
from neural_recommendation.presentation.routers import auth, recommendations, users

setup_logging()

app = FastAPI()

app.include_router(users.router)
app.include_router(auth.router)
app.include_router(recommendations.router)


@app.get("/", status_code=HTTPStatus.OK, response_model=Message)
def read_root():
    return {"message": "Olá Mundo!"}
