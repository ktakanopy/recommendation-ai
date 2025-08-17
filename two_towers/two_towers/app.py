from http import HTTPStatus

from fastapi import FastAPI

from two_towers.applications.interfaces.schemas import Message
from two_towers.infrastructure.logging.logger import setup_logging
from two_towers.presentation.routers import auth, recommendations, users

setup_logging()

app = FastAPI()

app.include_router(users.router)
app.include_router(auth.router)
app.include_router(recommendations.router)


@app.get("/", status_code=HTTPStatus.OK, response_model=Message)
def read_root():
    return {"message": "Olá Mundo!"}
