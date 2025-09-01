from contextlib import contextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import pytest_asyncio
import torch
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pwdlib import PasswordHash
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.postgres import PostgresContainer

from neural_recommendation.app import app
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.domain.ports.services.auth_service import AuthService
from neural_recommendation.infrastructure.persistence.database import get_session
from neural_recommendation.infrastructure.persistence.models import User, table_registry

pwd_context = PasswordHash.recommended()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_user_features():
    return {
        "user_id": [1, 2, 3],
        "age": [25, 30, 35],
        "gender": ["M", "F", "M"],
        "occupation": ["Engineer", "Teacher", "Doctor"],
    }


@pytest.fixture
def sample_item_features():
    return {
        "item_id": [101, 102, 103],
        "title": ["Movie A", "Movie B", "Movie C"],
        "genres": ["Action", "Comedy", "Drama"],
        "year": [2020, 2021, 2022],
    }


@pytest.fixture
def sample_ratings():
    return {
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [101, 102, 101, 103, 102],
        "rating": [4.0, 3.5, 5.0, 2.0, 4.5],
        "timestamp": [1640995200, 1641081600, 1641168000, 1641254400, 1641340800],
    }


@pytest.fixture
def mock_model():
    model = Mock()
    model.user_tower = Mock()
    model.item_tower = Mock()
    model.user_tower.return_value = torch.randn(3, 64)
    model.item_tower.return_value = torch.randn(3, 64)
    model.eval = Mock()
    return model


@pytest.fixture
def mock_data_manager():
    data_manager = Mock()
    data_manager.load_user_features.return_value = {
        "user_id": [1, 2, 3],
        "age": [25, 30, 35],
        "gender": ["M", "F", "M"],
    }
    data_manager.load_item_features.return_value = {
        "item_id": [101, 102, 103],
        "title": ["Movie A", "Movie B", "Movie C"],
        "genres": ["Action", "Comedy", "Drama"],
    }
    data_manager.load_ratings.return_value = {
        "user_id": [1, 1, 2],
        "item_id": [101, 102, 101],
        "rating": [4.0, 3.5, 5.0],
    }
    return data_manager


@pytest.fixture
def mock_model_inference_manager():
    manager = Mock()
    manager.load_model.return_value = None
    manager.get_user_embeddings.return_value = torch.randn(1, 64)
    manager.get_item_embeddings.return_value = torch.randn(100, 64)
    manager.compute_similarities.return_value = torch.randn(100)
    return manager


class BaseIntegrationTest:
    """Base class for integration tests with common setup"""

    @pytest_asyncio.fixture
    async def postgres_engine(self):
        """Create test database engine"""
        with PostgresContainer("postgres:16", driver="psycopg") as postgres:
            engine = create_async_engine(postgres.get_connection_url())

            async with engine.begin() as conn:
                await conn.run_sync(table_registry.metadata.create_all)

            yield engine

            async with engine.begin() as conn:
                await conn.run_sync(table_registry.metadata.drop_all)
            await engine.dispose()

    @pytest_asyncio.fixture
    async def test_session(self, postgres_engine):
        """Create test database session"""
        async with AsyncSession(postgres_engine, expire_on_commit=False) as session:
            yield session

    @pytest_asyncio.fixture
    async def client(self, test_session):
        """Create test HTTP client with database override"""

        async def override_get_session():
            yield test_session

        app.dependency_overrides[get_session] = override_get_session

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

        app.dependency_overrides.clear()


@pytest.fixture
def token(client, user):
    response = client.post(
        "/auth/token",
        data={"username": user.email, "password": user.clean_password},
    )
    return response.json()["access_token"]


@pytest_asyncio.fixture
async def user(session):
    user = User(
        username="Teste",
        email="teste@test.com",
        password=pwd_context.hash("testtest"),
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    # Add clean_password as a dynamic attribute for testing
    setattr(user, "clean_password", "testtest")

    return user


@pytest_asyncio.fixture
async def session():
    with PostgresContainer("postgres:16", driver="psycopg") as postgres:
        engine = create_async_engine(postgres.get_connection_url())

        async with engine.begin() as conn:
            await conn.run_sync(table_registry.metadata.create_all)

        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

        async with engine.begin() as conn:
            await conn.run_sync(table_registry.metadata.drop_all)


@contextmanager
def _mock_db_time(*, model, time=datetime(2024, 1, 1)):
    def fake_time_hook(mapper, connection, target):
        if hasattr(target, "created_at"):
            target.created_at = time

    event.listen(model, "before_insert", fake_time_hook)

    yield time

    event.remove(model, "before_insert", fake_time_hook)


@pytest.fixture
def mock_db_time():
    return _mock_db_time


# Shared fixtures for use case testing
@pytest.fixture
def mock_user_repository():
    """Mock user repository for use case te ing"""
    return AsyncMock(spec=UserRepository)


@pytest.fixture
def mock_auth_service():
    """Mock auth service for use case testing"""
    return MagicMock(spec=AuthService)
