from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from neural_recommendation.infrastructure.config.settings import Settings


class _EngineStore:
    engine: Optional[AsyncEngine] = None


def set_engine(engine: AsyncEngine) -> None:
    _EngineStore.engine = engine


def get_engine() -> AsyncEngine:
    if _EngineStore.engine is None:
        settings = Settings()
        _EngineStore.engine = create_async_engine(settings.DATABASE_URL)
    return _EngineStore.engine


async def get_session():
    engine = get_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session


async def dispose_engine() -> None:
    if _EngineStore.engine is not None:
        await _EngineStore.engine.dispose()
        _EngineStore.engine = None
