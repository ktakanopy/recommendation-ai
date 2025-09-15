from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from neural_recommendation.infrastructure.config.settings import Settings

_engine: Optional[AsyncEngine] = None


def set_engine(engine: AsyncEngine) -> None:
    global _engine
    _engine = engine


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = Settings()
        _engine = create_async_engine(settings.DATABASE_URL)
    return _engine


async def get_session():
    engine = get_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session


async def dispose_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
