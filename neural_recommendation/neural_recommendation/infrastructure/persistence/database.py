from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from neural_recommendation.infrastructure.config.settings import Settings


@lru_cache()
def get_engine():
    settings = Settings()
    return create_async_engine(settings.DATABASE_URL)


async def get_session():
    engine = get_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
