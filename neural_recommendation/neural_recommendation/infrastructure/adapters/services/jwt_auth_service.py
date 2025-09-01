from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from jwt import DecodeError, ExpiredSignatureError, decode, encode
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.user import User as DomainUser
from neural_recommendation.domain.ports.services.auth_service import AuthService
from neural_recommendation.infrastructure.config.settings import Settings
from neural_recommendation.infrastructure.persistence.models import User as SQLUser


class JWTAuthService(AuthService):
    def __init__(self, session: AsyncSession, settings: Settings):
        self.session = session
        self.settings = settings
        self.pwd_context = PasswordHash.recommended()

    def hash_password(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, user: DomainUser) -> str:
        expire = datetime.now(tz=ZoneInfo("UTC")) + timedelta(minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode = {"sub": user.email, "exp": expire}
        encoded_jwt = encode(to_encode, self.settings.SECRET_KEY, algorithm=self.settings.ALGORITHM)
        return encoded_jwt

    async def get_current_user(self, token: str) -> Optional[DomainUser]:
        try:
            payload = decode(token, self.settings.SECRET_KEY, algorithms=[self.settings.ALGORITHM])
            subject_email = payload.get("sub")

            if not subject_email:
                return None

        except (DecodeError, ExpiredSignatureError):
            return None

        sql_user = await self.session.scalar(select(SQLUser).where(SQLUser.email == subject_email))

        if not sql_user:
            return None

        return DomainUser(
            id=sql_user.id,
            username=sql_user.username,
            email=sql_user.email,
            password_hash=sql_user.password,
            created_at=sql_user.created_at,
        )
