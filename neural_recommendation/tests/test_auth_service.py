from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.user import User as DomainUser
from neural_recommendation.infrastructure.adapters.services.jwt_auth_service import JWTAuthService
from neural_recommendation.infrastructure.config.settings import Settings
from neural_recommendation.infrastructure.persistence.models import User as SQLUser

from .conftest import BaseIntegrationTest


class TestJWTAuthService(BaseIntegrationTest):
    """Tests for JWT authentication service"""

    @pytest_asyncio.fixture
    async def test_session(self, postgres_engine):
        """Create test database session"""
        async with AsyncSession(postgres_engine, expire_on_commit=False) as session:
            yield session

    @pytest.fixture
    def test_settings(self):
        """Create test settings"""
        return Settings(
            DATABASE_URL="postgresql+psycopg://app_user:app_password@localhost:5433/app_db",
            SECRET_KEY="test-secret-key-for-testing",
            ALGORITHM="HS256",
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
        )

    @pytest.fixture
    def auth_service(self, test_session, test_settings):
        """Create auth service instance"""
        return JWTAuthService(test_session, test_settings)

    def test_hash_password(self, auth_service):
        """Test password hashing"""
        password = "testpassword123"
        hashed = auth_service.hash_password(password)

        assert hashed != password
        assert len(hashed) > 0

    def test_verify_password_correct(self, auth_service):
        """Test password verification with correct password"""
        password = "testpassword123"
        hashed = auth_service.hash_password(password)

        assert auth_service.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self, auth_service):
        """Test password verification with incorrect password"""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        hashed = auth_service.hash_password(password)

        assert auth_service.verify_password(wrong_password, hashed) is False

    def test_create_access_token(self, auth_service):
        """Test access token creation"""
        user = DomainUser(id=1, username="testuser", email="test@example.com", password_hash="hashed_password")

        token = auth_service.create_access_token(user)

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode token to verify contents
        decoded = jwt.decode(token, "test-secret-key-for-testing", algorithms=["HS256"])
        assert decoded["sub"] == "test@example.com"
        assert "exp" in decoded

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, auth_service, test_session):
        """Test getting current user with valid token"""
        # Create a user in database first
        sql_user = SQLUser(username="tokenuser", email="token@example.com", password="hashed_password")
        test_session.add(sql_user)
        await test_session.commit()
        await test_session.refresh(sql_user)

        # Create domain user and token
        domain_user = DomainUser(
            id=sql_user.id, username="tokenuser", email="token@example.com", password_hash="hashed_password"
        )

        token = auth_service.create_access_token(domain_user)

        # Get current user from token
        current_user = await auth_service.get_current_user(token)

        assert current_user is not None
        assert current_user.email == "token@example.com"
        assert current_user.username == "tokenuser"
        assert current_user.id == sql_user.id

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, auth_service):
        """Test getting current user with invalid token"""
        invalid_token = "invalid.token.here"

        current_user = await auth_service.get_current_user(invalid_token)
        assert current_user is None

    @pytest.mark.asyncio
    async def test_get_current_user_nonexistent_user(self, auth_service, test_settings):
        """Test getting current user when user doesn't exist in database"""
        # Create token for non-existent user
        fake_user = DomainUser(id=999, username="fakeuser", email="fake@example.com", password_hash="password")

        token = auth_service.create_access_token(fake_user)

        # Try to get current user
        current_user = await auth_service.get_current_user(token)
        assert current_user is None

    @pytest.mark.asyncio
    async def test_get_current_user_expired_token(self, test_session):
        """Test getting current user with expired token"""

        # Create settings with very short expiry
        expired_settings = Settings(
            DATABASE_URL="postgresql+psycopg://app_user:app_password@localhost:5433/app_db",
            SECRET_KEY="test-secret-key-for-testing",
            ALGORITHM="HS256",
            ACCESS_TOKEN_EXPIRE_MINUTES=1,  # 1 minute expiry
        )

        auth_service = JWTAuthService(test_session, expired_settings)

        user = DomainUser(id=1, username="expireduser", email="expired@example.com", password_hash="password")

        # Create token that will expire soon
        auth_service.create_access_token(user)

        # Manually create an expired token by manipulating the JWT

        # Create an expired token manually
        expire = datetime.now(tz=ZoneInfo("UTC")) - timedelta(minutes=1)  # Already expired
        expired_payload = {"sub": user.email, "exp": expire}
        expired_token = jwt.encode(expired_payload, "test-secret-key-for-testing", algorithm="HS256")

        # Token should be expired, so get_current_user should return None
        current_user = await auth_service.get_current_user(expired_token)
        assert current_user is None

    def test_password_hashing_consistency(self, auth_service):
        """Test that same password produces different hashes but verifies correctly"""
        password = "testpassword123"

        hash1 = auth_service.hash_password(password)
        hash2 = auth_service.hash_password(password)

        # Hashes should be different (due to salt)
        assert hash1 != hash2

        # But both should verify correctly
        assert auth_service.verify_password(password, hash1) is True
        assert auth_service.verify_password(password, hash2) is True
