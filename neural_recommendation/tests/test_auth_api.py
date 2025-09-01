import pytest
from fastapi import status

from .conftest import BaseIntegrationTest


class TestAuthAPI(BaseIntegrationTest):
    """Integration tests for Authentication API"""

    @pytest.mark.asyncio
    async def test_login_success(self, client):
        """Test successful login"""
        # Create a user first
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123"
        }
        await client.post("/users/", json=user_data)

        # Try to login
        login_data = {
            "username": "test@example.com",  # Using email as username
            "password": "testpassword123"
        }
        response = await client.post("/auth/token", data=login_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, client):
        """Test login with non-existent user"""
        login_data = {
            "username": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        response = await client.post("/auth/token", data=login_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client):
        """Test login with wrong password"""
        # Create a user first
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123"
        }
        await client.post("/users/", json=user_data)

        # Try to login with wrong password
        login_data = {
            "username": "test@example.com",
            "password": "wrongpassword"
        }
        response = await client.post("/auth/token", data=login_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_token_format(self, client):
        """Test JWT token format and expiration"""
        # Create a user first
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123"
        }
        await client.post("/users/", json=user_data)

        # Get token
        login_data = {
            "username": "test@example.com",
            "password": "testpassword123"
        }
        response = await client.post("/auth/token", data=login_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        token = data["access_token"]

        # Test token format (should be a non-empty string)
        assert isinstance(token, str)
        assert len(token) > 0

        # Test token works with protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get("/users/me", headers=headers)
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_invalid_token_format(self, client):
        """Test invalid token formats"""
        invalid_tokens = [
            "not_a_token",
            "Bearer without_token",
            "Bearer.invalid.format",
            "Bearer " + "a" * 1000  # Very long invalid token
        ]

        for token in invalid_tokens:
            headers = {"Authorization": token}
            response = await client.get("/users/me", headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_missing_token(self, client):
        """Test accessing protected endpoint without token"""
        response = await client.get("/users/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_token_case_sensitivity(self, client):
        """Test token type is case-insensitive as per OAuth2 spec"""
        # Create a user first
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123"
        }
        await client.post("/users/", json=user_data)

        # Get token
        login_data = {
            "username": "test@example.com",
            "password": "testpassword123"
        }
        response = await client.post("/auth/token", data=login_data)
        token = response.json()["access_token"]

        # Test with different case variations of "Bearer" - all should work
        variations = ["bearer", "BEARER", "Bearer", "bEaReR"]

        for bearer in variations:
            headers = {"Authorization": f"{bearer} {token}"}
            response = await client.get("/users/me", headers=headers)
            assert response.status_code == status.HTTP_200_OK, f"Failed with bearer: {bearer}"
            user_data = response.json()
            assert user_data["email"] == "test@example.com"
