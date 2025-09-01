import pytest
from fastapi import status

from .conftest import BaseIntegrationTest


class TestUserAPI(BaseIntegrationTest):
    """Integration tests for User API endpoints"""

    @pytest.mark.asyncio
    async def test_create_user_success(self, client):
        """Test successful user creation"""
        user_data = {"username": "testuser", "email": "test@example.com", "password": "testpassword123"}

        response = await client.post("/users/", json=user_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert "id" in data
        assert "password" not in data  # Password should not be returned

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self, client):
        """Test creating user with duplicate username"""
        user_data = {"username": "duplicate", "email": "first@example.com", "password": "password123"}

        # Create first user
        response1 = await client.post("/users/", json=user_data)
        assert response1.status_code == status.HTTP_201_CREATED

        # Try to create second user with same username
        user_data["email"] = "second@example.com"
        response2 = await client.post("/users/", json=user_data)
        assert response2.status_code == status.HTTP_409_CONFLICT

    @pytest.mark.asyncio
    async def test_create_user_invalid_email(self, client):
        """Test creating user with invalid email"""
        user_data = {"username": "testuser", "email": "invalid-email", "password": "password123"}

        response = await client.post("/users/", json=user_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_get_users(self, client):
        """Test retrieving all users"""
        # Create test users
        users_data = [
            {"username": "user1", "email": "user1@example.com", "password": "pass1"},
            {"username": "user2", "email": "user2@example.com", "password": "pass2"},
        ]

        for user_data in users_data:
            await client.post("/users/", json=user_data)

        response = await client.get("/users/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "users" in data
        assert len(data["users"]) == 2
        usernames = [user["username"] for user in data["users"]]
        assert "user1" in usernames
        assert "user2" in usernames

    @pytest.mark.asyncio
    async def test_get_users_with_pagination(self, client):
        """Test retrieving users with pagination"""
        # Create test users
        for i in range(5):
            user_data = {"username": f"user{i}", "email": f"user{i}@example.com", "password": "password"}
            await client.post("/users/", json=user_data)

        # Test pagination
        response = await client.get("/users/?offset=0&limit=3")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["users"]) == 3

    @pytest.mark.asyncio
    async def test_get_users_pagination_edge_cases(self, client):
        """Test pagination edge cases"""
        # Create test users
        for i in range(3):
            user_data = {"username": f"user{i}", "email": f"user{i}@example.com", "password": "password"}
            await client.post("/users/", json=user_data)

        # Test negative offset
        response = await client.get("/users/?offset=-1")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test negative limit
        response = await client.get("/users/?limit=-1")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test offset beyond available data
        response = await client.get("/users/?offset=100")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["users"]) == 0

        # Test zero limit
        response = await client.get("/users/?limit=0")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_update_user_requires_auth(self, client):
        """Test that updating user requires authentication"""
        # Create user
        user_data = {"username": "updateuser", "email": "update@example.com", "password": "password123"}
        create_response = await client.post("/users/", json=user_data)
        created_user = create_response.json()

        # Try to update without authentication
        update_data = {"username": "updateduser", "email": "updated@example.com", "password": "newpassword"}
        response = await client.put(f"/users/{created_user['id']}", json=update_data)

        # Should require authentication
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_delete_user_requires_auth(self, client):
        """Test that deleting user requires authentication"""
        # Create user
        user_data = {"username": "deleteuser", "email": "delete@example.com", "password": "password123"}
        create_response = await client.post("/users/", json=user_data)
        created_user = create_response.json()

        # Try to delete without authentication
        response = await client.delete(f"/users/{created_user['id']}")

        # Should require authentication
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
