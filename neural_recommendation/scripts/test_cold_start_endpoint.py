import argparse
import random
import sys
from typing import List, Optional

import requests

from neural_recommendation.applications.interfaces.schemas import RatingSchema


def create_user(
    base_url: str, username: str, email: str, password: str, age: int, gender: str, occupation: int, timeout: float
) -> tuple[bool, Optional[int]]:
    url = base_url.rstrip("/") + "/users/"
    payload = {
        "username": username,
        "email": email,
        "password": password,
        "age": age,
        "gender": gender,
        "occupation": occupation,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    return r.json()


def login(base_url: str, email: str, password: str, timeout: float) -> str | None:
    url = base_url.rstrip("/") + "/auth/token"
    data = {"username": email, "password": password}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(url, data=data, headers=headers, timeout=timeout)
    if r.status_code == 200:
        return r.json().get("access_token")
    return None


def get_current_user_id(base_url: str, token: str, timeout: float) -> Optional[int]:
    url = base_url.rstrip("/") + "/users/me"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code == 200:
        try:
            return int(r.json().get("id"))
        except Exception:
            return None
    return None


def fetch_movies(base_url: str, limit: int, timeout: float) -> List[dict]:
    url = base_url.rstrip("/") + f"/movies?offset=0&limit={limit}"
    r = requests.get(url, timeout=timeout)
    if r.status_code == 200:
        return r.json().get("movies", [])
    return []


def fetch_onboarding_movies(base_url: str, num_recommendations: int, timeout: float) -> List[dict]:
    url = base_url.rstrip("/") + "/recommendations/onboarding-movies"
    payload = {"num_recommendations": num_recommendations}
    r = requests.get(url, json=payload, timeout=timeout)
    print(r.json())
    if r.status_code == 200:
        return r.json().get("recommendations", [])
    return []


def create_rating(base_url: str, ratings: List[RatingSchema], timeout: float) -> bool:
    url = base_url.rstrip("/") + "/ratings/"
    # headers = {"Authorization": f"Bearer {token}"}
    payload = [rating.model_dump(mode="json") for rating in ratings]
    r = requests.post(url, json=payload, timeout=timeout)
    return r.json()


def recommend_cold_start(base_url: str, user_id: int, num_recommendations: int, timeout: float) -> Optional[dict]:
    url = base_url.rstrip("/") + "/recommendations/cold-start"
    payload = {"user_id": user_id, "num_recommendations": num_recommendations}
    r = requests.post(url, json=payload, timeout=timeout)
    return r.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://localhost:8000")
    parser.add_argument("--num_users", type=int, default=2)
    parser.add_argument("--ratings_per_user", type=int, default=5)
    parser.add_argument("--movies_fetch_limit", type=int, default=500)
    parser.add_argument("--password", type=str, default="password123")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--num_movies", type=int, default=5)
    args = parser.parse_args()

    onboarding_movies_by_genres = fetch_onboarding_movies(args.base_url, args.num_movies, args.timeout)
    movies = []
    for _, movie_ids in onboarding_movies_by_genres.items():
        movies.extend(movie_ids[:2])
    if not movies:
        print("No movies available to rate", file=sys.stderr)
        sys.exit(1)

    for i in range(1, args.num_users + 1):
        email_random_part = random.randint(1000, 9999)
        username = f"user_{i}_{email_random_part}"
        email = f"user_{i}_{email_random_part}@example.google.com"
        age = random.choice([18, 25, 35, 45])
        gender = random.choice(["M", "F"])
        occupation = random.randint(0, 20)
        print("** Creating user...")
        created = create_user(args.base_url, username, email, args.password, age, gender, occupation, args.timeout)
        print("Created user:", created)
        try:
            user_id = created["id"]
            picks = random.sample(movies, k=min(args.ratings_per_user, len(movies)))
            ratings = [
                RatingSchema(user_id=created["id"], movie_id=m["movie_id"], rating=random.choice([4.0, 4.5, 5.0]))
                for m in picks
            ]
            print("** Creating ratings...")
            created_rating = create_rating(args.base_url, ratings, args.timeout)
            print("Created ratings:", created_rating)
            print("Rated movies:")
            for r, m in zip(ratings, picks):
                genres = m.get("genres", [])
                if isinstance(genres, list):
                    genres_str = ", ".join(genres)
                else:
                    genres_str = str(genres)
                print(f"- {m['title']} (id={m['movie_id']}) [{genres_str}]: {r.rating}")
            print("** Creating recommendations...")

            cs = recommend_cold_start(args.base_url, user_id, num_recommendations=10, timeout=args.timeout)
            print("Recommendations:")
            print(cs)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
