import argparse
import sys
from typing import List

import pandas as pd
import requests


def read_movies(path: str) -> List[dict]:
    df = pd.read_csv(path, sep="::", header=None, engine="python", names=["movie_id", "title", "genres"])
    df["genres"] = df["genres"].fillna("").apply(lambda g: g.split("|") if g else [])
    return df[["title", "genres"]].to_dict(orient="records")


def insert_movies(base_url: str, movies: List[dict], dry_run: bool, limit: int, timeout: float) -> None:
    url = base_url.rstrip("/") + "/movies/"
    count = 0
    for payload in movies:
        if limit and count >= limit:
            break
        if dry_run:
            count += 1
            continue
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code in (200, 201):
                count += 1
                continue
            if r.status_code == 409:
                count += 1
                continue
            r.raise_for_status()
        except Exception:
            print(f"Failed to insert movie: {payload}", file=sys.stderr)
            continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://localhost:8000")
    parser.add_argument("--movies_path", type=str, default="data/ml-1m/movies.dat")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    try:
        rows = read_movies(args.movies_path)
    except Exception as e:
        print(f"Failed to read movies: {e}", file=sys.stderr)
        sys.exit(1)
    insert_movies(args.base_url, rows, args.dry_run, args.limit, args.timeout)


if __name__ == "__main__":
    main()


