import argparse

import requests


def delete_all_users(base_url: str, timeout: float) -> tuple[int, dict]:
    url = base_url.rstrip("/") + "/users"
    r = requests.delete(url, timeout=timeout)
    return r.status_code, r.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    status, body = delete_all_users(args.base_url, args.timeout)
    print(status)
    print(body)


if __name__ == "__main__":
    main()
