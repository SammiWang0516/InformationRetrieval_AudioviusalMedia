#!/usr/bin/env python3
"""Verify TMDB application authentication with one read-only request."""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = "https://api.themoviedb.org/3"
TEST_MOVIE_ID = 11  # Star Wars
DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_local_credentials() -> None:
    """Load supported TMDB variables from the adjacent .env file.

    Existing environment variables take precedence over values in the file.
    """
    try:
        with open(DOTENV_PATH, encoding="utf-8") as dotenv_file:
            lines = dotenv_file.readlines()
    except FileNotFoundError:
        return

    supported_names = {"TMDB_READ_TOKEN", "TMDB_API_KEY"}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()

        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in supported_names:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value and name not in os.environ:
            os.environ[name] = value


def main() -> int:
    load_local_credentials()
    read_token = os.environ.get("TMDB_READ_TOKEN", "").strip()
    api_key = os.environ.get("TMDB_API_KEY", "").strip()

    if not read_token and not api_key:
        print(
            "Missing credential. Set TMDB_READ_TOKEN (preferred) or "
            "TMDB_API_KEY.",
            file=sys.stderr,
        )
        return 2

    query = {"language": "en-US"}
    headers = {"Accept": "application/json", "User-Agent": "SI650-TMDB-Test/1.0"}

    if read_token:
        headers["Authorization"] = f"Bearer {read_token}"
        auth_method = "API Read Access Token"
    else:
        query["api_key"] = api_key
        auth_method = "v3 API key"

    url = (
        f"{BASE_URL}/movie/{TEST_MOVIE_ID}?"
        f"{urllib.parse.urlencode(query)}"
    )
    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            payload = json.load(error)
            message = payload.get("status_message", "Unknown TMDB error")
            code = payload.get("status_code", "unknown")
        except (json.JSONDecodeError, AttributeError):
            message = error.reason
            code = "unknown"
        print(
            f"TMDB request failed: HTTP {error.code}, "
            f"TMDB code {code}: {message}",
            file=sys.stderr,
        )
        return 1
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"Could not reach TMDB: {error}", file=sys.stderr)
        return 1

    result = {
        "authentication": auth_method,
        "id": payload.get("id"),
        "title": payload.get("title"),
        "original_language": payload.get("original_language"),
        "release_date": payload.get("release_date"),
        "genres": [genre.get("name") for genre in payload.get("genres", [])],
        "poster_path": payload.get("poster_path"),
    }
    print("TMDB authentication succeeded.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
