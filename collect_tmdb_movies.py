#!/usr/bin/env python3
"""Collect recent TMDB movie metadata without cleaning or downloading images.

The collector works in two steps:
1. Page through /discover/movie from newest to oldest.
2. Fetch and save a complete details response for each unique movie ID.

Every response is saved immediately, so the script can be stopped and resumed.
Existing valid JSON files are reused instead of downloaded again.
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


API_BASE_URL = "https://api.themoviedb.org/3"
# This is used when the user does not provide --target on the command line.
DEFAULT_TARGET = 6000
RESULTS_PER_PAGE = 20
TMDB_MAX_PAGE = 500

MOVIE_APPEND_FIELDS = (
    "credits",
    "keywords",
    "translations",
    "alternative_titles",
    "external_ids",
    "images",
    "release_dates",
    "videos",
)


def parse_args() -> argparse.Namespace:
    """Read command-line options and return them in one object."""
    script_folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Save recent TMDB movie metadata from newest to oldest."
    )

    # Example: --target 4000 becomes args.target == 4000.
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Number of unique movie detail records to save (default: 6000).",
    )
    parser.add_argument(
        "--cutoff-date",
        default=date.today().isoformat(),
        help="Latest allowed primary release date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="Language for the main movie details response (default: en-US).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds to wait after each API request (default: 0.25).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=TMDB_MAX_PAGE,
        help="Maximum discover pages to inspect (default: 500).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_folder / "data",
        help="Collection output directory (default: project_proposal/data).",
    )

    # argparse.Namespace is a simple object holding values such as
    # args.target, args.language, and args.output_dir.
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Check that command-line values are valid."""
    if not 1 <= args.target <= RESULTS_PER_PAGE * TMDB_MAX_PAGE:
        raise ValueError("--target must be between 1 and 10000.")
    if not 1 <= args.max_pages <= TMDB_MAX_PAGE:
        raise ValueError("--max-pages must be between 1 and 500.")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    try:
        date.fromisoformat(args.cutoff_date)
    except ValueError as error:
        raise ValueError("--cutoff-date must use YYYY-MM-DD format.") from error


def load_credentials() -> tuple[str | None, str | None]:
    """Load a TMDB token or API key from the adjacent .env file."""
    dotenv_path = Path(__file__).resolve().with_name(".env")
    load_dotenv(dotenv_path)

    read_token = os.getenv("TMDB_READ_TOKEN", "").strip() or None
    api_key = os.getenv("TMDB_API_KEY", "").strip() or None
    if not read_token and not api_key:
        raise RuntimeError(
            f"No TMDB credential found. Add TMDB_READ_TOKEN (preferred) "
            f"or TMDB_API_KEY to {dotenv_path}."
        )
    return read_token, api_key


def create_session(read_token: str | None) -> requests.Session:
    """Create a reusable HTTP session for TMDB requests."""
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "UMich-SI650-TMDB-Collector/1.0",
        }
    )
    if read_token:
        session.headers["Authorization"] = f"Bearer {read_token}"
    return session


def request_json(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    api_key: str | None,
    delay: float,
    max_attempts: int = 6,
) -> dict[str, Any]:
    """Request JSON from TMDB and retry temporary failures."""
    request_params = dict(params)
    if "Authorization" not in session.headers and api_key:
        request_params["api_key"] = api_key

    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, params=request_params, timeout=30)
        except requests.RequestException as error:
            if attempt == max_attempts:
                raise RuntimeError(f"Request failed after {attempt} attempts: {error}")
            wait_seconds = min(2 ** (attempt - 1), 30)
            print(f"Network error; retrying in {wait_seconds}s: {error}")
            time.sleep(wait_seconds)
            continue

        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == max_attempts:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else 2 ** (attempt - 1)
            except ValueError:
                wait_seconds = 2 ** (attempt - 1)
            wait_seconds = min(max(wait_seconds, 1), 60)
            print(
                f"TMDB returned HTTP {response.status_code}; "
                f"retrying in {wait_seconds:.1f}s."
            )
            time.sleep(wait_seconds)
            continue

        if not response.ok:
            try:
                message = response.json().get("status_message", response.text)
            except (ValueError, AttributeError):
                message = response.text
            raise RuntimeError(
                f"TMDB request failed with HTTP {response.status_code}: {message}"
            )

        payload = response.json()
        if delay:
            time.sleep(delay)
        return payload

    raise RuntimeError("Request retry loop ended unexpectedly.")


def read_valid_json(path: Path) -> dict[str, Any] | None:
    """Read an existing JSON object, or return None if unusable."""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON safely so an interruption does not leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(path)


def collect(args: argparse.Namespace) -> int:
    """Collect and save raw discovery pages and movie details."""
    validate_args(args)
    read_token, api_key = load_credentials()
    session = create_session(read_token)

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    discover_dir = raw_dir / "discover" / "movie"
    details_dir = raw_dir / "details" / "movie"
    summary_path = output_dir / "collection_summary.json"

    # exist_ok=True means an existing folder is accepted. parents=True creates
    # missing parent folders such as data/raw at the same time.
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    discover_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()
    saved_ids: list[int] = []
    failures: list[dict[str, Any]] = []
    pages_examined = 0
    started_at = datetime.now(timezone.utc)

    print(f"Output directory: {output_dir}")
    print(
        f"Collecting {args.target} movies released on or before "
        f"{args.cutoff_date}, newest first."
    )

    try:
        for page_number in range(1, args.max_pages + 1):
            if len(saved_ids) >= args.target:
                break

            discover_path = discover_dir / f"page_{page_number:04d}.json"
            discover_payload = read_valid_json(discover_path)
            if discover_payload is None:
                discover_payload = request_json(
                    session,
                    "discover/movie",
                    {
                        "page": page_number,
                        "sort_by": "primary_release_date.desc",
                        "primary_release_date.lte": args.cutoff_date,
                        "include_adult": "false",
                        "include_video": "false",
                        "language": args.language,
                    },
                    api_key,
                    args.delay,
                )
                write_json_atomic(discover_path, discover_payload)

            pages_examined = page_number
            results = discover_payload.get("results", [])
            if not results:
                print(f"No results on discover page {page_number}; stopping.")
                break

            print(
                f"Discover page {page_number}: {len(results)} results; "
                f"{len(saved_ids)}/{args.target} details saved."
            )

            for item in results:
                if len(saved_ids) >= args.target:
                    break

                movie_id = item.get("id")
                if not isinstance(movie_id, int) or movie_id in seen_ids:
                    continue
                seen_ids.add(movie_id)

                detail_path = details_dir / f"{movie_id}.json"
                detail_payload = read_valid_json(detail_path)
                if detail_payload is None:
                    try:
                        detail_payload = request_json(
                            session,
                            f"movie/{movie_id}",
                            {
                                "language": args.language,
                                "append_to_response": ",".join(MOVIE_APPEND_FIELDS),
                                "include_image_language": "en,null",
                            },
                            api_key,
                            args.delay,
                        )
                        write_json_atomic(detail_path, detail_payload)
                    except Exception as error:
                        failures.append({"movie_id": movie_id, "error": str(error)})
                        print(f"  Failed movie {movie_id}: {error}", file=sys.stderr)
                        continue

                saved_ids.append(movie_id)
                title = detail_payload.get("title") or detail_payload.get("original_title")
                print(f"  [{len(saved_ids)}/{args.target}] {movie_id}: {title}")

    except KeyboardInterrupt:
        print("\nCollection interrupted. Saved files will be reused on the next run.")
    finally:
        finished_at = datetime.now(timezone.utc)
        summary = {
            "collection_type": "tmdb_recent_movies_raw_metadata",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "target_detail_records": args.target,
            "saved_detail_records_this_run": len(saved_ids),
            "unique_ids_seen_this_run": len(seen_ids),
            "discover_pages_examined_this_run": pages_examined,
            "cutoff_date": args.cutoff_date,
            "sort_by": "primary_release_date.desc",
            "language": args.language,
            "include_adult": False,
            "include_video": False,
            "appended_resources": list(MOVIE_APPEND_FIELDS),
            "failure_count": len(failures),
            "failures": failures,
        }
        write_json_atomic(summary_path, summary)
        print(f"Summary saved to: {summary_path}")

    return 0 if len(saved_ids) >= args.target else 1


def main() -> int:
    """Run the collector and convert expected errors into readable messages."""
    try:
        return collect(parse_args())
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
