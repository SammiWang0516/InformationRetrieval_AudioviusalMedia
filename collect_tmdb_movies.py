#!/usr/bin/env python3
"""Collect movies with usable English and Chinese metadata from TMDB.

The script searches one release year at a time. Within each year, TMDB sorts
movies by popularity. A movie is accepted only when it has:

1. A usable English title and overview.
2. A zh-CN or zh-TW translation with both a title and overview. (for future multi-lingual queries)
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


API_BASE_URL = "https://api.themoviedb.org/3"

# Used when --target is not provided on the command line.
# hit 5000 movies since conditions are already set
DEFAULT_TARGET = 5000

# TMDB returns at most 500 pages for one discover query.
TMDB_MAX_PAGE = 500

# These related sections are added to each movie detail response.
# in the details folder under raw folder
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

# add arguments after the execution of script
def parse_args() -> argparse.Namespace:
    """Read command-line options and return them in one object."""
    script_folder = Path(__file__).resolve().parent
    current_year = date.today().year

    parser = argparse.ArgumentParser(
        description=(
            "Collect movies with English metadata and complete zh-CN or "
            "zh-TW translations."
        )
    )

    # Example: --target 1000 becomes args.target == 1000.
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Number of qualifying movies to collect (default: 5000).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=current_year,
        help=f"Newest release year to search (default: {current_year}).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=1900,
        help="Oldest release year to search (default: 1900).",
    )
    parser.add_argument(
        "--cutoff-date",
        default=date.today().isoformat(),
        help="Do not include releases after this YYYY-MM-DD date.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds to wait after each API request (default: 0.25).",
    )
    parser.add_argument(
        "--max-pages-per-year",
        type=int,
        default=TMDB_MAX_PAGE,
        help="Maximum discover pages to scan in each year (default: 500).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_folder / "data" / "bilingual_movies",
        help="Output folder (default: data/bilingual_movies).",
    )

    # Namespace is a simple object containing values such as args.target.
    return parser.parse_args()

# function checking whether the input arguments are correct or not
def validate_args(args: argparse.Namespace) -> None:
    """Check that command-line values are valid."""
    if args.target < 1:
        raise ValueError("--target must be at least 1.")
    if args.start_year < args.end_year:
        raise ValueError("--start-year must be greater than or equal to --end-year.")
    if not 1 <= args.max_pages_per_year <= TMDB_MAX_PAGE:
        raise ValueError("--max-pages-per-year must be between 1 and 500.")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    try:
        cutoff = date.fromisoformat(args.cutoff_date)
    except ValueError as error:
        raise ValueError("--cutoff-date must use YYYY-MM-DD format.") from error
    if args.start_year > cutoff.year:
        raise ValueError("--start-year cannot be later than the cutoff year.")

# load the api key or token from tmdb website
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
            "User-Agent": "UMich-SI650-TMDB-Collector/2.0",
        }
    )
    if read_token:
        session.headers["Authorization"] = f"Bearer {read_token}"
    return session

# request the json file from tmdb server
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

        # 429 means rate limited. 5xx means a temporary server problem.
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


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON without leaving a partial file if interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(path)


def has_text(value: Any) -> bool:
    """Return True when a value contains non-whitespace text."""
    return isinstance(value, str) and bool(value.strip())


def complete_translation_regions(
    movie: dict[str, Any], language: str, allowed_regions: set[str]
) -> list[str]:
    """Return regions having both a translated title and overview."""
    matching_regions: list[str] = []
    translations = movie.get("translations", {}).get("translations", [])

    for translation in translations:
        if translation.get("iso_639_1") != language:
            continue

        region = translation.get("iso_3166_1")
        if region not in allowed_regions:
            continue

        translated_data = translation.get("data", {})
        has_title = has_text(translated_data.get("title"))
        has_overview = has_text(translated_data.get("overview"))

        if has_title and has_overview:
            matching_regions.append(region)

    return sorted(set(matching_regions))


def evaluate_movie(movie: dict[str, Any]) -> tuple[bool, str, list[str]]:
    """Decide whether a movie has the required bilingual metadata."""
    # The main response was requested with language=en-US. These are the
    # English fields that will later form the retrieval document.
    if not has_text(movie.get("title")):
        return False, "missing_english_title", []
    if not has_text(movie.get("overview")):
        return False, "missing_english_overview", []

    # A movie has one original_language, but it can have many translations.
    # Chinese uses language code zh plus a region such as CN or TW.
    chinese_regions = complete_translation_regions(movie, "zh", {"CN", "TW"})
    if not chinese_regions:
        return False, "missing_complete_zh_cn_or_zh_tw_translation", []

    return True, "accepted", chinese_regions


def year_date_range(year: int, cutoff: date) -> tuple[str, str] | None:
    """Return the searchable date range for one release year."""
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), cutoff)
    if start > end:
        return None
    return start.isoformat(), end.isoformat()

# where the magic starts and grab json file with conditions listed in the very beginning of this script
def collect(args: argparse.Namespace) -> int:
    """Collect candidates until the requested bilingual target is reached."""
    validate_args(args)
    cutoff = date.fromisoformat(args.cutoff_date)
    read_token, api_key = load_credentials()
    session = create_session(read_token)

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    discover_dir = raw_dir / "discover" / "movie"
    details_dir = raw_dir / "details" / "movie"
    accepted_path = output_dir / "accepted_movies.json"
    summary_path = output_dir / "collection_summary.json"

    # parents=True creates missing parents. exist_ok=True accepts folders that
    # already exist, which makes the collector safe to resume.
    output_dir.mkdir(parents=True, exist_ok=True)
    discover_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    accepted_movies: list[dict[str, Any]] = []
    accepted_ids: set[int] = set()
    seen_candidate_ids: set[int] = set()
    rejection_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    pages_scanned_by_year: dict[str, int] = {}
    started_at = datetime.now(timezone.utc)

    print(f"Output directory: {output_dir}")
    print(
        f"Target: {args.target} movies with English metadata and a complete "
        f"zh-CN or zh-TW translation."
    )

    try:
        # range needs end_year - 1 because its stopping value is excluded.
        for year in range(args.start_year, args.end_year - 1, -1):
            if len(accepted_movies) >= args.target:
                break

            searchable_range = year_date_range(year, cutoff)
            if searchable_range is None:
                continue
            start_date, end_date = searchable_range

            print(f"\nSearching {year}, sorted by popularity...")
            year_folder = discover_dir / f"year_{year}"
            year_folder.mkdir(parents=True, exist_ok=True)

            page_number = 1
            total_pages = args.max_pages_per_year

            while page_number <= min(total_pages, args.max_pages_per_year):
                if len(accepted_movies) >= args.target:
                    break

                discover_path = year_folder / f"page_{page_number:04d}.json"
                discover_payload = read_valid_json(discover_path)

                if discover_payload is None:
                    discover_payload = request_json(
                        session,
                        "discover/movie",
                        {
                            "page": page_number,
                            "sort_by": "popularity.desc",
                            "primary_release_date.gte": start_date,
                            "primary_release_date.lte": end_date,
                            "include_adult": "false",
                            "include_video": "false",
                            "language": "en-US",
                        },
                        api_key,
                        args.delay,
                    )
                    write_json_atomic(discover_path, discover_payload)

                # TMDB reports how many pages exist for this year's query.
                total_pages = min(
                    int(discover_payload.get("total_pages", 1)),
                    args.max_pages_per_year,
                    TMDB_MAX_PAGE,
                )
                pages_scanned_by_year[str(year)] = page_number
                results = discover_payload.get("results", [])

                if not results:
                    break

                print(
                    f"  Page {page_number}/{total_pages}: "
                    f"{len(accepted_movies)}/{args.target} accepted"
                )

                for candidate in results:
                    if len(accepted_movies) >= args.target:
                        break

                    movie_id = candidate.get("id")
                    if not isinstance(movie_id, int):
                        continue
                    if movie_id in seen_candidate_ids:
                        continue
                    seen_candidate_ids.add(movie_id)

                    detail_path = details_dir / f"{movie_id}.json"
                    movie = read_valid_json(detail_path)

                    if movie is None:
                        try:
                            movie = request_json(
                                session,
                                f"movie/{movie_id}",
                                {
                                    "language": "en-US",
                                    "append_to_response": ",".join(
                                        MOVIE_APPEND_FIELDS
                                    ),
                                    "include_image_language": "en,zh,null",
                                },
                                api_key,
                                args.delay,
                            )
                            write_json_atomic(detail_path, movie)
                        except Exception as error:
                            failures.append(
                                {"movie_id": movie_id, "error": str(error)}
                            )
                            print(
                                f"    Failed movie {movie_id}: {error}",
                                file=sys.stderr,
                            )
                            continue

                    accepted, reason, chinese_regions = evaluate_movie(movie)
                    if not accepted:
                        rejection_counts[reason] += 1
                        continue
                    if movie_id in accepted_ids:
                        continue

                    accepted_ids.add(movie_id)
                    accepted_movies.append(
                        {
                            "tmdb_id": movie_id,
                            "title": movie.get("title"),
                            "release_date": movie.get("release_date"),
                            "original_language": movie.get("original_language"),
                            "chinese_translation_regions": chinese_regions,
                            "raw_detail_file": str(
                                detail_path.relative_to(output_dir).as_posix()
                            ),
                        }
                    )
                    print(
                        f"    ACCEPTED [{len(accepted_movies)}/{args.target}] "
                        f"{movie.get('title')} ({', '.join(chinese_regions)})"
                    )

                page_number += 1

    except KeyboardInterrupt:
        print("\nCollection interrupted. Saved raw files will be reused next time.")
    finally:
        finished_at = datetime.now(timezone.utc)

        # This compact file identifies the qualifying records. Full metadata
        # remains in the raw detail files referenced by raw_detail_file.
        write_json_atomic(accepted_path, accepted_movies)

        summary = {
            "collection_type": "tmdb_english_chinese_bilingual_movies",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "target_accepted_movies": args.target,
            "accepted_movies": len(accepted_movies),
            "unique_candidates_checked": len(seen_candidate_ids),
            "start_year": args.start_year,                          # set the start year in argument or default
            "end_year": args.end_year,
            "cutoff_date": args.cutoff_date,
            "sort_by": "popularity.desc",                           # each year sorted by popularity
            "required_chinese_regions": ["CN", "TW"],               # grab only taiwan or chinese translation
            "required_translation_fields": ["title", "overview"],   # only title and overview need translation for now
            "pages_scanned_by_year": pages_scanned_by_year,         # each year in one page
            "rejection_counts": dict(rejection_counts),
            "failure_count": len(failures),
            "failures": failures,
        }
        write_json_atomic(summary_path, summary)

        print(f"\nAccepted-movie list: {accepted_path}")
        print(f"Collection summary: {summary_path}")

    if len(accepted_movies) >= args.target:
        return 0

    print(
        f"Only {len(accepted_movies)} qualifying movies were found in the "
        f"requested year range.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    """Run the collector and print readable errors."""
    try:
        args = parse_args()
        return collect(args)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
