#!/usr/bin/env python3
"""Convert accepted TMDB movie metadata into retrieval documents.

Input:
    data/bilingual_movies/accepted_movies.json
    data/bilingual_movies/raw/details/movie/*.json

Output:
    data/processed/documents.jsonl
    data/processed/collection_statistics.json
    data/processed/excluded_documents.jsonl

The output keeps display fields, Chinese translations, and one English
retrieval_text field. Only retrieval_text is intended for the first BM25 index.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# TMDB uses genre ID 16 for both animated movies and animated TV series.
ANIMATION_GENRE_ID = 16

# The alternative-title endpoint provides a country but not a language.
# These regions are used as a conservative source of English alternatives.
ENGLISH_TITLE_REGIONS = {"US", "GB", "AU", "NZ", "IE"}


def parse_args() -> argparse.Namespace:
    """Read command-line options and return them in one object."""
    script_folder = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Prepare normalized English movie documents from TMDB metadata."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_folder / "data" / "bilingual_movies",
        help="Folder containing accepted_movies.json and raw metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_folder / "data" / "processed",
        help="Folder for processed documents (default: data/processed).",
    )
    parser.add_argument(
        "--cast-limit",
        type=int,
        default=10,
        help="Maximum cast names stored per document (default: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N accepted movies for testing.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Check that command-line values are valid."""
    if args.cast_limit < 1:
        raise ValueError("--cast-limit must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")


def has_text(value: Any) -> bool:
    """Return True when a value contains non-whitespace text."""
    return isinstance(value, str) and bool(value.strip())


def read_json(path: Path) -> Any:
    """Read and return one JSON file."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON without leaving a partial file if interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """Write one JSON object per line without leaving a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
    temporary_path.replace(path)


def unique_text(values: list[Any]) -> list[str]:
    """Remove empty and repeated strings while keeping their order."""
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not has_text(value):
            continue
        text = value.strip()
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)

    return result


def extract_alternative_titles(movie: dict[str, Any]) -> list[str]:
    """Return likely English alternative titles."""
    titles = []
    for item in movie.get("alternative_titles", {}).get("titles", []):
        if item.get("iso_3166_1") in ENGLISH_TITLE_REGIONS:
            titles.append(item.get("title"))

    # Do not repeat the main English title as an alternative.
    main_title = (movie.get("title") or "").strip().casefold()
    return [title for title in unique_text(titles) if title.casefold() != main_title]


def extract_people(
    movie: dict[str, Any], cast_limit: int
) -> tuple[list[str], list[str], list[str]]:
    """Return top cast members, directors, and writers."""
    credits = movie.get("credits", {})

    # Cast is already ordered by billing order in most TMDB responses.
    cast_records = sorted(
        credits.get("cast", []),
        key=lambda person: person.get("order", 999999),
    )
    cast = unique_text([person.get("name") for person in cast_records])[:cast_limit]

    crew = credits.get("crew", [])
    directors = unique_text(
        [person.get("name") for person in crew if person.get("job") == "Director"]
    )
    writers = unique_text(
        [
            person.get("name")
            for person in crew
            if person.get("department") == "Writing"
        ]
    )

    return cast, directors, writers


def extract_chinese_translations(movie: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return complete zh-CN and zh-TW metadata."""
    result: dict[str, dict[str, Any]] = {}

    for translation in movie.get("translations", {}).get("translations", []):
        if translation.get("iso_639_1") != "zh":
            continue

        region = translation.get("iso_3166_1")
        if region not in {"CN", "TW"}:
            continue

        data = translation.get("data", {})
        if not has_text(data.get("title")) or not has_text(data.get("overview")):
            continue

        result[f"zh-{region}"] = {
            "title": data["title"].strip(),
            "overview": data["overview"].strip(),
            "tagline": (data.get("tagline") or "").strip(),
        }

    return result


def parse_release_year(release_date: Any) -> int | None:
    """Return the four-digit year from a TMDB release date."""
    if not has_text(release_date) or len(release_date) < 4:
        return None
    try:
        return int(release_date[:4])
    except ValueError:
        return None


def build_retrieval_text(document: dict[str, Any]) -> str:
    """Combine selected English fields into text for BM25."""
    sections: list[str] = []

    sections.append(document["title_en"])

    if document["alternative_titles_en"]:
        sections.append(
            "Alternative titles: " + ", ".join(document["alternative_titles_en"])
        )
    if document["overview_en"]:
        sections.append(document["overview_en"])
    if document["tagline_en"]:
        sections.append(document["tagline_en"])
    if document["genres"]:
        sections.append("Genres: " + ", ".join(document["genres"]))
    if document["keywords"]:
        sections.append("Keywords: " + ", ".join(document["keywords"]))
    if document["cast"]:
        sections.append("Cast: " + ", ".join(document["cast"]))
    if document["directors"]:
        sections.append("Directors: " + ", ".join(document["directors"]))
    if document["writers"]:
        sections.append("Writers: " + ", ".join(document["writers"]))
    if document["release_year"] is not None:
        sections.append(f"Release year: {document['release_year']}")

    # Add punctuation only when a section does not already end with it.
    sentences: list[str] = []
    for section in sections:
        sentence = section.strip()
        if sentence and not sentence.endswith((".", "!", "?")):
            sentence += "."
        sentences.append(sentence)

    return " ".join(sentences)


def normalize_movie(
    movie: dict[str, Any], raw_detail_file: str, cast_limit: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one raw TMDB response into a processed document."""
    movie_id = movie.get("id")
    if not isinstance(movie_id, int):
        return None, "missing_tmdb_id"
    if not has_text(movie.get("title")):
        return None, "missing_english_title"
    if not has_text(movie.get("overview")):
        return None, "missing_english_overview"

    translations_zh = extract_chinese_translations(movie)
    if not translations_zh:
        return None, "missing_complete_chinese_translation"

    genre_records = movie.get("genres", [])
    genres = unique_text([genre.get("name") for genre in genre_records])
    genre_ids = [
        genre.get("id")
        for genre in genre_records
        if isinstance(genre.get("id"), int)
    ]
    keywords = unique_text(
        [
            keyword.get("name")
            for keyword in movie.get("keywords", {}).get("keywords", [])
        ]
    )
    cast, directors, writers = extract_people(movie, cast_limit)
    release_date = (movie.get("release_date") or "").strip()

    document: dict[str, Any] = {
        "doc_id": f"movie:{movie_id}",
        "tmdb_id": movie_id,
        "media_type": "movie",
        "is_animation": ANIMATION_GENRE_ID in genre_ids,
        "title_en": movie["title"].strip(),
        "original_title": (movie.get("original_title") or "").strip(),
        "alternative_titles_en": extract_alternative_titles(movie),
        "overview_en": movie["overview"].strip(),
        "tagline_en": (movie.get("tagline") or "").strip(),
        "genres": genres,
        "keywords": keywords,
        "cast": cast,
        "directors": directors,
        "writers": writers,
        "creators": unique_text(directors + writers),
        "release_date": release_date,
        "release_year": parse_release_year(release_date),
        "original_language": movie.get("original_language"),
        "runtime_minutes": movie.get("runtime"),
        "poster_path": movie.get("poster_path"),
        "backdrop_path": movie.get("backdrop_path"),
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
        "translations_zh": translations_zh,
        "raw_detail_file": raw_detail_file,
        "tmdb_url": f"https://www.themoviedb.org/movie/{movie_id}",
    }

    # retrieval_text is created last because it uses the normalized fields.
    document["retrieval_text"] = build_retrieval_text(document)
    return document, None


def build_statistics(
    input_count: int,
    documents: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    cast_limit: int,
) -> dict[str, Any]:
    """Summarize the processed collection."""
    coverage: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    exclusion_counts = Counter(record["reason"] for record in excluded)
    release_years: list[int] = []

    for document in documents:
        if document["poster_path"]:
            coverage["poster"] += 1
        if document["backdrop_path"]:
            coverage["backdrop"] += 1
        if document["keywords"]:
            coverage["keywords"] += 1
        if document["cast"]:
            coverage["cast"] += 1
        if document["directors"]:
            coverage["directors"] += 1
        if document["writers"]:
            coverage["writers"] += 1
        if "zh-CN" in document["translations_zh"]:
            coverage["zh-CN"] += 1
        if "zh-TW" in document["translations_zh"]:
            coverage["zh-TW"] += 1
        if document["is_animation"]:
            coverage["animation"] += 1

        for genre in document["genres"]:
            genre_counts[genre] += 1
        language_counts[document["original_language"] or "unknown"] += 1
        if document["release_year"] is not None:
            release_years.append(document["release_year"])

    document_count = len(documents)

    def coverage_record(name: str) -> dict[str, Any]:
        count = coverage[name]
        percentage = round(100 * count / document_count, 1) if document_count else 0
        return {"count": count, "percentage": percentage}

    return {
        "input_accepted_entries": input_count,
        "documents_written": document_count,
        "documents_excluded": len(excluded),
        "cast_limit": cast_limit,
        "release_year_min": min(release_years) if release_years else None,
        "release_year_max": max(release_years) if release_years else None,
        "coverage": {
            name: coverage_record(name)
            for name in (
                "poster",
                "backdrop",
                "keywords",
                "cast",
                "directors",
                "writers",
                "zh-CN",
                "zh-TW",
                "animation",
            )
        },
        "top_genres": dict(genre_counts.most_common(20)),
        "original_languages": dict(language_counts.most_common()),
        "exclusion_reasons": dict(exclusion_counts),
    }


def prepare_documents(args: argparse.Namespace) -> int:
    """Read accepted metadata and write normalized retrieval documents."""
    validate_args(args)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    accepted_path = input_dir / "accepted_movies.json"

    if not accepted_path.exists():
        raise FileNotFoundError(f"Accepted-movie file not found: {accepted_path}")

    accepted_entries = read_json(accepted_path)
    if not isinstance(accepted_entries, list):
        raise ValueError("accepted_movies.json must contain a JSON list.")

    if args.limit is not None:
        accepted_entries = accepted_entries[: args.limit]

    documents: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    print(f"Input: {accepted_path}")
    print(f"Output: {output_dir}")
    print(f"Preparing {len(accepted_entries)} accepted movie records...")

    for entry_number, entry in enumerate(accepted_entries, start=1):
        movie_id = entry.get("tmdb_id")
        raw_relative_path = entry.get("raw_detail_file")

        if not has_text(raw_relative_path):
            excluded.append({"tmdb_id": movie_id, "reason": "missing_raw_path"})
            continue

        raw_path = input_dir / raw_relative_path
        if not raw_path.exists():
            excluded.append(
                {
                    "tmdb_id": movie_id,
                    "reason": "raw_file_not_found",
                    "raw_detail_file": raw_relative_path,
                }
            )
            continue

        try:
            movie = read_json(raw_path)
        except (OSError, json.JSONDecodeError) as error:
            excluded.append(
                {
                    "tmdb_id": movie_id,
                    "reason": "invalid_raw_json",
                    "error": str(error),
                    "raw_detail_file": raw_relative_path,
                }
            )
            continue

        if movie.get("id") != movie_id:
            excluded.append(
                {
                    "tmdb_id": movie_id,
                    "reason": "tmdb_id_mismatch",
                    "raw_detail_file": raw_relative_path,
                }
            )
            continue

        document, reason = normalize_movie(movie, raw_relative_path, args.cast_limit)
        if document is None:
            excluded.append(
                {
                    "tmdb_id": movie_id,
                    "reason": reason,
                    "raw_detail_file": raw_relative_path,
                }
            )
            continue

        if document["doc_id"] in seen_doc_ids:
            excluded.append(
                {
                    "tmdb_id": movie_id,
                    "reason": "duplicate_doc_id",
                    "raw_detail_file": raw_relative_path,
                }
            )
            continue

        seen_doc_ids.add(document["doc_id"])
        documents.append(document)
        print(
            f"  [{entry_number}/{len(accepted_entries)}] "
            f"{document['doc_id']} {document['title_en']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    excluded_path = output_dir / "excluded_documents.jsonl"
    statistics_path = output_dir / "collection_statistics.json"

    write_jsonl_atomic(documents_path, documents)
    write_jsonl_atomic(excluded_path, excluded)
    statistics = build_statistics(
        len(accepted_entries), documents, excluded, args.cast_limit
    )
    write_json_atomic(statistics_path, statistics)

    print(f"\nDocuments written: {len(documents)}")
    print(f"Documents excluded: {len(excluded)}")
    print(f"Documents file: {documents_path}")
    print(f"Statistics file: {statistics_path}")

    return 0 if documents else 1


def main() -> int:
    """Run document preparation and print readable errors."""
    try:
        args = parse_args()
        return prepare_documents(args)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
