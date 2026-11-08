# Multilingual Audiovisual Media Retrieval

This project studies retrieval of movies and television series when the query and document are written in different languages. The initial experiment uses Chinese queries to retrieve English documents. We plan to compare lexical retrieval, translated-query retrieval, and multilingual semantic retrieval.

The current work focuses on building the document collection from The Movie Database (TMDB). The collection script is implemented for movies. TV series will be added through the corresponding TMDB TV endpoints. Animation will be identified using TMDB genre ID `16`; it may be either a movie or a television series.

## Data Selection

TMDB does not provide a discovery filter for translation availability. The script therefore searches for candidate movies, requests the complete metadata for each candidate, and then checks its translations.

Movies are searched one release year at a time. Within each year, results are sorted by popularity because widely viewed titles are more likely to contain complete translations. A movie is included in the bilingual collection when it has:

- a nonempty English title;
- a nonempty English overview; and
- a `zh-CN` or `zh-TW` translation containing both a title and an overview.

The target count refers to movies that pass these conditions, not the number of candidates examined. The script may need to inspect substantially more than 5,000 candidates to collect 5,000 qualifying movies.

## Collection Pipeline

```mermaid
flowchart LR
    A["TMDB<br/>credential"] --> B["Search each year<br/>by popularity"]
    B --> C["Fetch and save<br/>raw metadata"]
    C --> D{"English and<br/>zh-CN or zh-TW?"}
    D -->|"Accepted"| E["accepted_movies<br/>manifest"]
    D -->|"Rejected"| R["Record<br/>rejection reason"]
    E --> F["Prepare English<br/>retrieval documents"]
    F --> G["Create Chinese queries<br/>and relevance labels"]
    G --> H["Random and BM25<br/>baselines"]
    H --> I["MRR and Recall@10<br/>preliminary results"]
    I --> J["Checkpoint 1<br/>November 9, 2026"]

    classDef completed fill:#d9ead3,stroke:#38761d,stroke-width:2px,color:#000000;
    classDef checkpoint fill:#fff2cc,stroke:#bf9000,stroke-width:2px,color:#000000;
    class A,B,C,D,E,F,R completed;
    class J checkpoint;
```

The green block shows the pipeline components that have been implemented and tested. The full 5,000-movie collection is still running. The checkpoint requires a cleaned collection, working baselines, and preliminary results; the query and evaluation block remains to be completed.

Raw candidate metadata is retained even when a movie does not pass the bilingual filter. This allows the collection process to be audited and resumed without requesting the same details again.

## Data Preprocessing

The TMDB detail responses are source data rather than retrieval documents. They contain nested credits, translations, image records, release information, and other fields that should not all be placed in the search index. Preprocessing converts each accepted raw response into one consistent document while leaving the original JSON unchanged.

The preprocessing script performs the following steps:

1. Read the IDs and raw-file locations from `accepted_movies.json`.
2. Validate the raw JSON and confirm that its TMDB ID matches the manifest.
3. Extract the English title, overview, genres, keywords, top cast, directors, writers, release year, and media type.
4. Preserve Chinese translations and display fields such as poster and backdrop paths.
5. Combine only the selected English fields into `retrieval_text` for BM25.
6. Write processed documents, collection statistics, and any exclusion reasons.

Chinese translations are stored for query preparation and a future interface, but they are not added to `retrieval_text`. This keeps the retrieval task Chinese query to English document.

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```dotenv
TMDB_READ_TOKEN=your_api_read_access_token
```

The `.env` file contains a credential and must not be committed to GitHub.

## Running the Collector

Test the pipeline with 20 qualifying movies from 2025:

```powershell
python collect_tmdb_movies.py --target 20 --start-year 2025 --end-year 2025
```

Collect up to 5,000 qualifying movies across release years:

```powershell
python collect_tmdb_movies.py `
  --target 5000 `
  --start-year 2026 `
  --end-year 1900 `
  --cutoff-date 2026-11-08
```

The collector saves each API response as it is received. If the program is interrupted, running the same command again reuses valid files already stored on disk. Previously accepted IDs are reconstructed from the saved metadata, so increasing the target from 20 to 5,000 does not create duplicate documents.

## Output Structure

```text
project_proposal/
├── .env
├── .gitignore
├── requirements.txt
├── collect_tmdb_movies.py
├── prepare_tmdb_documents.py
├── test_tmdb_api.py
└── data/
    ├── raw/                         # Original newest-to-oldest pilot collection
    ├── bilingual_movies/
    │   ├── accepted_movies.json     # IDs and basic information for accepted movies
    │   ├── collection_summary.json  # Counts, settings, failures, and rejection reasons
    │   └── raw/
    │       ├── discover/
    │       │   └── movie/
    │       │       ├── year_2026/
    │       │       └── year_2025/
    │       └── details/
    │           └── movie/
    │               ├── 123.json
    │               └── 456.json
    └── processed/
        ├── documents.jsonl
        ├── collection_statistics.json
        └── excluded_documents.jsonl
```

`accepted_movies.json` is an index of qualifying records. The complete metadata remains under `raw/details/movie/`. Image paths and video references are stored as metadata, but the associated media files are not downloaded during this stage.

## Running Preprocessing

After the collection finishes, convert the accepted raw records into normalized documents:

```powershell
python prepare_tmdb_documents.py
```

For a small test, process only the first five accepted movies:

```powershell
python prepare_tmdb_documents.py --limit 5
```

`documents.jsonl` contains one processed movie per line. The English `retrieval_text` field is intended for the first BM25 index. Chinese translations, poster paths, ratings, and provenance fields remain available for query preparation and interface development but are not included in the retrieval text.

## Planned Extension

The next collection step will add television series using `/discover/tv` and `/tv/{series_id}`. Each normalized record will use a composite identifier such as `movie:129` or `tv:1399` so movie and TV identifiers cannot conflict. Records with genre ID `16` will be marked with `is_animation: true` rather than treated as a separate TMDB media type.
