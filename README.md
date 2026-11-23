# Multilingual Audiovisual Media Retrieval

This project studies retrieval of movies and television series when the query and document are written in different languages, and when the query is an image rather than text. Documents are indexed in English. We plan to compare lexical retrieval, translated-query retrieval, and multilingual semantic retrieval, all scored with the same queries, relevance labels, and evaluation script.

## Current Scope

- **Query languages:** English and Chinese first. Spanish queries may be added later. Because documents stay in English, a new language needs only new queries, not a new index. The saved TMDB responses already contain Spanish translations for most movies (about 64% in a 300-movie sample), so no extra collection is needed to write Spanish queries.
- **Image queries:** in scope. A user submits a movie image and the system returns matching titles. Backdrops or screenshots are preferred over posters as queries, because a poster query would trivially match the identical poster in the collection.
- **Query set:** the 40 bilingual queries in `data/evaluation/checkpoint/` are the official evaluation set. The set will be expanded later.

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
    F --> G["40 bilingual queries<br/>and relevance labels"]
    G --> H["Random and BM25<br/>baselines"]
    H --> I["nDCG@10, MAP@10,<br/>MRR@10 results"]
    I --> J["Checkpoint 1<br/>November 9, 2026"]
    J --> K["Multilingual<br/>retrieval"]
    J --> L["Image<br/>retrieval"]

    classDef completed fill:#d9ead3,stroke:#38761d,stroke-width:2px,color:#000000;
    classDef checkpoint fill:#fff2cc,stroke:#bf9000,stroke-width:2px,color:#000000;
    classDef planned fill:#eeeeee,stroke:#999999,stroke-width:1px,stroke-dasharray:4 3,color:#000000;
    class A,B,C,D,E,F,R,G,H,I completed;
    class J checkpoint;
    class K,L planned;
```

Green blocks are implemented and tested; grey dashed blocks are planned. The collection contains 5,000 movies released 2024–2026, and the baselines have been evaluated on all 40 queries.

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
    ├── processed/
    │   ├── documents.jsonl
    │   ├── collection_statistics.json
    │   └── excluded_documents.jsonl
    └── evaluation/
        └── checkpoint/
            ├── queries_40_bilingual.csv       # 20 English/Chinese query pairs
            ├── human_relevance_judgments.csv  # 50 judged documents per query
            └── baseline_results.csv           # Per-query metrics for each method
```

The retrieval and evaluation scripts sit alongside the collection scripts in the project folder: `baseline_retrieval.py`, `prepare_evaluation.py`, and `evaluate_checkpoint.py`.

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

## Evaluation Set

`queries_40_bilingual.csv` contains 20 query groups. Each group has one English query and one Chinese query (Traditional characters) expressing the same information need, so results can be compared across languages on identical needs.

| Type | English | Chinese | Relevance |
|---|---|---|---|
| Known-item | 10 | 10 | The single movie in `target_doc_id` |
| Exploratory | 10 | 10 | Graded human labels, 1–5 |

For every query, 50 candidate documents were pooled from BM25 results for the query, BM25 results for its English pair, and random documents. Each candidate was labelled 1–5 by hand in `human_relevance_judgments.csv`. A label of 4 or 5 counts as relevant for MAP.

`prepare_evaluation.py` shows the pooling approach but does not reproduce the committed pool exactly: the committed file also pools the paired English query's BM25 results and uses different column names. The script that produced the committed pool still needs to be added.

## Baseline Retrieval

`baseline_retrieval.py` implements two baselines over the English `retrieval_text` field:

- **BM25** with an inverted index, `k1 = 1.2`, `b = 0.75`, and lowercase word tokenization.
- **Random**, a reproducible shuffle of the collection.

Search from the command line:

```powershell
python baseline_retrieval.py --query "a robot stranded on an island" --method bm25 --top_k 10
```

## Running the Evaluation

```powershell
python evaluate_checkpoint.py
```

The script evaluates only queries whose 50 candidates are all labelled, writes per-query metrics to `baseline_results.csv`, and prints means overall and by language and query type.

| Metric | Queries | What it measures |
|---|---|---|
| nDCG@10 | All | Ranking quality against the graded 1–5 labels (main metric) |
| MAP@10 | All | Precision at each relevant (4–5) result in the top 10 |
| MRR@10 | Known-item | How high the target movie appears; 0 if it is outside the top 10 |

MRR@10 uses only `target_doc_id`. Similar movies that were labelled 4 or 5 do not count, which matches the proposal's definition of known-item relevance. All metrics are computed against the judged pool rather than the full collection.

## Checkpoint Results

| Queries | Method | nDCG@10 | MAP@10 | MRR@10 |
|---|---|---|---|---|
| English known-item | BM25 | 0.860 | 0.724 | 0.950 |
| English exploratory | BM25 | 0.752 | 0.397 | — |
| Chinese known-item | BM25 | 0.000 | 0.000 | 0.000 |
| Chinese exploratory | BM25 | 0.000 | 0.000 | — |
| All 40 queries | BM25 | 0.403 | 0.280 | 0.475 |
| All 40 queries | Random | 0.012 | 0.003 | 0.000 |

BM25 works well for English queries and returns nothing useful for Chinese queries. The index contains only English text, and the tokenizer treats a run of Chinese characters as a single token, so Chinese queries share no terms with any document. Closing this gap is the purpose of the multilingual methods below. Report the per-language rows alongside the overall figures, because the overall mean hides the gap.

## Next Steps

Each new method plugs into the same evaluation: it takes a query, returns a top-10 list, and is scored by `evaluate_checkpoint.py` with the existing labels.

1. **Translated-query BM25:** translate each Chinese query into English, then run the existing BM25 index.
2. **Multilingual semantic retrieval:** embed queries and documents with a multilingual model so Chinese and English text share one vector space, then rank by similarity.
3. **Hybrid retrieval:** combine BM25 and semantic scores.
4. **Image retrieval:** download backdrop and poster images, index them with an image–text model such as CLIP, and create a small set of image queries.
5. **More queries:** expand the evaluation set, including Spanish queries, once the pipeline is stable.

New retrieved documents that are not in the judged pool will need labels before they can be scored.

## Planned Collection Extension

The next collection step will add television series using `/discover/tv` and `/tv/{series_id}`. Each normalized record will use a composite identifier such as `movie:129` or `tv:1399` so movie and TV identifiers cannot conflict. Records with genre ID `16` will be marked with `is_animation: true` rather than treated as a separate TMDB media type.
