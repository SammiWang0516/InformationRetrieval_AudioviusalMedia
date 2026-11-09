import csv
from pathlib import Path

from baseline_retrieval import (
    load_documents,
    build_index,
    BM25,
    Ranker,
    random_rank,
)


def load_queries(file_path: str) -> list[dict]:
    """Load evaluation queries from a CSV file."""

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def generate_annotation_pool(
    queries: list[dict],
    documents: dict,
    ranker: Ranker,
    bm25_top_k: int = 50,
    random_top_k: int = 20,
) -> list[dict]:
    """Generate pooled candidate documents for human annotation."""

    annotation_pool = []

    for query_index, query_data in enumerate(queries):
        query_id = query_data["query_id"]
        query_text = query_data["query"]

        # Retrieve documents using BM25 and Random baselines.
        bm25_results = ranker.query(query_text, bm25_top_k)
        random_results = random_rank(
            documents,
            top_k=random_top_k,
            seed=42 + query_index
        )

        # Combine results while keeping unique document IDs.
        pooled_documents = {}

        for method, results in [
            ("bm25", bm25_results),
            ("random", random_results),
        ]:
            for rank, (doc_id, score) in enumerate(results, start=1):
                if doc_id not in pooled_documents:
                    pooled_documents[doc_id] = {
                        "query_id": query_id,
                        "query": query_text,
                        "doc_id": doc_id,
                        "title_en": documents[doc_id].get("title_en", ""),
                        "overview_en": documents[doc_id].get("overview_en", ""),
                        "source": method,
                        "relevance": "",
                    }
                elif method not in pooled_documents[doc_id]["source"].split(";"):
                    pooled_documents[doc_id]["source"] += f";{method}"

        annotation_pool.extend(pooled_documents.values())

        print(
            f"{query_id}: {len(pooled_documents)} "
            "unique documents for annotation"
        )

    return annotation_pool


def save_annotation_pool(annotation_pool: list[dict], file_path: str):
    """Save pooled documents to a CSV file."""

    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_id",
        "query",
        "doc_id",
        "title_en",
        "overview_en",
        "source",
        "relevance",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotation_pool)


def main():
    """Build retrieval models and prepare human annotation data."""

    documents = load_documents("data/processed/documents.jsonl")
    queries = load_queries("data/evaluation/queries.csv")

    # Build the inverted index and initialize the BM25 ranker.
    index = build_index(documents)
    scorer = BM25(index)
    ranker = Ranker(index, scorer)

    # Generate the candidate pool for relevance judgments.
    annotation_pool = generate_annotation_pool(
        queries,
        documents,
        ranker
    )

    # Export candidates for manual evaluation.
    save_annotation_pool(
        annotation_pool,
        "data/evaluation/annotation_pool.csv"
    )

    print(f"\nLoaded documents: {len(documents)}")
    print(f"Loaded queries: {len(queries)}")
    print(f"Total annotation rows: {len(annotation_pool)}")
    print("Saved: data/evaluation/annotation_pool.csv")


if __name__ == "__main__":
    main()
