"""Evaluate retrieval methods on the bilingual checkpoint queries with human labels.

This script never generates or invents relevance labels. It skips queries until
all candidate rows for that query have a human score from 1 to 5.
Metrics are calculated with pooled judgments, not full-corpus relevance.

Methods other than the original baselines can retrieve documents that were not
in the judged pool. Those unjudged documents are treated as not relevant (gain
0), which is standard practice for pooled evaluation, and the judged_at_10
column records how many of the top 10 were actually labelled. A low value means
the pooled metrics for that method underestimate it and more labels are needed.

MRR@10 is reported for known-item queries only. It is computed against each
query's target_doc_id, so only the movie the query writer had in mind counts
as a hit, regardless of how similar movies were labelled. It does not depend on
the judged pool.
"""
import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from baseline_retrieval import BM25, Ranker, build_index, load_documents

METHOD_NAMES = {
    "bm25": "BM25",
    "random": "Random",
    "bm25_translated": "BM25+MT",
    "dense": "Dense",
}

# A missing label means "not judged"; it is scored like the lowest label (gain 0).
UNJUDGED_LABEL = 1


def read_csv(file_path: str) -> list[dict]:
    """Read a CSV file into dictionaries, preserving Chinese characters."""
    with open(file_path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def dcg_at_10(relevances: list[int]) -> float:
    """Calculate discounted cumulative gain from human labels of 1 to 5."""
    return sum(
        (2 ** (relevance - 1) - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:10], start=1)
    )


def evaluate_ranking(ranking: list[str], labels: dict[str, int]) -> dict:
    """Calculate pooled AP@10 and nDCG@10 for one fully judged query."""
    # An empty or short ranking simply contributes fewer retrieved items.
    retrieved_labels = [labels.get(doc_id, UNJUDGED_LABEL) for doc_id in ranking]
    relevant_in_pool = sum(value >= 4 for value in labels.values())
    hits = 0
    precision_sum = 0.0

    for rank, relevance in enumerate(retrieved_labels[:10], start=1):
        if relevance >= 4:
            hits += 1
            precision_sum += hits / rank

    ap_at_10 = (
        precision_sum / min(relevant_in_pool, 10)
        if relevant_in_pool else 0.0
    )
    ideal_labels = sorted(labels.values(), reverse=True)[:10]
    ideal_dcg = dcg_at_10(ideal_labels)
    ndcg_at_10 = dcg_at_10(retrieved_labels) / ideal_dcg if ideal_dcg else 0.0
    return {"pooled_AP_at_10": ap_at_10, "pooled_nDCG_at_10": ndcg_at_10}


def reciprocal_rank_at_10(ranking: list[str], target_doc_id: str) -> float:
    """Return 1/rank of the target document in the top 10, or 0 if absent."""
    for rank, doc_id in enumerate(ranking[:10], start=1):
        if doc_id == target_doc_id:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def print_summary(results: list[dict], label: str, methods: list[str]) -> None:
    """Print mean metrics per method for one group of result rows."""
    for method in methods:
        subset = [row for row in results if row["method"] == method]
        if not subset:
            continue
        mean_ap = mean([float(row["pooled_AP_at_10"]) for row in subset])
        mean_ndcg = mean([float(row["pooled_nDCG_at_10"]) for row in subset])
        mean_judged = mean([int(row["judged_at_10"]) for row in subset])
        reciprocal_ranks = [
            float(row["RR_at_10"]) for row in subset if row["RR_at_10"] != ""
        ]
        mrr_text = (
            f", MRR@10={mean(reciprocal_ranks):.4f}" if reciprocal_ranks else ""
        )
        print(
            f"  {label:<22} {method:<8} n={len(subset):<3} "
            f"nDCG@10={mean_ndcg:.4f}, MAP@10={mean_ap:.4f}{mrr_text}"
            f"  (judged@10={mean_judged:.1f})"
        )


def build_rankers(method_keys: list[str], documents: dict, args) -> dict:
    """Create a function per method that maps a query row to a top-10 doc list."""
    rankers = {}
    needs_bm25 = {"bm25", "bm25_translated"} & set(method_keys)
    bm25 = None
    if needs_bm25:
        index = build_index(documents)
        bm25 = Ranker(index, BM25(index))

    for key in method_keys:
        if key == "bm25":
            rankers[key] = lambda query: [d for d, _ in bm25.query(query["query"], 10)]

        elif key == "random":
            all_ids = sorted(documents)

            def random_ranker(query, all_ids=all_ids):
                random_ids = all_ids.copy()
                random.Random(42 + query["position"]).shuffle(random_ids)
                return random_ids[:10]

            rankers[key] = random_ranker

        elif key == "bm25_translated":
            translated_path = Path(args.translated)
            if not translated_path.is_file():
                raise FileNotFoundError(
                    f"{translated_path} not found. Run translate_queries.py first."
                )
            translated = {
                row["query_id"]: row["query_en"] for row in read_csv(args.translated)
            }
            rankers[key] = lambda query: [
                d for d, _ in bm25.query(translated[query["query_id"]], 10)
            ]

        elif key == "dense":
            from dense_retrieval import DenseRetriever

            dense = DenseRetriever(documents, args.dense_model)
            rankers[key] = lambda query: [d for d, _ in dense.query(query["query"], 10)]

        else:
            raise ValueError(f"Unknown method '{key}'. Choose from {list(METHOD_NAMES)}")

    return rankers


def main() -> None:
    """Load approved judgments and compare methods on the same fully judged queries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/documents.jsonl")
    parser.add_argument("--queries", default="data/evaluation/checkpoint/queries_40_bilingual.csv")
    parser.add_argument("--annotations", default="data/evaluation/checkpoint/human_relevance_judgments.csv")
    parser.add_argument("--out", default="data/evaluation/checkpoint/baseline_results.csv")
    parser.add_argument(
        "--methods",
        default="bm25,random",
        help=f"Comma-separated methods: {', '.join(METHOD_NAMES)}",
    )
    parser.add_argument(
        "--translated",
        default="data/evaluation/checkpoint/queries_translated.csv",
        help="Output of translate_queries.py, used by bm25_translated",
    )
    parser.add_argument("--dense_model", default="intfloat/multilingual-e5-base")
    args = parser.parse_args()

    method_keys = [key.strip() for key in args.methods.split(",") if key.strip()]
    queries = read_csv(args.queries)
    rows = read_csv(args.annotations)
    per_query_labels = defaultdict(dict)
    per_query_total = defaultdict(int)
    invalid = []

    for row in rows:
        query_id = row["query_id"]
        doc_id = row["doc_id"]
        per_query_total[query_id] += 1
        raw = row.get("human_relevance_1to5", "").strip()
        if raw:
            if raw not in {"1", "2", "3", "4", "5"}:
                invalid.append((query_id, doc_id, raw))
            else:
                per_query_labels[query_id][doc_id] = int(raw)

    if invalid:
        raise ValueError(f"Invalid human labels (first 5): {invalid[:5]}")

    for position, query in enumerate(queries):
        query["position"] = position

    completed = [
        query for query in queries
        if per_query_total[query["query_id"]] >= 50
        and len(per_query_labels[query["query_id"]]) == per_query_total[query["query_id"]]
    ]
    print(f"Fully judged queries: {len(completed)} / {len(queries)}")
    if not completed:
        print("No graded results yet: finish and review the human_relevance_1to5 column.")
        return

    documents = load_documents(args.data)
    rankers = build_rankers(method_keys, documents, args)
    results = []

    for query in completed:
        query_id = query["query_id"]
        labels = per_query_labels[query_id]
        target_doc_id = query.get("target_doc_id", "").strip()

        for key, ranker in rankers.items():
            ranking = ranker(query)[:10]
            metric = evaluate_ranking(ranking, labels)
            reciprocal_rank = (
                f"{reciprocal_rank_at_10(ranking, target_doc_id):.6f}"
                if query["type"] == "known-item" and target_doc_id else ""
            )
            results.append({
                "query_id": query_id,
                "language": query["language"],
                "type": query["type"],
                "method": METHOD_NAMES[key],
                "pooled_AP_at_10": f'{metric["pooled_AP_at_10"]:.6f}',
                "pooled_nDCG_at_10": f'{metric["pooled_nDCG_at_10"]:.6f}',
                "RR_at_10": reciprocal_rank,
                "judged_at_10": sum(doc_id in labels for doc_id in ranking),
                "judged_documents": len(labels),
            })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    # Compare methods only on the same set of fully judged queries.
    method_labels = [METHOD_NAMES[key] for key in method_keys]
    print("\nOverall:")
    print_summary(results, "all queries", method_labels)

    # Break results down so the language gap is not hidden by the average.
    print("\nBy language and query type:")
    groups = sorted({(row["language"], row["type"]) for row in results})
    for language, query_type in groups:
        group_rows = [
            row for row in results
            if row["language"] == language and row["type"] == query_type
        ]
        print_summary(group_rows, f"{language} / {query_type}", method_labels)

    print(f"\nResults saved: {output}")
    if len(completed) < len(queries):
        print("PARTIAL evaluation only. Do not describe these as 40-query results.")


if __name__ == "__main__":
    main()
