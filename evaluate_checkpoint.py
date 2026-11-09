"""Evaluate existing HW1-style BM25 and Random baselines with human labels.

This script never generates or invents relevance labels. It skips queries until
all candidate rows for that query have a human score from 1 to 5.
Metrics are calculated with pooled judgments, not full-corpus relevance.
"""
import argparse
import csv
from html import parser
import math
import random
from collections import defaultdict
from pathlib import Path

from baseline_retrieval import BM25, Ranker, build_index, load_documents


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
    # A missing result is treated as zero retrieved relevant items, not as
    # an unjudged document being declared irrelevant.
    retrieved_labels = [labels[doc_id] for doc_id in ranking]
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


def main() -> None:
    """Load approved judgments and compare BM25 with Random on matching queries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/documents.jsonl")
    parser.add_argument("--queries", default="data/evaluation/checkpoint/queries_40_bilingual.csv")
    parser.add_argument("--annotations", default="data/evaluation/checkpoint/human_relevance_judgments.csv") ###
    parser.add_argument("--out", default="data/evaluation/checkpoint/baseline_results.csv")
    args = parser.parse_args()

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
    index = build_index(documents)
    ranker = Ranker(index, BM25(index))
    all_ids = sorted(documents)
    query_positions = {q["query_id"]: i for i, q in enumerate(queries)}
    results = []

    for query in completed:
        query_id = query["query_id"]
        labels = per_query_labels[query_id]
        query_index = query_positions[query_id]
        bm25_ids = [doc_id for doc_id, _ in ranker.query(query["query"], 10)]
        random_ids = all_ids.copy()
        random.Random(42 + query_index).shuffle(random_ids)

        for method, ranking in (("BM25", bm25_ids), ("Random", random_ids[:10])):
            missing = [doc_id for doc_id in ranking if doc_id not in labels]
            if missing:
                raise ValueError(
                    f"Candidate pool is missing {method} results for {query_id}: {missing[:5]}"
                )
            metric = evaluate_ranking(ranking, labels)
            results.append({
                "query_id": query_id,
                "language": query["language"],
                "type": query["type"],
                "method": method,
                "pooled_AP_at_10": f'{metric["pooled_AP_at_10"]:.6f}',
                "pooled_nDCG_at_10": f'{metric["pooled_nDCG_at_10"]:.6f}',
                "judged_documents": len(labels),
            })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    # Compare methods only on the same set of fully judged queries.
    for method in ("BM25", "Random"):
        subset = [row for row in results if row["method"] == method]
        mean_ap = sum(float(row["pooled_AP_at_10"]) for row in subset) / len(subset)
        mean_ndcg = sum(float(row["pooled_nDCG_at_10"]) for row in subset) / len(subset)
        print(f"{method}: pooled MAP@10={mean_ap:.4f}, pooled nDCG@10={mean_ndcg:.4f}")

    print(f"Results saved: {output}")
    if len(completed) < len(queries):
        print("PARTIAL evaluation only. Do not describe these as 40-query results.")


if __name__ == "__main__":
    main()
