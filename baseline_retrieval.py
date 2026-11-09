
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


# =============================================================================
# 1. Tokenizer
# =============================================================================

def tokenize(text):
    """Convert text into lowercase tokens using regular expressions."""

    # Extract words and numbers, ignoring punctuation.
    return re.findall(r"\b\w+\b", text.lower())


# =============================================================================
# 2. Document Loading
# =============================================================================

def load_documents(file_path):
    """Load movie documents from a JSONL file."""

    documents = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Skip empty lines.
            if not line.strip():
                continue

            # Convert each JSON line into a Python dictionary.
            document = json.loads(line)

            # Use the stable TMDB document identifier.
            doc_id = document["doc_id"]

            # Store the complete movie record.
            documents[doc_id] = document

    return documents


# =============================================================================
# 3. Basic Inverted Index
# =============================================================================

class BasicInvertedIndex:
    """Store term postings and document statistics."""

    def __init__(self):
        # Map each term to its document IDs and term frequencies.
        self.index = defaultdict(dict)

        # Store document lengths.
        self.document_metadata = {}

        # Store document frequencies.
        self.term_metadata = {}

        # Track the total number of tokens.
        self.total_token_count = 0

    def add_doc(self, doc_id, tokens):
        """Add one tokenized document to the inverted index."""

        # Count how many times each term appears in this document.
        term_counts = Counter(tokens)

        # Store the length of this document.
        doc_length = len(tokens)
        self.document_metadata[doc_id] = {"length": doc_length}

        # Update the collection's total token count.
        self.total_token_count += doc_length

        for term, frequency in term_counts.items():
            # Record the term frequency for this document.
            self.index[term][doc_id] = frequency

            # Increase document frequency once per document.
            self.term_metadata[term] = (
                self.term_metadata.get(term, 0) + 1
            )

    def get_postings(self, term):
        """Return document IDs and term frequencies for a term."""

        return self.index.get(term, {})

    def get_doc_metadata(self, doc_id):
        """Return metadata for a document."""

        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term):
        """Return document frequency information for a term."""

        return {
            "doc_frequency": self.term_metadata.get(term, 0)
        }

    def get_statistics(self):
        """Return collection-level statistics."""

        number_of_documents = len(self.document_metadata)

        # Avoid division by zero for an empty collection.
        mean_length = (
            self.total_token_count / number_of_documents
            if number_of_documents > 0 else 0
        )

        return {
            "number_of_documents": number_of_documents,
            "mean_document_length": mean_length,
            "total_token_count": self.total_token_count,
        }


# =============================================================================
# 4. BM25 Relevance Scorer
# =============================================================================

class BM25:
    """Compute BM25 relevance scores using index statistics."""

    def __init__(self, index, k1=1.2, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def score(self, doc_id, query_term_counts, doc_term_counts):
        """Calculate the BM25 score for one document."""

        statistics = self.index.get_statistics()

        # Get collection-level statistics.
        N = statistics["number_of_documents"]
        avgdl = statistics["mean_document_length"]

        # Get the current document's length.
        doc_length = self.index.get_doc_metadata(doc_id)["length"]

        score = 0.0

        if N == 0 or avgdl == 0:
            return score

        for term in query_term_counts:
            # Get the term frequency in the current document.
            tf = doc_term_counts.get(term, 0)

            # Get the number of documents containing this term.
            df = self.index.get_term_metadata(term)["doc_frequency"]

            # Terms absent from the document do not contribute.
            if tf == 0 or df == 0:
                continue

            # Calculate the nonnegative BM25 inverse document frequency.
            idf = math.log(
                1 + (N - df + 0.5) / (df + 0.5)
            )

            # Normalize term frequency based on document length.
            length_norm = (
                1 - self.b + self.b * doc_length / avgdl
            )

            # Apply BM25 term frequency saturation.
            tf_weight = (
                tf * (self.k1 + 1)
                / (tf + self.k1 * length_norm)
            )

            # Add the term's contribution to the total score.
            score += idf * tf_weight

        return score


# =============================================================================
# 5. Ranker
# =============================================================================

class Ranker:
    """Retrieve candidate documents and rank them by BM25 score."""

    def __init__(self, index, scorer):
        self.index = index
        self.scorer = scorer

    def query(self, query, top_k=10):
        """Return the top-k documents ranked by BM25."""

        # Tokenize the user's query.
        query_tokens = tokenize(query)

        # Count query terms.
        query_term_counts = Counter(query_tokens)

        # Collect term frequencies for candidate documents.
        candidate_documents = defaultdict(dict)

        for term in query_term_counts:
            # Find documents that contain the query term.
            postings = self.index.get_postings(term)

            for doc_id, frequency in postings.items():
                candidate_documents[doc_id][term] = frequency

        results = []

        for doc_id, doc_term_counts in candidate_documents.items():
            # Compute the relevance score for this candidate.
            score = self.scorer.score(
                doc_id,
                query_term_counts,
                doc_term_counts
            )

            results.append((doc_id, score))

        # Rank by descending score; use doc_id to break ties.
        results.sort(key=lambda item: (-item[1], item[0]))

        return results[:top_k]


# =============================================================================
# 6. Random Baseline
# =============================================================================

def random_rank(documents, top_k=10, seed=42):
    """Return a reproducible random ranking of documents."""

    # Sort IDs before shuffling to keep results reproducible.
    doc_ids = sorted(documents.keys())

    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    return [(doc_id, 0.0) for doc_id in doc_ids[:top_k]]


# =============================================================================
# 7. Build the Retrieval System
# =============================================================================

def build_index(documents):
    """Create an inverted index using movie retrieval text."""

    index = BasicInvertedIndex()

    for doc_id, document in documents.items():
        # Use the English retrieval text prepared by the team.
        retrieval_text = document.get("retrieval_text", "")

        # Convert the retrieval text into tokens.
        tokens = tokenize(retrieval_text)

        # Add the tokenized movie to the inverted index.
        index.add_doc(doc_id, tokens)

    return index


# =============================================================================
# 8. Main Program
# =============================================================================

def main():
    """Load data, build the index, and execute a search query."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/processed/documents.jsonl",
        help="Path to the processed movie documents"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Search query"
    )

    parser.add_argument(
        "--method",
        choices=["bm25", "random"],
        default="bm25",
        help="Retrieval method"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results to return"
    )

    args = parser.parse_args()

    # Check that the dataset exists.
    if not Path(args.data).is_file():
        parser.error(f"Dataset not found: {args.data}")

    # Load the movie collection.
    documents = load_documents(args.data)

    print(f"Loaded documents: {len(documents)}")

    if args.method == "bm25":
        # Build the inverted index.
        index = build_index(documents)

        # Initialize BM25 and the ranker.
        scorer = BM25(index)
        ranker = Ranker(index, scorer)

        # Retrieve ranked results.
        results = ranker.query(args.query, args.top_k)

    else:
        # Generate a random baseline ranking.
        results = random_rank(documents, args.top_k)

    print(f"\nMethod: {args.method}")
    print(f"Query: {args.query}\n")

    # Display the top-ranked movie titles.
    for rank, (doc_id, score) in enumerate(results, start=1):
        # Get the English title of the movie.
        title = documents[doc_id].get("title_en", "Unknown")

        # Random ranking does not calculate a relevance score.
        if args.method == "random":
            print(f"{rank}. {title} [{doc_id}] | Score: N/A")

        # BM25 ranking displays the calculated relevance score.
        else:
            print(f"{rank}. {title} [{doc_id}] | Score: {score:.4f}")


if __name__ == "__main__":
    main()
