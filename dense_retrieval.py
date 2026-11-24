"""Multilingual dense retrieval over the English movie documents.

A multilingual embedding model maps queries in any supported language and the
English documents into one vector space. Documents are ranked by cosine
similarity to the query. The index stays English: a Chinese query is matched
directly against English text, with no translation step.

Two document text modes are supported:

- "full": the same retrieval_text that BM25 indexes, including cast, director,
  and writer names.
- "plot": title, overview, tagline, genres, and keywords only. Person names are
  left out because long name lists made Chinese queries favour Chinese-language
  films by language rather than by plot.

The document vectors are cached under data/indexes/ so the collection is
encoded only once per model and text mode.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline_retrieval import load_documents

DEFAULT_MODEL = "intfloat/multilingual-e5-base"


TEXT_MODES = ("full", "plot")


def is_e5_model(model_name: str) -> bool:
    """Return True for E5 models, which expect 'query: '/'passage: ' prefixes."""
    return "e5" in model_name.lower()


def build_plot_text(document: dict) -> str:
    """Combine the descriptive English fields, leaving out person names."""
    sections = [document.get("title_en", "")]
    if document.get("overview_en"):
        sections.append(document["overview_en"])
    if document.get("tagline_en"):
        sections.append(document["tagline_en"])
    if document.get("genres"):
        sections.append("Genres: " + ", ".join(document["genres"]))
    if document.get("keywords"):
        sections.append("Keywords: " + ", ".join(document["keywords"]))

    sentences = []
    for section in sections:
        sentence = section.strip()
        if sentence and not sentence.endswith((".", "!", "?")):
            sentence += "."
        if sentence:
            sentences.append(sentence)
    return " ".join(sentences)


def document_text(document: dict, text_mode: str) -> str:
    """Return the text to embed for one document in the chosen mode."""
    if text_mode == "full":
        return document.get("retrieval_text", "")
    if text_mode == "plot":
        return build_plot_text(document)
    raise ValueError(f"Unknown text mode '{text_mode}'. Choose from {TEXT_MODES}")


class DenseRetriever:
    """Rank English documents by embedding similarity to a query."""

    def __init__(
        self,
        documents: dict,
        model_name: str = DEFAULT_MODEL,
        text_mode: str = "plot",
        cache_dir: str = "data/indexes",
        batch_size: int = 8,
    ):
        from sentence_transformers import SentenceTransformer

        if text_mode not in TEXT_MODES:
            raise ValueError(f"Unknown text mode '{text_mode}'. Choose from {TEXT_MODES}")
        self.model_name = model_name
        self.text_mode = text_mode
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        self.doc_ids = sorted(documents)
        self.doc_vectors = self._load_or_encode(documents, cache_dir, batch_size)

    def _cache_paths(self, cache_dir: str) -> tuple[Path, Path]:
        """Return the vector and ID file paths for this model and text mode."""
        safe_name = f"{self.model_name.replace('/', '__')}_{self.text_mode}"
        folder = Path(cache_dir)
        return folder / f"dense_{safe_name}.npy", folder / f"dense_{safe_name}_ids.json"

    def _load_or_encode(self, documents: dict, cache_dir: str, batch_size: int):
        """Load cached document vectors, or encode the collection and cache it."""
        vector_path, ids_path = self._cache_paths(cache_dir)

        if vector_path.is_file() and ids_path.is_file():
            cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if cached_ids == self.doc_ids:
                print(f"Loaded cached document vectors: {vector_path}")
                return np.load(vector_path)
            print("Cached vectors do not match the current collection; re-encoding.")

        prefix = "passage: " if is_e5_model(self.model_name) else ""
        texts = [
            prefix + document_text(documents[doc_id], self.text_mode)
            for doc_id in self.doc_ids
        ]

        print(
            f"Encoding {len(texts)} documents with {self.model_name} "
            f"(text mode: {self.text_mode}) ..."
        )
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vector_path, vectors)
        ids_path.write_text(json.dumps(self.doc_ids), encoding="utf-8")
        print(f"Saved document vectors: {vector_path}")
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        """Embed one query into the shared vector space."""
        prefix = "query: " if is_e5_model(self.model_name) else ""
        return self.model.encode(
            [prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

    def query(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, cosine similarity) pairs for a query."""
        scores = self.doc_vectors @ self.encode_query(query)
        top = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]


def main() -> None:
    """Search the collection from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/documents.jsonl")
    parser.add_argument("--query", required=True, help="Search query in any language")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text_mode", choices=TEXT_MODES, default="plot")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    documents = load_documents(args.data)
    retriever = DenseRetriever(documents, args.model, args.text_mode)

    print(f"\nModel: {args.model} ({args.text_mode} text)\nQuery: {args.query}\n")
    for rank, (doc_id, score) in enumerate(retriever.query(args.query, args.top_k), start=1):
        title = documents[doc_id].get("title_en", "Unknown")
        print(f"{rank}. {title} [{doc_id}] | Cosine: {score:.4f}")


if __name__ == "__main__":
    main()
