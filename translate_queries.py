"""Translate non-English evaluation queries into English for query-translation BM25.

The translations are written to a CSV once, so evaluation is reproducible and
does not need the translation model at scoring time. English queries are copied
unchanged. The index stays English: only the query is translated.
"""
import argparse
import csv
from pathlib import Path

DEFAULT_MODELS = {
    "zh": "Helsinki-NLP/opus-mt-zh-en",
}


def read_csv(file_path: str) -> list[dict]:
    """Read a CSV file into dictionaries, preserving Chinese characters."""
    with open(file_path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def translate_texts(texts: list[str], model_name: str) -> list[str]:
    """Translate a batch of texts into English with a MarianMT model."""
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()

    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    generated = model.generate(**batch, num_beams=4, max_new_tokens=128)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main() -> None:
    """Translate every non-English query and save an English version of each."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        default="data/evaluation/checkpoint/queries_40_bilingual.csv",
    )
    parser.add_argument(
        "--out",
        default="data/evaluation/checkpoint/queries_translated.csv",
    )
    args = parser.parse_args()

    queries = read_csv(args.queries)
    translations: dict[str, tuple[str, str]] = {}

    # Translate each language with its own model, in one batch per language.
    languages = sorted({query["language"] for query in queries} - {"en"})
    for language in languages:
        if language not in DEFAULT_MODELS:
            raise ValueError(f"No translation model configured for '{language}'")
        model_name = DEFAULT_MODELS[language]
        subset = [query for query in queries if query["language"] == language]
        outputs = translate_texts([query["query"] for query in subset], model_name)
        for query, english in zip(subset, outputs):
            translations[query["query_id"]] = (english.strip(), model_name)
        print(f"Translated {len(subset)} '{language}' queries with {model_name}")

    rows = []
    for query in queries:
        english, model_name = translations.get(
            query["query_id"], (query["query"], "none (already English)")
        )
        rows.append({
            "query_id": query["query_id"],
            "group_id": query["group_id"],
            "language": query["language"],
            "query": query["query"],
            "query_en": english,
            "translation_model": model_name,
        })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
