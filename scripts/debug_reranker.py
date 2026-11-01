import json
from pathlib import Path

from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"

EVAL_DIR = ROOT / "data" / "eval"

with (EVAL_DIR / "evaluation_corpus.json").open(encoding="utf-8") as file:
    chunks = json.load(file)

with (EVAL_DIR / "golden_dataset.json").open(encoding="utf-8") as file:
    dataset = json.load(file)

retriever = RetrievalEngine(
    chunks=chunks,
    collection_name="debug_reranker",
)

reranker = CrossEncoderReranker()

target_ids = {"q007", "q019", "q027", "q010"}

for item in dataset:
    if item["id"] not in target_ids:
        continue

    question = item["question"]

    candidates = retriever.search_hybrid(question, top_k=6)
    reranked = reranker.rerank(question, candidates, top_k=6)

    print("\n" + "=" * 80)
    print(f"ID: {item['id']}")
    print(f"Question: {question}")
    print(f"Expected answerable: {item['answerable']}")

    for rank, chunk in enumerate(reranked, start=1):
        print(f"\nRank: {rank}")
        print(f"Source: {chunk.get('source')}")
        print(f"Score: {chunk.get('reranker_score'):.4f}")
        print(f"Text: {chunk.get('text')}")
