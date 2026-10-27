"""Step 3 - retrieval: the closest discipline rules and labelled examples for a page."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.schemas import PlanFeatures, RetrievedRef
from app.pipeline.llm import get_embeddings

K_RULES = 4
K_EXAMPLES = 4
ALWAYS_RULE = "not_a_plan"


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float  # cosine similarity, 1 = identical


@dataclass(frozen=True)
class Retrieval:
    rules: list[Hit]
    examples: list[Hit]

    def refs(self) -> list[RetrievedRef]:
        """The `retrieved` field of the API output."""
        out = [
            RetrievedRef(type="rule", id=h.doc.metadata["id"], source=h.doc.metadata.get("source"), score=h.score)
            for h in self.rules
        ]
        out += [
            RetrievedRef(
                type="example",
                file=h.doc.metadata["file"],
                label=h.doc.metadata["label"],
                source=h.doc.metadata.get("source") or None,
                score=h.score,
            )
            for h in self.examples
        ]
        return out

    def example_labels(self) -> list[str]:
        return [h.doc.metadata["label"] for h in self.examples]


class IndexMissingError(RuntimeError):
    pass


@lru_cache
def get_index() -> FAISS:
    path = get_settings().kb_index_dir
    if not (path / "index.faiss").exists():
        raise IndexMissingError(f"No FAISS index at {path}; run: python -m scripts.ingest_kb")
    # The pickle is written by our own ingest script, so loading it is safe.
    return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)


def _to_similarity(distance: float) -> float:
    # Vectors are unit length, so FAISS's squared L2 distance d = 2 - 2*cos.
    return round(1 - float(distance) / 2, 4)  # float(): FAISS returns numpy float32


def search(
    vector: list[float],
    doc_type: str,
    k: int,
    exclude_file: str | None = None,
    index: FAISS | None = None,
) -> list[Hit]:
    """Top-k documents of one type ('rule' or 'example') for a query vector."""
    index = index or get_index()
    # fetch_k = whole index: the type filter is applied after the search, so this
    # guarantees k results of the wanted type (the index is tiny).
    results = index.similarity_search_with_score_by_vector(
        vector, k=k + 1, filter={"type": doc_type}, fetch_k=index.index.ntotal
    )
    hits = [
        Hit(doc, _to_similarity(dist))
        for doc, dist in results
        if exclude_file is None or doc.metadata.get("file") != exclude_file
    ]
    return hits[:k]


def top_rules(vector: list[float], k: int = K_RULES, index: FAISS | None = None) -> list[Hit]:
    """Top-k rules for k different labels, plus the not_a_plan rule if it did not make the cut.

    A label can have several rule parts (fire_safety: sprinklers / evacuation /
    alarm); only its best-matching part is kept, so the context still covers k
    disciplines. "Is this a plan at all?" is the first question of every
    decision, so the rejection rule is always included (its text is about
    drawings, so invoices or photos do not always land near it).
    """
    index = index or get_index()
    ranked, seen = [], set()
    for hit in search(vector, "rule", index.index.ntotal, index=index):
        if hit.doc.metadata["id"] not in seen:
            seen.add(hit.doc.metadata["id"])
            ranked.append(hit)
    hits = ranked[:k]
    if all(h.doc.metadata["id"] != ALWAYS_RULE for h in hits):
        hits += [h for h in ranked if h.doc.metadata["id"] == ALWAYS_RULE]
    return hits


def retrieve(features: PlanFeatures, k_rules: int = K_RULES, k_examples: int = K_EXAMPLES) -> Retrieval:
    """Embed the page features once and search rules and examples separately."""
    vector = get_embeddings().embed_query(features.to_embedding_text())
    return Retrieval(rules=top_rules(vector, k_rules), examples=search(vector, "example", k_examples))
