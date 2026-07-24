"""MedCPT and BM25 retrieval with cache provenance validation."""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TEXT_COLUMNS = ("text", "translated_text", "guidelinetext_processed", "content")
CORPUS_METADATA_SCHEMA_VERSION = 2


def _read_corpus_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(
        f"Unsupported corpus file {path}; expected .pkl, .csv, .parquet, or .xlsx."
    )


def parse_rag_spec(spec: str) -> dict[str, Any]:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid RAG spec {spec!r}; expected name::source1|||source2::cache_dir."
        )
    name, sources, cache_dir = parts
    source_list = [Path(value) for value in sources.split("|||") if value]
    if not name or not source_list or not cache_dir:
        raise ValueError(f"Incomplete RAG spec: {spec!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(
            "RAG name may contain only letters, numbers, dot, underscore, and dash."
        )
    return {"name": name, "sources": source_list, "cache_dir": Path(cache_dir)}


def _source_identity(path: Path) -> dict[str, Any]:
    if path.is_file():
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {"path": str(path)}


def corpus_identity(sources: list[Path]) -> list[dict[str, Any]]:
    return [_source_identity(path) for path in sources]


def load_corpus(sources: list[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    pages: list[str] = []
    metadata: list[dict[str, Any]] = []
    for source in sources:
        if source.is_file():
            frame = _read_corpus_file(source)
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    f"Loading dataset source {source} requires datasets."
                ) from exc
            frame = load_dataset(str(source), split="train").to_pandas()
        column = next((name for name in TEXT_COLUMNS if name in frame), None)
        if column is None:
            raise ValueError(
                f"No supported text column in {source}; expected {TEXT_COLUMNS}."
            )
        for source_row, (_, row) in enumerate(frame.iterrows()):
            value = row[column]
            value = "" if pd.isna(value) else value
            text = str(value).strip()
            if text:
                pages.append(text)
                row_metadata = row.drop(labels=[column]).to_dict()
                row_metadata["corpus_path"] = str(source.resolve())
                row_metadata["corpus_row"] = source_row
                row_metadata["text_column"] = column
                metadata.append(row_metadata)
    if not pages:
        raise ValueError("The retrieval corpus contains no non-empty documents.")
    return pages, metadata


def _manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")) == expected


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def embed_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
    *,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch
    from tqdm import tqdm

    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="Embedding",
            leave=False,
        ):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state[:, 0, :]
            normalized = torch.nn.functional.normalize(hidden, p=2, dim=1)
            embeddings.append(normalized.detach().float().cpu().numpy())
    return np.concatenate(embeddings).astype(np.float32)


def load_or_build_medcpt_corpus(
    *,
    sources: list[Path],
    cache_dir: Path,
    article_encoder_path: str,
    device: Any,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "embeddings": cache_dir / "medcpt_pages.npy",
        "pages": cache_dir / "medcpt_pages.pkl",
        "metadata": cache_dir / "medcpt_metadata.pkl",
        "manifest": cache_dir / "medcpt_manifest.json",
    }
    manifest = {
        "backend": "medcpt",
        "metadata_schema": CORPUS_METADATA_SCHEMA_VERSION,
        "sources": corpus_identity(sources),
        "article_encoder": article_encoder_path,
        "max_length": max_length,
    }
    data_files_exist = all(
        paths[key].is_file() for key in ("embeddings", "pages", "metadata")
    )
    if data_files_exist and _manifest_matches(paths["manifest"], manifest):
        return {
            "pages": pd.read_pickle(paths["pages"]),
            "metadata": pd.read_pickle(paths["metadata"]),
            "embeddings": np.load(paths["embeddings"]),
        }
    if data_files_exist:
        raise ValueError(
            f"MedCPT cache provenance differs in {cache_dir}. "
            "This includes caches created with an older corpus metadata "
            "schema. Use a new cache directory or remove and rebuild the "
            "stale cache."
        )
    from transformers import AutoModel, AutoTokenizer

    pages, metadata = load_corpus(sources)
    tokenizer = AutoTokenizer.from_pretrained(
        article_encoder_path,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        article_encoder_path,
        trust_remote_code=True,
    ).to(device)
    embeddings = embed_texts(
        model,
        tokenizer,
        pages,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )
    np.save(paths["embeddings"], embeddings)
    pd.to_pickle(pages, paths["pages"])
    pd.to_pickle(metadata, paths["metadata"])
    _atomic_json(manifest, paths["manifest"])
    del model
    return {"pages": pages, "metadata": metadata, "embeddings": embeddings}


def _dense_topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("Retrieval k must be positive.")
    k = min(k, len(scores))
    if k == len(scores):
        return np.argsort(-scores)
    indices = np.argpartition(-scores, k - 1)[:k]
    return indices[np.argsort(-scores[indices])]


def dense_topk(query_embedding: np.ndarray, embeddings: np.ndarray, k: int) -> np.ndarray:
    scores = embeddings @ query_embedding
    return _dense_topk_indices(scores, k)


def dense_topk_with_scores(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = embeddings @ query_embedding
    indices = _dense_topk_indices(scores, k)
    return indices, scores[indices].astype(np.float32)


def medcpt_rerank(
    tokenizer: Any,
    model: Any,
    query: str,
    documents: list[str],
    device: Any,
    *,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch

    if not documents:
        return np.array([], dtype=np.float32)
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(documents), batch_size):
            pairs = [[query, document] for document in documents[start : start + batch_size]]
            encoded = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits.reshape(-1)
            scores.append(logits.detach().float().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def load_or_build_bm25_corpus(
    *,
    sources: list[Path],
    cache_dir: Path,
    k1: float,
    b: float,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "bm25_index.pkl"
    pages_path = cache_dir / "bm25_pages.pkl"
    metadata_path = cache_dir / "bm25_metadata.pkl"
    manifest_path = cache_dir / "bm25_manifest.json"
    manifest = {
        "backend": "bm25",
        "metadata_schema": CORPUS_METADATA_SCHEMA_VERSION,
        "sources": corpus_identity(sources),
        "k1": k1,
        "b": b,
        "tokenizer": "lowercase_alphanumeric_v1",
    }
    data_files_exist = all(
        path.is_file() for path in (index_path, pages_path, metadata_path)
    )
    if data_files_exist and _manifest_matches(manifest_path, manifest):
        with index_path.open("rb") as file:
            index = pickle.load(file)
        return {
            **index,
            "pages": pd.read_pickle(pages_path),
            "metadata": pd.read_pickle(metadata_path),
        }
    if data_files_exist:
        raise ValueError(
            f"BM25 cache provenance differs in {cache_dir}. "
            "This includes caches created with an older corpus metadata "
            "schema. Use a new cache directory or remove and rebuild the "
            "stale cache."
        )
    pages, metadata = load_corpus(sources)
    postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    document_lengths = np.zeros(len(pages), dtype=np.float32)
    document_frequencies: Counter[str] = Counter()
    for document_id, page in enumerate(pages):
        frequencies = Counter(bm25_tokenize(page))
        document_lengths[document_id] = sum(frequencies.values())
        for term, frequency in frequencies.items():
            postings[term].append((document_id, frequency))
            document_frequencies[term] += 1
    count = len(pages)
    inverse_document_frequency = {
        term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequencies.items()
    }
    index = {
        "postings": dict(postings),
        "idf": inverse_document_frequency,
        "document_lengths": document_lengths,
        "average_length": float(document_lengths.mean()),
        "k1": k1,
        "b": b,
    }
    with index_path.open("wb") as file:
        pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)
    pd.to_pickle(pages, pages_path)
    pd.to_pickle(metadata, metadata_path)
    _atomic_json(manifest, manifest_path)
    return {**index, "pages": pages, "metadata": metadata}


def bm25_scores(pack: dict[str, Any], query: str) -> np.ndarray:
    scores = np.zeros(len(pack["pages"]), dtype=np.float32)
    average_length = max(pack["average_length"], 1e-6)
    for term, query_frequency in Counter(bm25_tokenize(query)).items():
        for document_id, term_frequency in pack["postings"].get(term, []):
            length = pack["document_lengths"][document_id]
            denominator = term_frequency + pack["k1"] * (
                1 - pack["b"] + pack["b"] * length / average_length
            )
            scores[document_id] += (
                query_frequency
                * pack["idf"][term]
                * term_frequency
                * (pack["k1"] + 1)
                / max(denominator, 1e-12)
            )
    return scores


def bm25_topk(pack: dict[str, Any], query: str, k: int) -> np.ndarray:
    scores = bm25_scores(pack, query)
    nonzero = np.flatnonzero(scores)
    if not len(nonzero):
        return np.array([], dtype=np.int64)
    order = nonzero[np.argsort(-scores[nonzero])]
    return order[:k].astype(np.int64)


def rank_candidates(
    *,
    backend: str,
    pack: dict[str, Any],
    candidate_indices: np.ndarray,
    query: str,
    cross_tokenizer: Any = None,
    cross_model: Any = None,
    device: Any = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    indices, _ = rank_candidates_with_scores(
        backend=backend,
        pack=pack,
        candidate_indices=candidate_indices,
        query=query,
        cross_tokenizer=cross_tokenizer,
        cross_model=cross_model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return indices


def rank_candidates_with_scores(
    *,
    backend: str,
    pack: dict[str, Any],
    candidate_indices: np.ndarray,
    query: str,
    cross_tokenizer: Any = None,
    cross_model: Any = None,
    device: Any = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(candidate_indices):
        return candidate_indices, np.array([], dtype=np.float32)
    if backend == "bm25":
        scores = bm25_scores(pack, query)[candidate_indices]
    elif backend == "medcpt":
        documents = [pack["pages"][int(index)] for index in candidate_indices]
        scores = medcpt_rerank(
            cross_tokenizer,
            cross_model,
            query,
            documents,
            device,
            batch_size=batch_size,
            max_length=max_length,
        )
    else:
        raise ValueError(f"Unsupported retrieval backend: {backend}")
    order = np.argsort(-scores)
    return (
        candidate_indices[order].astype(np.int64),
        scores[order].astype(np.float32),
    )
