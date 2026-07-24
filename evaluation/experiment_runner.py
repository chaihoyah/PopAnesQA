"""Shared model loading, profile extraction, retrieval, and generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable

try:
    from .evaluation_utils import (
        apply_chat_template,
        prompt_with_budget,
        retrieval_query_text,
    )
    from .retrieval_utils import (
        bm25_scores,
        dense_topk_with_scores,
        embed_texts,
        rank_candidates_with_scores,
    )
except ImportError:
    from evaluation_utils import (
        apply_chat_template,
        prompt_with_budget,
        retrieval_query_text,
    )
    from retrieval_utils import (
        bm25_scores,
        dense_topk_with_scores,
        embed_texts,
        rank_candidates_with_scores,
    )


@dataclass
class RetrievalModels:
    query_tokenizer: Any = None
    query_model: Any = None
    cross_tokenizer: Any = None
    cross_model: Any = None


CONDITIONS = {
    "base_rag": ("full", "full", None),
    "prof_rag": ("question", "combined", "profile"),
    "q_qp": ("question", "combined", None),
    "q_q": ("question", "question", None),
}


def load_generator(
    model_path: str,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_length: int,
    download_dir: str | None,
) -> tuple[Any, Any]:
    from vllm import LLM

    model = LLM(
        model=model_path,
        download_dir=download_dir,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_length,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_medcpt_models(
    *,
    query_encoder: str,
    cross_encoder: str,
    device: Any,
) -> RetrievalModels:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    query_tokenizer = AutoTokenizer.from_pretrained(
        query_encoder,
        trust_remote_code=True,
    )
    query_model = AutoModel.from_pretrained(
        query_encoder,
        trust_remote_code=True,
    ).to(device)
    cross_tokenizer = AutoTokenizer.from_pretrained(
        cross_encoder,
        trust_remote_code=True,
    )
    cross_model = AutoModelForSequenceClassification.from_pretrained(
        cross_encoder,
        trust_remote_code=True,
    ).to(device)
    query_model.eval()
    cross_model.eval()
    return RetrievalModels(
        query_tokenizer,
        query_model,
        cross_tokenizer,
        cross_model,
    )


PROFILE_PROMPT = """You are a clinical information extraction assistant.

Your task is to separate a clinical question into two parts:
1). PATIENT_PROFILE - Basic demographic information about the patient, such as age, sex, or weight. Do not include any clinical information in this part, such as clinical history or lab values.
2). QUESTION_CORE - The remaining clinical question content describing the medical situation, condition, procedure, or decision.

Return the output strictly in following JSON format without any explanation and reasoning.
{{
    "PATIENT_PROFILE": "extracted patient profile",
    "QUESTION_CORE": "extracted question core"
}}

If no patient profile exists, return:
PATIENT_PROFILE = "general patient"
QUESTION_CORE = original question

Clinical Question:
{question}

Output:
"""


def parse_profile(text: str, original: str) -> tuple[str, str]:
    cleaned = text.strip()
    if "```" in cleaned:
        pieces = cleaned.split("```")
        cleaned = pieces[1].removeprefix("json").strip() if len(pieces) > 1 else cleaned
    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return "general patient", original
    profile = str(value.get("PATIENT_PROFILE", "")).strip()
    core = str(value.get("QUESTION_CORE", "")).strip()
    if not profile or "general patient" in profile.lower():
        return "general patient", original
    return profile, core or original


def ensure_profiles(
    frame: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    max_tokens: int,
    seed: int,
) -> pd.DataFrame:
    if {
        "extracted_patient_profile",
        "extracted_question_core",
    }.issubset(frame.columns) and frame[
        ["extracted_patient_profile", "extracted_question_core"]
    ].notna().all().all():
        return frame.copy()
    from vllm import SamplingParams

    output = frame.copy()
    profiles: list[str] = []
    cores: list[str] = []
    parameters = SamplingParams(
        temperature=0.0,
        seed=seed,
        max_tokens=max_tokens,
    )
    for start in tqdm(
        range(0, len(output), batch_size),
        desc="Extracting profile/core",
        leave=False,
    ):
        batch = output.iloc[start : start + batch_size]
        originals = [retrieval_query_text(row) for _, row in batch.iterrows()]
        prompts = [
            apply_chat_template(
                tokenizer,
                PROFILE_PROMPT.format(question=original),
            )
            for original in originals
        ]
        generated = model.generate(prompts, parameters)
        for result, original in zip(generated, originals):
            profile, core = parse_profile(result.outputs[0].text, original)
            profiles.append(profile)
            cores.append(core)
    output["extracted_patient_profile"] = profiles
    output["extracted_question_core"] = cores
    return output


def _query(kind: str, profile: str, core: str, full: str) -> str:
    if kind == "profile":
        return profile
    if kind == "question":
        return core
    if kind == "combined":
        return f"{profile} {core}".strip()
    if kind == "full":
        return full
    raise ValueError(f"Unknown query component: {kind}")


def _cache_result(
    cache: dict[tuple[Any, ...], Any] | None,
    key: tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    if cache is None:
        return None
    value = cache.get(key)
    if value is None:
        return None
    indices, scores = value
    return np.asarray(indices), np.asarray(scores)


def _stage_documents(
    pack: dict[str, Any],
    indices: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    documents = []
    metadata = pack.get("metadata", [])
    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        corpus_index = int(index)
        document_metadata = (
            dict(metadata[corpus_index]) if corpus_index < len(metadata) else {}
        )
        documents.append(
            {
                "rank": rank,
                "corpus_index": corpus_index,
                "score": float(score),
                "text": str(pack["pages"][corpus_index]),
                "metadata": document_metadata,
            }
        )
    return documents


def retrieve(
    *,
    condition: str,
    backend: str,
    pack: dict[str, Any],
    models: RetrievalModels,
    profile: str,
    core: str,
    full: str,
    device: Any,
    first_k: int,
    middle_k: int,
    final_k: int,
    query_max_length: int,
    rerank_batch_size: int,
    cross_max_length: int,
    cache: dict[tuple[Any, ...], Any] | None = None,
) -> dict[str, Any]:
    initial_kind, middle_kind, final_kind = CONDITIONS[condition]
    initial_query = _query(initial_kind, profile, core, full)
    retrieval_key = ("retrieve", backend, initial_query, first_k)
    initial_result = _cache_result(cache, retrieval_key)
    if initial_result is None:
        if backend == "bm25":
            all_scores = bm25_scores(pack, initial_query)
            nonzero = np.flatnonzero(all_scores)
            candidates = nonzero[np.argsort(-all_scores[nonzero])][:first_k]
            candidates = candidates.astype(np.int64)
            candidate_scores = all_scores[candidates].astype(np.float32)
        elif backend == "medcpt":
            query_embedding = embed_texts(
                models.query_model,
                models.query_tokenizer,
                [initial_query],
                device,
                batch_size=1,
                max_length=query_max_length,
            )[0]
            candidates, candidate_scores = dense_topk_with_scores(
                query_embedding,
                pack["embeddings"],
                first_k,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        if cache is not None:
            cache[retrieval_key] = (candidates, candidate_scores)
    else:
        candidates, candidate_scores = initial_result

    middle_query = _query(middle_kind, profile, core, full)
    middle_key = ("rank", backend, middle_query, tuple(candidates.tolist()))
    middle_result = _cache_result(cache, middle_key)
    if middle_result is None:
        ranked, ranked_scores = rank_candidates_with_scores(
            backend=backend,
            pack=pack,
            candidate_indices=candidates,
            query=middle_query,
            cross_tokenizer=models.cross_tokenizer,
            cross_model=models.cross_model,
            device=device,
            batch_size=rerank_batch_size,
            max_length=cross_max_length,
        )
        if cache is not None:
            cache[middle_key] = (ranked, ranked_scores)
    else:
        ranked, ranked_scores = middle_result
    middle_indices = ranked[:middle_k]
    middle_scores = ranked_scores[:middle_k]

    final_reranked = final_kind is not None
    if final_reranked:
        final_query = _query(final_kind, profile, core, full)
        final_key = ("rank", backend, final_query, tuple(middle_indices.tolist()))
        final_result = _cache_result(cache, final_key)
        if final_result is None:
            final_ranked, final_ranked_scores = rank_candidates_with_scores(
                backend=backend,
                pack=pack,
                candidate_indices=middle_indices,
                query=final_query,
                cross_tokenizer=models.cross_tokenizer,
                cross_model=models.cross_model,
                device=device,
                batch_size=rerank_batch_size,
                max_length=cross_max_length,
            )
            if cache is not None:
                cache[final_key] = (final_ranked, final_ranked_scores)
        else:
            final_ranked, final_ranked_scores = final_result
        final_indices = final_ranked[:final_k]
        final_scores = final_ranked_scores[:final_k]
        final_query_kind = final_kind
    else:
        final_indices = middle_indices[:final_k]
        final_scores = middle_scores[:final_k]
        final_query = middle_query
        final_query_kind = middle_kind

    return {
        "condition": condition,
        "backend": backend,
        "initial": {
            "query_kind": initial_kind,
            "query": initial_query,
            "requested_k": first_k,
            "documents": _stage_documents(
                pack,
                candidates,
                candidate_scores,
            ),
        },
        "middle": {
            "query_kind": middle_kind,
            "query": middle_query,
            "requested_k": middle_k,
            "documents": _stage_documents(
                pack,
                middle_indices,
                middle_scores,
            ),
        },
        "final": {
            "query_kind": final_query_kind,
            "query": final_query,
            "requested_k": final_k,
            "reranked": final_reranked,
            "documents": _stage_documents(
                pack,
                final_indices,
                final_scores,
            ),
        },
    }


def generate_answers(
    frame: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    retrieved: list[list[str]],
    *,
    batch_size: int,
    max_model_length: int,
    max_output_tokens: int,
    max_chars_each: int,
    seed: int,
) -> list[str]:
    from vllm import SamplingParams

    if len(frame) != len(retrieved):
        raise ValueError("Retrieved-context count does not match evaluation rows.")
    parameters = SamplingParams(
        temperature=0.0,
        seed=seed,
        max_tokens=max_output_tokens,
    )
    answers: list[str] = []
    for start in tqdm(
        range(0, len(frame), batch_size),
        desc="Generating answers",
        leave=False,
    ):
        batch = frame.iloc[start : start + batch_size]
        prompts = [
            prompt_with_budget(
                tokenizer,
                row,
                retrieved[start + offset],
                max_model_length=max_model_length,
                max_output_tokens=max_output_tokens,
                max_chars_each=max_chars_each,
            )
            for offset, (_, row) in enumerate(batch.iterrows())
        ]
        generated = model.generate(prompts, parameters)
        answers.extend(result.outputs[0].text.strip() for result in generated)
    return answers
