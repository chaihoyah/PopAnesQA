"""Run PopAnesQA Baseline, Base-RAG, Prof-RAG, and optional Gold-RAG."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .evaluation_utils import (
        evaluate_predictions,
        load_evaluation_data,
        retrieval_query_text,
        stable_tag,
        update_metrics_csv,
    )
    from .experiment_runner import (
        RetrievalModels,
        ensure_profiles,
        generate_answers,
        load_generator,
        load_medcpt_models,
        retrieve,
    )
    from .retrieval_utils import (
        load_or_build_bm25_corpus,
        load_or_build_medcpt_corpus,
        parse_rag_spec,
    )
except ImportError:
    from evaluation_utils import evaluate_predictions, load_evaluation_data, retrieval_query_text, stable_tag, update_metrics_csv
    from experiment_runner import RetrievalModels, ensure_profiles, generate_answers, load_generator, load_medcpt_models, retrieve
    from retrieval_utils import load_or_build_bm25_corpus, load_or_build_medcpt_corpus, parse_rag_spec


def atomic_pickle(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temporary)
    temporary.replace(path)


def extraction_identity(data_path: Path, model_path: str, population: str) -> dict[str, Any]:
    stat = data_path.stat()
    return {
        "data_path": str(data_path.resolve()),
        "data_size": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "model_path": model_path,
        "target_population": population,
    }


def load_or_create_profiles(
    frame: pd.DataFrame,
    *,
    data_path: Path,
    model_path: str,
    model: Any,
    tokenizer: Any,
    cache_path: Path,
    population: str,
    batch_size: int,
    extraction_max_tokens: int,
    seed: int,
) -> pd.DataFrame:
    identity = extraction_identity(data_path, model_path, population)
    if cache_path.is_file():
        cached = pd.read_pickle(cache_path)
        if cached.attrs.get("extraction_identity") != identity:
            raise ValueError(
                f"Extraction cache does not match this run: {cache_path}"
            )
        if len(cached) != len(frame):
            raise ValueError(f"Extraction cache row count differs: {cache_path}")
        return cached
    extracted = ensure_profiles(
        frame,
        model,
        tokenizer,
        batch_size=batch_size,
        max_tokens=extraction_max_tokens,
        seed=seed,
    )
    extracted.attrs["extraction_identity"] = identity
    atomic_pickle(extracted, cache_path)
    return extracted


def gold_contexts(frame: pd.DataFrame, path: Path, text_column: str) -> list[list[str]]:
    gold = pd.read_pickle(path)
    required = {"guideline", "page", text_column}
    missing = sorted(required - set(gold.columns))
    if missing:
        raise ValueError(f"Gold guideline data are missing columns: {missing}")
    if not {"guideline", "page"}.issubset(frame.columns):
        raise ValueError("Gold-RAG requires guideline and page in evaluation data.")
    lookup: dict[tuple[str, str], str] = {}
    for _, row in gold.iterrows():
        key = (_join_key(row["guideline"]), _join_key(row["page"]))
        text = str(row[text_column]).strip()
        if key in lookup and lookup[key] != text:
            raise ValueError(f"Duplicate gold guideline/page with different text: {key}")
        lookup[key] = text
    contexts = []
    missing_keys = []
    for _, row in frame.iterrows():
        key = (_join_key(row["guideline"]), _join_key(row["page"]))
        text = lookup.get(key, "")
        contexts.append([text] if text else [])
        if not text:
            missing_keys.append(key)
    if missing_keys:
        raise ValueError(
            f"{len(missing_keys)} evaluation rows have no gold context; "
            f"examples: {missing_keys[:5]}"
        )
    return contexts


def _join_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    return str(int(numeric)) if numeric.is_integer() else text


def result_path(
    output_dir: Path,
    *,
    model_tag: str,
    dataset_tag: str,
    method: str,
    backend: str | None = None,
    rag_name: str | None = None,
) -> Path:
    if method == "baseline":
        return output_dir / "baseline" / model_tag / f"{dataset_tag}.pkl"
    label = "_".join(part for part in (rag_name, backend) if part)
    return output_dir / label / method / model_tag / f"{dataset_tag}.pkl"


def save_or_load_result(
    *,
    source: pd.DataFrame,
    path: Path,
    prediction_column: str,
    answers: list[str] | None,
    skip_existing: bool,
) -> pd.DataFrame:
    if path.is_file() and skip_existing:
        result = pd.read_pickle(path)
        if prediction_column not in result or len(result) != len(source):
            raise ValueError(f"Existing result is incompatible: {path}")
        print(f"[SKIP] Loaded existing result: {path}")
        return result
    if path.exists():
        raise FileExistsError(f"Result exists; use --skip-existing: {path}")
    if answers is None:
        raise ValueError("New result requires generated answers.")
    result = source.copy()
    result[prediction_column] = answers
    atomic_pickle(result, path)
    print(f"Saved: {path}")
    return result


def prepare_retrieval(args: argparse.Namespace, device: Any) -> tuple[dict, RetrievalModels]:
    specs = [parse_rag_spec(value) for value in args.rag_spec]
    packs: dict[tuple[str, str], dict[str, Any]] = {}
    models = RetrievalModels()
    if not specs or not any(
        method in {"base_rag", "prof_rag"} for method in args.methods
    ):
        return packs, models
    if "medcpt" in args.backends:
        for required in (
            "medcpt_query_encoder",
            "medcpt_article_encoder",
            "medcpt_cross_encoder",
        ):
            if not getattr(args, required):
                raise ValueError(f"--{required.replace('_', '-')} is required.")
        models = load_medcpt_models(
            query_encoder=args.medcpt_query_encoder,
            cross_encoder=args.medcpt_cross_encoder,
            device=device,
        )
    for spec in specs:
        for backend in args.backends:
            cache = spec["cache_dir"] / backend
            if backend == "medcpt":
                pack = load_or_build_medcpt_corpus(
                    sources=spec["sources"],
                    cache_dir=cache,
                    article_encoder_path=args.medcpt_article_encoder,
                    device=device,
                    batch_size=args.embedding_batch_size,
                    max_length=args.document_max_length,
                )
            else:
                pack = load_or_build_bm25_corpus(
                    sources=spec["sources"],
                    cache_dir=cache,
                    k1=args.bm25_k1,
                    b=args.bm25_b,
                )
            packs[(spec["name"], backend)] = pack
    return packs, models


def run(args: argparse.Namespace) -> None:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("[INFO] CUDA_VISIBLE_DEVICES is not set; using visible system GPUs.")
    import torch

    device = torch.device(args.retrieval_device)
    packs, retrieval_models = prepare_retrieval(args, device)
    specs = [parse_rag_spec(value) for value in args.rag_spec]
    all_metrics: list[dict[str, Any]] = []

    for model_path in args.model_paths:
        model_name = stable_tag(model_path)
        generator, tokenizer = load_generator(
            model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_length=args.max_model_length,
            download_dir=args.model_download_dir,
        )
        for data_path in args.data_paths:
            dataset_name = stable_tag(str(data_path))
            if args.target_population.lower() != "all":
                dataset_name += f"-{args.target_population.lower()}"
            frame = load_evaluation_data(data_path, args.target_population)
            if any(
                method in {"base_rag", "prof_rag"} for method in args.methods
            ):
                extraction_cache = (
                    args.output_dir
                    / "_extractions"
                    / model_name
                    / f"{dataset_name}.pkl"
                )
                frame = load_or_create_profiles(
                    frame,
                    data_path=data_path,
                    model_path=model_path,
                    model=generator,
                    tokenizer=tokenizer,
                    cache_path=extraction_cache,
                    population=args.target_population,
                    batch_size=args.batch_size,
                    extraction_max_tokens=args.extraction_max_tokens,
                    seed=args.seed,
                )

            if "baseline" in args.methods:
                path = result_path(
                    args.output_dir,
                    model_tag=model_name,
                    dataset_tag=dataset_name,
                    method="baseline",
                )
                answers = None
                if not (path.is_file() and args.skip_existing):
                    answers = generate_answers(
                        frame,
                        generator,
                        tokenizer,
                        [[] for _ in range(len(frame))],
                        batch_size=args.batch_size,
                        max_model_length=args.max_model_length,
                        max_output_tokens=args.answer_max_tokens,
                        max_chars_each=args.max_chars_each,
                        seed=args.seed,
                    )
                result = save_or_load_result(
                    source=frame,
                    path=path,
                    prediction_column="baseline_answer",
                    answers=answers,
                    skip_existing=args.skip_existing,
                )
                all_metrics.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "method": "baseline",
                        **evaluate_predictions(result, "baseline_answer"),
                    }
                )

            if "gold_rag" in args.methods:
                if args.gold_guideline is None:
                    raise ValueError("gold_rag requires --gold-guideline.")
                contexts = gold_contexts(
                    frame,
                    args.gold_guideline,
                    args.gold_text_column,
                )
                path = result_path(
                    args.output_dir,
                    model_tag=model_name,
                    dataset_tag=dataset_name,
                    method="gold_rag",
                    rag_name="gold",
                )
                answers = None if path.is_file() and args.skip_existing else generate_answers(
                    frame,
                    generator,
                    tokenizer,
                    contexts,
                    batch_size=args.batch_size,
                    max_model_length=args.max_model_length,
                    max_output_tokens=args.answer_max_tokens,
                    max_chars_each=args.max_chars_each,
                    seed=args.seed,
                )
                result = save_or_load_result(
                    source=frame,
                    path=path,
                    prediction_column="gold_rag_answer",
                    answers=answers,
                    skip_existing=args.skip_existing,
                )
                all_metrics.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "method": "gold_rag",
                        **evaluate_predictions(result, "gold_rag_answer"),
                    }
                )

            for spec in specs:
                for backend in args.backends:
                    for method in ("base_rag", "prof_rag"):
                        if method not in args.methods:
                            continue
                        path = result_path(
                            args.output_dir,
                            model_tag=model_name,
                            dataset_tag=dataset_name,
                            method=method,
                            backend=backend,
                            rag_name=spec["name"],
                        )
                        answers = None
                        candidates: list[list[str]] = []
                        middle_contexts: list[list[str]] = []
                        contexts: list[list[str]] = []
                        trajectories: list[dict[str, Any]] = []
                        if not (path.is_file() and args.skip_existing):
                            pack = packs[(spec["name"], backend)]
                            for _, row in frame.iterrows():
                                trajectory = retrieve(
                                    condition=method,
                                    backend=backend,
                                    pack=pack,
                                    models=retrieval_models,
                                    profile=str(row["extracted_patient_profile"]),
                                    core=str(row["extracted_question_core"]),
                                    full=retrieval_query_text(row),
                                    device=device,
                                    first_k=args.first_k,
                                    middle_k=args.middle_k,
                                    final_k=args.final_k,
                                    query_max_length=args.query_max_length,
                                    rerank_batch_size=args.rerank_batch_size,
                                    cross_max_length=args.cross_max_length,
                                )
                                trajectories.append(trajectory)
                                candidates.append(
                                    [
                                        document["text"]
                                        for document in trajectory["initial"]["documents"]
                                    ]
                                )
                                middle_contexts.append(
                                    [
                                        document["text"]
                                        for document in trajectory["middle"]["documents"]
                                    ]
                                )
                                contexts.append(
                                    [
                                        document["text"]
                                        for document in trajectory["final"]["documents"]
                                    ]
                                )
                            answers = generate_answers(
                                frame,
                                generator,
                                tokenizer,
                                contexts,
                                batch_size=args.batch_size,
                                max_model_length=args.max_model_length,
                                max_output_tokens=args.answer_max_tokens,
                                max_chars_each=args.max_chars_each,
                                seed=args.seed,
                            )
                        result = save_or_load_result(
                            source=frame,
                            path=path,
                            prediction_column=f"{method}_answer",
                            answers=answers,
                            skip_existing=args.skip_existing,
                        )
                        if answers is not None:
                            result["candidate_texts"] = candidates
                            result["middle_texts"] = middle_contexts
                            result["retrieved_texts"] = contexts
                            result["retrieval_trajectory"] = trajectories
                            atomic_pickle(result, path)
                        all_metrics.append(
                            {
                                "model": model_name,
                                "dataset": dataset_name,
                                "method": method,
                                "backend": backend,
                                "corpus": spec["name"],
                                **evaluate_predictions(
                                    result,
                                    f"{method}_answer",
                                ),
                            }
                        )
        del generator
        torch.cuda.empty_cache()

    metrics_path = args.output_dir / "metrics.csv"
    update_metrics_csv(
        all_metrics,
        metrics_path,
        key_columns=["model", "dataset", "method", "backend", "corpus"],
    )
    print(f"Saved metrics: {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-paths", nargs="+", required=True)
    parser.add_argument("--data-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rag-spec", action="append", default=[])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("baseline", "base_rag", "prof_rag", "gold_rag"),
        default=("baseline", "base_rag", "prof_rag"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("medcpt", "bm25"),
        default=("medcpt",),
    )
    parser.add_argument("--gold-guideline", type=Path)
    parser.add_argument("--gold-text-column", default="summary_text")
    parser.add_argument("--target-population", default="all")
    parser.add_argument("--model-download-dir")
    parser.add_argument("--retrieval-device", default="cuda:0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--answer-max-tokens", type=int, default=3000)
    parser.add_argument("--extraction-max-tokens", type=int, default=512)
    parser.add_argument("--max-chars-each", type=int, default=4500)
    parser.add_argument("--first-k", type=int, default=64)
    parser.add_argument("--middle-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=1)
    parser.add_argument("--query-max-length", type=int, default=512)
    parser.add_argument("--document-max-length", type=int, default=512)
    parser.add_argument("--cross-max-length", type=int, default=512)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--medcpt-query-encoder")
    parser.add_argument("--medcpt-article-encoder")
    parser.add_argument("--medcpt-cross-encoder")
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(method in {"base_rag", "prof_rag"} for method in args.methods) and not args.rag_spec:
        raise ValueError("RAG methods require at least one --rag-spec.")
    numeric_values = {
        "--first-k": args.first_k,
        "--middle-k": args.middle_k,
        "--final-k": args.final_k,
        "--max-model-length": args.max_model_length,
        "--answer-max-tokens": args.answer_max_tokens,
    }
    invalid = [name for name, value in numeric_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.middle_k > args.first_k or args.final_k > args.middle_k:
        raise ValueError("Require first-k >= middle-k >= final-k.")
    run(args)


if __name__ == "__main__":
    main()
