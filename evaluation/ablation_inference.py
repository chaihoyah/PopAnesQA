"""Run ranking-stage or profile-replacement ablations for Prof-RAG."""

from __future__ import annotations

import argparse
import random
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
    from .inference_evaluation import atomic_pickle
    from .retrieval_utils import (
        load_or_build_bm25_corpus,
        load_or_build_medcpt_corpus,
        parse_rag_spec,
    )
except ImportError:
    from evaluation_utils import evaluate_predictions, load_evaluation_data, retrieval_query_text, stable_tag, update_metrics_csv
    from experiment_runner import RetrievalModels, ensure_profiles, generate_answers, load_generator, load_medcpt_models, retrieve
    from inference_evaluation import atomic_pickle
    from retrieval_utils import load_or_build_bm25_corpus, load_or_build_medcpt_corpus, parse_rag_spec


RANKING_CONDITIONS = {
    "q_q": "Q retrieval → Q reranking",
    "q_qp": "Q retrieval → Q+P reranking",
    "prof_rag": "Q retrieval → Q+P reranking → P reranking",
}
PROFILE_CONDITIONS = ("original", "random", "within_group", "cross_group")


def load_extractions(
    frame: pd.DataFrame,
    path: Path | None,
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_tokens: int,
    seed: int,
) -> pd.DataFrame:
    if path is None:
        return ensure_profiles(
            frame,
            model,
            tokenizer,
            batch_size=batch_size,
            max_tokens=max_tokens,
            seed=seed,
        )
    extracted = pd.read_pickle(path)
    required = {"extracted_patient_profile", "extracted_question_core"}
    missing = sorted(required - set(extracted.columns))
    if missing:
        raise ValueError(f"Extraction data are missing columns: {missing}")
    if len(extracted) != len(frame):
        raise ValueError("Extraction and evaluation row counts differ.")
    if "question" in extracted and not extracted["question"].astype(str).equals(
        frame["question"].astype(str)
    ):
        raise ValueError("Extraction rows do not align with evaluation questions.")
    result = frame.copy()
    result["extracted_patient_profile"] = extracted[
        "extracted_patient_profile"
    ].to_numpy()
    result["extracted_question_core"] = extracted[
        "extracted_question_core"
    ].to_numpy()
    return result


def replacement_profiles(
    frame: pd.DataFrame,
    condition: str,
    *,
    seed: int,
    pool_limit: int,
) -> list[str]:
    originals = frame["extracted_patient_profile"].astype(str).tolist()
    if condition == "original":
        return originals
    if "target_population" not in frame:
        raise ValueError("Profile ablation requires target_population.")
    populations = frame["target_population"].astype(str).tolist()
    invalid = sorted(set(populations) - {"Adult", "Pediatric", "General"})
    if invalid:
        raise ValueError(
            f"Unknown target_population values in profile ablation: {invalid}."
        )
    rng = random.Random(seed)
    pools = {}
    for population in ("Pediatric", "Adult"):
        population_profiles = list(
            dict.fromkeys(
                profile
                for profile, row_population in zip(originals, populations)
                if row_population == population and profile != "general patient"
            )
        )
        rng.shuffle(population_profiles)
        pools[population] = population_profiles[:pool_limit]
    all_profiles = list(dict.fromkeys(pools["Pediatric"] + pools["Adult"]))
    rng.shuffle(all_profiles)
    replacements = []
    for index, (original, population) in enumerate(zip(originals, populations)):
        # This reproduces the original ablation: General rows follow the Adult
        # branch, while candidate pools themselves contain only Adult/Pediatric
        # profiles.
        profile_group = "Pediatric" if population == "Pediatric" else "Adult"
        if condition == "random":
            candidates = [value for value in all_profiles if value != original]
        elif condition == "within_group":
            candidates = [
                value for value in pools[profile_group] if value != original
            ]
        elif condition == "cross_group":
            opposite = (
                "Adult" if profile_group == "Pediatric" else "Pediatric"
            )
            candidates = pools[opposite]
        else:
            raise ValueError(f"Unknown profile condition: {condition}")
        if not candidates:
            raise ValueError(
                f"No replacement candidate for row {index}, condition={condition}."
            )
        replacements.append(rng.choice(candidates))
    return replacements


def prepare_pack(
    args: argparse.Namespace,
    device: Any,
) -> tuple[dict[str, Any], RetrievalModels, str]:
    spec = parse_rag_spec(args.rag_spec)
    models = RetrievalModels()
    cache = spec["cache_dir"] / args.backend
    if args.backend == "medcpt":
        required = (
            args.medcpt_query_encoder,
            args.medcpt_article_encoder,
            args.medcpt_cross_encoder,
        )
        if not all(required):
            raise ValueError("MedCPT backend requires all three encoder paths.")
        models = load_medcpt_models(
            query_encoder=args.medcpt_query_encoder,
            cross_encoder=args.medcpt_cross_encoder,
            device=device,
        )
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
    return pack, models, spec["name"]


def run_condition(
    frame: pd.DataFrame,
    *,
    retrieval_condition: str,
    profiles: list[str],
    pack: dict[str, Any],
    retrieval_models: RetrievalModels,
    generator: Any,
    tokenizer: Any,
    device: Any,
    args: argparse.Namespace,
    retrieval_cache: dict[tuple[Any, ...], Any],
) -> pd.DataFrame:
    candidates: list[list[str]] = []
    middle_contexts: list[list[str]] = []
    contexts: list[list[str]] = []
    trajectories: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        trajectory = retrieve(
            condition=retrieval_condition,
            backend=args.backend,
            pack=pack,
            models=retrieval_models,
            profile=profiles[position],
            core=str(row["extracted_question_core"]),
            full=retrieval_query_text(row),
            device=device,
            first_k=args.first_k,
            middle_k=args.middle_k,
            final_k=args.final_k,
            query_max_length=args.query_max_length,
            rerank_batch_size=args.rerank_batch_size,
            cross_max_length=args.cross_max_length,
            cache=retrieval_cache,
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
    result = frame.copy()
    result["ablation_profile"] = profiles
    result["candidate_texts"] = candidates
    result["middle_texts"] = middle_contexts
    result["retrieved_texts"] = contexts
    result["retrieval_trajectory"] = trajectories
    result["ablation_answer"] = answers
    return result


def run(args: argparse.Namespace) -> None:
    import torch

    device = torch.device(args.retrieval_device)
    frame = load_evaluation_data(args.data_path, args.target_population)
    generator, tokenizer = load_generator(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_length=args.max_model_length,
        download_dir=args.model_download_dir,
    )
    frame = load_extractions(
        frame,
        args.extractions,
        model=generator,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_tokens=args.extraction_max_tokens,
        seed=args.seed,
    )
    pack, retrieval_models, corpus_name = prepare_pack(args, device)
    model_name = stable_tag(args.model_path)
    dataset_name = stable_tag(str(args.data_path))
    if args.target_population.lower() != "all":
        dataset_name += f"-{args.target_population.lower()}"
    metrics = []
    retrieval_cache: dict[tuple[Any, ...], Any] = {}

    if args.command == "ranking":
        experiments = [
            (name, name, frame["extracted_patient_profile"].astype(str).tolist())
            for name in args.conditions
        ]
    else:
        experiments = [
            (
                name,
                "prof_rag",
                replacement_profiles(
                    frame,
                    name,
                    seed=args.seed,
                    pool_limit=args.profile_pool_limit,
                ),
            )
            for name in args.conditions
        ]

    for name, retrieval_condition, profiles in experiments:
        output_path = (
            args.output_dir
            / args.command
            / name
            / args.backend
            / corpus_name
            / model_name
            / f"{dataset_name}.pkl"
        )
        if output_path.is_file() and args.skip_existing:
            result = pd.read_pickle(output_path)
            if len(result) != len(frame) or "ablation_answer" not in result:
                raise ValueError(f"Incompatible existing output: {output_path}")
            print(f"[SKIP] Loaded: {output_path}")
        elif output_path.exists():
            raise FileExistsError(
                f"Output exists; use --skip-existing: {output_path}"
            )
        else:
            result = run_condition(
                frame,
                retrieval_condition=retrieval_condition,
                profiles=profiles,
                pack=pack,
                retrieval_models=retrieval_models,
                generator=generator,
                tokenizer=tokenizer,
                device=device,
                args=args,
                retrieval_cache=retrieval_cache,
            )
            atomic_pickle(result, output_path)
            print(f"Saved: {output_path}")
        metrics.append(
            {
                "ablation": args.command,
                "condition": name,
                "backend": args.backend,
                "corpus": corpus_name,
                "model": model_name,
                "dataset": dataset_name,
                **evaluate_predictions(result, "ablation_answer"),
            }
        )
    metrics_path = args.output_dir / args.command / "metrics.csv"
    update_metrics_csv(
        metrics,
        metrics_path,
        key_columns=[
            "ablation",
            "condition",
            "backend",
            "corpus",
            "model",
            "dataset",
        ],
    )
    print(f"Saved metrics: {metrics_path}")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--extractions", type=Path)
    parser.add_argument("--rag-spec", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("medcpt", "bm25"), default="medcpt")
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
    parser.add_argument("--profile-pool-limit", type=int, default=120)
    parser.add_argument("--skip-existing", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ranking = subparsers.add_parser("ranking")
    add_shared_arguments(ranking)
    ranking.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(RANKING_CONDITIONS),
        default=tuple(RANKING_CONDITIONS),
    )
    profile = subparsers.add_parser("profile")
    add_shared_arguments(profile)
    profile.add_argument(
        "--conditions",
        nargs="+",
        choices=PROFILE_CONDITIONS,
        default=PROFILE_CONDITIONS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.middle_k > args.first_k or args.final_k > args.middle_k:
        raise ValueError("Require first-k >= middle-k >= final-k.")
    if args.profile_pool_limit <= 0:
        raise ValueError("--profile-pool-limit must be positive.")
    run(args)


if __name__ == "__main__":
    main()
