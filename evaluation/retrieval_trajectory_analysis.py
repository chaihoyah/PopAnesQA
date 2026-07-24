"""Analyze source composition across initial, middle, and final retrieval stages.

New evaluation outputs contain a structured ``retrieval_trajectory`` column.
Legacy text-only columns (``top_64``/``top_8``/``top_1`` or
``candidate_texts``/``middle_texts``/``retrieved_texts``) are also supported
when source corpora are supplied for exact-text source matching.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TEXT_COLUMNS = (
    "text",
    "article_abstract",
    "translated_text",
    "guidelinetext_processed",
    "content",
)
STAGES = ("initial", "middle", "final")
LEGACY_STAGE_COLUMNS = {
    "initial": ("candidate_texts", "top_64"),
    "middle": ("middle_texts", "top_8"),
    "final": ("retrieved_texts", "top_1"),
}


def parse_source_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid corpus {value!r}; expected SOURCE=PATH."
        )
    source, path_text = value.split("=", 1)
    source = source.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", source):
        raise argparse.ArgumentTypeError(
            "SOURCE may contain only letters, numbers, dot, underscore, and dash."
        )
    if not path_text.strip():
        raise argparse.ArgumentTypeError(f"Corpus path is empty: {value!r}")
    return source, Path(path_text.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        dest="results",
        action="append",
        type=Path,
        required=True,
        help="Evaluation result file. Repeat to analyze multiple experiments.",
    )
    parser.add_argument(
        "--corpus",
        dest="corpora",
        action="append",
        type=parse_source_spec,
        default=[],
        metavar="SOURCE=PATH",
        help=(
            "Source corpus used to classify legacy text-only results. Repeat "
            "for Pediatric and General corpora."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-column",
        default="retrieval_trajectory",
    )
    parser.add_argument(
        "--population-column",
        default="target_population",
    )
    parser.add_argument("--output-prefix", default="retrieval_trajectory")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_table(path: Path) -> pd.DataFrame:
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
        f"Unsupported table {path}; expected .pkl, .csv, .parquet, or .xlsx."
    )


def detect_text_column(frame: pd.DataFrame) -> str:
    column = next((name for name in TEXT_COLUMNS if name in frame), None)
    if column is None:
        raise ValueError(
            f"No text column found; expected one of {list(TEXT_COLUMNS)}."
        )
    return column


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).strip().encode("utf-8")).hexdigest()


def build_source_lookup(
    corpora: list[tuple[str, Path]],
) -> dict[str, tuple[str, ...]]:
    memberships: defaultdict[str, set[str]] = defaultdict(set)
    for source, path in corpora:
        if not path.is_file():
            raise FileNotFoundError(f"Corpus not found: {path}")
        frame = read_table(path)
        text_column = detect_text_column(frame)
        for value in frame[text_column].dropna():
            text = str(value).strip()
            if text:
                memberships[text_hash(text)].add(source)
    return {
        digest: tuple(sorted(sources))
        for digest, sources in memberships.items()
    }


def parse_nested(value: Any) -> Any:
    if not isinstance(value, (str, dict, list, tuple, set)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped.lower() == "nan":
        return None
    if stripped[0] not in "[{(":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value


def normalize_membership(value: Any) -> tuple[str, ...]:
    value = parse_nested(value)
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.split("|") if "|" in value else [value]
    elif isinstance(value, Iterable):
        parts = list(value)
    else:
        parts = [value]
    return tuple(
        dict.fromkeys(
            str(part).strip()
            for part in parts
            if str(part).strip()
        )
    )


def document_membership(
    document: dict[str, Any],
    source_lookup: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    metadata = parse_nested(document.get("metadata")) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    membership = normalize_membership(metadata.get("source_membership"))
    if membership:
        return membership
    source = normalize_membership(metadata.get("source"))
    if source:
        return source
    return source_lookup.get(text_hash(document.get("text", "")), ())


def source_category(membership: tuple[str, ...]) -> str:
    if not membership:
        return "unknown"
    if len(membership) > 1:
        return "both"
    return membership[0]


def legacy_documents(
    row: pd.Series,
    stage: str,
) -> tuple[list[dict[str, Any]], int | None, dict[str, Any]]:
    column = next(
        (name for name in LEGACY_STAGE_COLUMNS[stage] if name in row.index),
        None,
    )
    if column is None:
        return [], None, {}
    values = parse_nested(row[column])
    if values is None:
        return [], None, {}
    if isinstance(values, str):
        values = [values]
    documents = [
        {
            "rank": rank,
            "corpus_index": None,
            "score": None,
            "text": str(text),
            "metadata": {},
        }
        for rank, text in enumerate(values, start=1)
    ]
    return documents, len(documents), {}


def trajectory_stage(
    row: pd.Series,
    trajectory_column: str,
    stage: str,
) -> tuple[list[dict[str, Any]], int | None, dict[str, Any]]:
    if trajectory_column in row.index:
        trajectory = parse_nested(row[trajectory_column])
        if isinstance(trajectory, dict):
            payload = trajectory.get(stage)
            if isinstance(payload, dict):
                documents = payload.get("documents") or []
                if isinstance(documents, list):
                    return (
                        documents,
                        payload.get("requested_k"),
                        {
                            "query_kind": payload.get("query_kind"),
                            "query": payload.get("query"),
                            "reranked": payload.get("reranked"),
                        },
                    )
    return legacy_documents(row, stage)


def flatten_results(
    results: list[Path],
    *,
    trajectory_column: str,
    population_column: str,
    source_lookup: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for result_path in results:
        if not result_path.is_file():
            raise FileNotFoundError(f"Result not found: {result_path}")
        frame = read_table(result_path)
        for row_position, (row_index, row) in enumerate(frame.iterrows()):
            trajectory = parse_nested(row.get(trajectory_column))
            if not isinstance(trajectory, dict):
                trajectory = {}
            population = (
                row.get(population_column, "unknown")
                if population_column in frame
                else "unknown"
            )
            question_id = row.get("id", row.get("question_id", row_index))
            for stage in STAGES:
                documents, requested_k, stage_info = trajectory_stage(
                    row,
                    trajectory_column,
                    stage,
                )
                observed_k = len(documents)
                for fallback_rank, document in enumerate(documents, start=1):
                    if not isinstance(document, dict):
                        document = {
                            "text": str(document),
                            "metadata": {},
                        }
                    membership = document_membership(document, source_lookup)
                    records.append(
                        {
                            "result_file": str(result_path.resolve()),
                            "result_name": result_path.stem,
                            "result_row": row_position,
                            "source_row_index": str(row_index),
                            "question_id": str(question_id),
                            "target_population": str(population),
                            "condition": trajectory.get("condition"),
                            "backend": trajectory.get("backend"),
                            "stage": stage,
                            "query_kind": stage_info.get("query_kind"),
                            "query": stage_info.get("query"),
                            "reranked": stage_info.get("reranked"),
                            "requested_k": requested_k,
                            "observed_k": observed_k,
                            "rank": document.get("rank", fallback_rank),
                            "corpus_index": document.get("corpus_index"),
                            "score": document.get("score"),
                            "document_source": source_category(membership),
                            "source_membership": "|".join(membership),
                            "text": str(document.get("text", "")),
                            "metadata": document.get("metadata") or {},
                        }
                    )
    if not records:
        raise ValueError(
            "No retrieval documents found. Expected retrieval_trajectory or "
            "legacy top-k text columns."
        )
    return pd.DataFrame(records)


def question_source_counts(documents: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "result_file",
        "result_name",
        "result_row",
        "question_id",
        "target_population",
        "stage",
        "requested_k",
        "observed_k",
    ]
    counts = (
        documents.groupby([*keys, "document_source"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["percentage"] = (
        counts["count"]
        .div(counts["observed_k"].where(counts["observed_k"] > 0))
        .mul(100)
        .round(2)
    )
    return counts


def population_summary(documents: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "result_file",
        "result_name",
        "target_population",
        "stage",
    ]
    summary = (
        documents.groupby([*keys, "document_source"], dropna=False)
        .agg(
            document_count=("text", "size"),
            question_count=("result_row", "nunique"),
        )
        .reset_index()
    )
    totals = summary.groupby(keys)["document_count"].transform("sum")
    summary["document_percentage"] = (
        summary["document_count"].div(totals).mul(100).round(2)
    )
    return summary


def warn_about_source_metadata(documents: pd.DataFrame) -> None:
    unknown_fraction = documents["document_source"].eq("unknown").mean()
    both_fraction = documents["document_source"].eq("both").mean()
    if unknown_fraction > 0:
        print(
            f"[warn] {unknown_fraction:.1%} of retrieved documents have no "
            "source metadata."
        )
    if both_fraction == 0:
        print(
            "[warn] No multi-source documents were found. If this analysis "
            "uses the combined deduplicated corpus, rebuild it with "
            "process_retrieval_corpus.py so source_membership is recorded."
        )


def ensure_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output exists. Use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )


def atomic_pickle(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_pickle(temporary)
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    serializable = frame.copy()
    if "metadata" in serializable:
        serializable["metadata"] = serializable["metadata"].map(
            lambda value: json.dumps(value, ensure_ascii=False, default=str)
        )
    serializable.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_prefix):
        raise ValueError(
            "--output-prefix may contain only letters, numbers, dot, "
            "underscore, and dash."
        )
    source_names = [source for source, _ in args.corpora]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Corpus SOURCE names must be unique.")

    documents_path = args.output_dir / f"{args.output_prefix}_documents.pkl"
    documents_csv_path = args.output_dir / f"{args.output_prefix}_documents.csv"
    counts_path = args.output_dir / f"{args.output_prefix}_question_counts.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    outputs = [
        documents_path,
        documents_csv_path,
        counts_path,
        summary_path,
    ]
    ensure_outputs_available(outputs, args.overwrite)

    source_lookup = build_source_lookup(args.corpora)
    documents = flatten_results(
        args.results,
        trajectory_column=args.trajectory_column,
        population_column=args.population_column,
        source_lookup=source_lookup,
    )
    warn_about_source_metadata(documents)
    question_counts = question_source_counts(documents)
    summary = population_summary(documents)

    atomic_pickle(documents, documents_path)
    atomic_csv(documents, documents_csv_path)
    atomic_csv(question_counts, counts_path)
    atomic_csv(summary, summary_path)

    print(
        f"Analyzed {documents['result_row'].nunique():,} result rows and "
        f"{len(documents):,} retrieved documents."
    )
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
