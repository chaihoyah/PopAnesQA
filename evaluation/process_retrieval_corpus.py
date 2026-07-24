"""Chunk one or more PubMed corpora for retrieval.

Each input is supplied as ``SOURCE=PATH``. The script writes one chunked file
per source and a combined corpus that is deduplicated by document text before
chunking. No private data or model path is embedded in the code.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".pkl", ".parquet"}
TEXT_COLUMN_CANDIDATES = (
    "text",
    "article_abstract",
    "translated_text",
    "guidelinetext_processed",
)
CHUNK_COLUMNS = (
    "source",
    "doc_id",
    "source_index",
    "chunk_id",
    "token_start",
    "token_end",
    "token_count",
    "document_token_count",
    "text",
)
PROVENANCE_COLUMNS = {*CHUNK_COLUMNS, "source_membership"}


def parse_input_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid input {value!r}; expected SOURCE=PATH."
        )
    source, path_text = value.split("=", 1)
    source = source.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", source):
        raise argparse.ArgumentTypeError(
            "SOURCE may contain only letters, numbers, dot, underscore, and dash."
        )
    path = Path(path_text.strip())
    if not path_text.strip():
        raise argparse.ArgumentTypeError(f"Input path is empty: {value!r}")
    return source, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=parse_input_spec,
        required=True,
        metavar="SOURCE=PATH",
        help=(
            "Input corpus with its output/source label. Repeat for Pediatric "
            "and General corpora."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="MedCPT article encoder path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from the tokenizer repository.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-suffix",
        choices=(".pkl", ".csv", ".parquet"),
        default=".pkl",
    )
    parser.add_argument(
        "--combined-name",
        default="total_pubmed",
        help="Stem for the combined output file.",
    )
    parser.add_argument(
        "--text-column",
        help=(
            "Explicit text column for every input. By default, each input is "
            "auto-detected independently."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--keep-combined-duplicates",
        action="store_true",
        help="Do not remove exact duplicate documents in the combined corpus.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.overlap < 0:
        raise ValueError("--overlap cannot be negative.")
    if args.overlap >= args.chunk_size:
        raise ValueError("--overlap must be smaller than --chunk-size.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.combined_name):
        raise ValueError(
            "--combined-name may contain only letters, numbers, dot, "
            "underscore, and dash."
        )
    sources = [source for source, _ in args.inputs]
    if len(sources) != len(set(sources)):
        raise ValueError("Input SOURCE names must be unique.")
    if args.combined_name in set(sources):
        raise ValueError("--combined-name must differ from every input SOURCE.")
    for _, path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Input corpus not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported input suffix for {path}; expected "
                f"{sorted(SUPPORTED_SUFFIXES)}."
            )


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def detect_text_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame:
            raise ValueError(f"Requested text column not found: {requested!r}")
        return requested
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in frame:
            return candidate
    raise ValueError(
        "No text column found; expected one of "
        f"{list(TEXT_COLUMN_CANDIDATES)} or pass --text-column."
    )


def prepare_documents(
    frame: pd.DataFrame,
    *,
    source: str,
    text_column: str,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    dropped_empty = 0
    metadata_columns = [
        column
        for column in frame.columns
        if column not in {*PROVENANCE_COLUMNS, text_column}
    ]
    for position, (source_index, row) in enumerate(frame.iterrows()):
        value = row[text_column]
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            dropped_empty += 1
            continue
        prepared: dict[str, Any] = {
            "source": source,
            "doc_id": position,
            "source_index": str(source_index),
            "source_membership": (source,),
            "text": text,
        }
        for column in metadata_columns:
            prepared[column] = row[column]
        rows.append(prepared)
    return pd.DataFrame(rows), dropped_empty


def token_chunks(
    text: str,
    tokenizer: Any,
    *,
    chunk_size: int,
    overlap: int,
) -> tuple[list[tuple[str, int, int, int]], int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)
    if total_tokens == 0:
        return [], 0
    if total_tokens <= chunk_size:
        return [(text, 0, total_tokens, total_tokens)], total_tokens

    stride = chunk_size - overlap
    chunks: list[tuple[str, int, int, int]] = []
    for start in range(0, total_tokens, stride):
        end = min(start + chunk_size, total_tokens)
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if chunk_text:
            chunks.append((chunk_text, start, end, len(chunk_ids)))
        if end >= total_tokens:
            break
    return chunks, total_tokens


def chunk_documents(
    documents: pd.DataFrame,
    tokenizer: Any,
    *,
    chunk_size: int,
    overlap: int,
) -> pd.DataFrame:
    required = {"source", "doc_id", "source_index", "text"}
    missing = sorted(required - set(documents.columns))
    if missing:
        raise ValueError(f"Prepared documents are missing columns: {missing}")

    chunked_rows: list[dict[str, Any]] = []
    for _, document in documents.iterrows():
        chunks, total_tokens = token_chunks(
            str(document["text"]),
            tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        metadata = document.drop(labels=["text"]).to_dict()
        for chunk_id, (text, start, end, count) in enumerate(chunks):
            chunked_rows.append(
                {
                    **metadata,
                    "chunk_id": chunk_id,
                    "token_start": start,
                    "token_end": end,
                    "token_count": count,
                    "document_token_count": total_tokens,
                    "text": text,
                }
            )
    metadata_columns = [
        column
        for column in documents.columns
        if column not in {"source", "doc_id", "source_index", "text"}
    ]
    return pd.DataFrame(
        chunked_rows,
        columns=[*CHUNK_COLUMNS, *metadata_columns],
    )


def atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if path.suffix == ".pkl":
        frame.to_pickle(temporary)
    elif path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        raise ValueError(f"Unsupported output suffix: {path.suffix}")
    temporary.replace(path)


def atomic_write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def input_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        source: args.output_dir / f"{source}_chunked{args.output_suffix}"
        for source, _ in args.inputs
    }
    paths[args.combined_name] = (
        args.output_dir
        / f"{args.combined_name}_chunked{args.output_suffix}"
    )
    return paths


def ensure_outputs_available(
    outputs: dict[str, Path],
    manifest_path: Path,
    *,
    overwrite: bool,
) -> None:
    candidates = [*outputs.values(), manifest_path]
    existing = [path for path in candidates if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output exists. Use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    outputs = output_paths(args)
    manifest_path = args.output_dir / "retrieval_corpus_manifest.json"
    ensure_outputs_available(
        outputs,
        manifest_path,
        overwrite=args.overwrite,
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Corpus chunking requires transformers. Install project "
            "dependencies before running this command."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    documents_by_source: dict[str, pd.DataFrame] = {}
    chunks_by_source: dict[str, pd.DataFrame] = {}
    source_reports: dict[str, dict[str, Any]] = {}
    for source, path in args.inputs:
        frame = read_dataframe(path)
        text_column = detect_text_column(frame, args.text_column)
        documents, dropped_empty = prepare_documents(
            frame,
            source=source,
            text_column=text_column,
        )
        chunked = chunk_documents(
            documents,
            tokenizer,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        atomic_write_dataframe(chunked, outputs[source])
        documents_by_source[source] = documents
        chunks_by_source[source] = chunked
        source_reports[source] = {
            "input": input_identity(path),
            "text_column": text_column,
            "input_rows": len(frame),
            "documents": len(documents),
            "dropped_empty": dropped_empty,
            "chunks": len(chunked),
            "output": str(outputs[source].resolve()),
        }
        print(
            f"{source}: {len(frame):,} input rows -> "
            f"{len(documents):,} documents -> {len(chunked):,} chunks"
        )

    combined_documents = pd.concat(
        list(documents_by_source.values()),
        ignore_index=True,
        sort=False,
    )
    combined_before_deduplication = len(combined_documents)
    if not args.keep_combined_duplicates:
        memberships = (
            combined_documents.groupby("text", sort=False)["source"]
            .agg(lambda values: tuple(dict.fromkeys(str(value) for value in values)))
            .to_dict()
        )
        combined_documents["source_membership"] = combined_documents["text"].map(
            memberships
        )
        combined_documents = combined_documents.drop_duplicates(
            subset=["text"],
            keep="first",
        ).reset_index(drop=True)
    combined_chunked = chunk_documents(
        combined_documents,
        tokenizer,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    chunk_memberships: defaultdict[str, set[str]] = defaultdict(set)
    for source, chunked in chunks_by_source.items():
        for text in chunked["text"].dropna().unique():
            chunk_memberships[str(text)].add(source)
    combined_chunked["source_membership"] = combined_chunked["text"].map(
        lambda text: tuple(sorted(chunk_memberships.get(str(text), ())))
    )
    atomic_write_dataframe(combined_chunked, outputs[args.combined_name])
    print(
        f"{args.combined_name}: {combined_before_deduplication:,} documents -> "
        f"{len(combined_documents):,} unique documents -> "
        f"{len(combined_chunked):,} chunks"
    )

    manifest = {
        "tokenizer": args.tokenizer,
        "trust_remote_code": args.trust_remote_code,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "stride": args.chunk_size - args.overlap,
        "combined_deduplicated": not args.keep_combined_duplicates,
        "sources": source_reports,
        "combined": {
            "name": args.combined_name,
            "documents_before_deduplication": combined_before_deduplication,
            "documents_after_deduplication": len(combined_documents),
            "chunks": len(combined_chunked),
            "chunks_in_multiple_sources": int(
                combined_chunked["source_membership"]
                .map(lambda values: len(values) > 1)
                .sum()
            ),
            "output": str(outputs[args.combined_name].resolve()),
        },
    }
    atomic_write_json(manifest, manifest_path)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
