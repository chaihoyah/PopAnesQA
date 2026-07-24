"""Parse guideline PDFs into page-level HTML with Upstage Document Parse.

The public API endpoint is not a secret. Authentication is read from an
environment variable, while all local paths are provided through CLI arguments.
The output preserves the ``guideline``, ``page``, and ``content`` columns used
by ``make_qa.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, TypeVar

import pandas as pd


DEFAULT_API_URL = "https://api.upstage.ai/v1/document-digitization"
SUPPORTED_OUTPUT_SUFFIXES = {".csv", ".xlsx", ".pkl", ".parquet"}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse guideline PDFs into page-level HTML."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing guideline PDF files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output path ending in .pkl, .parquet, .csv, or .xlsx.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Upstage Document Parse endpoint.",
    )
    parser.add_argument(
        "--api-key-env",
        default="UPSTAGE_API_KEY",
        help="Environment variable containing the Upstage API key.",
    )
    parser.add_argument("--model", default="document-parse")
    parser.add_argument("--ocr", default="force")
    parser.add_argument(
        "--base64-encoding",
        default="['table']",
        help="Value sent as the base64_encoding form field.",
    )
    parser.add_argument(
        "--raw-response-dir",
        type=Path,
        help="Optional directory in which each API JSON response is preserved.",
    )
    parser.add_argument(
        "--page-header-mode",
        choices=("all", "chapter"),
        default="all",
        help=(
            "Use every <header> as a page boundary (the original study "
            "behavior), or only headers matching 'Chapter N'."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument(
        "--expected-pdf-count",
        type=int,
        default=0,
        help="Optional expected PDF count; 0 disables the check.",
    )
    parser.add_argument(
        "--expected-page-count",
        type=int,
        default=0,
        help="Optional expected parsed page count; 0 disables the check.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Keep successful PDFs instead of failing when any PDF cannot be parsed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs and raw responses.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
    if args.output_file.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        raise ValueError(
            "--output-file must end in one of: "
            + ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        )
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive.")
    if args.retry_backoff_seconds < 0 or args.request_delay_seconds < 0:
        raise ValueError("Retry and request delays cannot be negative.")


def progress(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def is_page_header(header_html: str) -> bool:
    """Return whether a header fragment matches the repeated page header."""
    if not isinstance(header_html, str):
        return False

    opening_tag = re.match(
        r"\s*<header\b(?P<attributes>[^>]*)>",
        header_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not opening_tag:
        return False

    text = html.unescape(re.sub(r"<[^>]+>", " ", header_html))
    text = re.sub(r"\s+", " ", text).strip()
    has_chapter_number = re.search(r"\bChapter\s+\d+\b", text, re.IGNORECASE)

    # Upstage may normalize style whitespace differently between documents.
    attributes = opening_tag.group("attributes")
    style_match = re.search(
        r"\bstyle\s*=\s*(['\"])(?P<style>.*?)\1",
        attributes,
        flags=re.IGNORECASE | re.DOTALL,
    )
    style = (
        re.sub(r"\s+", "", style_match.group("style").lower())
        if style_match
        else ""
    )
    style_ok = not style or "font-size:18px" in style
    return bool(has_chapter_number and style_ok)


def split_by_page_header(
    html_text: str,
    page_header_mode: str = "all",
) -> tuple[list[str], int]:
    """Split raw HTML at all headers or only validated chapter headers.

    Content before the first accepted header is retained as part of the first
    page instead of being discarded.
    """
    if not isinstance(html_text, str) or not html_text.strip():
        raise ValueError("Document Parse returned empty HTML.")

    header_pattern = re.compile(
        r"<header\b[^>]*>.*?</header\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    accepted_positions: list[int] = []
    for match in header_pattern.finditer(html_text):
        if page_header_mode == "all" or is_page_header(match.group(0)):
            accepted_positions.append(match.start())

    if not accepted_positions:
        return [html_text], 0

    # Page 1 begins at the start of the response so any preamble is preserved.
    boundaries = [0, *accepted_positions[1:], len(html_text)]
    pages = [
        html_text[boundaries[index] : boundaries[index + 1]]
        for index in range(len(boundaries) - 1)
        if html_text[boundaries[index] : boundaries[index + 1]].strip()
    ]
    return pages, len(accepted_positions)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_document_parse(
    session: Any,
    pdf_path: Path,
    *,
    api_url: str,
    api_key: str,
    model: str,
    ocr: str,
    base64_encoding: str,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    headers = {"Authorization": f"Bearer {api_key}"}
    form_data = {
        "ocr": ocr,
        "base64_encoding": base64_encoding,
        "model": model,
    }

    for attempt in range(1, max_retries + 1):
        try:
            with pdf_path.open("rb") as document:
                response = session.post(
                    api_url,
                    headers=headers,
                    files={
                        "document": (
                            pdf_path.name,
                            document,
                            "application/pdf",
                        )
                    },
                    data=form_data,
                    timeout=timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("API response must be a JSON object.")
            content = payload.get("content")
            if not isinstance(content, dict) or not isinstance(
                content.get("html"), str
            ):
                raise ValueError("API response is missing content.html.")
            if not content["html"].strip():
                raise ValueError("API response contains empty content.html.")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            delay = retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"{pdf_path.name}: attempt {attempt}/{max_retries} failed "
                f"({type(exc).__name__}: {exc}); retrying in {delay:.1f}s."
            )
            time.sleep(delay)

    assert last_error is not None
    raise RuntimeError(
        f"{pdf_path.name} failed after {max_retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        df.to_pickle(temporary_path)
    elif suffix == ".parquet":
        df.to_parquet(temporary_path, index=False)
    elif suffix == ".csv":
        df.to_csv(temporary_path, index=False)
    elif suffix == ".xlsx":
        df.to_excel(temporary_path, index=False)
    else:
        raise ValueError(f"Unsupported output suffix: {suffix}")
    temporary_path.replace(path)


def output_sidecars(output_file: Path) -> dict[str, Path]:
    return {
        "manifest": output_file.with_name(f"{output_file.stem}_manifest.csv"),
        "config": output_file.with_name(f"{output_file.stem}_config.json"),
    }


def ensure_outputs_available(
    output_file: Path,
    sidecars: dict[str, Path],
    raw_response_paths: list[Path],
    overwrite: bool,
) -> None:
    candidates = [output_file, *sidecars.values(), *raw_response_paths]
    existing = [path for path in candidates if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists. Use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing[:10])
        )


def write_config(args: argparse.Namespace, path: Path) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    # Store only the environment-variable name, never its secret value.
    atomic_write_json(config, path)


def main() -> None:
    args = parse_args()
    validate_args(args)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"API key environment variable is not set: {args.api_key_env}"
        )

    pdf_paths = sorted(
        (
            path
            for path in args.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.lower(),
    )
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {args.input_dir}")
    if args.expected_pdf_count and len(pdf_paths) != args.expected_pdf_count:
        raise ValueError(
            f"Expected {args.expected_pdf_count:,} PDFs, found {len(pdf_paths):,}."
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    sidecars = output_sidecars(args.output_file)
    raw_response_paths: list[Path] = []
    if args.raw_response_dir:
        args.raw_response_dir.mkdir(parents=True, exist_ok=True)
        raw_response_paths = [
            args.raw_response_dir / f"{pdf_path.stem}.json"
            for pdf_path in pdf_paths
        ]
    ensure_outputs_available(
        args.output_file,
        sidecars,
        raw_response_paths,
        args.overwrite,
    )
    write_config(args, sidecars["config"])

    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "Guideline PDF parsing requires requests."
        ) from exc

    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    session = requests.Session()

    for pdf_path in progress(pdf_paths, desc="Parsing guideline PDFs"):
        file_hash = sha256_file(pdf_path)
        try:
            payload = request_document_parse(
                session,
                pdf_path,
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                ocr=args.ocr,
                base64_encoding=args.base64_encoding,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
            html = payload["content"]["html"]
            pages, accepted_header_count = split_by_page_header(
                html,
                page_header_mode=args.page_header_mode,
            )

            if args.raw_response_dir:
                raw_path = args.raw_response_dir / f"{pdf_path.stem}.json"
                atomic_write_json(payload, raw_path)
            else:
                raw_path = None

            rows.extend(
                {
                    "guideline": pdf_path.name,
                    "page": page_number,
                    "content": page_content,
                }
                for page_number, page_content in enumerate(pages, start=1)
            )
            manifest_rows.append(
                {
                    "guideline": pdf_path.name,
                    "source_sha256": file_hash,
                    "source_bytes": pdf_path.stat().st_size,
                    "parsed_pages": len(pages),
                    "accepted_page_headers": accepted_header_count,
                    "html_characters": len(html),
                    "raw_response": str(raw_path) if raw_path else "",
                    "status": "complete",
                    "error": "",
                }
            )
            time.sleep(args.request_delay_seconds)
        except Exception as exc:
            manifest_rows.append(
                {
                    "guideline": pdf_path.name,
                    "source_sha256": file_hash,
                    "source_bytes": pdf_path.stat().st_size,
                    "parsed_pages": 0,
                    "accepted_page_headers": 0,
                    "html_characters": 0,
                    "raw_response": "",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    session.close()
    guideline_df = pd.DataFrame(
        rows,
        columns=["guideline", "page", "content"],
    )
    manifest_df = pd.DataFrame(manifest_rows)
    atomic_write_dataframe(guideline_df, args.output_file)
    atomic_write_dataframe(manifest_df, sidecars["manifest"])

    failed_count = int(manifest_df["status"].eq("failed").sum())
    no_header_count = int(
        (
            manifest_df["status"].eq("complete")
            & manifest_df["accepted_page_headers"].eq(0)
        ).sum()
    )
    print(f"Input PDFs: {len(pdf_paths):,}")
    print(f"Successfully parsed PDFs: {len(pdf_paths) - failed_count:,}")
    print(f"Failed PDFs: {failed_count:,}")
    print(f"Parsed page-level rows: {len(guideline_df):,}")
    print(f"Documents without accepted page headers: {no_header_count:,}")
    print(f"Saved parsed guidelines to: {args.output_file}")
    print(f"Saved parse manifest to: {sidecars['manifest']}")

    if args.expected_page_count and len(guideline_df) != args.expected_page_count:
        raise ValueError(
            f"Expected {args.expected_page_count:,} parsed pages, "
            f"found {len(guideline_df):,}."
        )
    if failed_count and not args.allow_partial:
        raise RuntimeError(
            f"{failed_count:,} PDFs failed to parse. Successful rows and the "
            f"failure manifest were saved; rerun with --allow-partial only if "
            f"partial output is intentional."
        )


if __name__ == "__main__":
    main()
