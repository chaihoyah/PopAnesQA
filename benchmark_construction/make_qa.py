"""Build PopAnesQA from parsed guideline pages with Gemini Batch API.

The pipeline deliberately separates local preparation, remote submission,
result collection, and validated merging.  This makes every stage resumable
without embedding private data paths or API credentials in the repository.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:
    from .batch_utils import (
        atomic_write_dataframe,
        collect_stage_batches,
        ensure_writable,
        load_json,
        load_stage_results,
        read_dataframe,
        sha256_file,
        stage_artifacts,
        submit_stage_batches,
        write_request_batches,
    )
    from .prompts import (
        STAGE1_SYSTEM,
        STAGE1_USER,
        STAGE2_SYSTEM,
        STAGE2_USER,
        STAGE3_SYSTEM,
        STAGE3_USER,
    )
    from .qa_postprocess import (
        build_final_qa,
        build_stage3_groups,
        explode_stage2_rules,
        rebalance_correct_answers,
        validate_stage1_response,
        validate_stage2_response,
        validate_stage3_response,
    )
except ImportError:
    from batch_utils import (
        atomic_write_dataframe,
        collect_stage_batches,
        ensure_writable,
        load_json,
        load_stage_results,
        read_dataframe,
        sha256_file,
        stage_artifacts,
        submit_stage_batches,
        write_request_batches,
    )
    from prompts import (
        STAGE1_SYSTEM,
        STAGE1_USER,
        STAGE2_SYSTEM,
        STAGE2_USER,
        STAGE3_SYSTEM,
        STAGE3_USER,
    )
    from qa_postprocess import (
        build_final_qa,
        build_stage3_groups,
        explode_stage2_rules,
        rebalance_correct_answers,
        validate_stage1_response,
        validate_stage2_response,
        validate_stage3_response,
    )


DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
STAGES = ("stage1", "stage2", "stage3")
ERROR_COLUMNS = ("key", "result_file", "error", "raw_response")


def clean_html_content(raw_html: Any) -> str:
    """Convert parsed page HTML to compact plain text for the LLM prompts."""
    if raw_html is None or (not isinstance(raw_html, str) and pd.isna(raw_html)):
        return ""
    source = str(raw_html)
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", source)
        text = html.unescape(text)
    else:
        soup = BeautifulSoup(source, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def make_batch_request(
    key: str,
    prompt: str,
    system_message: str,
    temperature: float,
) -> dict[str, Any]:
    return {
        "key": key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "system_instruction": {
                "parts": [{"text": system_message}],
            },
            "generation_config": {
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        },
    }


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    description: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{description} is missing columns: {missing}")


def verify_prepared_input(
    *,
    input_file: Path,
    work_dir: Path,
    stage: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    artifacts = stage_artifacts(work_dir, stage)
    manifest = load_json(artifacts["prepare_manifest"])
    if manifest.get("stage") != stage:
        raise ValueError(
            f"Prepare manifest stage is {manifest.get('stage')!r}, not {stage!r}."
        )
    current_hash = sha256_file(input_file)
    if current_hash != manifest.get("input_sha256"):
        raise ValueError(
            f"{stage} input has changed since request preparation. "
            "Prepare the requests again before merging."
        )
    request_index = pd.read_csv(artifacts["request_index"])
    require_columns(
        request_index,
        {"key", "row_position"},
        description=f"{stage} request index",
    )
    if request_index["key"].duplicated().any():
        raise ValueError(f"{stage} request index contains duplicate keys.")
    return manifest, request_index


def validate_results(
    *,
    stage: str,
    work_dir: Path,
    request_index: pd.DataFrame,
    validator: Callable[[str, Any], Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    parsed_results, parse_errors = load_stage_results(
        stage=stage,
        work_dir=work_dir,
    )
    expected_keys = set(request_index["key"].astype(str))
    result_keys = set(parsed_results)
    errors: list[dict[str, str]] = list(parse_errors)

    for key in sorted(result_keys - expected_keys):
        errors.append(
            {
                "key": key,
                "result_file": "",
                "error": "Unexpected result key.",
                "raw_response": json.dumps(
                    parsed_results[key],
                    ensure_ascii=False,
                ),
            }
        )

    invalid_keys = {str(error.get("key", "")) for error in errors}
    validated: dict[str, Any] = {}
    for key in request_index["key"].astype(str):
        if key not in parsed_results:
            if key not in invalid_keys:
                errors.append(
                    {
                        "key": key,
                        "result_file": "",
                        "error": "Expected result key is missing.",
                        "raw_response": "",
                    }
                )
            continue
        try:
            validated[key] = validator(key, parsed_results[key])
        except (TypeError, ValueError) as exc:
            errors.append(
                {
                    "key": key,
                    "result_file": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_response": json.dumps(
                        parsed_results[key],
                        ensure_ascii=False,
                    ),
                }
            )

    error_frame = pd.DataFrame(errors, columns=ERROR_COLUMNS)
    return validated, error_frame


def write_merge_outputs(
    *,
    outputs: list[tuple[pd.DataFrame, Path | None]],
    errors: pd.DataFrame,
    errors_file: Path,
    overwrite: bool,
    allow_incomplete: bool,
) -> None:
    paths = [path for _, path in outputs if path is not None]
    paths.append(errors_file)
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Merge output paths must be distinct.")
    ensure_writable(paths, overwrite)
    for frame, path in outputs:
        if path is not None:
            atomic_write_dataframe(frame, path)
            print(f"Saved {len(frame):,} rows to: {path}")
    atomic_write_dataframe(errors, errors_file)
    print(f"Saved {len(errors):,} validation errors to: {errors_file}")
    if not errors.empty and not allow_incomplete:
        raise RuntimeError(
            f"{len(errors):,} result(s) failed validation. Outputs were saved "
            "for inspection; rerun with --allow-incomplete to accept partial data."
        )


def prepare_stage1(args: argparse.Namespace) -> None:
    frame = read_dataframe(args.input)
    require_columns(
        frame,
        {"guideline", "page", "content"},
        description="Parsed guideline data",
    )
    requests: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    skipped = 0
    for row_position, (_, row) in enumerate(frame.iterrows()):
        processed = clean_html_content(row["content"])
        if len(processed) < args.min_characters:
            skipped += 1
            continue
        key = f"stage1_row_{row_position:06d}"
        prompt = STAGE1_USER.format(input_text=processed)
        requests.append(
            make_batch_request(
                key,
                prompt,
                STAGE1_SYSTEM,
                args.temperature,
            )
        )
        index_rows.append(
            {
                "key": key,
                "row_position": row_position,
                "source_index": str(frame.index[row_position]),
                "guideline": row["guideline"],
                "page": row["page"],
                "clean_characters": len(processed),
            }
        )

    manifest = write_request_batches(
        stage="stage1",
        input_file=args.input,
        work_dir=args.work_dir,
        requests=requests,
        request_index=pd.DataFrame(index_rows),
        batch_size=args.batch_size,
        metadata={
            "input_row_count": len(frame),
            "skipped_short_row_count": skipped,
            "min_characters": args.min_characters,
            "temperature": args.temperature,
        },
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {manifest['request_count']:,} Stage 1 requests in "
        f"{manifest['batch_count']:,} batches; skipped {skipped:,} short pages."
    )


def merge_stage1(args: argparse.Namespace) -> None:
    frame = read_dataframe(args.input)
    require_columns(
        frame,
        {"guideline", "page", "content"},
        description="Parsed guideline data",
    )
    _, request_index = verify_prepared_input(
        input_file=args.input,
        work_dir=args.work_dir,
        stage="stage1",
    )
    validated, errors = validate_results(
        stage="stage1",
        work_dir=args.work_dir,
        request_index=request_index,
        validator=lambda _key, value: validate_stage1_response(value),
    )

    annotated = frame.copy()
    annotated["gemini_stage1_response"] = None
    for row in request_index.itertuples(index=False):
        key = str(row.key)
        if key in validated:
            annotated.at[
                annotated.index[int(row.row_position)],
                "gemini_stage1_response",
            ] = validated[key]

    for field in ("is_relevant", "target_population", "category", "reason"):
        annotated[field] = annotated["gemini_stage1_response"].map(
            lambda response: (
                response.get(field) if isinstance(response, dict) else None
            )
        )
    annotated["guidelinetext_processed"] = annotated["content"].map(
        clean_html_content
    )
    relevant = annotated.loc[annotated["is_relevant"].eq(True)].copy()
    relevant.reset_index(drop=True, inplace=True)

    write_merge_outputs(
        outputs=[
            (relevant, args.output),
            (annotated, args.annotated_output),
        ],
        errors=errors,
        errors_file=args.errors_output,
        overwrite=args.overwrite,
        allow_incomplete=args.allow_incomplete,
    )


def prepare_stage2(args: argparse.Namespace) -> None:
    frame = read_dataframe(args.input)
    require_columns(
        frame,
        {
            "guideline",
            "page",
            "guidelinetext_processed",
            "target_population",
            "category",
            "reason",
        },
        description="Stage 1 relevant data",
    )
    requests: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(frame.iterrows()):
        processed = clean_html_content(row["guidelinetext_processed"])
        if not processed:
            raise ValueError(
                f"Stage 2 row {row_position} has empty guidelinetext_processed."
            )
        key = f"stage2_row_{row_position:06d}"
        prompt = STAGE2_USER.format(
            target_population=row["target_population"],
            category=row["category"],
            input_text=processed,
        )
        requests.append(
            make_batch_request(
                key,
                prompt,
                STAGE2_SYSTEM,
                args.temperature,
            )
        )
        index_rows.append(
            {
                "key": key,
                "row_position": row_position,
                "source_index": str(frame.index[row_position]),
                "guideline": row["guideline"],
                "page": row["page"],
            }
        )

    manifest = write_request_batches(
        stage="stage2",
        input_file=args.input,
        work_dir=args.work_dir,
        requests=requests,
        request_index=pd.DataFrame(index_rows),
        batch_size=args.batch_size,
        metadata={
            "input_row_count": len(frame),
            "temperature": args.temperature,
        },
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {manifest['request_count']:,} Stage 2 requests in "
        f"{manifest['batch_count']:,} batches."
    )


def merge_stage2(args: argparse.Namespace) -> None:
    frame = read_dataframe(args.input)
    _, request_index = verify_prepared_input(
        input_file=args.input,
        work_dir=args.work_dir,
        stage="stage2",
    )
    validated, errors = validate_results(
        stage="stage2",
        work_dir=args.work_dir,
        request_index=request_index,
        validator=lambda _key, value: validate_stage2_response(value),
    )

    annotated = frame.copy()
    annotated["gemini_stage2_response"] = None
    for row in request_index.itertuples(index=False):
        key = str(row.key)
        if key in validated:
            annotated.at[
                annotated.index[int(row.row_position)],
                "gemini_stage2_response",
            ] = validated[key]
    logic = explode_stage2_rules(annotated)
    empty_rule_pages = sum(
        isinstance(response, dict) and not response.get("rules")
        for response in annotated["gemini_stage2_response"]
    )
    print(
        f"Expanded Stage 2 into {len(logic):,} rule rows; "
        f"{empty_rule_pages:,} validated pages contained no rules."
    )

    write_merge_outputs(
        outputs=[
            (logic, args.output),
            (annotated, args.annotated_output),
        ],
        errors=errors,
        errors_file=args.errors_output,
        overwrite=args.overwrite,
        allow_incomplete=args.allow_incomplete,
    )


def prepare_stage3(args: argparse.Namespace) -> None:
    logic = read_dataframe(args.input)
    grouped = build_stage3_groups(logic)
    if grouped.empty:
        raise ValueError("No Stage 3 guideline-page groups were created.")

    grouped_file = args.work_dir / "stage3_grouped_input.pkl"
    ensure_writable([grouped_file], args.overwrite)
    requests: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(grouped.iterrows()):
        rule_payload = {
            "rules": row["rules"],
            "merge_preference": "merge_only_if_clinically_realistic",
            "merge_goal": (
                "Create one coherent vignette from compatible rules on this page."
            ),
        }
        prompt = STAGE3_USER.format(
            target_population=row["target_population"],
            category=row["category"],
            input_text=row["guidelinetext_processed"],
            rules_json=json.dumps(
                rule_payload,
                ensure_ascii=False,
                indent=2,
            ),
        )
        requests.append(
            make_batch_request(
                row["key"],
                prompt,
                STAGE3_SYSTEM,
                args.temperature,
            )
        )
        index_rows.append(
            {
                "key": row["key"],
                "row_position": row_position,
                "guideline": row["guideline"],
                "page": row["page"],
                "rule_count": len(row["rules"]),
            }
        )

    atomic_write_dataframe(grouped, grouped_file)
    try:
        manifest = write_request_batches(
            stage="stage3",
            input_file=args.input,
            work_dir=args.work_dir,
            requests=requests,
            request_index=pd.DataFrame(index_rows),
            batch_size=args.batch_size,
            metadata={
                "input_rule_row_count": len(logic),
                "grouped_input_file": str(grouped_file.resolve()),
                "group_count": len(grouped),
                "temperature": args.temperature,
            },
            overwrite=args.overwrite,
        )
    except Exception:
        if grouped_file.exists():
            grouped_file.unlink()
        raise
    print(
        f"Prepared {manifest['request_count']:,} Stage 3 page groups in "
        f"{manifest['batch_count']:,} batches."
    )


def merge_stage3(args: argparse.Namespace) -> None:
    manifest, request_index = verify_prepared_input(
        input_file=args.input,
        work_dir=args.work_dir,
        stage="stage3",
    )
    grouped_file = Path(manifest["grouped_input_file"])
    grouped = read_dataframe(grouped_file)
    require_columns(
        grouped,
        {"key", "target_population", "category"},
        description="Saved Stage 3 groups",
    )
    grouped_by_key = grouped.set_index("key", drop=False)

    def validator(key: str, value: Any) -> list[dict[str, Any]]:
        if key not in grouped_by_key.index:
            raise ValueError("No Stage 3 group metadata were found for this key.")
        row = grouped_by_key.loc[key]
        return validate_stage3_response(
            value,
            expected_population=row["target_population"],
            expected_category=row["category"],
        )

    validated, errors = validate_results(
        stage="stage3",
        work_dir=args.work_dir,
        request_index=request_index,
        validator=validator,
    )
    final_qa = build_final_qa(grouped, validated)
    write_merge_outputs(
        outputs=[(final_qa, args.output)],
        errors=errors,
        errors_file=args.errors_output,
        overwrite=args.overwrite,
        allow_incomplete=args.allow_incomplete,
    )


def rebalance(args: argparse.Namespace) -> None:
    frame = read_dataframe(args.input)
    balanced = rebalance_correct_answers(frame, seed=args.seed)
    ensure_writable([args.output], args.overwrite)
    atomic_write_dataframe(balanced, args.output)
    counts = balanced["correct_answer"].value_counts().sort_index().to_dict()
    print(f"Saved {len(balanced):,} balanced questions to: {args.output}")
    print(f"Correct-answer counts: {counts}")


def submit(args: argparse.Namespace) -> None:
    submit_stage_batches(
        stage=args.stage,
        work_dir=args.work_dir,
        model=args.model,
        api_key_env=args.api_key_env,
        display_name_prefix=args.display_name_prefix or args.stage,
        resume=args.resume,
    )


def collect(args: argparse.Namespace) -> None:
    collect_stage_batches(
        stage=args.stage,
        work_dir=args.work_dir,
        api_key_env=args.api_key_env,
        wait=args.wait,
        poll_interval_seconds=args.poll_interval_seconds,
        wait_timeout_seconds=args.wait_timeout_seconds,
        overwrite=args.overwrite,
    )


def add_prepare_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_temperature: float,
) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=default_temperature)
    parser.add_argument("--overwrite", action="store_true")


def add_merge_arguments(
    parser: argparse.ArgumentParser,
    *,
    annotated: bool,
) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    if annotated:
        parser.add_argument("--annotated-output", type=Path)
    parser.add_argument("--errors-output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1_prepare = subparsers.add_parser(
        "prepare-stage1",
        help="Create Stage 1 relevance-classification request batches.",
    )
    add_prepare_arguments(stage1_prepare, default_temperature=0.0)
    stage1_prepare.add_argument("--min-characters", type=int, default=100)
    stage1_prepare.set_defaults(handler=prepare_stage1)

    stage2_prepare = subparsers.add_parser(
        "prepare-stage2",
        help="Create Stage 2 clinical-rule extraction request batches.",
    )
    add_prepare_arguments(stage2_prepare, default_temperature=0.0)
    stage2_prepare.set_defaults(handler=prepare_stage2)

    stage3_prepare = subparsers.add_parser(
        "prepare-stage3",
        help="Group extracted rules and create Stage 3 MCQ request batches.",
    )
    add_prepare_arguments(stage3_prepare, default_temperature=0.7)
    stage3_prepare.set_defaults(handler=prepare_stage3)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Upload prepared JSONL files and submit Gemini batch jobs.",
    )
    submit_parser.add_argument("--stage", choices=STAGES, required=True)
    submit_parser.add_argument("--work-dir", type=Path, required=True)
    submit_parser.add_argument("--model", default=DEFAULT_MODEL)
    submit_parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    submit_parser.add_argument("--display-name-prefix")
    submit_parser.add_argument("--resume", action="store_true")
    submit_parser.set_defaults(handler=submit)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Check batch jobs and download completed result JSONL files.",
    )
    collect_parser.add_argument("--stage", choices=STAGES, required=True)
    collect_parser.add_argument("--work-dir", type=Path, required=True)
    collect_parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    collect_parser.add_argument("--wait", action="store_true")
    collect_parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=30.0,
    )
    collect_parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=86400.0,
    )
    collect_parser.add_argument("--overwrite", action="store_true")
    collect_parser.set_defaults(handler=collect)

    stage1_merge = subparsers.add_parser(
        "merge-stage1",
        help="Validate Stage 1 results and retain relevant pages.",
    )
    add_merge_arguments(stage1_merge, annotated=True)
    stage1_merge.set_defaults(handler=merge_stage1)

    stage2_merge = subparsers.add_parser(
        "merge-stage2",
        help="Validate and expand Stage 2 rules into one rule per row.",
    )
    add_merge_arguments(stage2_merge, annotated=True)
    stage2_merge.set_defaults(handler=merge_stage2)

    stage3_merge = subparsers.add_parser(
        "merge-stage3",
        help="Validate Stage 3 questions and retain page provenance.",
    )
    add_merge_arguments(stage3_merge, annotated=False)
    stage3_merge.set_defaults(handler=merge_stage3)

    rebalance_parser = subparsers.add_parser(
        "rebalance",
        help="Evenly redistribute correct answers across a/b/c/d positions.",
    )
    rebalance_parser.add_argument("--input", type=Path, required=True)
    rebalance_parser.add_argument("--output", type=Path, required=True)
    rebalance_parser.add_argument("--seed", type=int, default=42)
    rebalance_parser.add_argument("--overwrite", action="store_true")
    rebalance_parser.set_defaults(handler=rebalance)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if hasattr(args, "min_characters") and args.min_characters < 0:
        raise ValueError("--min-characters cannot be negative.")
    if hasattr(args, "temperature") and args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")
    args.handler(args)


if __name__ == "__main__":
    main()
