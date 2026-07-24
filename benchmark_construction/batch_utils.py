"""Shared local I/O and Gemini Batch API utilities for construction stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_DATA_SUFFIXES = {".csv", ".xlsx", ".pkl", ".parquet"}
TERMINAL_JOB_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}


def read_dataframe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(
        f"Unsupported data file {path}; expected one of "
        f"{sorted(SUPPORTED_DATA_SUFFIXES)}."
    )


def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_DATA_SUFFIXES:
        raise ValueError(
            f"Unsupported output suffix {path.suffix!r}; expected one of "
            f"{sorted(SUPPORTED_DATA_SUFFIXES)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
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
    temporary_path.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_path.replace(path)


def atomic_write_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists. Use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing[:10])
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_artifacts(work_dir: Path, stage: str) -> dict[str, Path]:
    return {
        "prepare_manifest": work_dir / f"{stage}_prepare_manifest.json",
        "request_index": work_dir / f"{stage}_request_index.csv",
        "job_manifest": work_dir / f"{stage}_job_manifest.json",
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in: {path}")
    return payload


def write_request_batches(
    *,
    stage: str,
    input_file: Path,
    work_dir: Path,
    requests: list[dict[str, Any]],
    request_index: pd.DataFrame,
    batch_size: int,
    metadata: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not requests:
        raise ValueError(f"No {stage} requests were prepared.")
    if len(requests) != len(request_index):
        raise ValueError(
            f"Request/index length mismatch: {len(requests)} vs "
            f"{len(request_index)}."
        )
    if "key" not in request_index:
        raise ValueError("Request index must contain a key column.")
    if request_index["key"].duplicated().any():
        raise ValueError("Request index contains duplicate keys.")
    request_keys = [request.get("key") for request in requests]
    if request_keys != request_index["key"].tolist():
        raise ValueError("Request order/keys do not match the request index.")

    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts = stage_artifacts(work_dir, stage)
    stale_request_files = sorted(work_dir.glob(f"{stage}_requests_batch_*.jsonl"))
    stale_result_files = sorted(work_dir.glob(f"{stage}_results_batch_*.jsonl"))
    ensure_writable(
        [
            artifacts["prepare_manifest"],
            artifacts["request_index"],
            artifacts["job_manifest"],
            *stale_request_files,
            *stale_result_files,
        ],
        overwrite,
    )
    if overwrite:
        stale_files = [
            artifacts["job_manifest"],
            *stale_request_files,
            *stale_result_files,
        ]
        for stale_file in stale_files:
            if not stale_file.exists():
                continue
            stale_file.unlink()

    request_files: list[Path] = []
    for batch_number, start in enumerate(
        range(0, len(requests), batch_size),
        start=1,
    ):
        batch_requests = requests[start : start + batch_size]
        request_path = work_dir / f"{stage}_requests_batch_{batch_number:03d}.jsonl"
        with request_path.open("w", encoding="utf-8") as file:
            for request in batch_requests:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")
        request_files.append(request_path)

    atomic_write_dataframe(request_index, artifacts["request_index"])
    manifest = {
        "stage": stage,
        "input_file": str(input_file.resolve()),
        "input_sha256": sha256_file(input_file),
        "request_count": len(requests),
        "batch_size": batch_size,
        "batch_count": len(request_files),
        "request_index_file": str(artifacts["request_index"].resolve()),
        "request_files": [str(path.resolve()) for path in request_files],
        **metadata,
    }
    atomic_write_json(manifest, artifacts["prepare_manifest"])
    return manifest


def create_genai_client(api_key_env: str) -> tuple[Any, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"API key environment variable is not set: {api_key_env}"
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Gemini batch processing requires the google-genai package."
        ) from exc
    return genai.Client(api_key=api_key), types


def job_state_name(job: Any) -> str:
    state = getattr(job, "state", None)
    if state is None:
        return "JOB_STATE_UNSPECIFIED"
    return str(getattr(state, "name", state))


def submit_stage_batches(
    *,
    stage: str,
    work_dir: Path,
    model: str,
    api_key_env: str,
    display_name_prefix: str,
    resume: bool,
) -> None:
    artifacts = stage_artifacts(work_dir, stage)
    prepare_manifest = load_json(artifacts["prepare_manifest"])
    request_files = [Path(path) for path in prepare_manifest["request_files"]]
    for request_file in request_files:
        if not request_file.is_file():
            raise FileNotFoundError(f"Request batch not found: {request_file}")

    job_manifest_path = artifacts["job_manifest"]
    if job_manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"Job manifest already exists: {job_manifest_path}. "
                "Use --resume only to continue a partial submission."
            )
        job_manifest = load_json(job_manifest_path)
        if job_manifest.get("model") != model:
            raise ValueError(
                "The existing job manifest uses a different model: "
                f"{job_manifest.get('model')!r}."
            )
    else:
        job_manifest = {
            "stage": stage,
            "model": model,
            "api_key_environment": api_key_env,
            "prepare_manifest": str(artifacts["prepare_manifest"].resolve()),
            "batches": [],
        }

    submitted = {
        batch["request_file"]
        for batch in job_manifest.get("batches", [])
        if batch.get("job_name")
    }
    client, types = create_genai_client(api_key_env)
    for batch_number, request_file in enumerate(request_files, start=1):
        request_file_resolved = str(request_file.resolve())
        if request_file_resolved in submitted:
            print(f"{stage} batch {batch_number:03d} already submitted; skipping.")
            continue

        uploaded_file = client.files.upload(
            file=str(request_file),
            config=types.UploadFileConfig(
                display_name=(
                    f"{display_name_prefix}-requests-{batch_number:03d}"
                ),
                mime_type="jsonl",
            ),
        )
        batch_job = client.batches.create(
            model=model,
            src=uploaded_file.name,
            config={
                "display_name": f"{display_name_prefix}-job-{batch_number:03d}"
            },
        )
        job_manifest["batches"].append(
            {
                "batch_number": batch_number,
                "request_file": request_file_resolved,
                "uploaded_file_name": uploaded_file.name,
                "job_name": batch_job.name,
                "last_known_state": job_state_name(batch_job),
                "local_result_file": "",
            }
        )
        atomic_write_json(job_manifest, job_manifest_path)
        print(
            f"Submitted {stage} batch {batch_number:03d}: "
            f"{batch_job.name} ({job_state_name(batch_job)})"
        )
    print(f"Saved job manifest to: {job_manifest_path}")


def wait_for_job(
    client: Any,
    job_name: str,
    *,
    wait: bool,
    poll_interval_seconds: float,
    wait_timeout_seconds: float,
) -> Any:
    started = time.monotonic()
    while True:
        job = client.batches.get(name=job_name)
        state = job_state_name(job)
        if state in TERMINAL_JOB_STATES or not wait:
            return job
        if time.monotonic() - started >= wait_timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for {job_name}; last state was {state}."
            )
        print(f"{job_name}: {state}; checking again in {poll_interval_seconds:.1f}s.")
        time.sleep(poll_interval_seconds)


def collect_stage_batches(
    *,
    stage: str,
    work_dir: Path,
    api_key_env: str,
    wait: bool,
    poll_interval_seconds: float,
    wait_timeout_seconds: float,
    overwrite: bool,
) -> None:
    if poll_interval_seconds <= 0 or wait_timeout_seconds <= 0:
        raise ValueError("Polling interval and wait timeout must be positive.")

    artifacts = stage_artifacts(work_dir, stage)
    job_manifest = load_json(artifacts["job_manifest"])
    batches = job_manifest.get("batches", [])
    if not batches:
        raise ValueError("Job manifest contains no submitted batches.")

    client, _ = create_genai_client(api_key_env)
    incomplete: list[str] = []
    failed: list[str] = []
    for batch in batches:
        batch_number = int(batch["batch_number"])
        result_path = work_dir / f"{stage}_results_batch_{batch_number:03d}.jsonl"
        if (
            result_path.exists()
            and batch.get("local_result_file")
            and not overwrite
        ):
            print(f"{stage} batch {batch_number:03d} already collected; skipping.")
            continue
        if result_path.exists() and not overwrite:
            raise FileExistsError(
                f"Untracked result file already exists: {result_path}"
            )

        job = wait_for_job(
            client,
            batch["job_name"],
            wait=wait,
            poll_interval_seconds=poll_interval_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
        )
        state = job_state_name(job)
        batch["last_known_state"] = state
        if state == "JOB_STATE_SUCCEEDED":
            destination = getattr(job, "dest", None)
            result_file_name = getattr(destination, "file_name", None)
            if not result_file_name:
                failed.append(
                    f"batch {batch_number:03d}: succeeded without dest.file_name"
                )
            else:
                content = client.files.download(file=result_file_name)
                atomic_write_bytes(content, result_path)
                batch["remote_result_file_name"] = result_file_name
                batch["local_result_file"] = str(result_path.resolve())
                print(
                    f"Collected {stage} batch {batch_number:03d}: "
                    f"{result_path} ({len(content):,} bytes)"
                )
        elif state in TERMINAL_JOB_STATES:
            failed.append(
                f"batch {batch_number:03d}: {state}: {getattr(job, 'error', None)}"
            )
        else:
            incomplete.append(f"batch {batch_number:03d}: {state}")
        atomic_write_json(job_manifest, artifacts["job_manifest"])

    if incomplete:
        raise RuntimeError(
            "Some jobs are not complete. Run collect again later or use --wait: "
            f"{incomplete}"
        )
    if failed:
        raise RuntimeError(f"Some jobs failed or had no output file: {failed}")
    print(f"All {stage} result files are available.")


def parse_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}."
                )
            rows.append(value)
    return rows


def extract_response_text(result: dict[str, Any]) -> str:
    try:
        return result["response"]["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        api_error = result.get("error")
        if api_error:
            raise ValueError(f"Batch response contains API error: {api_error}") from exc
        raise ValueError("Batch response is missing candidate text.") from exc


def parse_json_value(text: str) -> Any:
    candidate = text.strip()
    fenced = re.fullmatch(
        r"\s*```(?:json)?\s*([\[{].*[\]}])\s*```\s*",
        candidate,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidate = fenced.group(1)
    else:
        starts = [
            position
            for position in (candidate.find("{"), candidate.find("["))
            if position >= 0
        ]
        start = min(starts) if starts else -1
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)


def load_stage_results(
    *,
    stage: str,
    work_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    artifacts = stage_artifacts(work_dir, stage)
    prepare_manifest = load_json(artifacts["prepare_manifest"])
    expected_batch_count = int(prepare_manifest["batch_count"])
    expected_paths = {
        work_dir / f"{stage}_results_batch_{batch_number:03d}.jsonl"
        for batch_number in range(1, expected_batch_count + 1)
    }
    result_paths = sorted(work_dir.glob(f"{stage}_results_batch_*.jsonl"))
    actual_paths = set(result_paths)
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        unexpected = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(
            f"{stage} result files do not match the prepared batches. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )

    result_by_key: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for result_path in result_paths:
        for result in parse_jsonl_file(result_path):
            key = result.get("key")
            if not isinstance(key, str):
                errors.append(
                    {
                        "key": "",
                        "result_file": str(result_path),
                        "error": "Result is missing a string key.",
                        "raw_response": json.dumps(result, ensure_ascii=False),
                    }
                )
                continue
            if key in result_by_key:
                errors.append(
                    {
                        "key": key,
                        "result_file": str(result_path),
                        "error": "Duplicate result key.",
                        "raw_response": json.dumps(result, ensure_ascii=False),
                    }
                )
                continue
            try:
                result_by_key[key] = parse_json_value(extract_response_text(result))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(
                    {
                        "key": key,
                        "result_file": str(result_path),
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_response": json.dumps(result, ensure_ascii=False),
                    }
                )
    return result_by_key, errors
