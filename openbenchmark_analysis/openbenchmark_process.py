"""Classify public medical QA benchmarks with a single LLM.

This script does not redistribute benchmark data or model weights. Users must
provide local cache/input/output paths and obtain the MedBullets CSV separately.
Run this script once for each classifier model used in the majority vote.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


HF_DATASETS = (
    "openlifescienceai/mmlu_clinical_knowledge",
    "openlifescienceai/mmlu_professional_medicine",
    "openlifescienceai/medmcqa",
    "openlifescienceai/medqa",
    "TsinghuaC3I/MedXpertQA",
)

QUESTION_TYPES = {"DIAGNOSIS", "MANAGEMENT", "BASIC_SCIENCE", "OTHER"}
TARGET_POPULATIONS = {"Pediatric", "Adult", "General"}

PROMPT_CLASSIFY_SYSTEM = (
    "You are a medical question reviewer specializing in Anesthesiology."
)

PROMPT_CLASSIFY_USER = """### Instruction:
Analyze the following question and options. Classify the question type, target
population, and whether the primary subject is anesthesiology or pain medicine.

Return ONLY a valid JSON object. Do not include any extra text.

### Question type must be ONE of:
- "DIAGNOSIS": asks to identify the most likely diagnosis/cause or differential diagnosis.
- "MANAGEMENT": asks what to do (treatment, next step, testing, dose, disposition,
  guideline-based decision, etc.).
- "BASIC_SCIENCE": tests factual knowledge (physiology, pharmacology, anatomy,
  definitions, mechanisms, etc.), not primarily a clinical decision.
- "OTHER": does not fit the categories above (research methods, ethics, law,
  statistics, administration, or unclear).

Choose the single most appropriate question type even if multiple intents exist.

### Target Population Classification:
1. Classify as "Pediatric" IF AND ONLY IF:
- The question explicitly mentions a patient under 18 years old (e.g., infant,
  neonate, toddler, child, adolescent, boy, girl, or pediatric).
- OR the disease/condition described is exclusively pediatric.
- OR all options are strictly related to pediatric conditions.

2. Classify as "Adult" if:
- The question explicitly mentions a patient 18 years or older.
- OR the text refers to adults, elderly, or geriatric patients.

3. Classify as "General" if:
- No age group is explicitly mentioned.
- OR the scenario is mixed or unclear.

### Anesthesiology related:
Set "is_anesthesia" = true ONLY IF:
- The primary subject is anesthesia or pain medicine.
- The question would most naturally be answered by an anesthesiologist rather
  than by a surgeon, internist, pediatrician, or emergency physician.

Typical true cases include:
- Choice of anesthesia or sedation technique
- Airway management as an anesthesia task
- Anesthetic drugs and their perioperative use
- Perioperative anesthesia complications
- Regional or neuraxial anesthesia
- Post-anesthesia recovery and monitoring
- Pain medicine procedures or management

Set "is_anesthesia" = false IF:
- The main subject is surgery, internal medicine, pediatrics, emergency
  medicine, or critical care, even if anesthesia, surgery, ICU care, or
  procedures are mentioned.
- Anesthesia is only a minor background detail.
- The question mainly concerns diagnosis or treatment of a disease rather than
  anesthesia itself.

### Output JSON format
{{
  "question_type": "DIAGNOSIS|MANAGEMENT|BASIC_SCIENCE|OTHER",
  "target_population": "Pediatric|Adult|General",
  "is_anesthesia": true,
  "reason": "Short justification"
}}

### Input:
{data}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one LLM classifier over the five public QA benchmarks."
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        required=True,
        help="Local Hugging Face dataset cache directory.",
    )
    parser.add_argument(
        "--medbullets-csv",
        type=Path,
        required=True,
        help="User-provided MedBullets CSV file.",
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Hugging Face model ID or a local model directory.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        required=True,
        help="Directory used by vLLM for model downloads/cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which model classifications are saved.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional output stem. By default, it is derived from the model name.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help='Optional CUDA device list, for example "0,1" or "0,1,2,3".',
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--expected-total-items",
        type=int,
        default=10_718,
        help="Expected merged item count; set to 0 to disable the check.",
    )
    parser.add_argument(
        "--dataset-revision",
        help="Optional Hugging Face dataset revision applied to all HF datasets.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only locally cached Hugging Face datasets and model files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def medmcqa_process(row: pd.Series) -> dict[str, Any]:
    return {
        "Correct Answer": "",
        "Correct Option": "",
        "Options": {
            "A": row["opa"],
            "B": row["opb"],
            "C": row["opc"],
            "D": row["opd"],
        },
        "Question": row["question"],
    }


def medxpertqa_process(row: pd.Series) -> dict[str, Any]:
    return {
        "Correct Answer": row["options"][row["label"]],
        "Correct Option": row["label"],
        "Options": row["options"],
        "Question": row["question"],
    }


def medbullets_process(row: pd.Series) -> dict[str, Any]:
    return {
        "Correct Answer": row["answer"],
        "Correct Option": row["answer_idx"],
        "Options": {
            "A": row["opa"],
            "B": row["opb"],
            "C": row["opc"],
            "D": row["opd"],
            "E": row["ope"],
        },
        "Question": row["question"],
        "Explanation": row["explanation"],
    }


def normalize_hf_dataset(
    dataset_name: str,
    dataset_cache_dir: Path,
    dataset_revision: str | None,
) -> pd.DataFrame:
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {"cache_dir": str(dataset_cache_dir)}
    if dataset_revision:
        load_kwargs["revision"] = dataset_revision

    if dataset_name == "TsinghuaC3I/MedXpertQA":
        dataset = load_dataset(dataset_name, "Text", **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, **load_kwargs)

    if "test" not in dataset:
        raise ValueError(f"{dataset_name} does not contain a test split.")

    df = dataset["test"].to_pandas()
    if dataset_name == "openlifescienceai/medmcqa":
        require_columns(
            df,
            {"id", "question", "opa", "opb", "opc", "opd"},
            dataset_name,
        )
        normalized = pd.DataFrame(
            {
                "id": df["id"],
                "data": df.apply(medmcqa_process, axis=1),
                "subject_name": df.get("subject_name"),
                "medical_task": None,
            }
        )
    elif dataset_name == "TsinghuaC3I/MedXpertQA":
        require_columns(
            df,
            {"id", "question", "options", "label", "medical_task"},
            dataset_name,
        )
        normalized = pd.DataFrame(
            {
                "id": df["id"],
                "data": df.apply(medxpertqa_process, axis=1),
                "subject_name": None,
                "medical_task": df["medical_task"],
            }
        )
    else:
        require_columns(df, {"id", "data"}, dataset_name)
        normalized = pd.DataFrame(
            {
                "id": df["id"],
                "data": df["data"],
                "subject_name": df.get("subject_name"),
                "medical_task": df.get("medical_task"),
            }
        )

    normalized["source"] = dataset_name
    return normalized


def normalize_medbullets(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"MedBullets CSV not found: {csv_path}. "
            "Obtain the dataset separately and pass its local path."
        )

    df = pd.read_csv(csv_path)
    require_columns(
        df,
        {
            "link",
            "question",
            "answer",
            "answer_idx",
            "opa",
            "opb",
            "opc",
            "opd",
            "ope",
            "explanation",
        },
        "MedBullets CSV",
    )
    return pd.DataFrame(
        {
            "id": df["link"],
            "data": df.apply(medbullets_process, axis=1),
            "subject_name": None,
            "medical_task": None,
            "source": "medbullets_op5",
        }
    )


def load_benchmarks(args: argparse.Namespace) -> pd.DataFrame:
    frames = [
        normalize_hf_dataset(
            dataset_name,
            args.dataset_cache_dir,
            args.dataset_revision,
        )
        for dataset_name in HF_DATASETS
    ]
    frames.append(normalize_medbullets(args.medbullets_csv))

    total_df = pd.concat(frames, ignore_index=True)
    if total_df["id"].isna().any():
        raise ValueError("At least one benchmark item has a missing ID.")
    if total_df["id"].duplicated().any():
        duplicated = total_df.loc[total_df["id"].duplicated(), "id"].head().tolist()
        raise ValueError(f"Benchmark IDs must be globally unique; duplicates: {duplicated}")
    if total_df["data"].isna().any():
        raise ValueError("At least one benchmark item has missing normalized data.")
    if args.expected_total_items and len(total_df) != args.expected_total_items:
        raise ValueError(
            f"Expected {args.expected_total_items:,} items, found {len(total_df):,}. "
            "Check dataset versions and the MedBullets input."
        )

    print(f"Total items after merge: {len(total_df):,}")
    print("Items by source:")
    print(total_df["source"].value_counts().sort_index().to_string())
    return total_df


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fence_match = re.fullmatch(
        r"\s*```(?:json)?\s*(\{.*\})\s*```\s*",
        candidate,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        candidate = fence_match.group(1)

    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(candidate[start : end + 1])

    if not isinstance(result, dict):
        raise ValueError("Model output must decode to a JSON object.")
    return result


def validate_classification(result: dict[str, Any]) -> dict[str, Any]:
    required = {"question_type", "target_population", "is_anesthesia", "reason"}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"Classification is missing fields: {missing}")
    if result["question_type"] not in QUESTION_TYPES:
        raise ValueError(f"Invalid question_type: {result['question_type']!r}")
    if result["target_population"] not in TARGET_POPULATIONS:
        raise ValueError(
            f"Invalid target_population: {result['target_population']!r}"
        )
    if not isinstance(result["is_anesthesia"], bool):
        raise ValueError("is_anesthesia must be a JSON boolean.")
    if not isinstance(result["reason"], str):
        raise ValueError("reason must be a string.")
    return result


def atomic_pickle(df: pd.DataFrame, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_pickle(temporary_path)
    temporary_path.replace(output_path)


def classify_benchmarks(
    total_df: pd.DataFrame,
    args: argparse.Namespace,
    raw_output_path: Path,
) -> pd.DataFrame:
    import torch
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model_name_or_path}")
    model = LLM(
        model=args.model_name_or_path,
        download_dir=str(args.model_cache_dir),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_seq_len_to_capture=min(2048, args.max_model_len),
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
    )
    tokenizer = model.get_tokenizer()
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    model_outputs: list[str] = []
    for start in tqdm(
        range(0, len(total_df), args.batch_size),
        desc="Classifying",
    ):
        batch_df = total_df.iloc[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": PROMPT_CLASSIFY_SYSTEM},
                    {
                        "role": "user",
                        "content": PROMPT_CLASSIFY_USER.format(
                            data=json.dumps(
                                row["data"],
                                indent=2,
                                ensure_ascii=False,
                            )
                        ),
                    },
                ],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            for _, row in batch_df.iterrows()
        ]
        with torch.no_grad():
            generated = model.generate(prompts, sampling_params)
        model_outputs.extend(output.outputs[0].text.strip() for output in generated)

    if len(model_outputs) != len(total_df):
        raise RuntimeError(
            f"Expected {len(total_df)} model outputs, received {len(model_outputs)}."
        )

    classified = total_df.copy()
    classified["model_class"] = model_outputs
    atomic_pickle(classified, raw_output_path)
    print(f"Saved raw model outputs to: {raw_output_path}")

    parsed_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for idx, output in enumerate(model_outputs):
        try:
            parsed_rows.append(validate_classification(extract_json_object(output)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            parse_errors.append(
                {
                    "row_index": idx,
                    "id": str(total_df.iloc[idx]["id"]),
                    "error": f"{type(exc).__name__}: {exc}",
                    "model_output": output,
                }
            )

    if parse_errors:
        error_path = raw_output_path.with_suffix(".parse_errors.jsonl")
        with error_path.open("w", encoding="utf-8") as file:
            for error in parse_errors:
                file.write(json.dumps(error, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"{len(parse_errors)} outputs failed validation. "
            f"Raw outputs were preserved at {raw_output_path}; "
            f"details are in {error_path}."
        )

    classified["question_type"] = [row["question_type"] for row in parsed_rows]
    classified["target_population"] = [
        row["target_population"] for row in parsed_rows
    ]
    classified["is_anesthesia_model"] = [
        row["is_anesthesia"] for row in parsed_rows
    ]
    classified["model_reason"] = [row["reason"] for row in parsed_rows]
    return classified


def safe_output_stem(model_name_or_path: str) -> str:
    stem = model_name_or_path.rstrip("/\\").replace("\\", "/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = args.output_name or safe_output_stem(args.model_name_or_path)
    output_path = args.output_dir / f"openbench_{output_stem}_inference.pkl"
    raw_output_path = args.output_dir / f"openbench_{output_stem}_raw.pkl"
    config_path = args.output_dir / f"openbench_{output_stem}_config.json"

    existing = [path for path in (output_path, raw_output_path, config_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists. Use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    total_df = load_benchmarks(args)
    classified = classify_benchmarks(total_df, args, raw_output_path)
    atomic_pickle(classified, output_path)
    print(f"Saved {len(classified):,} validated classifications to: {output_path}")


if __name__ == "__main__":
    main()
