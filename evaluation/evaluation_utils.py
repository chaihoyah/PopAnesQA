"""Data-schema, prompting, parsing, and metric utilities."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


POPULATIONS = ("Adult", "Pediatric", "General")
OPTION_LABELS = ("a", "b", "c", "d")


def stable_tag(value: str) -> str:
    path = Path(value)
    if "models--" in value:
        base = value.split("models--", 1)[1].split("snapshots", 1)[0]
    else:
        base = path.name or value
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_")
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{clean or 'item'}-{digest}"


def _nested(value: Any, name: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{name} contains invalid serialized data.") from exc


def get_options(row: pd.Series) -> dict[str, str]:
    """Return the options actually paired with the evaluated answer key."""
    candidates: list[Any] = []
    if "new_options" in row and not _missing(row["new_options"]):
        candidates.append(row["new_options"])
    if "options" in row and not _missing(row["options"]):
        candidates.append(row["options"])
    for candidate in candidates:
        value = _nested(candidate, "options")
        if isinstance(value, dict):
            normalized = {str(k).lower(): str(v) for k, v in value.items()}
            if set(normalized) == set(OPTION_LABELS):
                return {label: normalized[label] for label in OPTION_LABELS}
    raise ValueError("Row has no valid four-option mapping.")


def get_correct_answer(row: pd.Series) -> str:
    """Support canonical PopAnesQA and legacy rebalanced column names."""
    if (
        "new_options" in row
        and not _missing(row["new_options"])
        and "new_correct_answer" in row
        and not _missing(row["new_correct_answer"])
    ):
        value = row["new_correct_answer"]
    elif "correct_answer" in row and not _missing(row["correct_answer"]):
        value = row["correct_answer"]
    elif "final_answer" in row and not _missing(row["final_answer"]):
        value = row["final_answer"]
    else:
        raise ValueError(
            "Row has no answer key; expected correct_answer, "
            "new_correct_answer, or final_answer."
        )
    answer = str(value).strip().lower().strip("()")
    if answer not in OPTION_LABELS:
        raise ValueError(f"Invalid answer label: {value!r}")
    return answer


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def retrieval_query_text(row: pd.Series) -> str:
    """Reproduce the original retrieval query without answer options."""
    if (
        "patient_profile" in row
        and "scenario" in row
        and not _missing(row["patient_profile"])
        and not _missing(row["scenario"])
    ):
        return (
            f"{row['patient_profile']} {row['scenario']} {row['question']}"
        )
    return str(row["question"])


def format_context(texts: Iterable[str], max_chars_each: int) -> str:
    blocks = []
    for rank, text in enumerate(texts, start=1):
        cleaned = str(text).strip()
        if len(cleaned) > max_chars_each:
            clipped = cleaned[:max_chars_each]
            cleaned = clipped.rsplit(" ", 1)[0] + "..."
        blocks.append(f"[{rank}] Text:\n{cleaned}")
    if not blocks:
        return ""
    noun = "excerpt" if len(blocks) == 1 else "excerpts"
    return f"### Retrieved {noun} (top-{len(blocks)})\n" + "\n\n".join(blocks)


def build_mcqa_prompt(row: pd.Series, context: str = "") -> str:
    options = get_options(row)
    option_text = "\n".join(
        f"({label.upper()}): {options[label]}" for label in OPTION_LABELS
    )
    patient = (
        str(row["patient_profile"]).strip()
        if "patient_profile" in row and not _missing(row["patient_profile"])
        else ""
    )
    scenario = (
        str(row["scenario"]).strip()
        if "scenario" in row and not _missing(row["scenario"])
        else ""
    )
    clinical_parts = []
    if patient:
        clinical_parts.append(f"### Patient profile: {patient}")
    if scenario:
        clinical_parts.append(f"### Clinical scenario: {scenario}")
    clinical = "\n\n".join(clinical_parts)
    if not context:
        return f"""### Instruction: The following is a scenario-based multiple-choice question for anesthesiology specialists. Carefully consider the patient profile and clinical scenario, and choose the most accurate answer from the options provided. Be sure to follow the output structure and provide an appropriate explanation.

### Output structure:
- Explanation: Your step-by-step reasoning
- Final answer: Your final answer (e.g. (A), (B), ...)

{clinical}

### Question: {row['question']}
{option_text}

### Answer:
""".strip()
    return f"""### Instruction: The following is a scenario-based multiple-choice question for anesthesiology specialists. Carefully consider the patient profile and clinical scenario, and choose the most accurate answer from the options provided. If the retrieved excerpts are included, use them as supporting evidence to help identify the correct answer when relevant. Be sure to follow the output structure and provide an appropriate explanation.

### Output structure:
- Explanation: Your step-by-step reasoning
- Final answer: Your final answer (e.g. (A), (B), ...)

{clinical}

### Question: {row['question']}
{option_text}

{context}

### Answer:
""".strip()


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except (AttributeError, TypeError, ValueError):
            return prompt


def prompt_with_budget(
    tokenizer: Any,
    row: pd.Series,
    retrieved_texts: list[str],
    *,
    max_model_length: int,
    max_output_tokens: int,
    max_chars_each: int,
) -> str:
    input_budget = max_model_length - max_output_tokens
    if input_budget <= 0:
        raise ValueError("max_output_tokens must be smaller than max_model_length.")
    current_chars = max_chars_each
    while True:
        context = format_context(retrieved_texts, current_chars)
        prompt = apply_chat_template(tokenizer, build_mcqa_prompt(row, context))
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count <= input_budget:
            return prompt
        if not retrieved_texts or current_chars <= 128:
            raise ValueError(
                f"Prompt requires {token_count} tokens but input budget is "
                f"{input_budget}. Increase --max-model-length or reduce output."
            )
        current_chars = max(128, int(current_chars * 0.75))


FINAL_ANSWER_PATTERN = re.compile(
    r"final\s*answer\s*:?\s*(?:option\s*)?[\(\[]?\s*([A-D])\s*[\)\]]?",
    re.IGNORECASE,
)
PAREN_ANSWER_PATTERN = re.compile(r"\(([A-D])\)", re.IGNORECASE)


def parse_mcqa_answer(text: Any) -> str | None:
    value = str(text)
    explicit = list(FINAL_ANSWER_PATTERN.finditer(value))
    if explicit:
        return explicit[-1].group(1).lower()
    parenthesized = list(PAREN_ANSWER_PATTERN.finditer(value))
    return parenthesized[-1].group(1).lower() if parenthesized else None


def evaluate_predictions(df: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    if prediction_column not in df:
        raise ValueError(f"Missing prediction column: {prediction_column}")
    totals = {population: 0 for population in POPULATIONS}
    correct = {population: 0 for population in POPULATIONS}
    overall_correct = 0
    unparsed = 0
    for _, row in df.iterrows():
        prediction = parse_mcqa_answer(row[prediction_column])
        expected = get_correct_answer(row)
        if prediction is None:
            unparsed += 1
        is_correct = prediction == expected
        overall_correct += int(is_correct)
        if "target_population" in row and not _missing(row["target_population"]):
            population = str(row["target_population"]).strip()
            if population not in POPULATIONS:
                raise ValueError(f"Unknown target_population: {population!r}")
            totals[population] += 1
            correct[population] += int(is_correct)
    metrics: dict[str, Any] = {
        "n": len(df),
        "correct": overall_correct,
        "unparsed": unparsed,
        "accuracy": 100.0 * overall_correct / len(df) if len(df) else float("nan"),
    }
    for population in POPULATIONS:
        key = population.lower()
        metrics[f"n_{key}"] = totals[population]
        metrics[f"accuracy_{key}"] = (
            100.0 * correct[population] / totals[population]
            if totals[population]
            else float("nan")
        )
    return metrics


def load_evaluation_data(path: Path, population: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_pickle(path)
    required = {"question"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation data are missing columns: {missing}")
    population_normalized = population.lower()
    if population_normalized != "all":
        lookup = {value.lower(): value for value in POPULATIONS}
        if population_normalized not in lookup:
            raise ValueError(
                "--target-population must be all, adult, pediatric, or general."
            )
        if "target_population" not in frame:
            raise ValueError("Population filtering requires target_population.")
        frame = frame[
            frame["target_population"].eq(lookup[population_normalized])
        ]
    frame = frame.reset_index(drop=True)
    for _, row in frame.iterrows():
        get_options(row)
        get_correct_answer(row)
    return frame


def update_metrics_csv(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    key_columns: list[str],
) -> None:
    current = pd.DataFrame(rows)
    if path.is_file():
        previous = pd.read_csv(path)
        combined = pd.concat([previous, current], ignore_index=True)
    else:
        combined = current
    keys = [column for column in key_columns if column in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
        combined = combined.sort_values(keys, kind="stable", na_position="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
