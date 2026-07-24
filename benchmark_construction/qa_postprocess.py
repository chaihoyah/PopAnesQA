"""Validation and deterministic post-processing for PopAnesQA construction."""

from __future__ import annotations

import ast
import copy
import json
import random
from collections import Counter
from typing import Any

import pandas as pd


TARGET_POPULATIONS = {"Pediatric", "Adult", "General"}
CATEGORIES = {
    "Pre-op Assessment",
    "Pharmacology & Fluid",
    "Airway & Equipment",
    "Crisis & Complication",
    "Post-op & Pain",
}
QUESTION_TYPES = {"Management", "Calculation", "Diagnosis"}
DIFFICULTIES = {"Hard", "Medium", "Easy"}
OPTION_LABELS = ("a", "b", "c", "d")


def validate_stage1_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Stage 1 response must be a JSON object.")
    required = {"is_relevant", "target_population", "category", "reason"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Stage 1 response is missing fields: {missing}")
    if not isinstance(value["is_relevant"], bool):
        raise ValueError("is_relevant must be a JSON boolean.")
    if value["target_population"] not in TARGET_POPULATIONS:
        raise ValueError(
            f"Invalid target_population: {value['target_population']!r}"
        )
    if value["category"] is not None and value["category"] not in CATEGORIES:
        raise ValueError(f"Invalid category: {value['category']!r}")
    if value["is_relevant"] and value["category"] is None:
        raise ValueError("Relevant Stage 1 responses must include a category.")
    if not isinstance(value["reason"], str):
        raise ValueError("reason must be a string.")
    return value


def validate_stage2_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Stage 2 response must be a JSON object.")
    rules = value.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Stage 2 response must contain a rules list.")

    normalized_rules: list[dict[str, Any]] = []
    required = {
        "condition_base",
        "action",
        "triggers",
        "source_fragment",
        "notes",
    }
    for rule_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"Rule {rule_index} must be a JSON object.")
        missing = sorted(required - set(rule))
        if missing:
            raise ValueError(f"Rule {rule_index} is missing fields: {missing}")
        for field in ("condition_base", "action", "source_fragment"):
            if not isinstance(rule[field], str) or not rule[field].strip():
                raise ValueError(
                    f"Rule {rule_index} field {field!r} must be non-empty text."
                )
        if not isinstance(rule["triggers"], list) or not rule["triggers"]:
            raise ValueError(
                f"Rule {rule_index} triggers must be a non-empty list."
            )
        for trigger_index, trigger in enumerate(rule["triggers"]):
            if not isinstance(trigger, dict):
                raise ValueError(
                    f"Rule {rule_index} trigger {trigger_index} must be an object."
                )
            criterion = trigger.get("criterion")
            if not isinstance(criterion, str) or not criterion.strip():
                raise ValueError(
                    f"Rule {rule_index} trigger {trigger_index} must contain "
                    "non-empty criterion text."
                )
        if rule["notes"] is not None and not isinstance(rule["notes"], str):
            raise ValueError(f"Rule {rule_index} notes must be text or null.")
        normalized_rules.append(rule)

    return {"rules": normalized_rules}


def explode_stage2_rules(
    annotated_df: pd.DataFrame,
    response_column: str = "gemini_stage2_response",
) -> pd.DataFrame:
    """Expand each Stage 2 rule into one row while retaining page provenance."""
    required_columns = {
        "guideline",
        "page",
        "guidelinetext_processed",
        "target_population",
        "category",
        "reason",
        response_column,
    }
    missing = sorted(required_columns - set(annotated_df.columns))
    if missing:
        raise ValueError(f"Stage 2 annotated data are missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for _, source_row in annotated_df.iterrows():
        response = source_row[response_column]
        if not isinstance(response, dict):
            continue
        rules = response.get("rules")
        if not isinstance(rules, list):
            continue
        for rule_number, rule in enumerate(rules, start=1):
            rows.append(
                {
                    "guideline": source_row["guideline"],
                    "page": source_row["page"],
                    "guidelinetext_processed": source_row[
                        "guidelinetext_processed"
                    ],
                    "condition": rule["condition_base"],
                    "action": rule["action"],
                    "triggers": rule["triggers"],
                    "source_fragment": rule["source_fragment"],
                    "notes": rule["notes"],
                    "target_population": source_row["target_population"],
                    "category": source_row["category"],
                    "reason": source_row["reason"],
                    "rule_number_on_page": rule_number,
                    "gemini_stage2_response": response,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "guideline",
            "page",
            "guidelinetext_processed",
            "condition",
            "action",
            "triggers",
            "source_fragment",
            "notes",
            "target_population",
            "category",
            "reason",
            "rule_number_on_page",
            "gemini_stage2_response",
        ],
    )


def _one_value(group: pd.DataFrame, column: str, group_key: str) -> Any:
    values = group[column].dropna().tolist()
    unique: list[Any] = []
    for value in values:
        if not any(_values_equal(value, existing) for existing in unique):
            unique.append(value)
    if len(unique) != 1:
        raise ValueError(
            f"Group {group_key!r} must have exactly one {column!r} value; "
            f"found {len(unique)}."
        )
    return unique[0]


def _values_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
        return bool(result) if not hasattr(result, "all") else bool(result.all())
    except Exception:
        return False


def _deserialize_nested(value: Any, field: str) -> Any:
    """Recover list/dict values after a round trip through CSV."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{field} contains invalid serialized data.") from exc


def build_stage3_groups(logic_df: pd.DataFrame) -> pd.DataFrame:
    """Create one Stage 3 request row per guideline page."""
    required = {
        "guideline",
        "page",
        "guidelinetext_processed",
        "condition",
        "action",
        "triggers",
        "source_fragment",
        "notes",
        "target_population",
        "category",
        "reason",
    }
    missing = sorted(required - set(logic_df.columns))
    if missing:
        raise ValueError(f"Stage 2 logic data are missing columns: {missing}")
    if logic_df[["guideline", "page"]].isna().any().any():
        raise ValueError("Stage 2 logic data contain missing guideline/page values.")

    grouped_rows: list[dict[str, Any]] = []
    for group_number, ((guideline, page), group) in enumerate(
        logic_df.groupby(["guideline", "page"], sort=False, dropna=False),
        start=1,
    ):
        key = f"guideline_page_{group_number:06d}"
        rules = [
            {
                "condition_base": row["condition"],
                "action": row["action"],
                "triggers": _deserialize_nested(row["triggers"], "triggers"),
                "source_fragment": row["source_fragment"],
                "notes": None if pd.isna(row["notes"]) else row["notes"],
            }
            for _, row in group.iterrows()
        ]
        grouped_rows.append(
            {
                "key": key,
                "guideline": guideline,
                "page": page,
                "guideline_page_key": f"{guideline}::{page}",
                "target_population": _one_value(
                    group, "target_population", key
                ),
                "category": _one_value(group, "category", key),
                "reason": _one_value(group, "reason", key),
                "guidelinetext_processed": _one_value(
                    group, "guidelinetext_processed", key
                ),
                "rules": rules,
                "source_row_indices": group.index.tolist(),
            }
        )
    return pd.DataFrame(grouped_rows)


def _normalize_options(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        value = _deserialize_nested(value, "options")
    if not isinstance(value, dict):
        raise ValueError("options must be a JSON object.")
    normalized = {str(key).lower(): option for key, option in value.items()}
    if set(normalized) != set(OPTION_LABELS):
        raise ValueError(
            f"options must contain exactly {list(OPTION_LABELS)}; "
            f"found {sorted(normalized)}."
        )
    if not all(isinstance(option, str) and option.strip() for option in normalized.values()):
        raise ValueError("Every option must contain non-empty text.")
    return {label: normalized[label] for label in OPTION_LABELS}


def validate_stage3_response(
    value: Any,
    *,
    expected_population: str,
    expected_category: str,
) -> list[dict[str, Any]]:
    """Validate one generated MCQ; a list is accepted for legacy outputs."""
    candidates = value if isinstance(value, list) else [value]
    if not candidates:
        raise ValueError("Stage 3 response contains no generated questions.")

    required = {
        "question_type",
        "target_population",
        "category",
        "patient_profile",
        "scenario",
        "question",
        "options",
        "correct_answer",
        "explanation",
        "difficulty",
    }
    normalized_questions: list[dict[str, Any]] = []
    for question_index, question in enumerate(candidates):
        if not isinstance(question, dict):
            raise ValueError(f"Generated question {question_index} must be an object.")
        missing = sorted(required - set(question))
        if missing:
            raise ValueError(
                f"Generated question {question_index} is missing: {missing}"
            )
        if question["question_type"] not in QUESTION_TYPES:
            raise ValueError(
                f"Invalid question_type: {question['question_type']!r}"
            )
        if question["target_population"] != expected_population:
            raise ValueError(
                "Generated target_population does not match its Stage 2 group."
            )
        if question["category"] != expected_category:
            raise ValueError("Generated category does not match its Stage 2 group.")
        if question["difficulty"] not in DIFFICULTIES:
            raise ValueError(f"Invalid difficulty: {question['difficulty']!r}")
        for field in (
            "patient_profile",
            "scenario",
            "question",
            "explanation",
        ):
            if not isinstance(question[field], str) or not question[field].strip():
                raise ValueError(f"{field} must contain non-empty text.")

        options = _normalize_options(question["options"])
        correct_answer = str(question["correct_answer"]).lower()
        if correct_answer not in OPTION_LABELS:
            raise ValueError(
                f"correct_answer must be one of {OPTION_LABELS}; "
                f"found {question['correct_answer']!r}."
            )
        normalized = dict(question)
        normalized["options"] = options
        normalized["correct_answer"] = correct_answer
        normalized_questions.append(normalized)
    return normalized_questions


def build_final_qa(
    grouped_df: pd.DataFrame,
    response_by_key: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in grouped_df.iterrows():
        key = group["key"]
        if key not in response_by_key:
            continue
        questions = validate_stage3_response(
            response_by_key[key],
            expected_population=group["target_population"],
            expected_category=group["category"],
        )
        for question in questions:
            rows.append(
                {
                    "guideline_page_key": group["guideline_page_key"],
                    "guideline": group["guideline"],
                    "page": group["page"],
                    "guidelinetext_processed": group[
                        "guidelinetext_processed"
                    ],
                    "condition": [
                        rule["condition_base"] for rule in group["rules"]
                    ],
                    "action": [rule["action"] for rule in group["rules"]],
                    "triggers": [rule["triggers"] for rule in group["rules"]],
                    "source_fragment": [
                        rule["source_fragment"] for rule in group["rules"]
                    ],
                    "notes": [rule["notes"] for rule in group["rules"]],
                    "target_population": group["target_population"],
                    "category": group["category"],
                    "reason": group["reason"],
                    "gemini_stage3_response": question,
                    "question_type": question["question_type"],
                    "patient_profile": question["patient_profile"],
                    "scenario": question["scenario"],
                    "question": question["question"],
                    "options": question["options"],
                    "correct_answer": question["correct_answer"],
                    "explanation": question["explanation"],
                    "difficulty": question["difficulty"],
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values(
            ["guideline_page_key"],
            kind="stable",
            inplace=True,
        )
        result.reset_index(drop=True, inplace=True)
    return result


def rebalance_correct_answers(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Redistribute correct answer positions as evenly as mathematically possible."""
    required = {"options", "correct_answer"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"QA data are missing columns: {missing}")

    total = len(df)
    base, remainder = divmod(total, len(OPTION_LABELS))
    targets = {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(OPTION_LABELS)
    }
    used: Counter[str] = Counter()
    rng = random.Random(seed)

    new_options_list: list[dict[str, str]] = []
    new_correct_list: list[str] = []
    old_options_list: list[dict[str, str]] = []
    old_correct_list: list[str] = []

    for row_index, row in df.iterrows():
        options = _normalize_options(row["options"])
        old_correct = str(row["correct_answer"]).lower()
        if old_correct not in OPTION_LABELS:
            raise ValueError(
                f"Row {row_index} has invalid correct_answer: {old_correct!r}"
            )

        candidates = [
            label for label in OPTION_LABELS if used[label] < targets[label]
        ]
        if not candidates:
            raise RuntimeError("Answer-position target allocation was exhausted.")
        new_correct = rng.choice(candidates)
        used[new_correct] += 1

        correct_text = options[old_correct]
        distractors = [
            option for label, option in options.items() if label != old_correct
        ]
        rng.shuffle(distractors)
        distractor_iter = iter(distractors)
        new_options = {
            label: (
                correct_text if label == new_correct else next(distractor_iter)
            )
            for label in OPTION_LABELS
        }

        old_options_list.append(copy.deepcopy(options))
        old_correct_list.append(old_correct)
        new_options_list.append(new_options)
        new_correct_list.append(new_correct)

    if dict(Counter(new_correct_list)) != {
        label: count for label, count in targets.items() if count
    }:
        raise RuntimeError("Rebalanced answer counts do not match their targets.")

    result = df.copy()
    result["options_before_rebalance"] = old_options_list
    result["correct_answer_before_rebalance"] = old_correct_list
    result["options"] = new_options_list
    result["correct_answer"] = new_correct_list
    return result
