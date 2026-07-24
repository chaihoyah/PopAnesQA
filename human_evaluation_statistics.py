"""Compute PopAnesQA expert-rating statistics and inter-rater agreement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


ORDINAL_METRICS = (
    "Guideline Consistency",
    "Medical Validity",
    "One-Best-Answer Clarity",
    "Distractor Plausibility",
)
TARGET_METRIC = "Target Population Appropriateness"
TARGET_VALUES = {"Correct": 1, "Incorrect": 0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator1", type=Path, required=True)
    parser.add_argument("--annotator2", type=Path, required=True)
    parser.add_argument(
        "--key-column",
        default="data_id",
        help="Unique item key shared by both annotator files.",
    )
    parser.add_argument(
        "--expected-items",
        type=int,
        default=100,
        help="Expected matched item count; set to 0 to disable the check.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional summary output ending in .csv or .json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.annotator1, args.annotator2):
        if not path.is_file():
            raise FileNotFoundError(f"Annotator file not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Annotator input must be CSV: {path}")
    if args.expected_items < 0:
        raise ValueError("--expected-items cannot be negative.")
    if args.output_file is not None:
        if args.output_file.suffix.lower() not in {".csv", ".json"}:
            raise ValueError("--output-file must end in .csv or .json.")
        if args.output_file.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists; use --overwrite to replace it: {args.output_file}"
            )


def load_ratings(path: Path, key_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={key_column: str})
    required = {key_column, *ORDINAL_METRICS, TARGET_METRIC}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    result = frame[[key_column, *ORDINAL_METRICS, TARGET_METRIC]].copy()
    result[key_column] = result[key_column].astype(str).str.strip()
    if result[key_column].eq("").any():
        raise ValueError(f"{path} contains empty item keys.")
    if result[key_column].duplicated().any():
        duplicated = result.loc[
            result[key_column].duplicated(keep=False),
            key_column,
        ].head().tolist()
        raise ValueError(f"{path} contains duplicate item keys: {duplicated}")

    for metric in ORDINAL_METRICS:
        try:
            numeric = pd.to_numeric(result[metric], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {metric} must contain integer ratings.") from exc
        if not numeric.between(1, 5).all() or not (numeric % 1).eq(0).all():
            invalid = sorted(set(result.loc[~numeric.isin(range(1, 6)), metric]))
            raise ValueError(
                f"{path}: {metric} must use integer ratings 1-5; "
                f"found {invalid}."
            )
        result[metric] = numeric.astype(int)

    target = result[TARGET_METRIC].astype(str).str.strip()
    invalid_target = sorted(set(target) - set(TARGET_VALUES))
    if invalid_target:
        raise ValueError(
            f"{path}: {TARGET_METRIC} must be Correct or Incorrect; "
            f"found {invalid_target}."
        )
    result[TARGET_METRIC] = target
    return result


def align_annotators(
    annotator1: pd.DataFrame,
    annotator2: pd.DataFrame,
    key_column: str,
) -> pd.DataFrame:
    keys1 = set(annotator1[key_column])
    keys2 = set(annotator2[key_column])
    if keys1 != keys2:
        raise ValueError(
            "Annotator item keys differ: "
            f"{len(keys1 - keys2)} only in annotator1 and "
            f"{len(keys2 - keys1)} only in annotator2."
        )
    return annotator1.merge(
        annotator2,
        on=key_column,
        how="inner",
        validate="one_to_one",
        suffixes=("_annotator1", "_annotator2"),
        sort=False,
    )


def cohen_kappa(
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    weighting: Optional[str] = None,
) -> float:
    """Compute unweighted or quadratic-weighted Cohen's kappa."""
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape:
        raise ValueError("Kappa inputs must have the same shape.")
    if left_values.ndim != 1 or len(left_values) == 0:
        raise ValueError("Kappa inputs must be non-empty one-dimensional arrays.")

    labels = sorted(set(left_values.tolist()) | set(right_values.tolist()))
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for left_value, right_value in zip(left_values, right_values):
        confusion[
            label_to_index[left_value],
            label_to_index[right_value],
        ] += 1

    observed = confusion / confusion.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    row_indices, column_indices = np.indices(confusion.shape)
    if weighting is None:
        weights = (row_indices != column_indices).astype(np.float64)
    elif weighting == "quadratic":
        denominator = max(len(labels) - 1, 1)
        weights = ((row_indices - column_indices) / denominator) ** 2
    else:
        raise ValueError(f"Unsupported kappa weighting: {weighting}")

    observed_disagreement = float((weights * observed).sum())
    expected_disagreement = float((weights * expected).sum())
    if math.isclose(expected_disagreement, 0.0):
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def summarize(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in ORDINAL_METRICS:
        left = aligned[f"{metric}_annotator1"].to_numpy()
        right = aligned[f"{metric}_annotator2"].to_numpy()
        pooled = np.concatenate([left, right]).astype(np.float64)
        rows.append(
            {
                "metric": metric,
                "scale": "ordinal_1_5",
                "n_items": len(aligned),
                "n_ratings": len(pooled),
                "mean": float(pooled.mean()),
                "std": float(pooled.std(ddof=1)),
                "accuracy": float("nan"),
                "exact_agreement": float(np.mean(left == right)),
                "kappa": cohen_kappa(left, right, weighting="quadratic"),
                "kappa_weighting": "quadratic",
                "within_one": float(np.mean(np.abs(left - right) <= 1)),
            }
        )

    left_target = aligned[f"{TARGET_METRIC}_annotator1"].map(TARGET_VALUES).to_numpy()
    right_target = aligned[f"{TARGET_METRIC}_annotator2"].map(TARGET_VALUES).to_numpy()
    pooled_target = np.concatenate([left_target, right_target])
    rows.append(
        {
            "metric": TARGET_METRIC,
            "scale": "binary",
            "n_items": len(aligned),
            "n_ratings": len(pooled_target),
            "mean": float("nan"),
            "std": float("nan"),
            "accuracy": float(pooled_target.mean()),
            "exact_agreement": float(np.mean(left_target == right_target)),
            "kappa": cohen_kappa(left_target, right_target),
            "kappa_weighting": "unweighted",
            "within_one": float("nan"),
        }
    )
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        if row["scale"] == "ordinal_1_5":
            print(
                f"{row['metric']}: mean={row['mean']:.4f}; SD={row['std']:.4f}; "
                f"quadratic kappa={row['kappa']:.4f}; "
                f"exact agreement={row['exact_agreement']:.4f}; "
                f"within one={row['within_one']:.4f}"
            )
        else:
            print(
                f"{row['metric']}: accuracy={row['accuracy']:.4f}; "
                f"kappa={row['kappa']:.4f}; "
                f"exact agreement={row['exact_agreement']:.4f}"
            )


def atomic_write(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if path.suffix.lower() == ".csv":
        summary.to_csv(temporary, index=False)
    else:
        records = [
            {
                key: None if pd.isna(value) else value
                for key, value in record.items()
            }
            for record in summary.to_dict(orient="records")
        ]
        temporary.write_text(
            json.dumps(
                records,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    annotator1 = load_ratings(args.annotator1, args.key_column)
    annotator2 = load_ratings(args.annotator2, args.key_column)
    aligned = align_annotators(annotator1, annotator2, args.key_column)
    if args.expected_items and len(aligned) != args.expected_items:
        raise ValueError(
            f"Expected {args.expected_items} matched items, found {len(aligned)}."
        )

    print(f"Matched items: {len(aligned):,}")
    summary = summarize(aligned)
    print_summary(summary)
    if args.output_file is not None:
        atomic_write(summary, args.output_file)
        print(f"Saved summary: {args.output_file}")


if __name__ == "__main__":
    main()
