"""Extract PubMed abstracts for the retrieval corpus.

This script performs one of two searches:

- General: each anesthesia keyword retrieves at most 1,500 PMIDs.
- Pediatric: each anesthesia keyword is combined separately with
  ``neonate``, ``infant``, ``children``, and ``pediatric``; each combination
  retrieves at most 500 PMIDs.

All paths are supplied by the user. The script stores canonical PMIDs and
query manifests so that deduplication and corpus counts remain auditable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import pandas as pd


DEFAULT_PEDIATRIC_MODIFIERS = ("neonate", "infant", "children", "pediatric")
SUPPORTED_OUTPUT_SUFFIXES = {".csv", ".xlsx", ".pkl", ".parquet"}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve the General or Pediatric PubMed corpus."
    )
    parser.add_argument(
        "--population-type",
        choices=("General", "Pediatric"),
        required=True,
    )
    parser.add_argument(
        "--keyword-file",
        type=Path,
        required=True,
        help="Excel file whose non-empty cells contain anesthesia keywords.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output path ending in .csv, .xlsx, .pkl, or .parquet.",
    )
    parser.add_argument(
        "--guideline-pmids-file",
        type=Path,
        required=True,
        help="Text file containing one gold-guideline PMID per line.",
    )
    parser.add_argument(
        "--skip-keyword-rows",
        type=int,
        default=2,
        help="Number of data rows to discard after reading the Excel header.",
    )
    parser.add_argument("--start-date", default="2015/01/01")
    parser.add_argument("--end-date", default="2024/12/31")
    parser.add_argument(
        "--general-limit-per-keyword",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--pediatric-limit-per-modifier",
        type=int,
        default=500,
        help="Maximum results for each keyword-modifier combination.",
    )
    parser.add_argument(
        "--pediatric-modifiers",
        nargs="+",
        default=list(DEFAULT_PEDIATRIC_MODIFIERS),
    )
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Initial retry delay; subsequent delays grow exponentially.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.34,
        help="Delay after each successful PubMed request.",
    )
    parser.add_argument(
        "--expected-unique-pmids",
        type=int,
        default=0,
        help=(
            "Optional expected unique PMID count before guideline exclusion; "
            "0 disables the check."
        ),
    )
    parser.add_argument(
        "--expected-output-pmids",
        type=int,
        default=0,
        help=(
            "Optional expected PMID count after guideline exclusion; "
            "0 disables the check."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Save partial output instead of failing when requests remain unresolved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output and its manifest files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.keyword_file.is_file():
        raise FileNotFoundError(f"Keyword file not found: {args.keyword_file}")
    if not args.guideline_pmids_file.is_file():
        raise FileNotFoundError(
            f"Guideline PMID file not found: {args.guideline_pmids_file}"
        )
    if args.output_file.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
        raise ValueError(
            "--output-file must end in one of: "
            + ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        )
    if args.skip_keyword_rows < 0:
        raise ValueError("--skip-keyword-rows cannot be negative.")
    for name in (
        "general_limit_per_keyword",
        "pediatric_limit_per_modifier",
        "page_size",
        "max_retries",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.retry_backoff_seconds < 0 or args.request_delay_seconds < 0:
        raise ValueError("Retry and request delays cannot be negative.")

    cleaned_modifiers = [
        str(modifier).strip().lower()
        for modifier in args.pediatric_modifiers
        if str(modifier).strip()
    ]
    args.pediatric_modifiers = list(dict.fromkeys(cleaned_modifiers))
    if args.population_type == "Pediatric" and not args.pediatric_modifiers:
        raise ValueError("At least one pediatric modifier is required.")


def load_keywords(path: Path, skip_data_rows: int) -> list[str]:
    keyword_df = pd.read_excel(path)
    if skip_data_rows:
        keyword_df = keyword_df.iloc[skip_data_rows:]

    usable_columns = [
        column
        for column in keyword_df.columns
        if not str(column).strip().lower().startswith("unnamed:")
    ]
    if not usable_columns:
        raise ValueError("The keyword file contains no usable columns.")

    keywords: list[str] = []
    for column in usable_columns:
        for value in keyword_df[column].dropna():
            keyword = str(value).strip()
            if keyword:
                keywords.append(keyword)

    unique_keywords = list(dict.fromkeys(keywords))
    if not unique_keywords:
        raise ValueError("No keywords were found in the keyword file.")

    print(
        f"Loaded {len(unique_keywords):,} unique keywords "
        f"from {len(usable_columns):,} columns."
    )
    return unique_keywords


def load_guideline_pmids(path: Path) -> set[str]:
    guideline_pmids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            pmid = raw_line.split("#", maxsplit=1)[0].strip()
            if pmid:
                guideline_pmids.add(pmid)

    if not guideline_pmids:
        raise ValueError(f"No guideline PMIDs were found in: {path}")
    print(f"Loaded {len(guideline_pmids):,} gold-guideline PMIDs.")
    return guideline_pmids


def retry_call(
    operation: Callable[[], T],
    *,
    description: str,
    max_retries: int,
    initial_backoff_seconds: float,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:  # The API may raise several transport exceptions.
            last_error = exc
            if attempt == max_retries:
                break
            delay = initial_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"{description} failed on attempt {attempt}/{max_retries}: "
                f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s."
            )
            time.sleep(delay)

    assert last_error is not None
    raise RuntimeError(
        f"{description} failed after {max_retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def progress(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def fetch_query_pmids(
    fetcher: Any,
    *,
    query: str,
    limit: int,
    start_date: str,
    end_date: str,
    page_size: int,
    max_retries: int,
    retry_backoff_seconds: float,
    request_delay_seconds: float,
) -> list[str]:
    collected: list[str] = []
    retstart = 0

    while len(collected) < limit:
        request_size = min(page_size, limit - len(collected))

        def request() -> list[str]:
            return fetcher.pmids_for_query(
                query,
                since=start_date,
                until=end_date,
                retstart=retstart,
                retmax=request_size,
            )

        batch = retry_call(
            request,
            description=f"PMID search for {query!r} at retstart={retstart}",
            max_retries=max_retries,
            initial_backoff_seconds=retry_backoff_seconds,
        )
        batch = [str(pmid).strip() for pmid in batch if str(pmid).strip()]
        if not batch:
            break

        collected.extend(batch)
        retstart += len(batch)
        time.sleep(request_delay_seconds)

        if len(batch) < request_size:
            break

    # Preserve PubMed result order while removing duplicates within the query.
    return list(dict.fromkeys(collected))


def build_queries(
    population_type: str,
    keywords: list[str],
    pediatric_modifiers: list[str],
    general_limit: int,
    pediatric_limit: int,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for keyword in keywords:
        if population_type == "General":
            queries.append(
                {
                    "base_keyword": keyword,
                    "modifier": None,
                    "query": keyword,
                    "limit": general_limit,
                }
            )
        else:
            for modifier in pediatric_modifiers:
                queries.append(
                    {
                        "base_keyword": keyword,
                        "modifier": modifier,
                        "query": f"{modifier} {keyword}",
                        "limit": pediatric_limit,
                    }
                )
    return queries


def search_all_pmids(
    fetcher: Any,
    queries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    pmid_metadata: dict[str, dict[str, Any]] = {}
    query_records: list[dict[str, Any]] = []

    for query_spec in progress(queries, desc="Searching PubMed queries"):
        try:
            query_pmids = fetch_query_pmids(
                fetcher,
                query=query_spec["query"],
                limit=query_spec["limit"],
                start_date=args.start_date,
                end_date=args.end_date,
                page_size=args.page_size,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                request_delay_seconds=args.request_delay_seconds,
            )
            status = "complete"
            error = ""
        except RuntimeError as exc:
            query_pmids = []
            status = "failed"
            error = str(exc)

        for pmid in query_pmids:
            if pmid not in pmid_metadata:
                pmid_metadata[pmid] = {
                    "matched_queries": [],
                    "matched_keywords": [],
                    "matched_modifiers": [],
                }
            metadata = pmid_metadata[pmid]
            if query_spec["query"] not in metadata["matched_queries"]:
                metadata["matched_queries"].append(query_spec["query"])
            if query_spec["base_keyword"] not in metadata["matched_keywords"]:
                metadata["matched_keywords"].append(query_spec["base_keyword"])
            modifier = query_spec["modifier"]
            if modifier and modifier not in metadata["matched_modifiers"]:
                metadata["matched_modifiers"].append(modifier)

        query_records.append(
            {
                **query_spec,
                "population_type": args.population_type,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "retrieved_pmids": len(query_pmids),
                "status": status,
                "error": error,
            }
        )

    return pmid_metadata, pd.DataFrame(query_records)


def fetch_articles(
    fetcher: Any,
    pmid_metadata: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    articles: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for pmid, metadata in progress(
        pmid_metadata.items(),
        total=len(pmid_metadata),
        desc="Fetching PubMed articles",
    ):
        try:
            article = retry_call(
                lambda pmid=pmid: fetcher.article_by_pmid(pmid),
                description=f"Article retrieval for PMID {pmid}",
                max_retries=args.max_retries,
                initial_backoff_seconds=args.retry_backoff_seconds,
            )
            articles.append(
                {
                    "pmid": pmid,
                    "article_title": article.title,
                    "article_abstract": article.abstract,
                    "year": article.year,
                    "article_url": article.url,
                    "population_query": args.population_type,
                    "matched_queries": json.dumps(
                        metadata["matched_queries"],
                        ensure_ascii=False,
                    ),
                    "matched_keywords": json.dumps(
                        metadata["matched_keywords"],
                        ensure_ascii=False,
                    ),
                    "matched_modifiers": json.dumps(
                        metadata["matched_modifiers"],
                        ensure_ascii=False,
                    ),
                }
            )
            time.sleep(args.request_delay_seconds)
        except RuntimeError as exc:
            failures.append({"pmid": pmid, "error": str(exc)})

    return pd.DataFrame(
        articles,
        columns=[
            "pmid",
            "article_title",
            "article_abstract",
            "year",
            "article_url",
            "population_query",
            "matched_queries",
            "matched_keywords",
            "matched_modifiers",
        ],
    ), pd.DataFrame(
        failures,
        columns=["pmid", "error"],
    )


def output_sidecars(output_file: Path) -> dict[str, Path]:
    return {
        "queries": output_file.with_name(f"{output_file.stem}_queries.csv"),
        "pmids": output_file.with_name(f"{output_file.stem}_pmids.csv"),
        "excluded_guidelines": output_file.with_name(
            f"{output_file.stem}_excluded_guideline_pmids.csv"
        ),
        "failures": output_file.with_name(f"{output_file.stem}_failures.csv"),
        "config": output_file.with_name(f"{output_file.stem}_config.json"),
    }


def ensure_outputs_available(
    output_file: Path,
    sidecars: dict[str, Path],
    overwrite: bool,
) -> None:
    paths = [output_file, *sidecars.values()]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists. Use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )


def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(temporary_path, index=False)
    elif suffix == ".xlsx":
        df.to_excel(temporary_path, index=False)
    elif suffix == ".pkl":
        df.to_pickle(temporary_path)
    elif suffix == ".parquet":
        df.to_parquet(temporary_path, index=False)
    else:
        raise ValueError(f"Unsupported output suffix: {suffix}")
    temporary_path.replace(path)


def pmid_manifest(pmid_metadata: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pmid, metadata in pmid_metadata.items():
        rows.append(
            {
                "pmid": pmid,
                "matched_queries": json.dumps(
                    metadata["matched_queries"],
                    ensure_ascii=False,
                ),
                "matched_keywords": json.dumps(
                    metadata["matched_keywords"],
                    ensure_ascii=False,
                ),
                "matched_modifiers": json.dumps(
                    metadata["matched_modifiers"],
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "pmid",
            "matched_queries",
            "matched_keywords",
            "matched_modifiers",
        ],
    )


def exclude_guideline_pmids(
    pmid_metadata: dict[str, dict[str, Any]],
    guideline_pmids: set[str],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    retained: dict[str, dict[str, Any]] = {}
    excluded: dict[str, dict[str, Any]] = {}
    for pmid, metadata in pmid_metadata.items():
        if pmid in guideline_pmids:
            excluded[pmid] = metadata
        else:
            retained[pmid] = metadata

    excluded_df = pmid_manifest(excluded)
    excluded_df["exclusion_reason"] = "gold_guideline"
    return retained, excluded_df


def write_config(args: argparse.Namespace, path: Path) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    sidecars = output_sidecars(args.output_file)
    ensure_outputs_available(args.output_file, sidecars, args.overwrite)
    write_config(args, sidecars["config"])

    # Importing here keeps ``--help`` usable before optional API dependencies
    # are installed.
    try:
        from metapub import PubMedFetcher
    except ImportError as exc:
        raise ImportError(
            "PubMed extraction requires metapub. Install the project "
            "dependencies before running this script."
        ) from exc

    keywords = load_keywords(args.keyword_file, args.skip_keyword_rows)
    guideline_pmids = load_guideline_pmids(args.guideline_pmids_file)
    queries = build_queries(
        args.population_type,
        keywords,
        args.pediatric_modifiers,
        args.general_limit_per_keyword,
        args.pediatric_limit_per_modifier,
    )
    print(
        f"Prepared {len(queries):,} queries for {args.population_type}: "
        f"{len(keywords):,} base keywords."
    )

    fetcher = PubMedFetcher()
    searched_pmid_metadata, query_df = search_all_pmids(fetcher, queries, args)
    n_unique_before_exclusion = len(searched_pmid_metadata)
    pmid_metadata, excluded_guideline_df = exclude_guideline_pmids(
        searched_pmid_metadata,
        guideline_pmids,
    )
    pmid_df = pmid_manifest(pmid_metadata)
    atomic_write_dataframe(query_df, sidecars["queries"])
    atomic_write_dataframe(pmid_df, sidecars["pmids"])
    atomic_write_dataframe(
        excluded_guideline_df,
        sidecars["excluded_guidelines"],
    )

    failed_queries = query_df["status"].eq("failed").sum()
    print(
        "Unique PMIDs before guideline exclusion: "
        f"{n_unique_before_exclusion:,}"
    )
    print(f"Excluded gold-guideline PMIDs: {len(excluded_guideline_df):,}")
    print(f"Unique PMIDs after guideline exclusion: {len(pmid_metadata):,}")
    print(f"Failed queries: {failed_queries:,}/{len(query_df):,}")

    if (
        args.expected_unique_pmids
        and n_unique_before_exclusion != args.expected_unique_pmids
    ):
        raise ValueError(
            f"Expected {args.expected_unique_pmids:,} unique PMIDs, "
            f"found {n_unique_before_exclusion:,} before guideline exclusion. "
            "Query and PMID manifests were saved."
        )
    if (
        args.expected_output_pmids
        and len(pmid_metadata) != args.expected_output_pmids
    ):
        raise ValueError(
            f"Expected {args.expected_output_pmids:,} output PMIDs, "
            f"found {len(pmid_metadata):,} after guideline exclusion. "
            "Query and PMID manifests were saved."
        )
    if failed_queries and not args.allow_partial:
        raise RuntimeError(
            f"{failed_queries:,} PubMed queries failed. Manifests were saved; "
            "rerun after resolving the failures or pass --allow-partial."
        )

    article_df, failure_df = fetch_articles(fetcher, pmid_metadata, args)
    atomic_write_dataframe(article_df, args.output_file)
    atomic_write_dataframe(failure_df, sidecars["failures"])

    print(f"Successfully fetched articles: {len(article_df):,}")
    print(f"Failed article retrievals: {len(failure_df):,}")
    print(
        "Articles without abstracts: "
        f"{article_df['article_abstract'].isna().sum():,}"
    )
    print(f"Saved corpus to: {args.output_file}")

    if len(failure_df) and not args.allow_partial:
        raise RuntimeError(
            f"{len(failure_df):,} articles could not be retrieved. "
            f"Successful rows and failure details were saved under "
            f"{args.output_file.parent}."
        )


if __name__ == "__main__":
    main()
