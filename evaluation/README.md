# PopAnesQA evaluation

This directory contains the local-model evaluation used for Baseline,
Base-RAG, Prof-RAG, retrieval ablations, and profile ablations. Model, data,
corpus, cache, and output paths are always supplied by command-line arguments
or shell environment variables.

GPT and Gemini adapters are intentionally not included. They use the same
prompts, answer parser, and metrics; only the provider-specific API call and
rate-limit handling differ.

The released Hugging Face benchmark is a Parquet `test` split, while the
current local inference runner accepts pandas Pickle input. Convert the
downloaded file before evaluation:

```python
import pandas as pd

pd.read_parquet("<downloaded_test_parquet>").to_pickle("<local_popanesqa.pkl>")
```

## PubMed corpus chunking

```bash
python -m evaluation.process_retrieval_corpus \
  --input pediatric_pubmed=<pediatric_pubmed.pkl> \
  --input general_pubmed=<general_pubmed.pkl> \
  --tokenizer <medcpt_article_encoder> \
  --output-dir <corpus_output_dir>
```

Each `--input` uses the form `SOURCE=PATH` and may be repeated. Pickle, CSV,
Excel, and Parquet inputs are supported; the text column is auto-detected from
`text`, `article_abstract`, `translated_text`, or
`guidelinetext_processed`. Use `--text-column` for another schema.

The default configuration uses 384-token chunks with 64-token overlap. It
writes a chunked file for every source, a document-level deduplicated
`total_pubmed_chunked.pkl`, and a provenance manifest. Existing outputs require
an explicit `--overwrite`. Paths and model IDs are never embedded in the
script.

## Main evaluation

```bash
python -m evaluation.inference_evaluation \
  --model-paths <generator_model> \
  --data-paths <benchmark.pkl> \
  --rag-spec "pubmed::<corpus.pkl>::<cache_dir>" \
  --output-dir <output_dir> \
  --methods baseline base_rag prof_rag \
  --backends medcpt \
  --medcpt-query-encoder <query_encoder> \
  --medcpt-article-encoder <article_encoder> \
  --medcpt-cross-encoder <cross_encoder>
```

Use `--backends bm25`, or both `medcpt bm25`, for lexical comparisons. Add
`gold_rag --gold-guideline <guideline.pkl>` to the method list only for the
explicit oracle/ground-truth-context experiment.

The benchmark reader supports the canonical `options` and `correct_answer`
columns and the legacy `new_options`, `new_correct_answer`, and `final_answer`
columns. Prompts are constructed at runtime, so a precomputed `final_prompt`
column is not required.

The retrieval query follows the original experiment: for scenario questions it
concatenates `patient_profile`, `scenario`, and `question`; answer options are
not included.

## Ranking ablation

```bash
python -m evaluation.ablation_inference ranking \
  --model-path <generator_model> \
  --data-path <benchmark.pkl> \
  --extractions <profile_extractions.pkl> \
  --rag-spec "pubmed::<corpus.pkl>::<cache_dir>" \
  --output-dir <output_dir> \
  --backend medcpt \
  --medcpt-query-encoder <query_encoder> \
  --medcpt-article-encoder <article_encoder> \
  --medcpt-cross-encoder <cross_encoder>
```

Ranking conditions:

- `q_q`: Q retrieval followed by Q reranking.
- `q_qp`: Q retrieval followed by Q+P reranking.
- `prof_rag`: Q retrieval, Q+P reranking, then P reranking.

Every RAG and ablation result stores the complete retrieval path in
`retrieval_trajectory`. The nested `initial`, `middle`, and `final` stages
contain the query type, query text, requested k, and ordered documents with
their corpus index, score, text, and corpus metadata. The legacy-compatible
`candidate_texts`, `middle_texts`, and `retrieved_texts` columns contain only
the stage texts.

## Retrieval trajectory analysis

```bash
python -m evaluation.retrieval_trajectory_analysis \
  --result <evaluation_result.pkl> \
  --output-dir <analysis_output_dir>
```

The command creates a document-level table, per-question source counts, and a
population-level source summary for the initial, middle, and final stages.
It warns when any retrieved documents have unknown source metadata or when no
multi-source (`both`) documents are found. The latter can be legitimate for a
single-source corpus, but for the combined corpus it usually means the corpus
predates the `source_membership` column and should be rebuilt.
New results already carry corpus source metadata. To analyze older files that
only contain `top_64`/`top_8`/`top_1` or the text-only compatibility columns,
provide the original corpora:

```bash
python -m evaluation.retrieval_trajectory_analysis \
  --result <legacy_result.pkl> \
  --corpus pediatric_pubmed=<pediatric_corpus.pkl> \
  --corpus general_pubmed=<general_corpus.pkl> \
  --output-dir <analysis_output_dir>
```

## Profile ablation

```bash
python -m evaluation.ablation_inference profile \
  --model-path <generator_model> \
  --data-path <benchmark_adult_or_pediatric.pkl> \
  --extractions <profile_extractions.pkl> \
  --rag-spec "pubmed::<corpus.pkl>::<cache_dir>" \
  --output-dir <output_dir> \
  --backend medcpt \
  --medcpt-query-encoder <query_encoder> \
  --medcpt-article-encoder <article_encoder> \
  --medcpt-cross-encoder <cross_encoder>
```

Profile conditions:

- `original`: the extracted profile.
- `random`: a different profile sampled from all eligible rows.
- `within_group`: a different profile from the same population.
- `cross_group`: a profile from the opposite Adult/Pediatric population.

To reproduce the original experiment, General questions follow the Adult
branch during profile replacement: `within_group` samples from the Adult pool
and `cross_group` samples from the Pediatric pool. The candidate pools
themselves contain profiles from Adult and Pediatric rows only. Candidate
profile pools are shuffled with the experiment seed and limited to 120
profiles per population by default; change this with `--profile-pool-limit`.

## Reproducibility behavior

- No GPU IDs, offline mode, data paths, or model paths are hard-coded.
- `CUDA_VISIBLE_DEVICES` and Hugging Face offline/cache settings are controlled
  by the caller.
- MedCPT and BM25 caches contain provenance manifests and reject a different
  corpus, metadata schema, encoder, or BM25 configuration. Caches created
  before metadata schema version 2 must be rebuilt so retrieval trajectories
  retain `source`, `doc_id`, `chunk_id`, and `source_membership`.
- Existing result files are never silently overwritten. `--skip-existing`
  loads them and recomputes the complete metrics table.
- Input prompts are reduced to fit
  `max_model_length - answer_max_tokens`; the default answer allowance is
  3000 tokens, matching the original experiment.
- Invalid model answers are counted in the `unparsed` metric.
