#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${DATA_PATH:?Set DATA_PATH}"
: "${CORPUS_PATH:?Set CORPUS_PATH}"
: "${CACHE_DIR:?Set CACHE_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${MEDCPT_QUERY_ENCODER:?Set MEDCPT_QUERY_ENCODER}"
: "${MEDCPT_ARTICLE_ENCODER:?Set MEDCPT_ARTICLE_ENCODER}"
: "${MEDCPT_CROSS_ENCODER:?Set MEDCPT_CROSS_ENCODER}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RAG_NAME="${RAG_NAME:-corpus}"
METHODS=(baseline base_rag prof_rag)
GOLD_ARGS=()
if [[ -n "${GOLD_GUIDELINE_PATH:-}" ]]; then
  METHODS+=(gold_rag)
  GOLD_ARGS+=(--gold-guideline "${GOLD_GUIDELINE_PATH}")
  GOLD_ARGS+=(--gold-text-column "${GOLD_TEXT_COLUMN:-summary_text}")
fi

python -m evaluation.inference_evaluation \
  --model-paths "${MODEL_PATH}" \
  --data-paths "${DATA_PATH}" \
  --rag-spec "${RAG_NAME}::${CORPUS_PATH}::${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --methods "${METHODS[@]}" \
  --backends medcpt \
  --medcpt-query-encoder "${MEDCPT_QUERY_ENCODER}" \
  --medcpt-article-encoder "${MEDCPT_ARTICLE_ENCODER}" \
  --medcpt-cross-encoder "${MEDCPT_CROSS_ENCODER}" \
  --first-k 64 \
  --middle-k 8 \
  --final-k 1 \
  --skip-existing \
  "${GOLD_ARGS[@]}"
