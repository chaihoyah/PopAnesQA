#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${DATA_PATH:?Set DATA_PATH}"
: "${CORPUS_PATH:?Set CORPUS_PATH}"
: "${CACHE_DIR:?Set CACHE_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RAG_NAME="${RAG_NAME:-corpus}"
FINAL_TOP_K="${FINAL_TOP_K:-1}"

python -m evaluation.inference_evaluation \
  --model-paths "${MODEL_PATH}" \
  --data-paths "${DATA_PATH}" \
  --rag-spec "${RAG_NAME}::${CORPUS_PATH}::${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --methods baseline base_rag prof_rag \
  --backends bm25 \
  --first-k 64 \
  --middle-k 8 \
  --final-k "${FINAL_TOP_K}" \
  --skip-existing
