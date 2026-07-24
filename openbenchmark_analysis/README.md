# Existing benchmark analysis

This directory reproduces the classification and composition analysis of
existing medical QA benchmarks used for comparison with PopAnesQA.

The scripts do not redistribute benchmark records or MedBullets. Users must
obtain each dataset under its original terms and provide local cache/model
paths.

## Classification

Run `openbenchmark_process.py` once for each of the three classifier models:

```bash
python -m openbenchmark_analysis.openbenchmark_process \
  --dataset-cache-dir <dataset_cache> \
  --medbullets-csv <medbullets.csv> \
  --model-name-or-path <classifier_model> \
  --model-cache-dir <model_cache> \
  --output-dir <classification_output>
```

Supported Hugging Face sources are defined in the script. The two MMLU
subsets are retained as separate sources during processing and combined as one
`MMLU` benchmark in the final analysis.

## Three-model aggregation

```bash
python -m openbenchmark_analysis.openbenchmark_analysis \
  --gemma-results <gemma.pkl> \
  --medgemma-results <medgemma.pkl> \
  --llama-results <llama.pkl> \
  --output-dir <analysis_output>
```

The aggregation validates unique IDs and source consistency, removes
classification disagreements, applies majority-vote filtering, and writes
composition tables. Add `--validate-paper-counts` to require the headline
counts reported in the paper.

Human-evaluation records for these third-party benchmarks are intentionally
not included.

