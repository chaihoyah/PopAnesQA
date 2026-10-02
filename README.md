# PopAnesQA

> 🎉 **News:** Our paper has been published in **Proceedings of Machine Learning Research (PMLR), Volume 340**.
> The paper is available at [PMLR](https://proceedings.mlr.press/v340/shin26a.html), and the PopAnesQA
> dataset is available on [Hugging Face](https://huggingface.co/datasets/BMILab/PopAnesQA).

PopAnesQA is a population-aware anesthesiology question-answering benchmark
and evaluation pipeline. The project studies clinical questions for which the
appropriate management can differ across Pediatric, Adult, and General
populations.

This repository contains code for:

- analyzing existing medical QA benchmarks;
- collecting and processing PubMed retrieval corpora;
- parsing guidelines and constructing PopAnesQA with a three-stage pipeline;
- evaluating Baseline, Base-RAG, and profile-aware RAG methods;
- running ranking and patient-profile ablations;
- analyzing the initial, intermediate, and final retrieval trajectory; and
- reproducing expert-rating agreement statistics.

The PopAnesQA benchmark itself is distributed separately through its
[Hugging Face dataset repository](https://huggingface.co/datasets/BMILab/PopAnesQA)
in Parquet format. Guideline PDFs, PubMed abstracts, third-party benchmark
data, model weights, and generated inference outputs are not redistributed
here.

## Repository structure

```text
.
├── benchmark_construction/   # Guideline parsing and three-stage QA generation
├── evaluation/               # Inference, retrieval, ablations, and trajectory analysis
├── openbenchmark_analysis/   # Classification and analysis of existing QA benchmarks
├── data/
│   ├── human_evaluation/     # Anonymized PopAnesQA expert ratings
│   ├── Anesthesia_keywords.xlsx
│   └── guideline_pmids.txt
├── pubmed_extract.py         # PubMed corpus collection and guideline-PMID exclusion
└── human_evaluation_statistics.py
```

Detailed instructions are available in:

- [Benchmark construction](benchmark_construction/README.md)
- [Evaluation](evaluation/README.md)
- [Existing benchmark analysis](openbenchmark_analysis/README.md)
- [Local data layout](data/README.md)
- [Expert ratings](data/human_evaluation/README.md)

## Requirements

- Python 3.9+
- A CUDA-capable environment for vLLM and MedCPT evaluation
- API credentials only for the stages that call Upstage or Gemini

Common dependencies include:

```text
pandas
numpy
tqdm
requests
beautifulsoup4
openpyxl
pyarrow
torch
transformers
datasets
vllm
google-genai
```

For Python 3.9, use `google-genai==1.47.0`; newer releases may require a newer
Python version. Install only the optional dependencies required by the
pipeline you plan to run.

## Data

The [Hugging Face release](https://huggingface.co/datasets/BMILab/PopAnesQA)
contains the 623-item PopAnesQA `test` split. The public schema contains
population, category, question type, patient profile, clinical scenario,
question, four answer options, and the answer label. The current export also
retains `condition`, `action`, and `triggers` as construction metadata.

The evaluation runner currently accepts a local pandas Pickle benchmark. After
downloading the Parquet release, convert it without changing nested columns:

```python
import pandas as pd

df = pd.read_parquet("<downloaded_test_parquet>")
df.to_pickle("<local_popanesqa.pkl>")
```

All data and model paths are supplied at runtime. Do not place private API keys
or machine-specific paths in source files.

## Workflow overview

### 1. Analyze existing medical QA benchmarks

Run one classification pass per model, then aggregate the three result files.
The code loads supported Hugging Face datasets from the user’s cache and
expects the user to obtain MedBullets separately.

```bash
python -m openbenchmark_analysis.openbenchmark_process --help
python -m openbenchmark_analysis.openbenchmark_analysis --help
```

### 2. Build PubMed corpora

```bash
python pubmed_extract.py \
  --population-type Pediatric \
  --keyword-file data/Anesthesia_keywords.xlsx \
  --guideline-pmids-file data/guideline_pmids.txt \
  --output-file <pediatric_pubmed.pkl>
```

Use `--population-type General` for the General corpus. The extraction script
removes PMIDs used by the gold guidelines and saves query/provenance manifests.

### 3. Construct PopAnesQA

The construction pipeline parses guideline PDFs, classifies relevant pages,
extracts clinical rules, generates multiple-choice questions, validates every
stage, and deterministically rebalances answer positions.

See [benchmark_construction/README.md](benchmark_construction/README.md).

### 4. Evaluate Baseline and RAG methods

```bash
python -m evaluation.inference_evaluation \
  --model-paths <generator_model> \
  --data-paths <local_popanesqa.pkl> \
  --rag-spec "pubmed::<retrieval_corpus.pkl>::<cache_dir>" \
  --output-dir <output_dir> \
  --methods baseline base_rag prof_rag \
  --backends medcpt \
  --medcpt-query-encoder <query_encoder> \
  --medcpt-article-encoder <article_encoder> \
  --medcpt-cross-encoder <cross_encoder>
```

The default retrieval trajectory is `top-64 → top-8 → top-1`. Each result
stores ordered text, corpus indices, scores, and metadata for the initial,
intermediate, and final stages.

See [evaluation/README.md](evaluation/README.md) for BM25 evaluation,
ablations, cache provenance rules, and trajectory analysis.

### 5. Reproduce expert-rating statistics

```bash
python human_evaluation_statistics.py \
  --annotator1 data/human_evaluation/popanesqa_annotator1.csv \
  --annotator2 data/human_evaluation/popanesqa_annotator2.csv \
  --output-file <human_evaluation_summary.csv>
```

The script matches the two files by their anonymized item key before computing
descriptive statistics and Cohen's kappa.

## Reproducibility and data policy

- Randomized steps expose a seed; the default experiment seed is 42.
- Input data, model directories, caches, GPU selection, and outputs are
  caller-controlled.
- Existing outputs are not silently overwritten.
- Retrieval caches validate corpus, encoder, configuration, and metadata
  schema provenance.
- PopAnesQA expert-rating files use anonymized item keys and contain no
  question text or annotator identity.
- `data/raw/` and `data/external_knowledge/` are ignored by Git.

## Intended use

PopAnesQA is intended for research evaluation. Its clinical vignettes and
model outputs must not be used as a substitute for professional medical
judgment or local clinical guidance.

## Citation

If you use PopAnesQA or this codebase, please cite:

```bibtex
@inproceedings{shin2026characterizing,
  title={Characterizing Population Gaps in Clinical Decision-Making: A Guideline-Based Benchmark and Population-Aware Retrieval Analysis in Anesthesiology},
  author={Shin, Chaiho and Park, Jung-Bin and Kim, Kwangsoo and Kim, Hee-Soo},
  booktitle={Machine Learning for Healthcare Conference},
  pages={1791--1833},
  year={2026},
  organization={PMLR}
}
```
