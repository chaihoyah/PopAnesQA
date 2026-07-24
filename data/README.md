# Data layout

Only small, redistributable metadata and anonymized expert ratings are tracked
in this repository.

```text
data/
├── Anesthesia_keywords.xlsx       # Search terms used for PubMed collection
├── guideline_pmids.txt            # PMIDs excluded from retrieval corpora
├── human_evaluation/              # Anonymized per-annotator ratings
├── raw/                           # Local/Hugging Face release files; Git-ignored
└── external_knowledge/            # Retrieval corpora and caches; Git-ignored
```

The PopAnesQA benchmark is released separately through Hugging Face as a
Parquet `test` split. Guideline PDFs, PubMed abstracts, MedBullets, other
third-party benchmark files, model weights, and experiment outputs must be
obtained by users under their respective terms.

All scripts receive data, cache, and output locations through command-line
arguments. The directory names above are organizational conventions, not
hard-coded runtime requirements.

