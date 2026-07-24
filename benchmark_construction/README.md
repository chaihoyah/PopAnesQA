# PopAnesQA benchmark construction

`make_qa.py` implements the three guideline-to-QA stages as a resumable
command-line pipeline. Input data, generated data, model names, and API keys
are not hard-coded.

The stages are:

1. Stage 1: classify guideline pages by relevance, population, and category.
2. Stage 2: extract actionable clinical rules from relevant pages.
3. Stage 3: group rules by guideline page and generate multiple-choice QA.
4. Rebalance: redistribute correct answers across `a`, `b`, `c`, and `d`.

Each model stage is split into `prepare`, `submit`, `collect`, and `merge`.
This keeps local data processing separate from asynchronous Gemini Batch API
jobs and permits safe resumption.

Guideline PDFs are not distributed with this repository. Users must obtain
them under their original terms and provide a local input directory.

## Requirements

- Python 3.9+
- `pandas`
- `beautifulsoup4`
- `google-genai` for `submit` and `collect`
- An Excel or Parquet engine only when using those formats

The current pipeline code is compatible with Python 3.9. When reproducing it
on Python 3.9, install `google-genai==1.47.0`; current releases require Python
3.10 or later.

Pickle is recommended for intermediate files because Stage 2 and Stage 3
contain nested lists and dictionaries. CSV is also supported; nested values
written by this pipeline are recovered when they are read back.

## Parse guideline PDFs

Set the Upstage API key outside the code and parse the local PDF directory:

```powershell
$env:UPSTAGE_API_KEY = "..."
python -m benchmark_construction.guideline_pdf_parsing `
  --input-dir <guideline_pdf_dir> `
  --output-file <parsed_guidelines.pkl> `
  --raw-response-dir <raw_response_dir>
```

The public API endpoint is defined in the script; only the credential is read
from the environment.

## Example workflow

Set the API key outside the code:

```powershell
$env:GEMINI_API_KEY = "..."
```

Prepare and run Stage 1:

```powershell
python -m benchmark_construction.make_qa prepare-stage1 `
  --input <parsed_guidelines.pkl> --work-dir <work_dir>
python -m benchmark_construction.make_qa submit `
  --stage stage1 --work-dir <work_dir>
python -m benchmark_construction.make_qa collect `
  --stage stage1 --work-dir <work_dir> --wait
python -m benchmark_construction.make_qa merge-stage1 `
  --input <parsed_guidelines.pkl> --work-dir <work_dir> `
  --output <stage1_relevant.pkl> --errors-output <stage1_errors.csv>
```

Run Stage 2:

```powershell
python -m benchmark_construction.make_qa prepare-stage2 `
  --input <stage1_relevant.pkl> --work-dir <work_dir>
python -m benchmark_construction.make_qa submit `
  --stage stage2 --work-dir <work_dir>
python -m benchmark_construction.make_qa collect `
  --stage stage2 --work-dir <work_dir> --wait
python -m benchmark_construction.make_qa merge-stage2 `
  --input <stage1_relevant.pkl> --work-dir <work_dir> `
  --output <stage2_rules.pkl> --errors-output <stage2_errors.csv>
```

Run Stage 3 and rebalance:

```powershell
python -m benchmark_construction.make_qa prepare-stage3 `
  --input <stage2_rules.pkl> --work-dir <work_dir>
python -m benchmark_construction.make_qa submit `
  --stage stage3 --work-dir <work_dir>
python -m benchmark_construction.make_qa collect `
  --stage stage3 --work-dir <work_dir> --wait
python -m benchmark_construction.make_qa merge-stage3 `
  --input <stage2_rules.pkl> --work-dir <work_dir> `
  --output <qa_before_rebalance.pkl> --errors-output <stage3_errors.csv>
python -m benchmark_construction.make_qa rebalance `
  --input <qa_before_rebalance.pkl> --output <popanesqa.pkl> --seed 42
```

Use `--annotated-output` with Stage 1 or Stage 2 merge to retain rejected or
empty-response pages for auditing. Merges fail after saving their outputs when
any result is missing or invalid; `--allow-incomplete` explicitly accepts a
partial dataset. `--overwrite` replaces existing local artifacts, and during
preparation also clears that stage's prior job manifest and downloaded results.

Prompts live in `prompts.py`, shared Batch API and file handling in
`batch_utils.py`, and deterministic validation/post-processing in
`qa_postprocess.py`.
