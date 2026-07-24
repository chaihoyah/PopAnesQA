# PopAnesQA expert ratings

The two CSV files contain anonymized ratings for the same 100 PopAnesQA items:

- `popanesqa_annotator1.csv`
- `popanesqa_annotator2.csv`

`data_id` is the anonymized PopAnesQA item key (for example,
`popanesqa_0001`). It does not contain a guideline filename or page number.

Each key is unique within an annotator file. The two files contain identical
key sets in identical order, and every key maps to an item in the released
623-question benchmark.

Rating columns:

- `Guideline Consistency`: ordinal score from 1 to 5
- `Medical Validity`: ordinal score from 1 to 5
- `One-Best-Answer Clarity`: ordinal score from 1 to 5
- `Distractor Plausibility`: ordinal score from 1 to 5
- `Target Population Appropriateness`: `Correct` or `Incorrect`

The files do not contain question text, answer text, free-text comments, or
annotator identity. Join ratings to PopAnesQA by `data_id`, and
assert key equality before calculating agreement rather than relying only on
row order.

From the repository root, reproduce the summary statistics with:

```bash
python human_evaluation_statistics.py \
  --annotator1 data/human_evaluation/popanesqa_annotator1.csv \
  --annotator2 data/human_evaluation/popanesqa_annotator2.csv \
  --output-file <human_evaluation_summary.csv>
```

The script validates unique and identical key sets before matching. It reports
the pooled mean and sample standard deviation, quadratic-weighted Cohen's
kappa, exact agreement, and within-one agreement for each ordinal item. Target
population appropriateness is reported as pooled accuracy, unweighted kappa,
and exact agreement.
