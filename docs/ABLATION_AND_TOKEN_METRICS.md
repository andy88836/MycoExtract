# Ablation Study and Token Metrics

This document defines the manuscript-facing ablation outputs for MycoExtract.

## Why This Experiment Exists

The ablation study quantifies whether the teacher-student architecture improves extraction quality over simpler configurations. It should be reported as an extraction-performance experiment, not as a biological result.

## Configurations

`scripts/run_ablation_study.py` produces these configurations:

- `kimi_only`: records extracted by the Kimi student branch.
- `deepseek_only`: records extracted by the DeepSeek student branch.
- `glm-4.7_only`: records extracted by the GLM-4.7 student branch, when enabled.
- `students_union`: all student records merged without teacher aggregation.
- `majority_vote`: only enzyme-substrate-mutation records supported by at least two student branches.
- `teacher_aggregation`: final teacher-aggregated records.

The recommended manuscript comparison is single models vs. `majority_vote` vs. `teacher_aggregation`.

## Metrics

The evaluator compares each configuration against the manual validation workbook (`gold_dataset.xlsx`) using DOI-level grouping.

Field-level metrics are computed for:

- `Enzyme`
- `Substrate`
- `Organism`
- `Km (μM)`
- `kcat (s⁻¹)`
- `kcat/Km (M⁻¹s⁻¹)`
- `Degrad. %`

For string fields, values are lowercased and whitespace-normalized before exact comparison.

For numerical fields, a prediction is correct when it is within 5% relative error of the gold value. This matches the current validation figure logic in `make_draft_figures.py`.

For each field and configuration, the script reports:

- `tp`: predicted field value matches gold.
- `fp`: predicted field value is absent from or inconsistent with gold.
- `fn`: gold field value was not correctly recovered.
- `precision`
- `recall`
- `f1`

Micro-averaged precision, recall, and F1 are computed by summing TP/FP/FN across all fields.

## Token Metrics

Token usage is collected by `src/utils/token_usage.py`.

The tracker records:

- provider
- model
- prompt tokens
- completion tokens
- total tokens
- request count
- whether usage came from API-reported values or offline estimation
- task label, such as `text_kimi`, `table_vision`, or `teacher_aggregation`

API-reported usage is used whenever available. If the provider response lacks usage metadata, the tracker estimates text tokens locally and marks that request as `estimated`.

## Outputs

Run:

```bash
python scripts/run_ablation_study.py ^
  --input-dir data/ground_truth/ground_truth_parse ^
  --gold-file ../gold_dataset.xlsx ^
  --config config/extraction_config_v7_expanded.yaml ^
  --out-dir results/ablation_study
```

For a smoke test:

```bash
python scripts/run_ablation_study.py ^
  --input-dir data/ground_truth/ground_truth_parse ^
  --gold-file ../gold_dataset.xlsx ^
  --config config/extraction_config_v7_expanded.yaml ^
  --out-dir results/ablation_smoke ^
  --limit 3
```

Main outputs:

- `ablation_predictions.json`: raw model/configuration predictions by paper.
- `ablation_metrics_by_field.csv`: field-level precision/recall/F1.
- `ablation_metrics_overall.csv`: micro-averaged metrics by configuration.
- `ablation_token_usage.json`: full token usage summary.
- `ablation_token_usage_records.csv`: one row per LLM call.
- `ablation_token_usage_by_task.csv`: token totals by task/provider/model.
- `fig_ablation_micro_f1.png`: bar chart for the manuscript.
- `fig_token_usage_by_task.png`: token-use panel for efficiency/cost discussion.

To re-evaluate saved predictions without re-running LLM calls:

```bash
python scripts/run_ablation_study.py ^
  --evaluate-only results/ablation_study/ablation_predictions.json ^
  --gold-file ../gold_dataset.xlsx ^
  --out-dir results/ablation_study
```

## Manuscript Use

Use `ablation_metrics_overall.csv` for the main ablation table and `fig_ablation_micro_f1.png` for the Results figure panel.

Use `ablation_token_usage_by_task.csv` to support a cost/efficiency panel showing how many tokens are spent by:

- student text extraction
- table text extraction
- table vision extraction
- teacher aggregation

If many rows are marked `estimated`, state that token values are estimated for providers that do not return usage metadata.
