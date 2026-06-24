# MycoExtract v8 运行手册

完整的提取 → 审核 → 准确率评估流程。本文档假定你已经完成 v8 重构（见 `REFACTOR_PROMPT_v8.md`）并跑过 smoke test。

---

## 0. 目录速查

```
D:\Doc_Projects\MycoExtract\MycoExtract\
├─ All_papers\                              # 待提取的 paper 解析包（gitignored）
├─ analysis_outputs\                        # 所有 run 的输出（gitignored）
├─ config\
│   └─ extraction_config_v8.yaml            # v8 默认 config
├─ prompts\
│   ├─ prompts_extract_from_text_v8.txt     # v8 文本提取 prompt
│   └─ prompts_extract_from_table_v8.txt    # v8 表格提取 prompt
├─ scripts\
│   ├─ run_all_papers_full_extraction.py    # 主运行脚本
│   └─ calculate_v8_accuracy.py             # 准确率计算（见 §5）
├─ src\
│   └─ utils\
│       ├─ quality_constraints.py           # mycotoxin 白名单 gate
│       ├─ row_level_validator.py           # 单位 / specific activity / stability validators
│       └─ sequence_enricher.py             # UniProt / PubChem 富化
├─ tests\test_v8_*.py                       # v8 单元测试
└─ .env                                     # 你的 LLM API keys（gitignored）
```

---

## 1. Pre-flight 检查清单（每次跑全量前）

```powershell
cd D:\Doc_Projects\MycoExtract\MycoExtract

# 1.1 v8 单元测试必须全绿
pytest tests/test_v8_*.py -v
# 期望: ============ 75 passed ============

# 1.2 .env 里的 API keys 没过期
Get-Content .env | Select-String "API_KEY"
# 应该看到 OPENAI_API_KEY / DEEPSEEK_API_KEY / MOONSHOT_API_KEY / ZHIPUAI_API_KEY 等

# 1.3 输入目录有内容
(Get-ChildItem All_papers -Directory).Count        # paper 数量

# 1.4 git 状态干净（没有未提交的代码改动）
git status --short
git log --oneline -3
# HEAD 应该是 dd9043c (v8 merge commit) 或更新
```

---

## 2. 输入文件位置 & 格式

每篇 paper 在 `All_papers/<paper_id>/` 下，至少需要：

```
All_papers/
└─ 10.1234_xxxxx-yyyyy/                     # paper 目录名 = doi 转义
   ├─ full.md                               # paper 全文 markdown
   └─ <paper_id>_content_list.json          # MinerU / 类似工具产出的结构化解析
```

**重要**：`*.pdf` 本身不在 git 里（已 gitignored）。如果你新增 paper：
1. 把解析后的 `<paper>/full.md` 与 `<paper>/*_content_list.json` 放到 `All_papers/<paper>/` 下
2. 不需要改任何代码，inventory builder 会自动识别

v8 当前不再使用历史金标准目录作为默认跳过集合。所有进入输入目录的候选文章会统一重新提取，并用 deterministic v8 quality score 重新评分。

---

## 3. 运行：3 种模式

### 3.1 Inventory only（不调 LLM，秒级，先看清家底）

```powershell
python scripts/run_all_papers_full_extraction.py --inventory-only
```

输出：
```
analysis_outputs/all_papers_full_extraction_<timestamp>/preflight/
├─ all_papers_inventory.csv          # 每篇 paper 的状态
└─ inventory_summary.json            # 总数 / 状态分布
```

`extraction_status` 列的值：
- `new_for_extraction` → 待提取
- `duplicate_pdf` → 重复，跳过

### 3.2 Smoke test（1 篇，几分钟，验证 pipeline）

```powershell
python scripts/run_all_papers_full_extraction.py `
    --limit 1 `
    --output-root analysis_outputs/v8_smoke_test
```

跑完检查 `analysis_outputs/v8_smoke_test/curated_exports/`，确认：
- `mycoextract_v8_all_records.csv` 有内容
- 列数 ≤ 70（v8 schema ~50 列）
- `measurement_type` 取值只在 `{kinetic, degradation, stability}` 中

### 3.3 全量跑

```powershell
python scripts/run_all_papers_full_extraction.py `
    --max-workers 1 `
    --output-root analysis_outputs/v8_full_$(Get-Date -Format "yyyyMMdd_HHmmss")
```

参数说明：
- `--max-workers N` — 并发 paper 数。1 最稳；4-8 看 LLM rate limit
- `--output-root PATH` — 不指定就自动用时间戳
- `--limit N` — 限定数量，调试用
- `--resume` — 跳过已处理的 paper（检测 `validated_outputs/<paper>_validated.json` 是否存在）
- `--config PATH` — 不指定就用 `config/extraction_config_v8.yaml`

时间预估：每篇 paper ~30 秒-2 分钟（看 paper 长度 + LLM 选择）。100 篇 ≈ 1-3 小时。

---

## 4. 输出位置与文件含义

每次 run 会创建一个独立目录：

```
analysis_outputs/<run_name>/
├─ preflight/
│   ├─ all_papers_inventory.csv             # paper 清单
│   └─ inventory_summary.json
├─ raw_outputs/
│   └─ <paper>.json                         # paper 级原始 JSON
├─ validated_outputs/
│   ├─ <paper>_validated.json               # 经过 row-level validator 的记录
│   └─ <paper>_validated.csv
├─ logs/
│   └─ <paper>.log                          # 每篇 paper 的 INFO 日志
├─ debug_traces/
│   └─ <paper>_trace.json                   # 完整 trace（含 LLM 调用细节）
├─ curated_exports/                         # ⭐ 主交付物
│   ├─ mycoextract_v8_all_records.csv       # 全部记录（含 rejected）
│   ├─ mycoextract_v8_primary_eligible.csv  # 通过所有 v8 filter 的高质量记录
│   ├─ mycoextract_v8_kinetics_core.csv     # measurement_type=kinetic 子集
│   ├─ mycoextract_v8_degradation_core.csv  # measurement_type=degradation 子集
│   ├─ mycoextract_v8_stability_core.csv    # measurement_type=stability 子集
│   ├─ mycoextract_v8_secondary_candidates.csv
│   ├─ mycoextract_v8_rejected_records.csv  # 被白名单 / measurement_type 拒绝
│   ├─ mycoextract_v8_manual_review_candidates.csv
│   └─ mycoextract_v8_enzyme_substrate_summary.csv
├─ metrics/
│   ├─ token_usage_audit_all_papers.csv     # token 使用 + 价格
│   ├─ runtime_summary_all_papers.csv
│   └─ model_usage_summary_all_papers.csv
└─ comparison_reports/
    ├─ all_papers_run_summary.csv           # 每篇 paper 的 status
    ├─ zero_record_unresolved_report.csv    # paper-level 提取返回 0 行的论文
    └─ all_papers_final_report.md           # ⭐ 总报告（先看这个）
```

**优先看**：`comparison_reports/all_papers_final_report.md` — 总览所有数字。

**核心数据交付**：`curated_exports/mycoextract_v8_primary_eligible.csv`。

---

## 5. 数据审核（人工标注 TP/FP）

### 5.1 准备审核样本

从 primary_eligible 里抽样（建议先抽 100-150 行）：

```powershell
python scripts/prepare_review_sample.py `
    --input  analysis_outputs/<run_name>/curated_exports/mycoextract_v8_primary_eligible.csv `
    --output analysis_outputs/<run_name>/human_review_v8.xlsx `
    --sample-size 150
```

> 这个脚本在下面 §7 一起给。会把 CSV 转成 xlsx 并加 `human_notes` / `Status` 两个空白列。

### 5.2 在 Excel 里标注

打开 `human_review_v8.xlsx`，每行做判断，填两列：

| 列名 | 取值规范 |
|---|---|
| `Status` | `TP` / `FP` / `TP_field_correction` / `FN`（仅金标准对照时用） |
| `human_notes` | 自由文字描述错误原因 |

**标注规则建议**（沿用上次 review 的标准）：

- `TP` — 整行内容正确（酶名、底物、measurement_type、数值、单位、条件）
- `TP_field_correction` — 主要内容正确但某个字段错了（如突变体丢失、产物错附）
- `FP` — 整行不该出现：
  - `substrate is not mycotoxin`
  - `非酶体系` / `non-enzymatic`
  - `Review article`
  - `no valuable data`
  - `specific activity miscoded as degradation_efficiency`
  - 其他原因
- `FN` — 仅在与金标准对照时用（金标准里有但系统没抽到）

### 5.3 审核完成后存档

把标注好的 xlsx 存到 `analysis_outputs/<run_name>/human_review_v8.xlsx`，**不要 commit 到 git**（已经 gitignored）。

---

## 6. 准确率计算

### 6.1 Precision（每次 run 都能算）

```powershell
python scripts/calculate_v8_accuracy.py `
    --review analysis_outputs/<run_name>/human_review_v8.xlsx `
    --output analysis_outputs/<run_name>/accuracy_report.md
```

输出 markdown 报告：
- 总体 Precision = TP / (TP + FP)
- 按 `measurement_type` 拆分
- 按 `eligibility_status` 拆分
- 按 FP 原因聚类（substrate / non-enzymatic / review / specific_activity 等）
- 与 v7 baseline 对比表

### 6.2 Recall（需要金标准对照）

如果你有 `gold_dataset_4_27.xlsx`（已有），可以算 recall：

```powershell
python scripts/calculate_v8_accuracy.py `
    --review analysis_outputs/<run_name>/human_review_v8.xlsx `
    --gold gold_dataset_4_27.xlsx `
    --output analysis_outputs/<run_name>/accuracy_report.md
```

报告会增加：
- Recall = TP / (TP + FN)
- F1 = 2 * P * R / (P + R)
- 漏抽分析（金标准里有但系统没抽到的记录）

### 6.3 与 baseline 对比

```powershell
python scripts/calculate_v8_accuracy.py `
    --review analysis_outputs/<run_name>/human_review_v8.xlsx `
    --baseline analysis_outputs/all_papers_full_extraction_20260430_142755/curated_exports/mycoextract_primary_eligible_all_papers.csv `
    --output analysis_outputs/<run_name>/accuracy_report.md
```

会出一张表：

| 指标 | v7 baseline | v8 | Δ |
|---|---|---|---|
| 总记录数 | 1170 | ? | ? |
| primary_eligible | 225 | ? | ? |
| 非霉菌毒素底物泄漏 | 21 | ? | ? |
| measurement_type 取值数 | 12 | 3 | -9 |
| 列数 | 134 | ~50 | -84 |
| FP rate | 52% | ? | ? |
| 自动填充 uniprot_id | 1 | ? | ? |

---

## 7. 配套脚本

`scripts/prepare_review_sample.py` 与 `scripts/calculate_v8_accuracy.py` 已经放在 `scripts/` 下，详见对应文件。

---

## 8. 排错

### 8.1 LLM API 报错 / 超时

看 `logs/<paper>.log`，最常见：
- `429 rate limit` → 降 `--max-workers` 或换 provider
- `401 unauthorized` → `.env` 里的 key 失效
- `timeout` → 该 paper 太长，可在 `extraction_parameters.timeout` 里调（v8 默认 600s）

可用 `--resume` 重跑漏的，已成功的不会重做。

### 8.2 提取出来的 record 字段名还是 v7 的

如果 `mycoextract_v8_*.csv` 里出现 `canonical_enzyme_name` / `enzyme_full_name` / `application_matrix_degradation` 等 v7 字段，说明 `paper_level_extractor.py` 那一层还在产出 v7 schema。短期 workaround：

```powershell
# 看 record 第一行所有非空字段
python -c @"
import pandas as pd
df = pd.read_csv('analysis_outputs/<run_name>/curated_exports/mycoextract_v8_all_records.csv')
print(df.iloc[0].dropna().to_dict())
"@
```

把输出贴出来，可能需要补 commit E（refactor `paper_level_extractor.py` 的内部 schema 后处理）。

### 8.3 stability_core.csv 是空的

可能 LLM 没识别 stability 数据。检查：
- 你跑的 paper 里是否真的有 stability 描述（不是所有 paper 都有）
- prompts/prompts_extract_from_text_v8.txt 的"Stability 提取示例"是否被 LLM 看到（有些 LLM 在 prompt 太长时会忽略后段）

可以人工挑一篇已知有 thermal stability 数据的 paper 单独跑（`--limit 1`）验证。

### 8.4 全量跑断在中间

`--resume` 接着跑，已 validated 的 paper 会跳过：

```powershell
python scripts/run_all_papers_full_extraction.py `
    --resume `
    --output-root analysis_outputs/<existing_run_name>
```

---

## 9. 一键完整流程

抓总：

```powershell
cd D:\Doc_Projects\MycoExtract\MycoExtract

# 0. 验证环境
pytest tests/test_v8_*.py -v

# 1. Smoke test
python scripts/run_all_papers_full_extraction.py --limit 1 `
    --output-root analysis_outputs/v8_smoke_test

# 2. 看 smoke 输出
Import-Csv analysis_outputs/v8_smoke_test/curated_exports/mycoextract_v8_all_records.csv |
    Group-Object measurement_type | Select-Object Name, Count

# 3. 全量跑
$runName = "v8_full_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python scripts/run_all_papers_full_extraction.py --max-workers 4 `
    --output-root "analysis_outputs/$runName"

# 4. 看总报告
Get-Content "analysis_outputs/$runName/comparison_reports/all_papers_final_report.md"

# 5. 准备审核样本
python scripts/prepare_review_sample.py `
    --input "analysis_outputs/$runName/curated_exports/mycoextract_v8_primary_eligible.csv" `
    --output "analysis_outputs/$runName/human_review_v8.xlsx" `
    --sample-size 150

# 6. 在 Excel 里标 TP/FP（人工，~2-4 小时）

# 7. 计算准确率
python scripts/calculate_v8_accuracy.py `
    --review "analysis_outputs/$runName/human_review_v8.xlsx" `
    --baseline "analysis_outputs/all_papers_full_extraction_20260430_142755/curated_exports/mycoextract_primary_eligible_all_papers.csv" `
    --gold gold_dataset_4_27.xlsx `
    --output "analysis_outputs/$runName/accuracy_report.md"

# 8. 看 accuracy report
Get-Content "analysis_outputs/$runName/accuracy_report.md"
```

---

## 10. 期望指标（v8 vs v7）

完成 v8 全量跑 + 人工审核后，对照下表评估是否达标：

| 指标 | v7 baseline | v8 目标 | 实际（待填）|
|---|---|---|---|
| 总记录数 | 1170 | 500-650 | ___ |
| primary_eligible | 225 | 200-280 | ___ |
| primary FP rate | 52% | ≤ 12% | ___ |
| 非霉菌毒素底物泄漏 | 21 | 0 | ___ |
| measurement_type 取值数 | 12 | 3 | ___ |
| schema 列数 | 134 | ~50 | ___ |
| zero_record_rescue 产出 | 274 | 0 | ___ |
| 自动填充 uniprot_id | 1/225 | ≥ 30/225 | ___ |
| stability 数据捕获 | ≤ 5 | ≥ 15 | ___ |
| 有 candidate_sequence | 0 | 自动填 | ___ |

低于目标的指标说明 v8 改动还需要进一步收紧；高于目标说明改太严了（可能 recall 受损）。
