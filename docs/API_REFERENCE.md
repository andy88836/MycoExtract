# API 参考文档

## 核心模块

### `src.pipeline.enhanced_pipeline`

#### `EnhancedExtractionPipeline`

主要提取流水线类。

**初始化参数**:

```python
pipeline = EnhancedExtractionPipeline(
    # 多模型配置
    use_paper_level_aggregation=True,
    kimi_client=client1,
    deepseek_client=client2,
    glm47_client=client3,
    glm46v_client=vision_client,
    aggregation_client=teacher_client,

    # 兼容配置
    text_client=text_client,
    multimodal_client=multimodal_client,
    review_client=review_client,

    # 提示词路径
    text_prompt_path="prompts/prompts_extract_from_text.txt",
    table_prompt_path="prompts/prompts_extract_from_table.txt",

    # 性能参数
    max_workers=2,
    context_overlap_sentences=0,

    # 功能开关
    use_full_md=True,
    enable_record_merge=True,
    enable_sequence_enrichment=True,
    require_sequence=False,
)
```

**方法**:

##### `run(paper_dirs, output_dir, progress_callback=None)`

运行提取流程。

**参数**:
- `paper_dirs` (List[str]): 论文目录路径列表
- `output_dir` (str): 输出目录路径
- `progress_callback` (Callable, optional): 进度回调函数

**返回**:
```python
{
    "results": {paper_name: records_list},
    "statistics": {
        "total_papers": int,
        "processed_papers": int,
        "failed_papers": int,
        "skipped_papers": int,
        "total_records": int,
        "avg_time_per_paper": float
    },
    "failed_papers": {paper_name: error_message}
}
```

---

### `src.pipeline.paper_level_prechecker`

#### `PaperLevelPrechecker`

论文级预检查器。

**初始化参数**:

```python
prechecker = PaperLevelPrechecker(
    min_mycotoxin_hits=1,      # 最少霉菌毒素命中数
    min_kinetics_hits=2,       # 最少动力学命中数
    enable_unit_check=True      # 启用单位检查
)
```

**方法**:

##### `should_skip_paper(paper_dir, full_md_path=None, max_chars=30000)`

判断论文是否应该跳过。

**参数**:
- `paper_dir` (Path): 论文目录路径
- `full_md_path` (Path, optional): full.md文件路径
- `max_chars` (int): 读取的最大字符数

**返回**:
```python
{
    "should_skip": bool,
    "reason": str,
    "mycotoxin_hits": int,
    "kinetics_hits": int,
    "unit_hits": int
}
```

##### `batch_check_papers(paper_dirs)`

批量检查论文列表。

**返回**:
```python
{
    "total": int,
    "passed": int,
    "skipped": int,
    "skip_rate": float,
    "results": {paper_name: result_dict}
}
```

---

### `src.utils.data_validator`

#### `DataValidator`

数据验证器，提供置信度评分功能。

**方法**:

##### `calculate_confidence(record) -> int`

计算记录的置信度分数。

**参数**:
- `record` (Dict): 提取的记录

**返回**:
- `int`: 置信度分数 (1-3)

**评分规则**:
```
Score 3: 核心要素 + ≥3重要要素 + ≥1奖励要素
Score 2: 核心要素 + ≥1重要要素
Score 1: 其他
```

---

### `src.llm_clients`

#### `build_client(provider, model_name, **kwargs)`

构建LLM客户端。

**参数**:
- `provider` (str): 提供商 (`moonshot`, `deepseek`, `zhipuai`, `openai`)
- `model_name` (str): 模型名称
- `**kwargs`: 额外参数 (api_key, base_url等)

**返回**: LLM客户端实例

**示例**:
```python
# Moonshot Kimi
kimi_client = build_client("moonshot", "kimi-k2-0905-preview")

# DeepSeek
deepseek_client = build_client("deepseek", "deepseek-chat")

# 智谱 GLM-4.7
glm47_client = build_client("zhipuai", "glm-4.7")

# 智谱 GLM-4.6V
glm46v_client = build_client("zhipuai", "glm-4.6v")

# OpenAI GPT-5.1
gpt_client = build_client("openai", "gpt-5.1")
```

---

## 数据结构

### 输入格式

#### 论文目录结构

```
paper_name/
├── full.md          # 必需：论文全文(Markdown)
├── images/          # 可选：图表文件夹
│   ├── figure1.png
│   └── table1.png
└── metadata.json    # 可选：元数据
```

#### full.md 格式要求

- 必须包含论文主要部分（摘要、引言、方法、结果）
- 推荐使用标准Markdown格式
- 图片使用相对路径引用

### 输出格式

#### combined_results.json

```json
[
  {
    "enzyme_name": "ZHD101",
    "enzyme_full_name": "Zearalenone hydrolase ZHD101",
    "EC_number": "3.1.1.B12",
    "substrate": "zearalenone",
    "mycotoxin_class": "zearalenone",
    "Km_value": 0.5,
    "Km_unit": "M",
    "kcat_value": 2.5,
    "kcat_unit": "s-1",
    "kcat_Km_value": 5000000,
    "kcat_Km_unit": "M-1s-1",
    "degradation_efficiency": 95.0,
    "time": 60,
    "time_unit": "min",
    "products": "non-toxic metabolites",
    "toxicity_change": "eliminated",
    "organism": "Clonostachys rosea",
    "strain": "IFO 7063",
    "pH": 7.5,
    "pH_range": "6.0-9.0",
    "temperature": 30,
    "temperature_unit": "C",
    "temperature_range": "25-35",
    "buffer_system": "phosphate buffer",
    "preparation_type": "purified",
    "source": "recombinant E. coli",
    "sequence_id": "AAK72148.1",
    "database": "UniProt",
    "gene_name": "zhd101",
    "confidence_score": 3
  }
]
```

#### quality_report.json

```json
{
  "total_records": 100,
  "score_distribution": {
    "3": {"count": 60, "percentage": 60.0},
    "2": {"count": 30, "percentage": 30.0},
    "1": {"count": 10, "percentage": 10.0}
  },
  "field_coverage": {
    "Km_value": {"count": 68, "percentage": 68.0},
    "kcat_value": {"count": 42, "percentage": 42.0},
    "degradation_efficiency": {"count": 31, "percentage": 31.0}
  },
  "quality_metrics": {
    "overall_accuracy": 0.89,
    "high_confidence_accuracy": 0.95,
    "medium_confidence_accuracy": 0.84
  }
}
```

---

## 配置参考

### 环境变量

```bash
# Moonshot (Kimi)
export MOONSHOT_API_KEY="your-api-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-api-key"

# 智谱AI
export ZHIPUAI_API_KEY="your-api-key"

# OpenAI
export OPENAI_API_KEY="your-api-key"
```

### YAML配置

```yaml
# config/extraction_config.yaml

llm_clients:
  kimi_client:
    provider: moonshot
    model_name: kimi-k2-0905-preview
    api_key: "${MOONSHOT_API_KEY}"
    base_url: null

  deepseek_client:
    provider: deepseek
    model_name: deepseek-chat
    api_key: "${DEEPSEEK_API_KEY}"

  glm47_client:
    provider: zhipuai
    model_name: glm-4.7
    api_key: "${ZHIPUAI_API_KEY}"

  glm46v_client:
    provider: zhipuai
    model_name: glm-4.6v
    api_key: "${ZHIPUAI_API_KEY}"

  aggregation_client:
    provider: openai
    model_name: gpt-5.1
    api_key: "${OPENAI_API_KEY}"

file_paths:
  prompt_text: prompts/prompts_extract_from_text.txt
  prompt_table: prompts/prompts_extract_from_table.txt

pipeline:
  max_workers: 2
  enable_record_merge: true
  enable_sequence_enrichment: true
  require_sequence: false

prechecker:
  min_mycotoxin_hits: 1
  min_kinetics_hits: 2
  enable_unit_check: true
```

---

## 错误处理

### 常见异常

#### `APIConnectionError`

API连接失败。检查网络和API密钥配置。

#### `RateLimitError`

超过API速率限制。降低并发数或增加重试延迟。

#### `ValidationError`

输入数据验证失败。检查输入格式。

#### `ExtractionError`

提取过程失败。查看日志获取详细错误信息。

### 错误处理示例

```python
try:
    pipeline.run(paper_dirs, output_dir)
except APIConnectionError as e:
    logger.error(f"API连接失败: {e}")
except RateLimitError as e:
    logger.warning(f"速率限制: {e}, 降低并发数重试...")
    pipeline.max_workers = 1
    pipeline.run(paper_dirs, output_dir)
except Exception as e:
    logger.error(f"未知错误: {e}")
```
