# MycoExtract

> **Multi-Agent LLM Pipeline for Automated Construction of Mycotoxin-Degrading Enzyme Kinetics Database**

> 多代理大语言模型流水线，用于自动化构建霉菌毒素降解酶动力学数据库

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [输出结果](#输出结果)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 项目简介

MycoExtract 是一个基于多代理大语言模型（LLM）的自动化数据提取流水线，专门用于从科学文献中提取霉菌毒素降解酶的动力学参数。系统采用教师-学生架构，结合两个学生模型、一个视觉模型和一个教师模型，实现了高精度、高效率的数据提取。

### 应用场景

- 构建**霉菌毒素降解酶**专业数据库
- 提取**酶动力学参数**（Km、kcat、kcat/Km）
- 分析**降解效率**和**毒性变化**数据
- 支持**固定化酶**和**粗酶制剂**数据提取

---

## 核心特性

### 🏗️ 多代理协作架构

```
┌─────────────────────────────────────────────────────────┐
│                   论文输入 (PDF/Markdown)                  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   视觉模型    │ │  学生模型1   │ │  学生模型2   │
│  (GLM-4.6V)  │ │   (Kimi)    │ │ (DeepSeek)  │
│  表格/图片    │ │   文本提取   │ │   文本提取   │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │                │                │
       └────────────────┴────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  教师模型        │
              │    (GPT-5.1)    │
              │  (聚合与质量控制) │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  最终结果        │
              │   + 置信度评分   │
              └─────────────────┘
```

**模型配置说明**：
- **学生模型**：Kimi（长上下文）+ DeepSeek（技术准确性）
- **视觉模型**：GLM-4.6V（表格和图片识别）
- **教师模型**：GPT-5.1（智能聚合与质量控制）

### ⚡ 预检查优化

- **30% 成本节省**：关键词匹配过滤不相关论文
- **零误杀率**：保守策略确保不错过相关文献
- **快速扫描**：0.3ms/论文 vs 3000ms/LLM调用

### 📊 三级置信度评分

| 分数 | 标准 | 准确率 |
|-----|------|--------|
| **3 (高)** | 核心要素 + ≥3重要要素 + ≥1加分项 | 95% |
| **2 (中)** | 核心要素 + ≥1重要要素 | 84% |
| **1 (低)** | 缺少核心要素 | 56% |

### 🔬 扩展数据模式 (v7.0)

- ✅ 接受**固定化酶**和**粗酶制剂**
- ✅ 提取**降解产物**信息
- ✅ 记录**毒性变化**数据
- ✅ 支持**6个独立重要元素**评分

---

## 系统架构

### 提取流程

```
Phase 0: 论文级预检查
    ├─ 霉菌毒素关键词匹配 (30+ 术语)
    ├─ 酶动力学关键词匹配 (20+ 术语)
    └─ 决策: 处理 或 跳过

Phase 1: 多模型提取
    ├─ 视觉模型 (GLM-4.6V) → 表格数据（共享）
    ├─ 学生模型 (Kimi) → 文本数据
    └─ 学生模型 (DeepSeek) → 文本数据

Phase 2: 聚合与质量控制
    ├─ 教师模型 (GPT-5.1) → 冲突解决与数据融合
    ├─ 置信度评分 (1-3分)
    ├─ 两步底物过滤
    └─ 单位规范化
```

---

## 安装指南

### 环境要求

- Python 3.8 或更高版本
- 依赖的 LLM API 密钥

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/andy88836/MycoExtract.git
cd MycoExtract
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置 API 密钥**

创建 `.env` 文件或直接修改 `config/extraction_config.yaml`：

```yaml
llm_clients:
  kimi_client:
    api_key: "your-moonshot-api-key"
  deepseek_client:
    api_key: "your-deepseek-api-key"
  glm46v_client:
    api_key: "your-zhipuai-api-key"
  aggregation_client:
    api_key: "your-openai-api-key"
```

---

## 快速开始

### 基本用法

```bash
python scripts/run_extraction.py \
  --input-dir data/papers \
  --output-dir results/$(date +%Y%m%d)_extraction \
  --max-workers 2
```

### 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|-----|------|------|--------|
| `--input-dir` | `-i` | 输入目录（必需） | - |
| `--output-dir` | `-o` | 输出目录 | results/extraction_TIMESTAMP |
| `--config` | `-c` | 配置文件路径 | config/extraction_config.yaml |
| `--limit` | `-n` | 限制处理数量（测试用） | 全部 |
| `--max-workers` | - | 最大并发数 | 2 |

### 使用扩展版配置（接受固定化/粗酶）

```bash
python scripts/run_extraction.py \
  -i data/papers \
  -c config/extraction_config_v7_expanded.yaml \
  -o results/v7_expanded
```

---

## 配置说明

### 配置文件结构

```yaml
# config/extraction_config.yaml

llm_clients:
  # 学生模型
  kimi_client:
    provider: moonshot
    model_name: kimi-k2-0905-preview
    api_key: "${MOONSHOT_API_KEY}"

  deepseek_client:
    provider: deepseek
    model_name: deepseek-chat
    api_key: "${DEEPSEEK_API_KEY}"

  # 视觉模型
  glm46v_client:
    provider: zhipuai
    model_name: glm-4.6v
    api_key: "${ZHIPUAI_API_KEY}"

  # 教师模型
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
```

---

## 输出结果

### 输出文件

```
results/extraction_YYYYMMDD_HHMMSS/
├── combined_results.json      # 所有提取结果（主要使用）
├── table_results.json         # 表格提取结果
├── text_results.json          # 文本提取结果
├── quality_report.json        # 质量分析报告
├── statistics.json            # 提取统计信息
└── precheck_stats.json        # 预检查统计
```

### 数据格式

每条记录包含以下字段：

```json
{
  "enzyme_name": "ZHD101",
  "substrate": "zearalenone",
  "Km_value": 0.5,
  "Km_unit": "M",
  "kcat_value": 2.5,
  "kcat_unit": "s-1",
  "kcat_Km_value": 5000000,
  "kcat_Km_unit": "M-1s-1",
  "degradation_efficiency": 95.0,
  "products": "non-toxic metabolites",
  "toxicity_change": "eliminated",
  "organism": "Clonostachys rosea",
  "pH": 7.5,
  "temperature": 30,
  "temperature_unit": "C",
  "preparation_type": "purified",
  "confidence_score": 3
}
```

---

## 项目结构

```
mycoextract/
├── README.md                   # 本文件
├── LICENSE                     # 许可证
├── requirements.txt            # Python依赖
├── setup.py                    # 安装脚本
├── config/                     # 配置文件
│   ├── extraction_config.yaml
│   └── extraction_config_v7_expanded.yaml
├── prompts/                    # 提示词模板
│   ├── prompts_extract_from_text.txt
│   ├── prompts_extract_from_table.txt
│   ├── prompts_extract_from_text_v7_expanded.txt
│   └── prompts_extract_from_table_v7_expanded.txt
├── scripts/                    # 运行脚本
│   └── run_extraction.py       # 主运行脚本
├── src/                        # 源代码
│   ├── __init__.py
│   ├── agents/                 # 代理模块
│   │   ├── aggregation_agent.py
│   │   ├── sequence_detective.py
│   │   └── cross_validator.py
│   ├── extractors/             # 提取器模块
│   │   ├── paper_level_extractor.py
│   │   └── multi_model_extractor.py
│   ├── llm_clients/            # LLM客户端
│   │   ├── providers.py
│   │   └── client_factory.py
│   ├── llm_extraction/         # LLM提取模块
│   │   ├── text_extractor.py
│   │   ├── table_extractor.py
│   │   └── postprocessor.py
│   ├── pipeline/               # 流水线模块
│   │   ├── enhanced_pipeline.py
│   │   ├── paper_level_prechecker.py
│   │   ├── content_filter.py
│   │   └── post_processor.py
│   └── utils/                  # 工具模块
│       ├── data_validator.py
│       ├── quality_analyzer.py
│       ├── sequence_enricher.py
│       ├── unit_normalizer.py
│       └── logging_config.py
├── docs/                       # 文档
│   ├── ARCHITECTURE.md         # 系统架构说明
│   ├── EXTRACTION_FLOW.md      # 提取流程详解
│   └── API_REFERENCE.md        # API参考
└── examples/                   # 示例代码
    ├── basic_usage.py
    └── custom_config.py
```

---

## 常见问题

### Q: 如何获取API密钥？

- **Moonshot (Kimi)**: https://platform.moonshot.cn/
- **DeepSeek**: https://platform.deepseek.com/
- **智谱AI (GLM)**: https://open.bigmodel.cn/
- **OpenAI (GPT)**: https://platform.openai.com/

### Q: 如何调整并发数？

GLM-4.6V 视觉模型的并发限制为 1，建议保持默认值。如需调整：

```bash
python scripts/run_extraction.py -i data/papers --max-workers 2
```

### Q: 如何只提取表格数据？

修改配置文件，设置 `skip_text: true`，或在代码中调整。

### Q: 输入数据格式要求什么？

输入目录应包含解析后的论文文件夹，每个文件夹包含：
- `full.md` - 论文全文（Markdown格式）
- `images/` - 图片文件夹（可选）
- `metadata.json` - 元数据（可选）

---

## 性能指标

| 指标 | 数值 |
|-----|------|
| 每篇处理时间 | ~15秒 |
| 每篇API成本 | ~¥30 |
| 预检查跳过率 | 30% |
| 高置信度准确率 | 95% |
| Km值覆盖率 | 68.6% |
| kcat值覆盖率 | 41.9% |

---

## 引用

如果您在研究中使用了 MycoExtract，请引用：

```bibtex
@software{mycoextract2024,
  title = {MycoExtract: A Multi-Agent LLM Pipeline for Automated
           Construction of Mycotoxin-Degrading Enzyme Database},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/andy88836/MycoExtract}
}
```

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

- GitHub：https://github.com/andy88836/MycoExtract

---

## 致谢

感谢以下 LLM 提供商的支持：
- Moonshot (Kimi)
- DeepSeek
- 智谱AI (GLM)
- OpenAI (GPT)
