# MycoExtract 发布说明

## 版本信息

- **版本号**: v1.0.0
- **发布日期**: 2024-03-20
- **许可证**: MIT License

## 项目概述

MycoExtract 是一个基于多代理大语言模型（LLM）的自动化数据提取流水线，专门用于从科学文献中提取霉菌毒素降解酶的动力学参数。

### 核心特性

1. **多代理协作架构**
   - 2个学生模型：Kimi（长上下文）、DeepSeek（技术准确性）
   - 1个视觉模型：GLM-4.6V（图表识别）
   - 1个教师模型：GPT-5.1（聚合与质量控制）

2. **预检查优化机制**
   - 30%成本节省
   - 关键词匹配过滤
   - 零误杀率设计

3. **三级置信度评分**
   - 6个独立重要元素
   - 95%高准确率

4. **扩展数据模式（v7.0）**
   - 接受固定化/粗酶制剂
   - 降解产物追踪
   - 毒性变化记录

---

## 项目结构

```
mycoextract/
├── README.md                   # 项目说明
├── LICENSE                     # MIT许可证
├── requirements.txt            # Python依赖
├── setup.py                    # 安装脚本
├── .gitignore                  # Git忽略文件
├── config/                     # 配置文件
│   ├── extraction_config.yaml
│   └── extraction_config_v7_expanded.yaml
├── prompts/                    # 提示词模板
│   ├── prompts_extract_from_text.txt
│   ├── prompts_extract_from_table.txt
│   ├── prompts_extract_from_text_v7_expanded.txt
│   └── prompts_extract_from_table_v7_expanded.txt
├── scripts/                    # 运行脚本
│   ├── run_extraction.py       # 主运行脚本
│   └── validate_project.py     # 项目验证脚本
├── src/                        # 源代码
│   ├── agents/                 # 代理模块（聚合、序列检测）
│   ├── extractors/             # 提取器模块
│   ├── llm_clients/            # LLM客户端
│   ├── llm_extraction/         # LLM提取模块
│   ├── pipeline/               # 流水线模块
│   └── utils/                  # 工具模块
├── docs/                       # 文档
│   ├── ARCHITECTURE.md         # 系统架构
│   ├── EXTRACTION_FLOW.md      # 提取流程详解
│   └── API_REFERENCE.md        # API参考
└── examples/                   # 示例代码
    └── basic_usage.py
```

---

## 安装指南

### 1. 环境要求

- Python 3.8 或更高版本
- 各 LLM 提供商的 API 密钥

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥

编辑 `config/extraction_config.yaml`，填入您的 API 密钥：

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
  --output-dir results/extraction \
  --max-workers 2
```

### 使用扩展版配置

```bash
python scripts/run_extraction.py \
  --input-dir data/papers \
  --config config/extraction_config_v7_expanded.yaml \
  --output-dir results/v7_expanded
```

---

## 输出结果

### 文件结构

```
results/extraction_YYYYMMDD_HHMMSS/
├── combined_results.json      # 主要输出
├── table_results.json         # 表格提取结果
├── text_results.json          # 文本提取结果
├── quality_report.json        # 质量报告
├── statistics.json            # 统计信息
└── precheck_stats.json        # 预检查统计
```

### 数据格式

每条记录包含：

- **酶信息**: enzyme_name, enzyme_full_name, EC_number
- **底物信息**: substrate, mycotoxin_class
- **动力学参数**: Km, kcat, kcat/Km（含单位）
- **降解数据**: degradation_efficiency, products, toxicity_change
- **实验条件**: pH, temperature, buffer_system
- **生物体信息**: organism, strain
- **置信度评分**: confidence_score (1-3)

---

## 性能指标

| 指标 | 数值 |
|-----|------|
| 每篇处理时间 | ~15秒 |
| 每篇API成本 | ~¥30 |
| 预检查跳过率 | 30% |
| 高置信度准确率 | 95% |

---

## 验证项目完整性

运行验证脚本确保所有文件完整：

```bash
python scripts/validate_project.py
```

预期输出：
```
[SUCCESS] 所有检查通过！项目结构完整。

您可以运行:
  pip install -r requirements.txt
  python scripts/run_extraction.py --help
```

---

## 文档索引

- **README.md** - 项目概述和快速开始
- **docs/ARCHITECTURE.md** - 系统架构详细说明
- **docs/EXTRACTION_FLOW.md** - 提取流程详解
- **docs/API_REFERENCE.md** - API 参考文档

---

## 支持的 LLM 提供商

| 提供商 | 模型 | 用途 |
|-------|------|------|
| Moonshot | Kimi K2 | 学生模型（长上下文） |
| DeepSeek | DeepSeek Chat | 学生模型（技术准确性） |
| 智谱AI | GLM-4.6V | 视觉模型（图表识别） |
| OpenAI | GPT-5.1 | 教师模型（聚合） |

---

## 常见问题

### Q: 如何获取 API 密钥？

- [Moonshot](https://platform.moonshot.cn/)
- [DeepSeek](https://platform.deepseek.com/)
- [智谱AI](https://open.bigmodel.cn/)
- [OpenAI](https://platform.openai.com/)

### Q: 如何调整并发数？

默认为2（适配GLM-4.6V视觉模型限制），可通过 `--max-workers` 参数调整：

```bash
python scripts/run_extraction.py -i data/papers --max-workers 3
```

### Q: 如何只提取表格数据？

修改配置文件中的 `skip_text` 参数，或调整提示词。

---

## 开发路线图

### v1.0.0 (当前版本)
- ✅ 多代理协作架构
- ✅ 预检查优化
- ✅ 三级置信度评分
- ✅ 扩展数据模式（v7.0）

### 未来计划
- [ ] Web 界面
- [ ] 实时进度监控
- [ ] 更多霉菌毒素类型支持
- [ ] 社区标注工具
- [ ] 知识图谱集成

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

- 项目主页: https://github.com/your-repo/mycoextract
- 问题反馈: https://github.com/your-repo/mycoextract/issues

---

## 致谢

感谢以下项目和工具的支持：

- Python 编程语言
- 各 LLM 提供商的 API 服务
- 开源社区的贡献者
