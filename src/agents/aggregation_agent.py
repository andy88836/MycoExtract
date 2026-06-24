"""
Aggregation Agent - 聚合多个模型的提取结果

基于文献思路：使用配置文件指定的teacher模型作为"裁判"，
对比多个"学生"模型的结果，结合原文进行智能聚合。

参考：
- 输入：原文 + 多个模型的提取结果
- 输出：聚合后的最优结果
- 模型：由配置文件的 aggregation_client 决定
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AggregationAgent:
    """
    聚合智能体 - 对比多个模型结果并生成最优答案
    
    参考 LLM-BioDataExtractor 的实现：
    - 传完整原文（不截断）
    - 传原始提取Prompt（让聚合模型知道提取目标）
    - 用配置的teacher模型做聚合
    
    新功能：工具调用能力
    - 当发现模型冲突时，可以主动调用工具查看原始表格（含图片）
    - 支持ReAct式推理：发现问题 → 调用工具 → 基于工具结果做决策
    """
    
    def __init__(self, llm_client, model_name: str = "configured-teacher", extraction_prompt: str = None, paper_dir: Optional[Path] = None, optimized: bool = False):
        """
        Args:
            llm_client: teacher LLM客户端
            model_name: 模型名称（用于日志）
            extraction_prompt: 原始提取Prompt（用于让聚合模型理解提取目标）
            paper_dir: 论文目录（用于工具调用获取表格图片）
            optimized: 是否使用优化模式（减少冗余输出，降低token消耗）
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.extraction_prompt = extraction_prompt or self._load_default_extraction_prompt()
        self.paper_dir = paper_dir
        self.optimized = optimized
        logger.info(f"Initialized AggregationAgent with {model_name}")
        logger.info(f"  Tool-calling enabled: {paper_dir is not None}")
        logger.info(f"  Optimized mode: {optimized} (reduced verbose output)")
    
    def _load_default_extraction_prompt(self) -> str:
        """加载默认的提取Prompt"""
        import os
        prompt_path = "prompts/prompts_extract_from_text_v7_expanded.txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Extract enzyme kinetics data including Km, kcat, substrate, pH, temperature."
    
    def aggregate(
        self,
        original_text: str,
        model_results: Dict[str, List[Dict]],
        doi: str = "unknown",
        paper_blocks: Optional[List[Dict]] = None,
        locked_table_records: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        聚合多个模型的提取结果

        Args:
            original_text: 原始文献文本（用于对照检查）
            model_results: {
                "kimi": [{record1}, {record2}, ...],
                "deepseek": [{record1}, {record2}, ...],
                "glm-4.6": [{record1}, {record2}, ...]
            }
            doi: 论文DOI
            paper_blocks: 论文块列表（用于工具调用获取表格）
            locked_table_records: 已锁定的表格提取记录（只读参考，用于去重和酶名修正）

        Returns:
            teacher-harmonized text record candidates
        """
        logger.info(f"[Aggregation Agent] Starting aggregation for {doi}")
        logger.info(f"  - Models: {list(model_results.keys())}")
        logger.info(f"  - Total records: {sum(len(r) for r in model_results.values())}")
        if locked_table_records:
            logger.info(f"  - Locked table records (reference): {len(locked_table_records)}")

        # 存储paper_blocks供工具调用使用
        self.paper_blocks = paper_blocks
        self.locked_table_records = locked_table_records or []

        # 构建prompt
        prompt = self._build_aggregation_prompt(original_text, model_results)
        
        # 🔥 新增：支持工具调用的多轮对话
        max_iterations = 3  # 最多允许3轮工具调用
        conversation_history = [
            {
                "role": "system",
                "content": "You are an expert scientific data curator specializing in enzyme kinetics. Your task is to aggregate and validate extraction results from multiple AI assistants. You have access to tools to resolve conflicts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        for iteration in range(max_iterations):
            try:
                # 调用LLM
                response = self.llm_client.chat(
                    messages=conversation_history,
                    temperature=0.1,
                    max_tokens=8000,
                    task="teacher_aggregation"
                )
                
                # 检查是否有工具调用请求
                tool_call = self._parse_tool_call(response)
                
                if tool_call:
                    # Agent请求调用工具
                    logger.info(f"  🔧 [Iteration {iteration + 1}] Tool call requested: {tool_call['name']}")
                    
                    # 执行工具
                    tool_result = self._execute_tool(tool_call)
                    
                    # 将工具结果加入对话历史
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"I need to verify the data. Calling tool: {tool_call['name']}({tool_call['arguments']})"
                    })
                    conversation_history.append({
                        "role": "user",
                        "content": f"Tool result:\n```json\n{json.dumps(tool_result, indent=2, ensure_ascii=False)}\n```\n\nNow please continue aggregation based on this verified information."
                    })
                    
                    logger.info(f"  ✓ Tool executed, continuing aggregation...")
                    continue  # 继续下一轮对话
                
                else:
                    # 没有工具调用，说明得到最终结果
                    aggregated_records = self._parse_response(response)
                    if not aggregated_records and any(model_results.values()):
                        logger.warning(
                            "  ⚠️ Teacher aggregation returned no parseable records; "
                            "using conservative student-record fallback"
                        )
                        aggregated_records = self._build_fallback_records(model_results)
                    
                    # 后处理验证；这些仍只是 teacher-harmonized text candidates，
                    # 不是 final database records。
                    aggregated_records = self._post_validate_records(aggregated_records)
                    
                    logger.info(f"  ✓ Teacher harmonized {len(aggregated_records)} text candidate records")
                    return aggregated_records
                    
            except Exception as e:
                logger.error(f"  ✗ Aggregation iteration {iteration + 1} failed: {e}")
                if iteration == max_iterations - 1:
                    # 最后一次尝试也失败，降级
                    if model_results:
                        fallback_model = list(model_results.keys())[0]
                        logger.warning(f"  ⚠️ Falling back to {fallback_model} text candidates")
                        return model_results[fallback_model]
                    return []
        
        # 超过最大迭代次数，降级
        logger.warning(f"  ⚠️ Max iterations reached, falling back to text candidates")
        return self._build_fallback_records(model_results)

    def _format_locked_table_records(self) -> str:
        """格式化 locked table records 为只读参考上下文。"""
        locked = getattr(self, 'locked_table_records', []) or []
        if not locked:
            return ""

        key_fields = [
            'enzyme_name', 'mutations', 'reported_enzyme_name', 'substrate',
            'measurement_type', 'Km_value', 'Km_unit', 'kcat_value', 'kcat_unit',
            'kcat_Km_value', 'kcat_Km_unit', 'degradation_efficiency',
            'organism', 'source_table_id',
        ]
        lines = ["## Locked Table Records (for reference only — do not rewrite these records)"]
        lines.append("")
        lines.append("These records were extracted from tables and treated as primary truth for their measurements.")
        lines.append("Use them to verify text candidates and correct enzyme names if text candidates have better identity info.")
        lines.append("")
        for i, record in enumerate(locked, 1):
            fields = ", ".join(
                f"{k}={record.get(k)}" for k in key_fields if record.get(k) not in (None, "", [])
            )
            lines.append(f"Table Record {i}: {fields}")
        lines.append("")
        return "\n".join(lines)

    def _build_aggregation_prompt(
        self,
        original_text: str,
        model_results: Dict[str, List[Dict]]
    ) -> str:
        """
        构建聚合提示词 - 优化版（减少50%长度）
        """
        # 截断超长文本
        max_text_length = 100000
        text_truncated = False
        if len(original_text) > max_text_length:
            original_text = original_text[:max_text_length]
            text_truncated = True

        # 格式化模型结果
        model_outputs = ""
        for model_name, records in model_results.items():
            model_outputs += f"\n### [{model_name.upper()}] ({len(records)} records)\n"
            if not records:
                model_outputs += "*No records*\n"
            else:
                for i, record in enumerate(records, 1):
                    model_outputs += f"Record {i}: "
                    # 只显示关键字段
                    key_fields = ['enzyme_name', 'reported_enzyme_name', 'substrate', 'measurement_type',
                                  'measurement_context_id', 'source_section',
                                  'condition_scope', 'Km_value', 'kcat_value', 'kcat_Km_value',
                                  'kinetic_unit_multiplier', 'kinetic_unit_source_text',
                                  'degradation_efficiency', 'organism', 'temperature_value', 'ph',
                                  'kinetic_temperature_value', 'kinetic_ph',
                                  'degradation_temperature_value', 'degradation_ph', 'notes']
                    fields_str = ", ".join([f"{k}={record.get(k)}" for k in key_fields if record.get(k)])
                    model_outputs += fields_str + "\n"

        # 精简版prompt
        prompt = f"""# Enzyme Data Aggregation

## Article Text
```
{original_text[:80000]}{"[...truncated...]" if text_truncated else ""}
```

{self._format_locked_table_records()}

## Model Outputs to Aggregate
{model_outputs}

## Your Task

Harmonize TEXT candidate records from Kimi, DeepSeek, and the configured third text model into teacher_harmonized_records.
Do not extract new table-derived records from the article text. Use article text only to verify or reject fields already present in text candidates.
Parsed table and table-image candidates are handled outside this teacher step as locked candidates.

### Core Rules:

1. **VERIFY values against article text** - Only use values explicitly stated in the article
2. **RESOLVE conflicts** - Choose the value supported by the article
3. **MERGE complementary data** - Combine different fields from different models
4. **REMOVE true duplicates** - One record per unique experimental measurement context, NOT merely per enzyme-substrate pair
5. **CORRECT obvious errors** - Fix units, decimals, etc.
6. **USE locked table records as reference** - If a locked table record exists for the same measurement (same substrate + kinetic values), the text candidate should use the same enzyme_name. If the table record has a mutation name as enzyme_name (e.g., "WT", "Q202E"), use the text candidate's enzyme_name to correct it.

### Measurement Context Attribution Patch v8.2

Each final record must represent one experimental measurement context. Only output `measurement_type: "kinetic"` or `measurement_type: "degradation"`.

A measurement context is defined by the same paper + same enzyme/system + same substrate + same measured metric + same matrix + same temperature + same pH + same time + same cofactor/buffer condition.

Do not merge kinetic parameters and degradation/conversion results unless they come from the same experiment under the same conditions. Do not put degradation time into kinetic rows. Do not put kinetic temperature/pH/time into degradation rows. Do not put generic enzyme activity assay conditions into toxin degradation or kinetic rows.

Only fill optimum pH/temperature when the article explicitly says optimum/optimal. Optimized degradation conditions are not enzyme optimum conditions.

If abstract and main text conflict, keep the relevant record as `kinetic` or `degradation` only when the metric is explicit, set `human_review_required: true`, and explain the conflicting statements in `notes`. Do not output `ambiguous_conflicting_source`.

Kinetic records must contain at least one of `Km_value`, `kcat_value`, or `kcat_Km_value`. Do not use Vmax as a primary kinetic metric in this project. If a source only reports Vmax and no Km/kcat/kcat_Km, return no kinetic record for that measurement.

Stability and optimum information are auxiliary annotations only. Do not output standalone `measurement_type = "stability"` or `measurement_type = "optimum_condition"`. If the same enzyme-mycotoxin record already has explicit kinetic or degradation metrics, you may attach `stability_*` and `optimum_*` auxiliary fields when the source clearly refers to the same enzyme system and assay target. Do not attach ABTS/guaiacol/DMP activity optimum to mycotoxin records unless explicitly stated as the mycotoxin transformation/degradation optimum.

Reject non-enzymatic physical/chemical/material systems from final records: magnetic beads alone, graphene oxide, nanocomposites, adsorbents, membranes, dialysis, UV/light/photolysis, ozone, plasma, MOF, photocatalysts, PMS/PDS/AOP, chemical catalysts, or binding-only systems. Keep immobilized enzymes only when a biological enzyme is explicitly present.

If a table/caption/footnote condition applies globally only by assumption, do not propagate it globally. Propagate it only when the source explicitly states it applies to the whole table/experiment and all rows share the same measurement type.

For table header multipliers, preserve `kinetic_unit_multiplier` and `kinetic_unit_source_text` when provided by a student extractor. If a header reports values in 10^3 M^-1 min^-1 and the cell value is 2.48, the final `kcat_Km_value` must be 2480 with `kinetic_unit_multiplier: 1000`. Do not rewrite this as "multiplied by 10".

### Targeted Semantic Guard Patch v8.3

Before accepting a student record, check whether the metric semantics match the target field. `degradation_efficiency` is only for chemical toxin reduction, residual toxin reduction, conversion, transformation, disappearance, removal, or degradation measured by a toxin assay such as HPLC, UPLC, LC-MS/LC-MS/MS, GC-MS, TLC, ELISA toxin concentration, fluorescence/UV toxin concentration, or an explicit degradation formula.

Do not use residual bioluminescence, ecotoxicity, cytotoxicity, cell viability, inhibition rate, LDH, ROS, DNA damage, animal performance, tissue residue, gut microbiota, biomarker enzyme activity, or residual toxicity endpoints as `degradation_efficiency`. If a table says "Ecotoxicity of reaction media" and the column is "Residual Bioluminescence, %", skip it from primary records.

If `degradation_efficiency` has a value, keep its unit, usually `%`. Preserve qualifiers in `notes`: `>95%`, `more than 95%`, `about 90%`, and `less than 20%` must not be silently converted into exact values.

Only keep `products` when the current paper directly identifies or measures products in its own experiment, such as current-paper LC-MS/MS, GC-MS, NMR, HPLC peak identification, m/z, fragment ions, molecular formula, or structure evidence. Do not copy products from Introduction, literature review, Discussion citations, "our previous study", "has been reported", "previously reported", or references. Prior-literature products may be mentioned in `notes` only.

Non-mycotoxin support assays such as ABTS, pNPP, paraoxon, and dye decolorization are secondary enzyme activity evidence, not primary mycotoxin kinetic/degradation records. Cell viability, LDH, ROS, DNA damage, and similar endpoints are cell-level detoxification evidence, not degradation efficiency. Since the final v8 output is primary `kinetic` or `degradation`, skip unsupported secondary evidence instead of forcing it into primary fields.

Distinguish reaction temperature from pretreatment temperature. If enzyme, lysate, crude extract, supernatant, or material was pre-incubated, heat-treated, boiled, or stored before the toxin assay, that temperature is stability pretreatment context, not ordinary degradation reaction temperature.

Do not calculate exact values not directly reported by the paper. Do not derive WT degradation efficiency from mutant values and fold-change statements. Fold-change may appear in `notes`, not as a primary measured value.

Keep enzyme system contexts separate: purified recombinant enzyme, crude lysate, cell-free extract, culture supernatant, fermentation supernatant, whole cell, immobilized enzyme, enzyme nanocomplex, and commercial enzyme. `enzyme_state` must reflect the actual form used in that measurement; purified recombinant enzyme is not immobilized unless the measurement used an immobilized preparation.

Preserve transformation type in `notes` or product context: phosphorylation, conjugation, glycosylation/glucosylation, acetylation, hydrolysis, oxidation/reduction, etc. Do not generalize every transformation to hydrolysis.

Do not convert thermodynamic analysis temperature rows into multiple primary Michaelis kinetic records unless each row is clearly an independent fitted mycotoxin kinetic experiment. Do not duplicate one kinetic fit across multiple thermodynamic temperature rows.

Before final output, if a retained row still has uncertainty, set `human_review_required: true` and explain it in `notes`, using phrases such as `wrong_metric_type_risk`, `product_from_prior_literature_risk`, `possible_table_coverage_FN`, or `stability_pretreatment_context`. If the schema has no suitable field, do not invent one; preserve the detail in `notes` and source evidence.

### ⚠️ CRITICAL: Fold Changes vs Absolute Values

**NEVER fill fold-change numbers (45-fold, 2×, 3× higher) into Km/kcat/kcat_Km fields!**

- If article ONLY says "45-fold increase" → Set kinetic field to `null`
- Only fill kinetic fields if article provides absolute values with proper units

### 📋 Required Output Schema (use EXACTLY these fields):

```json
[
  {{
    "enzyme_name": "Laccase",
    "enzyme_full_name": "Laccase from Trametes versicolor",
    "enzyme_type": "oxidoreductase",

    "ec_number": "1.10.3.2",
    "gene_name": "lacA",

    "uniprot_id": "Q9HDR6",
    "genbank_id": "AB123456",
    "pdb_id": "1GYC",
    "sequence": "MKTLV...",

    "organism": "Trametes versicolor",
    "strain": "MTCC 5155",
    "is_recombinant": true,
    "is_wild_type": false,
    "mutations": "E100A",

    "substrate": "Aflatoxin B1",
    "substrate_smiles": null,
    "substrate_concentration": null,

    "Km_value": 10.5,
    "Km_unit": "μM",
    "kcat_value": 120.0,
    "kcat_unit": "s⁻¹",
    "kcat_Km_value": 11428571.0,
    "kcat_Km_unit": "M⁻¹s⁻¹",
    "kinetic_unit_multiplier": null,
    "kinetic_unit_source_text": null,

    "degradation_efficiency": null,
    "reaction_time_value": null,
    "reaction_time_unit": null,

    "products": [{{"name": "AFQ1", "toxicity_change": "less toxic"}}],

    "temperature_value": 30.0,
    "temperature_unit": "°C",
    "ph": 5.0,
    "optimal_ph": "5.0",
    "optimal_temperature_value": 30.0,
    "optimal_temperature_unit": "°C",

    "notes": "Purified enzyme, immobilized on chitosan beads",
    "confidence_score": 5,

    "enzyme_state": "immobilized",
    "sequence_availability": "database_id",

    "measurement_type": "kinetic",
    "measurement_context_id": "doi|enzyme|substrate|kinetic|pH5.0|30C",
    "condition_scope": "kinetic_assay",
    "reported_enzyme_name": "Laccase",
    "canonical_enzyme_name": "Laccase",
    "enrichment_status": "not_attempted",
    "human_review_required": false,
    "error_flags": [],

    "kinetic_temperature_value": 30.0,
    "kinetic_temperature_unit": "°C",
    "kinetic_ph": 5.0,
    "kinetic_time_value": null,
    "kinetic_time_unit": null,

    "degradation_temperature_value": null,
    "degradation_temperature_unit": null,
    "degradation_ph": null,
    "degradation_time_value": null,
    "degradation_time_unit": null,

    "stability_note": null,
    "stability_metric": null,
    "stability_value": null,
    "stability_unit": null,
    "stability_temperature_value": null,
    "stability_temperature_unit": null,
    "stability_time_value": null,
    "stability_time_unit": null,

    "activity_assay_temperature_value": null,
    "activity_assay_temperature_unit": null,
    "activity_assay_ph": null,
    "activity_assay_time_value": null,
    "activity_assay_time_unit": null,

    "optimum_temperature_value": null,
    "optimum_temperature_unit": null,
    "optimum_temperature_range": null,
    "optimum_ph": null,
    "optimum_ph_range": null,
    "optimum_condition_target": null
  }}
]
```

### Field Types:
- `products`: Array of objects with `name` and `toxicity_change`
- `enzyme_state`: free|immobilized|crude|partially_purified|cell_free|commercial
- `sequence_availability`: full_sequence|database_id|gene_name_only|none
- `measurement_type`: kinetic|degradation
- `condition_scope`: kinetic_assay|degradation_assay|unknown
- `confidence_score`: 1-3 (3=highest quality, auto-calculated by system)
- Use `null` for missing values, never omit fields
"""

        return prompt

    def _parse_response(self, response: str) -> List[Dict]:
        """
        解析LLM返回的JSON结果

        注意：会过滤掉所有以_开头的内部字段（如_aggregation_notes、_model_comparison等）
        """
        content = response
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
        content = str(content or "").strip()

        try:
            parsed = self._parse_json_payload(content)
            records = self._records_from_parsed_payload(parsed)
            if not isinstance(records, list):
                records = []

            # 过滤掉内部字段（以_开头的字段）
            cleaned_records = []
            internal_fields = {'_aggregation_notes', '_model_comparison', '_confidence', '_source_location', '_ambiguity_flag'}

            for record in records:
                if not isinstance(record, dict):
                    continue
                cleaned_record = {k: v for k, v in record.items() if k not in internal_fields}
                cleaned_records.append(cleaned_record)

            return cleaned_records
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response content: {content[:500]}")
            return []

    def _parse_json_payload(self, content: str) -> Any:
        """
        Parse JSON from common LLM response shapes.

        The teacher model sometimes wraps JSON in prose or emits a top-level
        object such as {"records": [...]}. Use strict json.loads first, then
        fall back to fenced blocks and raw_decode from likely JSON starts.
        """
        import re

        candidates = [content]
        candidates.extend(re.findall(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE))

        decoder = json.JSONDecoder()
        last_error = None
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc

            for idx, char in enumerate(candidate):
                if char not in "[{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[idx:])
                    return parsed
                except json.JSONDecodeError as exc:
                    last_error = exc

        if last_error:
            raise last_error
        raise json.JSONDecodeError("No JSON payload found", content, 0)

    def _records_from_parsed_payload(self, payload: Any) -> List[Dict]:
        """Normalize accepted teacher JSON envelopes into a record list."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("records", "aggregated_records", "final_records", "data", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            if any(not str(k).startswith("_") for k in payload.keys()):
                return [payload]
        return []

    def _build_fallback_records(self, model_results: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Conservative fallback used only when teacher JSON cannot be parsed.

        Prefer the first non-empty student output and de-duplicate exact
        measurement-context duplicates. This preserves extracted evidence
        instead of turning a table-heavy paper into zero final records.
        """
        fallback_records: List[Dict] = []
        for records in model_results.values():
            if records:
                fallback_records = records
                break

        deduped: List[Dict] = []
        seen = set()
        for record in fallback_records:
            if not isinstance(record, dict):
                continue
            key = (
                record.get("reported_enzyme_name") or record.get("enzyme_name"),
                record.get("substrate"),
                record.get("measurement_type"),
                record.get("condition_scope"),
                record.get("Km_value"),
                record.get("kcat_value"),
                record.get("kcat_Km_value"),
                record.get("degradation_efficiency"),
            )
            if key in seen:
                continue
            seen.add(key)
            cleaned = dict(record)
            cleaned["human_review_required"] = True
            flags = cleaned.get("error_flags") or []
            if isinstance(flags, str):
                flags = [f.strip() for f in flags.replace("|", ";").split(";") if f.strip()]
            if "teacher_aggregation_parse_fallback" not in flags:
                flags.append("teacher_aggregation_parse_fallback")
            cleaned["error_flags"] = flags
            notes = cleaned.get("notes") or ""
            if "Teacher aggregation JSON parse fallback" not in notes:
                cleaned["notes"] = (notes + " | " if notes else "") + "Teacher aggregation JSON parse fallback; verify against source."
            deduped.append(cleaned)

        return deduped
    
    def _post_validate_records(self, records: List[Dict]) -> List[Dict]:
        """
        后处理验证：检查常见错误
        
        1. 检查kinetic参数是否有单位（防止fold change误填）
        2. 检查数值是否合理（防止数量级错误）
        3. 添加警告标记
        
        Args:
            records: 聚合后的记录
            
        Returns:
            验证并修正后的记录
        """
        validated_records = []
        
        # 定义合理的数值范围（用于异常检测）
        REASONABLE_RANGES = {
            'Km_value': (1e-6, 1000),       # 1 nM - 1 M
            'kcat_value': (0.001, 10000),   # 0.001 s⁻¹ - 10000 s⁻¹
            'kcat_Km_value': (0.001, 1e9),  # 0.001 M⁻¹s⁻¹ - 1e9 M⁻¹s⁻¹ (允许小于1的值)
        }

        def normalize_unit(unit: str) -> str:
            text = str(unit or "").strip().lower()
            replacements = {
                "⁻": "-",
                "−": "-",
                "–": "-",
                "¹": "1",
                "²": "2",
                "³": "3",
                "µ": "u",
                "μ": "u",
                "·": "",
                " ": "",
            }
            for src, dst in replacements.items():
                text = text.replace(src, dst)
            return text.replace("^-", "-")

        # 单位必须匹配的模式
        REQUIRED_UNITS = {
            'Km_unit': ['μM', 'mM', 'nM', 'M', 'uM', 'µM'],
            'kcat_unit': ['s⁻¹', 's-1', 'min⁻¹', 'min-1', 's^-1', 'min^-1', '/s', '/min'],
            'kcat_Km_unit': ['M⁻¹s⁻¹', 'M-1s-1', 'mM⁻¹s⁻¹', 'mM-1s-1',
                           'M^-1s^-1', 'mM^-1s^-1', '/M/s', '/mM/s',
                           's⁻¹mM⁻¹', 's^-1mM^-1', 's-1mM-1',
                           # 分钟单位（会被后续转换）
                           'M⁻¹ min⁻¹', 'M-1min-1', 'mM⁻¹ min⁻¹', 'mM-1min-1',
                           'M^-1min^-1', 'mM^-1min^-1', '/M/min', '/mM/min',
                           'min⁻¹mM⁻¹', 'min^-1mM^-1', 'min-1mM-1'],
        }
        
        for record in records:
            issues = []
            
            # 检查每个kinetic参数
            for value_field in ['Km_value', 'kcat_value', 'kcat_Km_value']:
                value = record.get(value_field)
                unit_field = value_field.replace('_value', '_unit')
                unit = record.get(unit_field)
                
                # 规则1：有值但无单位 → 标记但不删除（保留提取的数值）
                if value is not None and value != '' and (unit is None or unit == ''):
                    issues.append(f"⚠️ {value_field} has value ({value}) but missing unit")
                    logger.warning(f"  ⚠️ Record validation: {value_field}={value} has no unit")
                    # 不删除数值 — 由quality_tier的metric_unit字段组自然惩罚
                    notes_field = '_aggregation_notes' if not self.optimized else 'notes'
                    notes = record.get(notes_field, '')
                    record[notes_field] = notes + f" | WARNING: {value_field}={value} has no unit (preserved for review)"
                
                # 规则2：有值有单位，但单位不在允许列表中
                elif value is not None and unit is not None:
                    allowed_units = REQUIRED_UNITS.get(unit_field, [])
                    normalized_unit = normalize_unit(unit)
                    normalized_allowed_units = {normalize_unit(u) for u in allowed_units}
                    if allowed_units and normalized_unit not in normalized_allowed_units:
                        issues.append(f"⚠️ {value_field}={value} has invalid unit '{unit}'")
                        logger.warning(f"  ⚠️ Invalid unit: {value_field}={value} {unit}")
                        # 标记但不自动清空（可能是新单位格式）
                        if '_ambiguity_flag' not in record or not record['_ambiguity_flag']:
                            record['_ambiguity_flag'] = f"Unusual unit format: {unit_field}='{unit}'"
                
                # 规则3：数值超出合理范围
                if value is not None:
                    try:
                        val_float = float(value)
                        min_val, max_val = REASONABLE_RANGES.get(value_field, (0, 1e10))
                        if val_float < min_val or val_float > max_val:
                            issues.append(f"⚠️ {value_field}={value} outside typical range ({min_val}-{max_val})")
                            logger.warning(f"  ⚠️ Unusual value: {value_field}={value} (typical: {min_val}-{max_val})")
                            # 标记但不清空（可能是真实极端值）
                            if '_ambiguity_flag' not in record or not record['_ambiguity_flag']:
                                record['_ambiguity_flag'] = f"Unusual value: {value_field}={value} (check if correct)"
                    except (ValueError, TypeError):
                        pass
            
            # 规则4：检查fold change关键词泄露到notes以外的地方
            fold_keywords = ['fold', 'times', 'increase', 'higher than', '×', 'x']
            notes_fields = ['notes']
            if not self.optimized:
                notes_fields.append('_aggregation_notes')
            for field in ['enzyme_name', 'enzyme_full_name', 'substrate'] + notes_fields:
                field_value = str(record.get(field, ''))
                if any(kw in field_value.lower() for kw in fold_keywords):
                    # notes字段允许出现fold，其他字段不允许
                    if field not in notes_fields:
                        issues.append(f"⚠️ Field '{field}' contains fold-change keywords: '{field_value}'")
            
            # 如果有问题，降低置信度
            if issues:
                original_confidence = record.get('_confidence', 'medium')
                if original_confidence == 'high':
                    record['_confidence'] = 'medium'
                    logger.info(f"  → Downgraded confidence from 'high' to 'medium' due to validation issues")
                
                # 汇总问题到_ambiguity_flag
                if issues:
                    existing_flag = record.get('_ambiguity_flag', '')
                    new_flag = '; '.join(issues)
                    record['_ambiguity_flag'] = f"{existing_flag}; {new_flag}" if existing_flag else new_flag
            
            validated_records.append(record)
        
        return validated_records


    def _parse_tool_call(self, response: str) -> Optional[Dict]:
        """
        解析LLM响应中的工具调用请求
        
        Args:
            response: LLM响应
            
        Returns:
            工具调用字典，如果没有工具调用返回None
        """
        import re
        
        content = response
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
        
        # 查找JSON格式的工具调用
        tool_call_pattern = r'\{\s*"tool_call"\s*:\s*\{[^}]+\}\s*\}'
        match = re.search(tool_call_pattern, content, re.DOTALL)
        
        if match:
            try:
                tool_request = json.loads(match.group(0))
                return tool_request.get('tool_call')
            except json.JSONDecodeError:
                logger.warning("  ⚠️ Found tool_call pattern but failed to parse JSON")
                return None
        
        return None
    
    def _execute_tool(self, tool_call: Dict) -> Dict:
        """
        执行工具调用
        
        Args:
            tool_call: {"name": "verify_table_image", "arguments": {"table_id": "Table 1", "question": "..."}}
            
        Returns:
            工具执行结果
        """
        tool_name = tool_call.get('name')
        arguments = tool_call.get('arguments', {})
        
        if tool_name == 'get_table_with_image':
            return self._get_table_with_image(arguments.get('table_id', ''))
        elif tool_name == 'verify_table_image':
            return self._verify_table_image(
                arguments.get('table_id', ''),
                arguments.get('question', '')
            )
        else:
            logger.warning(f"  ⚠️ Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _get_table_with_image(self, table_id: str) -> Dict:
        """
        获取指定表格的完整信息（HTML + 图片）
        
        Args:
            table_id: 表格标识（如 "Table 1", "Table 2"）
            
        Returns:
            {
                "table_id": str,
                "caption": str,
                "html_content": str,
                "image_path": str,
                "footnotes": str
            }
        """
        if not self.paper_blocks:
            return {
                "error": "No paper_blocks available",
                "table_id": table_id
            }
        
        def get_caption_text(block):
            """从block中提取caption文本"""
            caption = block.get('table_caption') or block.get('caption', '')
            if isinstance(caption, list):
                return ' '.join(str(c) for c in caption)
            return str(caption)
        
        # 查找匹配的表格块
        table_block = None
        all_tables = [b for b in self.paper_blocks if b.get('type') == 'table']
        
        # 1. 精确匹配 "Table X"
        for block in all_tables:
            caption = get_caption_text(block)
            if table_id.lower() in caption.lower():
                table_block = block
                break
        
        # 2. 尝试数字匹配
        if not table_block:
            table_number = ''.join(filter(str.isdigit, table_id))
            if table_number:
                for block in all_tables:
                    caption = get_caption_text(block)
                    # 匹配 "Table 1", "Table1", "表1" 等
                    if f"table {table_number}" in caption.lower() or f"table{table_number}" in caption.lower():
                        table_block = block
                        break
        
        # 3. 如果还没找到，按顺序返回第 N 个表格
        if not table_block:
            table_number = ''.join(filter(str.isdigit, table_id))
            if table_number:
                idx = int(table_number) - 1  # Table 1 对应索引 0
                if 0 <= idx < len(all_tables):
                    table_block = all_tables[idx]
                    logger.info(f"  📋 Using table by index: Table {table_number} -> index {idx}")
        
        if not table_block:
            logger.warning(f"  ⚠️ Table not found: {table_id}")
            available = []
            for b in self.paper_blocks:
                if b.get('type') == 'table':
                    cap = b.get('table_caption') or b.get('caption', 'Unknown')
                    if isinstance(cap, list):
                        cap = ' '.join(str(c) for c in cap)
                    available.append(str(cap)[:50])
            return {
                "error": f"Table not found: {table_id}",
                "table_id": table_id,
                "available_tables": available
            }
        
        # 提取表格信息
        caption_text = get_caption_text(table_block)
        footnote = table_block.get('table_footnote') or table_block.get('footnote', '')
        if isinstance(footnote, list):
            footnote = ' '.join(str(f) for f in footnote)
        
        result = {
            "table_id": table_id,
            "caption": caption_text,
            "html_content": (table_block.get('table_body') or table_block.get('content', ''))[:2000],
            "footnotes": footnote,
            "image_path": None
        }
        
        # 获取表格图片路径
        img_path = (
            table_block.get('img_path') or 
            table_block.get('image_path') or 
            table_block.get('table_img')
        )
        
        if img_path and self.paper_dir:
            full_image_path = self.paper_dir / img_path
            if full_image_path.exists():
                result["image_path"] = str(full_image_path)
                result["image_available"] = True
            else:
                # 尝试在images子目录查找
                alt_path = self.paper_dir / 'images' / Path(img_path).name
                if alt_path.exists():
                    result["image_path"] = str(alt_path)
                    result["image_available"] = True
                else:
                    result["image_available"] = False
                    result["image_error"] = f"Image not found: {img_path}"
        
        logger.info(f"  ✓ Retrieved table info: {table_id}")
        logger.info(f"    Caption: {result['caption'][:80]}...")
        logger.info(f"    Image: {'Available' if result.get('image_available') else 'Not found'}")
        
        return result

    def _verify_table_image(self, table_id: str, question: str) -> Dict:
        """
        使用当前配置的teacher/vision-capable模型直接验证表格图片
        
        当发现模型结果冲突时，调用此工具让配置的模型直接"看"表格图片，
        验证具体的数值。
        
        Args:
            table_id: 表格标识（如 "Table 1", "Table 2"）
            question: 需要验证的具体问题（如 "What is the Km value for AFB1?"）
            
        Returns:
            {
                "table_id": str,
                "question": str,
                "answer": str,  # 配置模型看图后的回答
                "confidence": str
            }
        """
        import os
        
        # 先获取表格信息
        table_info = self._get_table_with_image(table_id)
        
        if table_info.get('error'):
            return table_info
        
        if not table_info.get('image_available'):
            return {
                "table_id": table_id,
                "question": question,
                "error": "Table image not available",
                "fallback": "Using HTML content for verification",
                "html_content": table_info.get('html_content', '')[:1500]
            }
        
        image_path = table_info.get('image_path')
        if not image_path or not os.path.exists(image_path):
            return {
                "table_id": table_id,
                "question": question,
                "error": f"Image file not found: {image_path}"
            }
        
        # 构建验证 prompt
        verification_prompt = f"""Please examine this table image carefully and answer the following question:

**Question:** {question}

**Context:**
- Table caption: {table_info.get('caption', 'N/A')}
- Table footnotes: {table_info.get('footnotes', 'N/A')[:500]}

**Instructions:**
1. Look at the table image directly
2. Find the relevant data
3. Report the exact values you see (including units)
4. If you cannot find the answer, say "Not found in table"
5. Be precise with numbers - report exactly what you see

**Your answer:**"""

        logger.info(f"  🔍 Verifying table image: {table_id}")
        logger.info(f"    Question: {question[:80]}...")
        
        try:
            answer = self.llm_client.chat(
                messages=[
                    {
                        "role": "user",
                        "text": verification_prompt,
                        "image_path": image_path,
                    }
                ],
                is_multimodal=True,
                max_tokens=500,
                temperature=0.1,
                task="aggregation_table_image_verification",
            )
            
            logger.info("  ✓ Image verification complete")
            logger.info(f"    Answer: {answer[:100]}...")
            
            return {
                "table_id": table_id,
                "question": question,
                "answer": answer,
                "source": f"{self.model_name} direct image analysis",
                "confidence": "high"
            }
            
        except Exception as e:
            logger.error(f"  ✗ Image verification failed: {e}")
            return {
                "table_id": table_id,
                "question": question,
                "error": f"Vision API call failed: {e}",
                "fallback": "Using HTML content",
                "html_content": table_info.get('html_content', '')[:1500]
            }


def test_aggregation_agent():
    """测试函数"""
    from src.llm_clients.providers import build_client
    
    # 初始化
    gpt51_client = build_client("openai", "gpt-5")
    agent = AggregationAgent(gpt51_client)
    
    # 模拟数据
    original_text = """
    The kinetic parameters of the purified enzyme were determined.
    The Km value was found to be 0.073 mM for AFB1, and the kcat
    was 0.65 s⁻¹ at pH 7.0 and 25°C.
    """
    
    model_results = {
        "kimi": [
            {"enzyme_name": "Esterase", "Km_value": 0.073, "Km_unit": "mM", "substrate": "AFB1"}
        ],
        "deepseek": [
            {"enzyme_name": "Esterase", "kcat_value": 0.65, "kcat_unit": "s⁻¹", "pH": 7.0}
        ],
        "glm-4.6": [
            {"enzyme_name": "Esterase", "Km_value": 0.073, "Km_unit": "mM", 
             "kcat_value": 0.65, "kcat_unit": "s⁻¹", "substrate": "AFB1", "pH": 7.0}
        ]
    }
    
    # 聚合
    result = agent.aggregate(original_text, model_results, doi="10.1234/test")
    
    print("Aggregated result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_aggregation_agent()
