"""
Aggregation Agent - 聚合多个模型的提取结果

基于文献思路：使用最强模型（GPT-5.1/Claude 3.5）作为"裁判"，
对比多个"学生"模型的结果，结合原文进行智能聚合。

参考：
- 输入：原文 + 多个模型的提取结果
- 输出：聚合后的最优结果
- 模型：GPT-5.1 (或 Claude 3.5 Sonnet)
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
    - 用强模型做聚合（GPT-5.1）
    
    新功能：工具调用能力
    - 当发现模型冲突时，可以主动调用工具查看原始表格（含图片）
    - 支持ReAct式推理：发现问题 → 调用工具 → 基于工具结果做决策
    """
    
    def __init__(self, llm_client, model_name: str = "GPT-5.1", extraction_prompt: str = None, paper_dir: Optional[Path] = None, optimized: bool = False):
        """
        Args:
            llm_client: 强大的LLM客户端（GPT-5.1或Claude 3.5）
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
        prompt_path = "prompts/prompts_extract_from_text.txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Extract enzyme kinetics data including Km, kcat, substrate, pH, temperature."
    
    def aggregate(
        self,
        original_text: str,
        model_results: Dict[str, List[Dict]],
        doi: str = "unknown",
        paper_blocks: Optional[List[Dict]] = None
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
            
        Returns:
            聚合后的最优记录列表
        """
        logger.info(f"[Aggregation Agent] Starting aggregation for {doi}")
        logger.info(f"  - Models: {list(model_results.keys())}")
        logger.info(f"  - Total records: {sum(len(r) for r in model_results.values())}")
        
        # 存储paper_blocks供工具调用使用
        self.paper_blocks = paper_blocks
        
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
                    
                    # 后处理验证
                    aggregated_records = self._post_validate_records(aggregated_records)
                    
                    logger.info(f"  ✓ Aggregated to {len(aggregated_records)} final records")
                    return aggregated_records
                    
            except Exception as e:
                logger.error(f"  ✗ Aggregation iteration {iteration + 1} failed: {e}")
                if iteration == max_iterations - 1:
                    # 最后一次尝试也失败，降级
                    fallback_model = list(model_results.keys())[0]
                    logger.warning(f"  ⚠️ Falling back to {fallback_model} results")
                    return model_results[fallback_model]
        
        # 超过最大迭代次数，降级
        logger.warning(f"  ⚠️ Max iterations reached, falling back")
        fallback_model = list(model_results.keys())[0]
        return model_results[fallback_model]
    
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
                    key_fields = ['enzyme_name', 'substrate', 'Km_value', 'kcat_value', 'kcat_Km_value',
                                  'organism', 'temperature_value', 'ph']
                    fields_str = ", ".join([f"{k}={record.get(k)}" for k in key_fields if record.get(k)])
                    model_outputs += fields_str + "\n"

        # 精简版prompt
        prompt = f"""# Enzyme Data Aggregation

## Article Text
```
{original_text[:80000]}{"[...truncated...]" if text_truncated else ""}
```

## Model Outputs to Aggregate
{model_outputs}

## Your Task

Aggregate extraction results from Kimi, DeepSeek, GLM-4.6 into final records.

### Core Rules:

1. **VERIFY values against article text** - Only use values explicitly stated in the article
2. **RESOLVE conflicts** - Choose the value supported by the article
3. **MERGE complementary data** - Combine different fields from different models
4. **REMOVE duplicates** - One record per unique enzyme-substrate pair
5. **CORRECT obvious errors** - Fix units, decimals, etc.

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

    "thermal_stability": null,
    "thermal_stability_unit": null,
    "thermal_stability_time": null,
    "thermal_stability_time_unit": null,

    "notes": "Purified enzyme, immobilized on chitosan beads",
    "confidence_score": 5,

    "enzyme_state": "immobilized",
    "sequence_availability": "database_id"
  }}
]
```

### Field Types:
- `products`: Array of objects with `name` and `toxicity_change`
- `enzyme_state`: free|immobilized|crude|partially_purified|cell_free|commercial
- `sequence_availability`: full_sequence|database_id|gene_name_only|none
- `confidence_score`: 1-3 (3=highest quality, auto-calculated by system)
- Use `null` for missing values, never omit fields
"""

        return prompt

    def _parse_response(self, response: str) -> List[Dict]:
        """
        解析LLM返回的JSON结果

        注意：会过滤掉所有以_开头的内部字段（如_aggregation_notes、_model_comparison等）
        """
        # 移除markdown代码块标记
        import re

        content = response
        if isinstance(response, dict) and 'content' in response:
            content = response['content']

        # 提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # 解析
        try:
            records = json.loads(content)
            if isinstance(records, dict):
                records = [records]

            # 过滤掉内部字段（以_开头的字段）
            cleaned_records = []
            internal_fields = {'_aggregation_notes', '_model_comparison', '_confidence', '_source_location', '_ambiguity_flag'}

            for record in records:
                cleaned_record = {k: v for k, v in record.items() if k not in internal_fields}
                cleaned_records.append(cleaned_record)

            return cleaned_records
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response content: {content[:500]}")
            return []
    
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
            'Vmax_value': (0.001, 10000),   # 0.001 - 10000 (各种单位)
            'kcat_Km_value': (0.001, 1e9),  # 0.001 M⁻¹s⁻¹ - 1e9 M⁻¹s⁻¹ (允许小于1的值)
        }

        # 单位必须匹配的模式
        REQUIRED_UNITS = {
            'Km_unit': ['μM', 'mM', 'nM', 'M', 'uM', 'µM'],
            'kcat_unit': ['s⁻¹', 's-1', 'min⁻¹', 'min-1', 's^-1', 'min^-1', '/s', '/min'],
            'Vmax_unit': ['μM/min', 'mM/min', 'nmol/min/mg', 'μmol/min/mg', 'U/mg',
                         'umol/min', 'nmol/min', 'μmol/min'],
            'kcat_Km_unit': ['M⁻¹s⁻¹', 'M-1s-1', 'mM⁻¹s⁻¹', 'mM-1s-1',
                           'M^-1s^-1', 'mM^-1s^-1', '/M/s', '/mM/s',
                           # 分钟单位（会被后续转换）
                           'M⁻¹ min⁻¹', 'M-1min-1', 'mM⁻¹ min⁻¹', 'mM-1min-1',
                           'M^-1min^-1', 'mM^-1min^-1', '/M/min', '/mM/min'],
        }
        
        for record in records:
            issues = []
            
            # 检查每个kinetic参数
            for value_field in ['Km_value', 'kcat_value', 'Vmax_value', 'kcat_Km_value']:
                value = record.get(value_field)
                unit_field = value_field.replace('_value', '_unit')
                unit = record.get(unit_field)
                
                # 规则1：有值但无单位 → 可能是fold change误填
                if value is not None and value != '' and (unit is None or unit == ''):
                    issues.append(f"⚠️ {value_field} has value ({value}) but missing unit - possible fold-change misuse!")
                    logger.warning(f"  ⚠️ Record validation: {value_field}={value} has no unit")
                    # 自动修正：清空该值
                    record[value_field] = None
                    record[unit_field] = None

                    # 添加到notes（根据优化模式选择字段）
                    notes_field = '_aggregation_notes' if not self.optimized else 'notes'
                    notes = record.get(notes_field, '')
                    record[notes_field] = notes + f" | AUTO-CORRECTED: Removed {value_field}={value} (no unit, likely fold-change)"
                
                # 规则2：有值有单位，但单位不在允许列表中
                elif value is not None and unit is not None:
                    allowed_units = REQUIRED_UNITS.get(unit_field, [])
                    if allowed_units and unit not in allowed_units:
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
        使用 GPT-5.1 多模态能力直接验证表格图片
        
        当发现模型结果冲突时，调用此工具让 GPT-5.1 直接"看"表格图片，
        验证具体的数值。
        
        Args:
            table_id: 表格标识（如 "Table 1", "Table 2"）
            question: 需要验证的具体问题（如 "What is the Km value for AFB1?"）
            
        Returns:
            {
                "table_id": str,
                "question": str,
                "answer": str,  # GPT-5.1 看图后的回答
                "confidence": str
            }
        """
        import base64
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
        
        # 读取并编码图片
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            return {
                "table_id": table_id,
                "question": question,
                "error": f"Failed to read image: {e}"
            }
        
        # 确定图片格式
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        
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
            # 使用 OpenAI Vision API 格式直接调用
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": verification_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            logger.info(f"  ✓ Image verification complete ({tokens_used} tokens)")
            logger.info(f"    Answer: {answer[:100]}...")
            
            return {
                "table_id": table_id,
                "question": question,
                "answer": answer,
                "source": "GPT-5.1 Vision (direct image analysis)",
                "tokens_used": tokens_used,
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
