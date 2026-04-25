"""
交叉验证Agent (Cross-Validator)

目标：独立验证提取的数据是否与原文一致

工作流：
1. 接收提取的记录 + 原文段落
2. 用GPT-5.1逐字段检查："原文中Km是多少？" → 比对
3. 输出验证结果：MATCH / MISMATCH / UNCERTAIN
4. 只有MATCH才能APPROVED

优势：
- 双重检查：提取一次，验证一次
- 减少幻觉：验证阶段专注找原文证据
- 高准确率：不一致的自动送HITL
"""

import re
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """单个字段的验证结果"""
    field: str
    extracted_value: any
    found_in_text: any
    status: str  # "MATCH", "MISMATCH", "UNCERTAIN", "NOT_FOUND"
    evidence: str  # 原文证据
    confidence: float


@dataclass
class CrossValidationResult:
    """整条记录的交叉验证结果"""
    record_id: str
    overall_status: str  # "PASS", "FAIL", "UNCERTAIN"
    field_validations: List[ValidationResult]
    mismatch_count: int
    match_count: int
    reasoning: str


class CrossValidatorAgent:
    """
    交叉验证Agent
    
    用另一个LLM独立验证提取的数据是否与原文一致
    """
    
    CRITICAL_FIELDS = [
        'Km_value', 'Km_unit',
        'Vmax_value', 'Vmax_unit', 
        'kcat_value', 'kcat_unit',
        'kcat_Km_value', 'kcat_Km_unit',
        'ph', 'temperature_value', 'temperature_unit',
        'substrate', 'enzyme_name', 'organism'
    ]
    
    def __init__(self, llm_client, config: Dict = None):
        self.llm_client = llm_client
        self.config = config or {}
    
    async def validate_batch(
        self, 
        records: List[Dict], 
        content_list: List[Dict]
    ) -> List[CrossValidationResult]:
        """
        批量验证记录
        
        Args:
            records: 提取的记录列表
            content_list: 论文的content_list
            
        Returns:
            验证结果列表
        """
        logger.info(f"[CrossValidator] Validating {len(records)} records...")
        
        results = []
        
        # 批量处理（3-5条一组）
        batch_size = 3
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            batch_results = await self._validate_batch_parallel(batch, content_list)
            results.extend(batch_results)
        
        # 统计
        passed = sum(1 for r in results if r.overall_status == "PASS")
        failed = sum(1 for r in results if r.overall_status == "FAIL")
        uncertain = sum(1 for r in results if r.overall_status == "UNCERTAIN")
        
        logger.info(f"  Validation: PASS={passed}, FAIL={failed}, UNCERTAIN={uncertain}")
        
        return results
    
    async def _validate_batch_parallel(
        self, 
        records: List[Dict], 
        content_list: List[Dict]
    ) -> List[CrossValidationResult]:
        """并行验证一批记录"""
        tasks = []
        for record in records:
            tasks.append(self.validate_single(record, content_list))
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def validate_single(
        self, 
        record: Dict, 
        content_list: List[Dict]
    ) -> CrossValidationResult:
        """
        验证单条记录
        
        策略：
        1. 提取记录对应的原文段落
        2. 用LLM逐字段验证："原文中Km是多少？"
        3. 比对提取值和验证值
        """
        try:
            # 获取原文上下文
            context = self._extract_context(record, content_list)
            
            if not context or len(context) < 100:
                return CrossValidationResult(
                    record_id=str(record.get('id', 'unknown')),
                    overall_status="UNCERTAIN",
                    field_validations=[],
                    mismatch_count=0,
                    match_count=0,
                    reasoning="无法获取原文上下文"
                )
            
            # 构建验证提示
            prompt = self._build_validation_prompt(record, context)
            
            # 调用LLM验证
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "你是数据验证专家。请仔细对比提取的数据和原文，找出不一致之处。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # 解析响应
            content = response.get("content", "{}")
            validation_data = self._parse_validation_response(content)
            
            # 构建验证结果
            field_validations = []
            match_count = 0
            mismatch_count = 0
            
            for field_name, field_result in validation_data.items():
                status = field_result.get("status", "UNCERTAIN")
                if status == "MATCH":
                    match_count += 1
                elif status == "MISMATCH":
                    mismatch_count += 1
                
                field_validations.append(ValidationResult(
                    field=field_name,
                    extracted_value=record.get(field_name),
                    found_in_text=field_result.get("found_value"),
                    status=status,
                    evidence=field_result.get("evidence", ""),
                    confidence=field_result.get("confidence", 0.5)
                ))
            
            # 判断整体状态
            if mismatch_count > 0:
                overall_status = "FAIL"
                reasoning = f"发现 {mismatch_count} 个字段与原文不符"
            elif match_count >= 3:  # 至少3个关键字段匹配
                overall_status = "PASS"
                reasoning = f"所有 {match_count} 个关键字段与原文一致"
            else:
                overall_status = "UNCERTAIN"
                reasoning = "无法充分验证"
            
            return CrossValidationResult(
                record_id=str(record.get('id', 'unknown')),
                overall_status=overall_status,
                field_validations=field_validations,
                mismatch_count=mismatch_count,
                match_count=match_count,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return CrossValidationResult(
                record_id=str(record.get('id', 'unknown')),
                overall_status="UNCERTAIN",
                field_validations=[],
                mismatch_count=0,
                match_count=0,
                reasoning=f"验证过程出错: {e}"
            )
    
    def _extract_context(self, record: Dict, content_list: List[Dict]) -> str:
        """提取记录对应的原文段落"""
        source_info = record.get('source_in_document', {})
        block_id_raw = source_info.get('block_id')
        
        if block_id_raw is None or not content_list:
            return ""
        
        # 解析block_id
        if isinstance(block_id_raw, str):
            match = re.search(r'(\d+)$', block_id_raw)
            block_id = int(match.group(1)) if match else 0
        else:
            block_id = int(block_id_raw)
        
        if block_id >= len(content_list):
            return ""
        
        # 提取窗口（±5个块）
        window_size = 5
        start_idx = max(0, block_id - window_size)
        end_idx = min(len(content_list), block_id + window_size + 1)
        
        text_parts = []
        for block in content_list[start_idx:end_idx]:
            block_type = block.get('type', '')
            if block_type == 'text':
                text_parts.append(block.get('text', ''))
            elif block_type == 'table':
                table_text = block.get('table_body') or block.get('text', '')
                text_parts.append(str(table_text)[:2000])
        
        return '\n\n'.join(text_parts)[:8000]
    
    def _build_validation_prompt(self, record: Dict, context: str) -> str:
        """构建验证提示"""
        
        # 提取关键字段
        fields_to_check = {}
        for field in self.CRITICAL_FIELDS:
            value = record.get(field)
            if value and value not in ('', None, 'Unknown', 'N/A'):
                fields_to_check[field] = value
        
        prompt = f"""请验证以下提取的数据是否与原文一致。

## 提取的数据
"""
        for field, value in fields_to_check.items():
            prompt += f"- {field}: {value}\n"
        
        prompt += f"""

## 原文段落
{context}

## 验证任务
请逐字段检查提取的数据是否与原文一致：
1. 在原文中找到对应的数值或信息
2. 比对提取值和原文值是否完全一致
3. 注意单位、数量级、小数点

## 输出格式（JSON）
```json
{{
    "Km_value": {{
        "status": "MATCH|MISMATCH|UNCERTAIN|NOT_FOUND",
        "found_value": "原文中的值",
        "evidence": "原文证据片段（20字内）",
        "confidence": 0.0-1.0
    }},
    "kcat_value": {{...}},
    ...
}}
```

状态说明：
- MATCH: 完全一致
- MISMATCH: 不一致（这是最重要的！）
- UNCERTAIN: 无法判断
- NOT_FOUND: 原文未提及
"""
        return prompt
    
    def _parse_validation_response(self, content: str) -> Dict:
        """解析验证响应"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return json.loads(content)
        except:
            logger.warning("Failed to parse validation response")
            return {}


# ============================================================================
# 便捷函数
# ============================================================================

def create_cross_validator(llm_client, config: Dict = None) -> CrossValidatorAgent:
    """创建交叉验证Agent"""
    return CrossValidatorAgent(llm_client, config)
