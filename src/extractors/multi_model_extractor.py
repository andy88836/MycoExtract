"""
多模型提取器 (Multi-Model Extractor)

支持多个LLM并行提取，通过投票机制提高准确性

架构：
- Text/Table: 3个文本模型投票（Kimi, DeepSeek, GLM-4.6）
- Figure: 1个多模态模型（GLM-4.6V）

投票规则：
- 3/3一致 → 高置信度
- 2/3一致 → 中等置信度（保留少数意见）
- 1/1/1分歧 → 低置信度（送HITL）
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VotingResult:
    """投票结果"""
    value: Any
    votes: int
    total_models: int
    confidence: str  # "high", "medium", "low"
    alternatives: List[Dict]  # 其他候选值


class MultiModelExtractor:
    """
    多模型提取器
    
    使用多个LLM并行提取，通过投票提高准确性
    """
    
    # 关键字段（需要投票的字段）
    VOTABLE_FIELDS = [
        'Km_value', 'Km_unit',
        'Vmax_value', 'Vmax_unit',
        'kcat_value', 'kcat_unit',
        'kcat_Km_value', 'kcat_Km_unit',
        'ph', 'temperature_value', 'temperature_unit',
        'substrate', 'enzyme_name', 'organism',
        'enzyme_full_name', 'ec_number'
    ]
    
    def __init__(self, text_models: Dict[str, Any], multimodal_model: Any = None):
        """
        Args:
            text_models: 文本模型字典 {"kimi": client1, "deepseek": client2, "glm-4.6": client3}
            multimodal_model: 多模态模型客户端（可选）
        """
        self.text_models = text_models
        self.multimodal_model = multimodal_model
        logger.info(f"MultiModelExtractor initialized with {len(text_models)} text models")
    
    async def extract_from_text(
        self, 
        text_content: str, 
        prompt_template: str,
        doi: str = None,
        block_id: int = None
    ) -> Dict[str, Any]:
        """
        多模型文本提取 + 投票
        
        Args:
            text_content: 文本内容
            prompt_template: 提示词模板
            doi: 论文DOI
            block_id: 块ID
            
        Returns:
            投票后的记录 + 投票信息
        """
        logger.info(f"  [Multi-Model] Extracting with {len(self.text_models)} models...")
        
        # 并行调用所有模型
        tasks = []
        model_names = []
        for name, client in self.text_models.items():
            tasks.append(self._extract_single(client, text_content, prompt_template, name))
            model_names.append(name)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤失败的结果
        valid_results = []
        valid_names = []
        for name, result in zip(model_names, results):
            if isinstance(result, Exception):
                logger.warning(f"    {name} extraction failed: {result}")
            elif result:
                valid_results.append(result)
                valid_names.append(name)
        
        if not valid_results:
            logger.error("  All models failed!")
            return []
        
        logger.info(f"  Successfully extracted from {len(valid_results)}/{len(self.text_models)} models")
        
        # 投票合并
        voted_records = self._vote_and_merge(valid_results, valid_names)
        
        # 添加元数据
        for record in voted_records:
            record['source_in_document'] = {
                'doi': doi,
                'source_type': 'text',
                'block_id': block_id
            }
        
        return voted_records
    
    async def extract_from_table(
        self,
        table_content: str,
        prompt_template: str,
        doi: str = None,
        block_id: int = None
    ) -> List[Dict]:
        """多模型表格提取（同文本）"""
        return await self.extract_from_text(table_content, prompt_template, doi, block_id)
    
    async def extract_from_figure(
        self,
        image_path: str,
        prompt_template: str,
        doi: str = None,
        block_id: int = None
    ) -> List[Dict]:
        """
        单模型图片提取（只有一个多模态模型）
        
        不需要投票
        """
        if not self.multimodal_model:
            logger.warning("  No multimodal model available for figure extraction")
            return []
        
        logger.info(f"  [Single-Model] Extracting from figure with multimodal model...")
        
        try:
            records = await self._extract_single_figure(
                self.multimodal_model, 
                image_path, 
                prompt_template
            )
            
            # 添加元数据
            for record in records:
                record['source_in_document'] = {
                    'doi': doi,
                    'source_type': 'figure',
                    'block_id': block_id
                }
                # 标记为单模型（无投票信息）
                record['_extraction_method'] = 'single_model'
            
            return records
            
        except Exception as e:
            logger.error(f"  Figure extraction failed: {e}")
            return []
    
    async def _extract_single(
        self, 
        client, 
        content: str, 
        prompt: str,
        model_name: str
    ) -> List[Dict]:
        """单个模型提取"""
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": "You are an expert in enzyme kinetics data extraction."},
                {"role": "user", "content": f"{prompt}\n\n{content}"}
            ]
            
            # 调用模型
            response = client.chat(messages=messages, temperature=0.1)
            
            # 解析JSON
            import re, json
            content_str = response.get("content", "")
            json_match = re.search(r'```json\s*(.*?)\s*```', content_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(content_str)
            
            records = data if isinstance(data, list) else [data]
            
            # 标记来源模型
            for record in records:
                record['_extracted_by'] = model_name
            
            return records
            
        except Exception as e:
            logger.warning(f"    {model_name} extraction error: {e}")
            raise
    
    async def _extract_single_figure(self, client, image_path: str, prompt: str) -> List[Dict]:
        """单个模型图片提取"""
        # 实现图片提取逻辑（多模态）
        # TODO: 根据你的multimodal_client实现
        pass
    
    def _vote_and_merge(
        self, 
        results: List[List[Dict]], 
        model_names: List[str]
    ) -> List[Dict]:
        """
        投票合并多个模型的提取结果
        
        策略：
        1. 按记录索引对齐（假设所有模型提取的记录数量相同）
        2. 逐字段投票
        3. 计算置信度
        """
        if not results:
            return []
        
        # 找出最大记录数
        max_records = max(len(r) for r in results)
        
        merged_records = []
        
        for record_idx in range(max_records):
            # 收集该位置所有模型的记录
            candidate_records = []
            for model_results in results:
                if record_idx < len(model_results):
                    candidate_records.append(model_results[record_idx])
            
            if not candidate_records:
                continue
            
            # 投票合并
            merged_record = self._vote_single_record(candidate_records, model_names)
            merged_records.append(merged_record)
        
        return merged_records
    
    def _vote_single_record(
        self, 
        records: List[Dict], 
        model_names: List[str]
    ) -> Dict:
        """
        单条记录的逐字段投票
        
        返回：合并后的记录 + 投票信息
        """
        merged = {}
        voting_details = {}
        consensus_fields = 0
        total_fields = 0
        
        for field in self.VOTABLE_FIELDS:
            # 收集所有模型在该字段的值
            values = []
            for record in records:
                val = record.get(field)
                if val and val not in ('', None, 'Unknown', 'N/A'):
                    values.append(val)
            
            if not values:
                continue
            
            total_fields += 1
            
            # 投票
            vote_result = self._vote_field(values, model_names[:len(records)])
            merged[field] = vote_result.value
            
            # 记录投票信息
            voting_details[field] = {
                'value': vote_result.value,
                'votes': f"{vote_result.votes}/{vote_result.total_models}",
                'confidence': vote_result.confidence,
                'alternatives': vote_result.alternatives
            }
            
            if vote_result.confidence == 'high':
                consensus_fields += 1
        
        # 添加投票元信息
        merged['_voting_info'] = {
            'models': model_names[:len(records)],
            'total_models': len(records),
            'consensus_rate': consensus_fields / total_fields if total_fields > 0 else 0,
            'details': voting_details
        }
        
        # 复制其他非投票字段（从第一个记录）
        for key, val in records[0].items():
            if key not in merged and not key.startswith('_'):
                merged[key] = val
        
        return merged
    
    def _vote_field(
        self, 
        values: List[Any], 
        model_names: List[str]
    ) -> VotingResult:
        """
        单个字段投票
        
        规则：
        - 3/3 一致 → high confidence
        - 2/3 多数 → medium confidence
        - 1/1/1 分歧 → low confidence
        """
        # 数值类型需要考虑浮点误差
        if isinstance(values[0], (int, float)):
            values = [self._normalize_number(v) for v in values]
        
        # 统计投票
        counter = Counter(values)
        most_common = counter.most_common()
        
        winner_value, winner_votes = most_common[0]
        total_votes = len(values)
        
        # 计算置信度
        if winner_votes == total_votes:
            confidence = 'high'  # 全部一致
        elif winner_votes >= total_votes * 0.6:
            confidence = 'medium'  # 多数
        else:
            confidence = 'low'  # 分歧严重
        
        # 收集其他候选值
        alternatives = []
        for value, votes in most_common[1:]:
            alternatives.append({
                'value': value,
                'votes': votes
            })
        
        return VotingResult(
            value=winner_value,
            votes=winner_votes,
            total_models=total_votes,
            confidence=confidence,
            alternatives=alternatives
        )
    
    def _normalize_number(self, value: Any) -> Any:
        """
        数值规范化（处理浮点误差）
        
        例如：0.073, 0.0730, 0.073000 → 0.073
        """
        if isinstance(value, float):
            # 保留4位小数
            return round(value, 4)
        return value


# ============================================================================
# 便捷函数
# ============================================================================

def create_multi_model_extractor(
    kimi_client,
    deepseek_client,
    glm46_client,
    glm46v_client=None
) -> MultiModelExtractor:
    """创建多模型提取器"""
    text_models = {
        'kimi': kimi_client,
        'deepseek': deepseek_client,
        'glm-4.6': glm46_client
    }
    return MultiModelExtractor(text_models, glm46v_client)
