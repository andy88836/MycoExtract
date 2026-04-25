"""
数据处理与审核系统

设计思路：
1. DataProcessor - 确定性处理（不需要LLM）
   - 数据清洗、验证
   - 记录合并（merge）
   - Schema规范化
   - 序列富集（调用API但不需要LLM判断）

2. ReviewerAgent - 真正的Agent（需要LLM + 联网工具）
   - 用LLM综合判断数据质量
   - 调用联网工具验证UniProt、SMILES
   - 决定是否需要人工审核

流程：
    Extract → DataProcessor → ReviewerAgent → HITL
"""

import os
import json
import logging
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

class ReviewDecision(Enum):
    """审核决定"""
    APPROVED = "approved"           # 自动通过
    NEEDS_REVIEW = "needs_review"   # 需要人工审核
    REJECTED = "rejected"           # 直接拒绝（数据严重错误）


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    records: List[Dict]
    stats: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass  
class ReviewResult:
    """审核结果"""
    decision: ReviewDecision
    records: List[Dict]
    summary: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    tool_calls: List[Dict] = field(default_factory=list)


# ============================================================================
# DataProcessor - 确定性处理（不需要LLM）
# ============================================================================

class DataProcessor:
    """
    数据处理器 - 执行确定性的数据处理任务
    
    不需要LLM，纯规则处理：
    1. 数据验证和清洗
    2. 酶信息合并
    3. 重复记录去重
    4. Schema规范化
    5. 序列富集（API调用）
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enable_merge = self.config.get("enable_merge", True)
        self.enable_enrichment = self.config.get("enable_enrichment", True)
        self.km_tolerance = self.config.get("km_tolerance", 0.1)
        self.auto_fill_threshold = self.config.get("auto_fill_threshold", 0.9)
    
    def process(self, records: List[Dict], context: Dict = None) -> ProcessingResult:
        """
        处理数据
        
        Args:
            records: 原始提取记录
            context: 上下文（DOI、原文等）
            
        Returns:
            ProcessingResult
        """
        logger.info(f"[DataProcessor] Processing {len(records)} records...")
        
        stats = {
            "input_count": len(records),
            "validated": 0,
            "quality_filtered": 0,
            "merged": 0,
            "enriched": 0
        }
        
        current_records = records
        
        try:
            # 1. 验证和清洗
            logger.info("  Step 1: Validating...")
            current_records = self._validate(current_records)
            stats["validated"] = len(current_records)
            
            # 1.5 质量约束过滤 (新增：序列可获取性、霉菌毒素底物、解毒验证)
            logger.info("  Step 1.5: Quality constraint filtering...")
            current_records, filter_stats = self._quality_filter(current_records)
            stats["quality_filtered"] = len(current_records)
            if filter_stats['rejected'] > 0:
                logger.info(f"    Filtered: {filter_stats['rejected']} rejected "
                           f"(no_seq={filter_stats['rejected_no_sequence']}, "
                           f"non_myco={filter_stats['rejected_non_mycotoxin']}, "
                           f"bioact={filter_stats['rejected_bioactivation']})")
            
            # 2. 合并酶信息
            logger.info("  Step 2: Merging enzyme info...")
            current_records = self._merge_enzyme_info(current_records)
            
            # 3. 去重合并
            if self.enable_merge:
                before = len(current_records)
                logger.info("  Step 3: Deduplicating...")
                current_records = self._deduplicate(current_records)
                stats["merged"] = before - len(current_records)
            
            # 4. Schema规范化
            logger.info("  Step 4: Normalizing schema...")
            current_records = self._normalize_schema(current_records)
            
            # 5. 序列富集（可选）
            if self.enable_enrichment:
                logger.info("  Step 5: Enriching sequences...")
                current_records, enrich_stats = self._enrich_sequences(current_records)
                stats["enriched"] = enrich_stats.get("auto_filled", 0)
            
            stats["output_count"] = len(current_records)
            
            logger.info(f"[DataProcessor] Done: {stats['input_count']} → {stats['output_count']} records")
            
            return ProcessingResult(
                success=True,
                records=current_records,
                stats=stats,
                message=f"Processed {len(current_records)} records"
            )
            
        except Exception as e:
            logger.error(f"[DataProcessor] Error: {e}")
            return ProcessingResult(
                success=False,
                records=records,
                stats=stats,
                message=str(e)
            )
    
    def _validate(self, records: List[Dict]) -> List[Dict]:
        """验证和清洗数据"""
        from src.utils.data_validator import DataValidator
        return DataValidator.validate_batch(records)
    
    def _quality_filter(self, records: List[Dict]) -> tuple[List[Dict], Dict]:
        """质量约束过滤：序列可获取性、霉菌毒素底物、解毒验证"""
        from src.utils.quality_constraints import QualityConstraintFilter
        
        quality_filter = QualityConstraintFilter(
            require_sequence=True,
            require_mycotoxin=True,
            check_detoxification=True,
            strict_mode=False
        )
        
        return quality_filter.filter_records(records)
    
    def _merge_enzyme_info(self, records: List[Dict]) -> List[Dict]:
        """合并同一论文中相同酶的信息"""
        from collections import defaultdict
        
        BASIC_FIELDS = ['enzyme_full_name', 'enzyme_type', 'ec_number', 'gene_name', 'organism']
        
        # 按enzyme_name分组
        groups = defaultdict(list)
        for r in records:
            name = r.get('enzyme_name', '')
            if name:
                groups[name].append(r)
        
        # 补全信息
        for name, group in groups.items():
            if len(group) <= 1:
                continue
            
            # 构建模板
            template = {}
            sorted_group = sorted(group, key=lambda x: x.get('confidence_score', 0), reverse=True)
            
            for field in BASIC_FIELDS:
                for r in sorted_group:
                    val = r.get(field)
                    if val and val not in ('', None, [], {}, 'Unknown'):
                        template[field] = val
                        break
            
            # 补全每条记录
            for r in group:
                for field, val in template.items():
                    if not r.get(field) or r.get(field) in ('', 'Unknown'):
                        r[field] = val
        
        return records
    
    def _deduplicate(self, records: List[Dict]) -> List[Dict]:
        """去重合并"""
        try:
            from src.pipeline.post_processor import RecordMerger
            merger = RecordMerger(km_tolerance=self.km_tolerance)
            return merger.merge_records(records)
        except ImportError:
            logger.warning("RecordMerger not available, skipping deduplication")
            return records
    
    def _normalize_schema(self, records: List[Dict]) -> List[Dict]:
        """Schema规范化"""
        try:
            from src.pipeline.post_processor import normalize_records_batch
            return normalize_records_batch(records)
        except ImportError:
            return records
    
    def _enrich_sequences(self, records: List[Dict]) -> Tuple[List[Dict], Dict]:
        """序列富集"""
        try:
            from src.utils.sequence_enricher import SequenceEnricher
            enricher = SequenceEnricher(auto_fill_threshold=self.auto_fill_threshold)
            return enricher.enrich_records(records, auto_fill=True, verbose=False)
        except ImportError:
            logger.warning("SequenceEnricher not available, skipping enrichment")
            return records, {}


# ============================================================================
# 联网验证工具
# ============================================================================

class WebVerificationTools:
    """
    联网验证工具集
    
    提供API调用验证数据的功能：
    - UniProt ID验证
    - SMILES验证（PubChem）
    - EC号验证
    - DOI验证
    """
    
    @staticmethod
    async def verify_uniprot_id(uniprot_id: str) -> Dict[str, Any]:
        """
        验证UniProt ID是否存在，并返回关键信息
        """
        if not uniprot_id:
            return {"valid": False, "error": "Empty UniProt ID"}
        
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
            params = {"format": "json"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "valid": True,
                            "uniprot_id": uniprot_id,
                            "protein_name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                            "organism": data.get("organism", {}).get("scientificName", ""),
                            "sequence_length": len(data.get("sequence", {}).get("value", "")),
                            "reviewed": data.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"
                        }
                    elif response.status == 404:
                        return {"valid": False, "error": f"UniProt ID {uniprot_id} not found"}
                    else:
                        return {"valid": False, "error": f"HTTP {response.status}"}
        except asyncio.TimeoutError:
            return {"valid": False, "error": "Timeout"}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    async def verify_smiles(smiles: str, compound_name: str = None) -> Dict[str, Any]:
        """
        验证SMILES是否有效，并尝试匹配化合物名称
        """
        if not smiles:
            return {"valid": False, "error": "Empty SMILES"}
        
        try:
            # 方法1：通过SMILES查询PubChem
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName,MolecularFormula/JSON"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                        return {
                            "valid": True,
                            "smiles": smiles,
                            "iupac_name": props.get("IUPACName", ""),
                            "formula": props.get("MolecularFormula", ""),
                            "compound_match": None  # 可以进一步匹配compound_name
                        }
                    else:
                        return {"valid": False, "error": "Invalid SMILES or not found in PubChem"}
        except asyncio.TimeoutError:
            return {"valid": False, "error": "Timeout"}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    async def verify_ec_number(ec_number: str) -> Dict[str, Any]:
        """
        验证EC号格式和有效性
        """
        import re
        
        if not ec_number:
            return {"valid": False, "error": "Empty EC number"}
        
        # 格式验证
        pattern = r'^(\d+)\.(\d+|-)\.(\d+|-)\.(\d+|-)$'
        match = re.match(pattern, ec_number.strip())
        
        if not match:
            return {"valid": False, "format_ok": False, "error": "Invalid EC number format"}
        
        main_class = int(match.group(1))
        if main_class < 1 or main_class > 7:
            return {"valid": False, "format_ok": True, "error": f"Invalid main class {main_class}"}
        
        ec_classes = {
            1: "Oxidoreductases",
            2: "Transferases",
            3: "Hydrolases",
            4: "Lyases",
            5: "Isomerases",
            6: "Ligases",
            7: "Translocases"
        }
        
        return {
            "valid": True,
            "format_ok": True,
            "ec_number": ec_number,
            "main_class": main_class,
            "class_name": ec_classes[main_class]
        }
    
    @staticmethod
    async def verify_doi(doi: str) -> Dict[str, Any]:
        """
        验证DOI并获取论文信息
        """
        if not doi:
            return {"valid": False, "error": "Empty DOI"}
        
        try:
            url = f"https://api.crossref.org/works/{doi}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        msg = data.get("message", {})
                        return {
                            "valid": True,
                            "doi": doi,
                            "title": msg.get("title", [""])[0],
                            "journal": msg.get("container-title", [""])[0],
                            "year": msg.get("published-print", {}).get("date-parts", [[None]])[0][0]
                        }
                    else:
                        return {"valid": False, "error": "DOI not found"}
        except asyncio.TimeoutError:
            return {"valid": False, "error": "Timeout"}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    async def batch_verify(records: List[Dict], doi: str = None) -> Dict[str, Any]:
        """
        批量验证记录中的关键字段
        
        Returns:
            {
                "doi_verification": {...},
                "uniprot_verifications": [{...}, ...],
                "smiles_verifications": [{...}, ...],
                "ec_verifications": [{...}, ...],
                "summary": {...}
            }
        """
        results = {
            "doi_verification": None,
            "uniprot_verifications": [],
            "smiles_verifications": [],
            "ec_verifications": [],
            "summary": {
                "total_records": len(records),
                "uniprot_valid": 0,
                "smiles_valid": 0,
                "ec_valid": 0
            }
        }
        
        # 验证DOI
        if doi:
            results["doi_verification"] = await WebVerificationTools.verify_doi(doi)
        
        # 批量验证记录（并发但限制速率）
        semaphore = asyncio.Semaphore(5)  # 最多5个并发
        
        async def verify_record(record: Dict, idx: int):
            async with semaphore:
                verifications = {"record_index": idx}
                
                # UniProt
                uniprot_id = record.get("uniprot_id")
                if uniprot_id:
                    verifications["uniprot"] = await WebVerificationTools.verify_uniprot_id(uniprot_id)
                
                # SMILES
                smiles = record.get("substrate_smiles")
                if smiles:
                    verifications["smiles"] = await WebVerificationTools.verify_smiles(
                        smiles, record.get("substrate")
                    )
                
                # EC号
                ec = record.get("ec_number")
                if ec:
                    verifications["ec"] = await WebVerificationTools.verify_ec_number(ec)
                
                return verifications
        
        # 并发执行
        tasks = [verify_record(r, i) for i, r in enumerate(records)]
        verifications = await asyncio.gather(*tasks)
        
        # 整理结果
        for v in verifications:
            if "uniprot" in v:
                results["uniprot_verifications"].append(v["uniprot"])
                if v["uniprot"].get("valid"):
                    results["summary"]["uniprot_valid"] += 1
            
            if "smiles" in v:
                results["smiles_verifications"].append(v["smiles"])
                if v["smiles"].get("valid"):
                    results["summary"]["smiles_valid"] += 1
            
            if "ec" in v:
                results["ec_verifications"].append(v["ec"])
                if v["ec"].get("valid"):
                    results["summary"]["ec_valid"] += 1
        
        return results


# ============================================================================
# ReviewerAgent - 真正的Agent（LLM + 联网工具）
# ============================================================================

class ReviewerAgent:
    """
    审核Agent - 使用LLM进行智能审核
    
    职责：
    1. 调用联网工具验证关键字段（UniProt、SMILES、EC号、DOI）
    2. 用LLM综合分析验证结果和数据质量
    3. 做出审核决定：自动通过 / 需要人工审核 / 拒绝
    4. 生成审核报告供HITL参考
    """
    
    # 从文件加载System Prompt
    @staticmethod
    def _load_system_prompt() -> str:
        """从prompts文件夹加载系统提示词"""
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "prompts_review_agent_system.txt"
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load system prompt from {prompt_file}: {e}")
            # 返回最基本的prompt作为fallback
            return "你是一位专业的生物化学数据审核专家，负责酶动力学数据的质量控制。"
    
    SYSTEM_PROMPT = None  # 将在首次使用时加载
    
    def __init__(self, llm_client, config: Dict = None):
        """
        Args:
            llm_client: LLM客户端，需要有chat()方法
            config: 配置
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.auto_approve_threshold = self.config.get("auto_approve_threshold", 0.85)
        self.enable_web_verification = self.config.get("enable_web_verification", True)
        
        # 加载System Prompt（如果还未加载）
        if ReviewerAgent.SYSTEM_PROMPT is None:
            ReviewerAgent.SYSTEM_PROMPT = ReviewerAgent._load_system_prompt()
    
    async def review(self, records: List[Dict], doi: str = None, content_list: List[Dict] = None) -> ReviewResult:
        """
        审核数据
        
        Args:
            records: 待审核的记录
            doi: 论文DOI
            content_list: 论文的content_list（用于根据block_id提取相关段落）
            
        Returns:
            ReviewResult
        """
        logger.info(f"[ReviewerAgent] Reviewing {len(records)} records...")
        if content_list:
            logger.info(f"  Content list available: {len(content_list)} blocks")
        
        tool_calls = []
        
        # Step 1: 联网验证
        verification_results = None
        if self.enable_web_verification:
            logger.info("  Step 1: Running web verification...")
            verification_results = await WebVerificationTools.batch_verify(records, doi)
            tool_calls.append({
                "tool": "batch_verify",
                "input": {"records": len(records), "doi": doi},
                "output": verification_results["summary"]
            })
            logger.info(f"    DOI valid: {verification_results['doi_verification'].get('valid') if verification_results['doi_verification'] else 'N/A'}")
            logger.info(f"    UniProt valid: {verification_results['summary']['uniprot_valid']}/{len(verification_results['uniprot_verifications'])}")
        
        # Step 2: 为每条记录构建审核上下文并用LLM判断
        logger.info("  Step 2: LLM review...")
        reviewed_records = []
        decisions = {"approved": 0, "needs_review": 0, "rejected": 0}
        
        for i, record in enumerate(records):
            # 构建上下文
            context = self._build_review_context(record, i, verification_results, content_list)
            
            # 调用LLM
            review = await self._llm_review(context)
            
            # 更新记录
            record["_review"] = review
            record["_review_decision"] = review.get("decision", "NEEDS_REVIEW")
            record["_review_confidence"] = review.get("confidence", 0.5)
            record["_review_issues"] = review.get("issues", [])
            record["_review_notes"] = review.get("notes_for_reviewer", "")
            
            reviewed_records.append(record)
            
            decision = review.get("decision", "NEEDS_REVIEW").lower()
            if decision == "approved":
                decisions["approved"] += 1
            elif decision == "rejected":
                decisions["rejected"] += 1
            else:
                decisions["needs_review"] += 1
        
        # Step 3: 生成总体报告
        logger.info(f"  Results: {decisions}")
        
        overall_decision = ReviewDecision.APPROVED
        if decisions["rejected"] > 0:
            overall_decision = ReviewDecision.REJECTED
        elif decisions["needs_review"] > 0:
            overall_decision = ReviewDecision.NEEDS_REVIEW
        
        return ReviewResult(
            decision=overall_decision,
            records=reviewed_records,
            summary={
                "total": len(records),
                "approved": decisions["approved"],
                "needs_review": decisions["needs_review"],
                "rejected": decisions["rejected"],
                "verification": verification_results["summary"] if verification_results else None
            },
            reasoning=f"Reviewed {len(records)} records: {decisions['approved']} approved, {decisions['needs_review']} need review, {decisions['rejected']} rejected",
            tool_calls=tool_calls
        )
    
    def _build_review_context(self, record: Dict, index: int, verification: Dict, content_list: List[Dict] = None) -> str:
        """构建单条记录的审核上下文"""
        
        # 基本信息
        context = f"""
Record #{index + 1}:

Basic Information:
- Enzyme: {record.get('enzyme_name', 'MISSING')}
- Full Name: {record.get('enzyme_full_name', 'N/A')}
- Organism: {record.get('organism', 'MISSING')}
- EC Number: {record.get('ec_number', 'N/A')}

Kinetic Data:
- Substrate: {record.get('substrate', 'MISSING')}
- Km: {record.get('Km_value', 'N/A')} {record.get('Km_unit', '')}
- Vmax: {record.get('Vmax_value', 'N/A')} {record.get('Vmax_unit', '')}
- kcat: {record.get('kcat_value', 'N/A')} {record.get('kcat_unit', '')}
- kcat/Km: {record.get('kcat_Km_value', 'N/A')} {record.get('kcat_Km_unit', '')}
- pH: {record.get('ph', 'N/A')}
- Temperature: {record.get('temperature_value', 'N/A')} {record.get('temperature_unit', '')}

Source Information:
- DOI: {record.get('source_in_document', {}).get('doi', 'N/A')}
- Source Type: {record.get('source_in_document', {}).get('source_type', 'N/A')}
- Block ID: {record.get('source_in_document', {}).get('block_id', 'N/A')}
- Notes: {record.get('notes', 'N/A')}

Sequence Information:
- UniProt ID: {record.get('uniprot_id', 'MISSING')}
- Sequence: {'Yes (' + str(len(record.get('sequence', ''))) + ' aa)' if record.get('sequence') else 'MISSING'}

Substrate Structure:
- SMILES: {record.get('substrate_smiles', 'N/A')}
"""
        
        # 智能提取相关段落（根据block_id）
        if content_list:
            relevant_text = self._extract_relevant_context(record, content_list)
            if relevant_text:
                context += f"""

=== 相关原文段落 ===
{relevant_text}

请对比原文验证以下内容：
1. 数值准确性：Km、Vmax、kcat等数值是否与原文一致
2. 实验条件：pH、温度是否正确提取
3. 酶来源：organism信息是否准确
4. 底物名称：substrate是否与原文匹配
5. 酶类型：是否为自由酶（检查是否有immobilized、nanocomplex等词）

如果发现数值不匹配或信息错误，请在reasoning中详细说明。
"""
        
        # 添加验证结果
        if verification:
            uniprot_v = verification.get("uniprot_verifications", [])
            smiles_v = verification.get("smiles_verifications", [])
            ec_v = verification.get("ec_verifications", [])
            
            if index < len(uniprot_v) and uniprot_v[index]:
                v = uniprot_v[index]
                context += f"""
UniProt Verification:
- Valid: {v.get('valid', False)}
- Protein: {v.get('protein_name', 'N/A')}
- Organism: {v.get('organism', 'N/A')}
- Error: {v.get('error', 'None')}
"""
            
            if index < len(ec_v) and ec_v[index]:
                v = ec_v[index]
                context += f"""
EC Number Verification:
- Valid: {v.get('valid', False)}
- Class: {v.get('class_name', 'N/A')}
- Error: {v.get('error', 'None')}
"""
        
        context += "\nPlease review this record and provide your decision."
        
        return context
    
    async def _llm_review(self, context: str) -> Dict:
        """调用LLM进行审核"""
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ]
            
            # 调用LLM
            if hasattr(self.llm_client, 'chat'):
                response = await asyncio.to_thread(
                    self.llm_client.chat,
                    messages=messages
                )
            elif hasattr(self.llm_client, 'generate'):
                prompt = f"{self.SYSTEM_PROMPT}\n\n{context}"
                response = await asyncio.to_thread(
                    self.llm_client.generate,
                    prompt=prompt
                )
            else:
                return self._default_review()
            
            # 解析JSON响应（支持多种格式）
            import re
            
            # 尝试1: 提取markdown代码块中的JSON
            json_match = re.search(r'```(?:json)?\s*({[^`]+})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    if "reasoning" in result:
                        result["reasoning"] = f"[GPT-5.1审核]\n\n{result['reasoning']}"
                    return result
                except:
                    pass
            
            # 尝试2: 直接搜索JSON对象
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*"reasoning"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                # 尝试更宽松的匹配（支持嵌套）
                json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*"decision"(?:[^{}]|\{[^{}]*\})*\}', response, re.DOTALL)
            
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if "reasoning" in result:
                        result["reasoning"] = f"[GPT-5.1审核]\n\n{result['reasoning']}"
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error: {e}")
            
            # 解析失败，返回原始响应供调试
            logger.warning(f"Could not parse LLM response. First 1000 chars: {response[:1000]}")
            return {
                "decision": "NEEDS_REVIEW",
                "confidence": 0.5,
                "issues": ["Could not parse LLM response"],
                "reasoning": f"[GPT-5.1审核]\n\n原始响应:\n{response[:1000]}"
            }
                
        except Exception as e:
            logger.error(f"LLM review failed: {e}")
            return self._default_review()
    
    def _default_review(self) -> Dict:
        """默认审核结果（LLM失败时使用）"""
        return {
            "decision": "NEEDS_REVIEW",
            "confidence": 0.5,
            "issues": ["LLM review failed, manual review required"],
            "reasoning": "Automatic review"
        }
    
    def _extract_relevant_context(self, record: Dict, content_list: List[Dict]) -> str:
        """
        智能提取与记录相关的原文段落
        
        根据record的source_in_document.block_id，提取该block前后的相关文本
        
        Args:
            record: 待审核的记录
            content_list: 论文的content_list (包含所有text/table/figure blocks)
            
        Returns:
            相关的原文段落（字符串）
        """
        try:
            source_info = record.get('source_in_document', {})
            block_id_raw = source_info.get('block_id')
            
            if block_id_raw is None:
                logger.warning(f"  [Context Extraction] No block_id found in record. source_in_document={source_info}")
                return ""
            
            if not content_list:
                logger.warning(f"  [Context Extraction] content_list is empty or None")
                return ""
            
            # 处理block_id：可能是整数或字符串（如"table_block_43"）
            if isinstance(block_id_raw, str):
                # 提取数字部分，例如 "table_block_43" -> 43
                import re
                match = re.search(r'(\d+)$', block_id_raw)
                if match:
                    block_id = int(match.group(1))
                else:
                    logger.warning(f"  [Context Extraction] Cannot parse block_id from string: {block_id_raw}")
                    return ""
            else:
                block_id = int(block_id_raw)
            
            logger.info(f"  [Context Extraction] Extracting context for block_id={block_id} (from {block_id_raw}), content_list has {len(content_list)} blocks")
            
            # 检查 block_id 是否超出范围
            if block_id >= len(content_list):
                logger.warning(f"  [Context Extraction] block_id={block_id} exceeds content_list length={len(content_list)}, using broader context")
                # 使用全局上下文（Methods + Results 部分）
                return self._extract_global_context(content_list, record)
            
            # 提取目标块及其前后的上下文窗口
            # 窗口大小：前后各5-7个块（扩大窗口以提供更多上下文）
            window_size = 7
            start_idx = max(0, block_id - window_size)
            end_idx = min(len(content_list), block_id + window_size + 1)
            
            relevant_blocks = content_list[start_idx:end_idx]
            
            # 提取文本内容
            text_parts = []
            for i, block in enumerate(relevant_blocks):
                actual_idx = start_idx + i
                block_type = block.get('type', 'unknown')
                
                # 添加块标识
                if actual_idx == block_id:
                    text_parts.append(f"\n>>> [Block {actual_idx}] *** 数据来源 *** <<<")
                else:
                    text_parts.append(f"\n[Block {actual_idx}] ({block_type})")
                
                # 提取文本内容
                if block_type == 'text':
                    text_content = block.get('text', '')
                    text_parts.append(text_content)
                elif block_type == 'table':
                    # 表格显示结构化信息（增加长度限制以包含完整表格）
                    table_parts = []
                    
                    # 添加表格标题
                    caption = block.get('table_caption')
                    if caption:
                        if isinstance(caption, list):
                            caption = ' '.join(caption)
                        table_parts.append(f"Caption: {caption}")
                    
                    # 提取表格内容（优先级：table_body > text > html）
                    table_content = block.get('table_body') or block.get('text') or block.get('html', '')
                    if table_content:
                        # 增加长度到3000字符以包含完整表格
                        table_parts.append(str(table_content)[:3000])
                    
                    # 添加表格脚注
                    footnote = block.get('table_footnote')
                    if footnote:
                        if isinstance(footnote, list):
                            footnote = ' '.join(footnote)
                        table_parts.append(f"Footnote: {footnote}")
                    
                    table_text = '\n'.join(table_parts)
                    text_parts.append(f"[TABLE]\n{table_text}")
                elif block_type in ('figure', 'image'):
                    # 图片只显示标题/说明
                    caption = block.get('caption', block.get('text', '[No caption]'))
                    text_parts.append(f"[FIGURE]\n{caption}")
            
            # 合并文本并限制长度
            full_text = '\n'.join(text_parts)
            max_length = 10000  # 限制在约2500 tokens（提供更完整的上下文）
            
            if len(full_text) > max_length:
                full_text = full_text[:max_length] + "\n\n[...相关段落已截断，已显示关键部分...]"
            
            logger.info(f"  [Context Extraction] Extracted {len(full_text)} characters from blocks {start_idx}-{end_idx}")
            
            # 如果提取的内容太少，使用全局上下文
            if len(full_text) < 500:
                logger.warning(f"  [Context Extraction] Extracted text too short ({len(full_text)} chars), using broader context")
                return self._extract_global_context(content_list, record)
            
            return full_text
            
        except Exception as e:
            logger.warning(f"Failed to extract relevant context: {e}")
            return ""
    
    def _extract_global_context(self, content_list: List[Dict], record: Dict) -> str:
        """
        提取全局上下文（当 block_id 不可用或窗口上下文太少时）
        
        策略：提取包含关键词的文本块（Methods, Materials, Results, 酶名, 生物名等）
        """
        enzyme_name = record.get('enzyme_name', '')
        organism = record.get('organism', '')
        substrate = record.get('substrate', '')
        
        relevant_blocks = []
        
        for idx, block in enumerate(content_list):
            block_type = block.get('type', '')
            if block_type not in ('text', 'table'):
                continue
            
            text_content = block.get('text', '').lower()
            
            # 检查是否包含关键词
            is_relevant = False
            
            # 章节关键词
            section_keywords = ['materials and methods', 'methods', 'experimental', 'results', 
                              'kinetic', 'enzyme assay', 'activity', 'characterization']
            for kw in section_keywords:
                if kw in text_content:
                    is_relevant = True
                    break
            
            # 酶名/生物名/底物匹配
            if not is_relevant and enzyme_name:
                if enzyme_name.lower() in text_content:
                    is_relevant = True
            if not is_relevant and organism:
                # 检查属名和种名
                org_parts = organism.split()
                if len(org_parts) >= 2:
                    if org_parts[0].lower() in text_content or org_parts[1].lower() in text_content:
                        is_relevant = True
            if not is_relevant and substrate:
                if substrate.lower() in text_content:
                    is_relevant = True
            
            # 数值模式（可能包含动力学数据）
            if not is_relevant:
                import re
                if re.search(r'\b\d+\.?\d*\s*(mM|μM|s⁻¹|min⁻¹|kDa|°C)', text_content):
                    is_relevant = True
            
            if is_relevant:
                relevant_blocks.append((idx, block))
        
        # 限制块数量
        if len(relevant_blocks) > 15:
            relevant_blocks = relevant_blocks[:15]
        
        # 构建文本
        text_parts = []
        for idx, block in relevant_blocks:
            block_type = block.get('type', 'unknown')
            text_parts.append(f"\n[Block {idx}] ({block_type})")
            
            if block_type == 'text':
                text_parts.append(block.get('text', ''))
            elif block_type == 'table':
                # 提取完整表格（包含标题和内容）
                table_parts = []
                caption = block.get('table_caption')
                if caption:
                    if isinstance(caption, list):
                        caption = ' '.join(caption)
                    table_parts.append(f"Caption: {caption}")
                
                table_content = block.get('table_body') or block.get('text') or block.get('html', '')
                if table_content:
                    table_parts.append(str(table_content)[:2500])
                
                table_text = '\n'.join(table_parts)
                text_parts.append(f"[TABLE]\n{table_text}")
        
        result = '\n'.join(text_parts)
        
        # 限制长度
        if len(result) > 12000:
            result = result[:12000] + "\n\n[...更多内容已截断...]"
        
        logger.info(f"  [Global Context] Extracted {len(result)} characters from {len(relevant_blocks)} relevant blocks")
        return result


# ============================================================================
# 完整流水线
# ============================================================================

class PostExtractionPipeline:
    """
    提取后处理流水线
    
    增强流程（加入交叉验证）：
    1. DataProcessor - 数据清洗、合并、规范化（不需要LLM）
    2. SequenceDetective - 智能序列检索（GLM-4.6 + UniProt API）
    3. ✨ CrossValidator - 交叉验证提取的数据与原文是否一致（GPT-5.1）
    4. ReviewerAgent - 最终审核（LLM + 联网验证）
    5. 输出到 HITL 系统
    """
    
    def __init__(self, llm_client=None, sequence_client=None, config: Dict = None):
        """
        Args:
            llm_client: LLM客户端（用于ReviewerAgent，如GPT-5.1）
            sequence_client: 序列侦探客户端（用于SequenceDetective，如GLM-4.6）
            config: 配置
        """
        self.config = config or {}
        
        # 数据处理器（不需要LLM）
        self.processor = DataProcessor(config)
        
        # 序列侦探Agent（使用GLM-4.6）
        self.sequence_detective = None
        if sequence_client:
            from src.agents.sequence_detective import SequenceDetectiveAgent
            self.sequence_detective = SequenceDetectiveAgent(sequence_client, config)
        
        # 交叉验证Agent（使用GPT-5.1）
        self.cross_validator = None
        if llm_client:
            from src.agents.cross_validator import CrossValidatorAgent
            self.cross_validator = CrossValidatorAgent(llm_client, config)
        
        # 审核Agent（需要LLM）
        self.reviewer = ReviewerAgent(llm_client, config) if llm_client else None
        
        logger.info("PostExtractionPipeline initialized")
        logger.info(f"  - DataProcessor: enabled")
        logger.info(f"  - SequenceDetective: {'enabled' if self.sequence_detective else 'disabled (no sequence client)'}")
        logger.info(f"  - CrossValidator: {'enabled' if self.cross_validator else 'disabled (no LLM client)'}")
        logger.info(f"  - ReviewerAgent: {'enabled' if self.reviewer else 'disabled (no LLM client)'}")
    
    async def process_async(
        self,
        records: List[Dict],
        doi: str = None,
        skip_review: bool = False,
        paper_text: str = None,
        content_list: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        异步处理流程
        
        Args:
            records: 原始提取记录
            doi: 论文DOI
            skip_review: 是否跳过审核
            paper_text: 论文原文（用于序列侦探）
            content_list: 论文内容块列表（用于Review Agent上下文提取）
            
        Returns:
            {
                "records": 处理后的记录,
                "processing_stats": 处理统计,
                "sequence_detective_result": 序列侦探结果,
                "review_result": 审核结果,
                "for_hitl": 需要人工审核的记录
            }
        """
        result = {
            "records": records,
            "processing_stats": None,
            "sequence_detective_result": None,
            "review_result": None,
            "for_hitl": []
        }
        
        # Step 1: 数据处理
        logger.info("\n" + "="*60)
        logger.info("📋 Stage 1: Data Processing")
        logger.info("="*60)
        
        proc_result = self.processor.process(records, {"doi": doi})
        result["records"] = proc_result.records
        result["processing_stats"] = proc_result.stats
        
        # Step 2: 序列侦探（在Review之前运行）
        if self.sequence_detective and paper_text:
            logger.info("\n" + "="*60)
            logger.info("🔬 Stage 2: Sequence Detective")
            logger.info("="*60)
            
            try:
                # 为每条记录运行序列侦探
                for record in result["records"]:
                    # 只对缺少sequence的记录运行
                    if not record.get('sequence') and not record.get('uniprot_id'):
                        logger.info(f"  🔍 Investigating: {record.get('enzyme_name', 'Unknown')} from {record.get('organism', 'Unknown')}")
                        
                        detective_result = await self.sequence_detective.investigate(
                            paper_text=paper_text,
                            enzyme_name=record.get('enzyme_name'),
                            organism=record.get('organism'),
                            existing_data=record
                        )
                        
                        # 将侦探结果存储到记录中
                        record['_sequence_detective'] = {
                            'confidence': detective_result.confidence,
                            'reasoning': detective_result.reasoning,
                            'fingerprint': {
                                'organism': detective_result.fingerprint.organism,
                                'strain': detective_result.fingerprint.strain,
                                'reference_strain': detective_result.fingerprint.reference_strain,
                                'enzyme_name': detective_result.fingerprint.enzyme_name,
                                'gene_name': detective_result.fingerprint.gene_name,
                                'genbank_id': detective_result.fingerprint.genbank_id,
                                'uniprot_id': detective_result.fingerprint.uniprot_id,
                                'ec_number': detective_result.fingerprint.ec_number,
                                'molecular_weight_kda': detective_result.fingerprint.molecular_weight_kda,
                                'gene_length_bp': detective_result.fingerprint.gene_length_bp
                            },
                            'best_match': None,
                            'candidates': []
                        }
                        
                        if detective_result.best_match:
                            record['_sequence_detective']['best_match'] = {
                                'uniprot_id': detective_result.best_match.entry_id,
                                'entry_name': detective_result.best_match.entry_name,
                                'protein_name': detective_result.best_match.protein_name,
                                'organism': detective_result.best_match.organism,
                                'gene_names': detective_result.best_match.gene_names,
                                'length': detective_result.best_match.length,
                                'mass': detective_result.best_match.mass,
                                'reviewed': detective_result.best_match.reviewed,
                                'sequence': detective_result.best_match.sequence,
                                'score': detective_result.best_match.score,
                                'match_reasons': detective_result.best_match.match_reasons
                            }
                            
                            # 如果置信度高，自动填充
                            if detective_result.confidence in ["High", "Medium"]:
                                logger.info(f"    ✅ Found: {detective_result.best_match.entry_id} (confidence: {detective_result.confidence})")
                                record['_sequence_detective_recommendation'] = detective_result.best_match.entry_id
                            else:
                                logger.info(f"    ⚠️ Low confidence: {detective_result.confidence}")
                        
                        # 保存候选列表
                        for cand in detective_result.candidates[:5]:  # 最多5个
                            record['_sequence_detective']['candidates'].append({
                                'uniprot_id': cand.entry_id,
                                'protein_name': cand.protein_name,
                                'organism': cand.organism,
                                'score': cand.score,
                                'reviewed': cand.reviewed
                            })
                        
                logger.info(f"  Sequence Detective completed for {len(result['records'])} records")
                
            except Exception as e:
                logger.error(f"Sequence Detective failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Step 3: 交叉验证（验证提取数据与原文是否一致）
        if self.cross_validator and content_list:
            logger.info("\n" + "="*60)
            logger.info("✅ Stage 3: Cross-Validation (Data vs Original Text)")
            logger.info("="*60)
            
            try:
                validation_results = await self.cross_validator.validate_batch(
                    result["records"], 
                    content_list
                )
                
                # 将验证结果附加到记录中
                for record, validation in zip(result["records"], validation_results):
                    record['_cross_validation'] = {
                        'status': validation.overall_status,
                        'match_count': validation.match_count,
                        'mismatch_count': validation.mismatch_count,
                        'reasoning': validation.reasoning,
                        'field_results': [
                            {
                                'field': fv.field,
                                'status': fv.status,
                                'extracted': fv.extracted_value,
                                'found': fv.found_in_text,
                                'evidence': fv.evidence
                            }
                            for fv in validation.field_validations
                        ]
                    }
                    
                    # 如果交叉验证FAIL，自动标记为NEEDS_REVIEW
                    if validation.overall_status == "FAIL":
                        record['_auto_flag'] = "CROSS_VALIDATION_FAIL"
                        logger.warning(f"    ⚠️ Record {record.get('id')}: {validation.mismatch_count} mismatches detected")
                
                logger.info(f"  Cross-Validation completed for {len(result['records'])} records")
                
            except Exception as e:
                logger.error(f"Cross-Validation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Step 4: 最终审核
        if not skip_review and self.reviewer:
            logger.info("\n" + "="*60)
            logger.info("🔍 Stage 4: Final Review")
            logger.info("="*60)
            
            review_result = await self.reviewer.review(result["records"], doi, content_list)
            result["review_result"] = {
                "decision": review_result.decision.value,
                "summary": review_result.summary,
                "reasoning": review_result.reasoning
            }
            result["records"] = review_result.records
            
            # 分离需要人工审核的记录
            result["for_hitl"] = [
                r for r in result["records"]
                if r.get("_review_decision", "").upper() != "APPROVED"
            ]
        else:
            # 没有审核，全部送HITL
            result["for_hitl"] = result["records"]
        
        logger.info("\n" + "="*60)
        logger.info("✅ Pipeline Complete")
        logger.info(f"   Total records: {len(result['records'])}")
        logger.info(f"   For HITL: {len(result['for_hitl'])}")
        logger.info("="*60)
        
        return result
    
    def process(
        self,
        records: List[Dict],
        doi: str = None,
        skip_review: bool = False,
        paper_text: str = None,
        content_list: List[Dict] = None
    ) -> Dict[str, Any]:
        """同步处理"""
        return asyncio.run(self.process_async(records, doi, skip_review, paper_text, content_list))


# ============================================================================
# 便捷函数
# ============================================================================

def create_pipeline(llm_client=None, sequence_client=None, config: Dict = None) -> PostExtractionPipeline:
    """创建流水线"""
    return PostExtractionPipeline(llm_client, sequence_client, config)


async def run_post_extraction(
    records: List[Dict],
    llm_client=None,
    doi: str = None,
    config: Dict = None
) -> Dict[str, Any]:
    """运行后处理流水线"""
    pipeline = create_pipeline(llm_client, config)
    return await pipeline.process_async(records, doi)
