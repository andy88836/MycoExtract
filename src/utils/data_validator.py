"""
Data validation and quality scoring for extracted enzyme records.
"""
import logging
from typing import Dict, Any, List, Optional
from src.utils.unit_normalizer import UnitNormalizer

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate and score extracted enzyme data."""
    
    # Schema v5.0 - Allowed Fields (Simplified for validation, actual schema is nested)
    # We will validate the nested structure dynamically
    
    @classmethod
    def validate_and_clean(cls, record: Dict[str, Any], block_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Validate, clean, and score a single record (v5.0 Flattened Schema).
        
        Args:
            record: The raw extracted record
            block_id: Optional block ID for traceability
            
        Returns:
            Cleaned record with confidence score, or None if rejected (Score 1)
        """
        if not record:
            return None
        
        # 1. Normalize Units (DISABLED - keep original units for HITL review)
        # record = UnitNormalizer.normalize_record(record)
        # Note: Unit conversion disabled to preserve original extracted data
        # for human review. Reviewers can verify against source documents.
        
        # 2. Calculate Confidence Score (1-5) - v5.0: Top-level field
        score = cls.calculate_confidence(record)
        record["confidence_score"] = score
        
        # 3. Add Block ID for traceability (optional metadata)
        if block_id:
            # Keep metadata for backward compatibility with pipeline
            if "metadata" not in record:
                record["metadata"] = {}
            if "source_location" not in record["metadata"]:
                record["metadata"]["source_location"] = {}
            record["metadata"]["source_location"]["block_id"] = block_id
        
        # 4. Filter Low Quality (Score 1 = Discard)
        if score <= 1:
            logger.warning(f"Record rejected (Score {score}): {cls._get_record_summary(record)}")
            return None
        
        return record
    
    @classmethod
    def calculate_confidence(cls, record: Dict[str, Any]) -> int:
        """
        Calculate confidence score on 1-3 scale for v7.0 Expanded Schema.

        ✅ v7.0优化：不再强制要求序列ID - 接受固定化酶/粗酶
        ✅ 降解数据库核心：降解效率、产物、毒性变化各自独立评分

        核心要素（必需，至少有1个）：
          - enzyme_name（酶名）
          - substrate（底物）
          - 数据指标：动力学参数（Km/kcat/kcat_Km）或降解率（degradation_efficiency）

        重要要素（各自独立，最多6个）：
          - kinetics_param（有动力学参数：Km/kcat/kcat_Km）⭐ +1分
          - degradation_efficiency（有降解率）⭐ +1分
          - products（降解产物）⭐ +1分
          - toxicity_change（毒性变化）⭐ +1分
          - organism_info（生物体信息） +1分
          - conditions（反应条件：pH、温度） +1分

        加分要素（最多3个）：
          - EC号、enzyme_full_name、uniprot_id/genbank_id/pdb_id/sequence

        Score 3 (好):
          - 核心要素齐全 + 重要要素≥3个 + 加分≥1个
          - 或：有完整降解三要素（降解率+产物+毒性变化）+ 加分≥1个

        Score 2 (中):
          - 核心要素齐全 + 重要要素≥1个

        Score 1 (差):
          - 只有核心要素，或缺少核心要素
        """
        # === 核心要素检查 ===
        has_enzyme_name = bool(record.get("enzyme_name"))
        has_substrate = bool(record.get("substrate"))

        if not has_enzyme_name or not has_substrate:
            return 1

        # === 动力学参数检查 ===
        has_km = record.get("Km_value") is not None
        has_kcat = record.get("kcat_value") is not None
        has_kcat_km = record.get("kcat_Km_value") is not None

        # 动力学参数（Km、kcat、kcat/Km）
        has_kinetics = has_km or has_kcat or has_kcat_km

        # 降解率（独立指标）
        has_degradation = record.get("degradation_efficiency") is not None

        # 至少需要有一个（动力学参数 OR 降解率）
        has_core_data = has_kinetics or has_degradation

        if not has_core_data:
            return 1

        # === 加分要素 ===
        has_ec_number = bool(record.get("ec_number"))
        has_enzyme_full_name = bool(record.get("enzyme_full_name"))
        has_uniprot = bool(record.get("uniprot_id"))
        has_genbank = bool(record.get("genbank_id"))
        has_pdb = bool(record.get("pdb_id"))
        has_sequence = bool(record.get("sequence"))
        has_sequence_id = has_uniprot or has_genbank or has_pdb or has_sequence

        bonus_count = sum([
            has_ec_number,
            has_enzyme_full_name,
            has_sequence_id
        ])

        # === 重要要素检查（各自独立评分） ===

        # 1. 动力学参数（独立评分）
        has_kinetics_param = has_km or has_kcat or has_kcat_km

        # 2. 降解率（独立评分）
        # 注意：has_degradation在上面已经定义了

        # 3. 降解产物（独立评分）
        products = record.get("products", [])
        has_products = isinstance(products, list) and len(products) > 0

        # 4. 毒性变化（独立评分）
        has_toxicity_change = False
        if has_products:
            for p in products:
                if isinstance(p, dict) and p.get("toxicity_change"):
                    has_toxicity_change = True
                    break

        # 5. 生物体信息
        has_organism = bool(record.get("organism"))
        has_strain = bool(record.get("strain"))
        has_organism_info = has_organism or has_strain

        # 6. 反应条件
        has_ph = record.get("ph") is not None or record.get("optimal_ph") is not None
        has_temp = record.get("temperature_value") is not None or record.get("optimal_temperature_value") is not None
        has_conditions = has_ph or has_temp

        # 重要要素计数（最多6个，各自独立）
        important_count = sum([
            has_kinetics_param,     # ⭐有动力学参数 = +1
            has_degradation,        # ⭐有降解率 = +1
            has_products,           # ⭐有降解产物 = +1
            has_toxicity_change,    # ⭐有毒性变化 = +1
            has_organism_info,      # 有生物体信息 = +1
            has_conditions          # 有反应条件 = +1
        ])

        # === 评分逻辑（3级） ===

        # Score 3 (好): 核心齐全 + 重要要素>=3 + 至少1个加分项
        if has_core_data and important_count >= 3 and bonus_count >= 1:
            return 3

        # 如果有完整的降解三要素（降解率+产物+毒性），要求可以放宽
        if has_degradation and has_products and has_toxicity_change and bonus_count >= 1:
            return 3

        # Score 2 (中): 核心齐全 + 至少1个重要要素
        if has_core_data and important_count >= 1:
            return 2

        # Score 1 (差): 只有核心要素
        return 1

    @classmethod
    def validate_batch(cls, records: List[Dict[str, Any]], block_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Batch validation."""
        if not records:
            return []
        
        if block_ids is None:
            block_ids = [None] * len(records)
        
        valid_records = []
        for i, record in enumerate(records):
            block_id = block_ids[i] if i < len(block_ids) else None
            cleaned = cls.validate_and_clean(record, block_id)
            if cleaned:
                valid_records.append(cleaned)
        
        return valid_records
    
    @classmethod
    def get_quality_summary(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate quality summary for v5.0 Flattened Schema."""
        if not records:
            return {"total": 0, "avg_score": 0, "distribution": {}}
        
        # 过滤掉非字典类型的记录
        valid_records = [r for r in records if isinstance(r, dict)]
        if not valid_records:
            return {"total": 0, "avg_score": 0, "distribution": {}}
        
        # v5.0: confidence_score is a top-level field, not nested in metadata
        scores = []
        for r in valid_records:
            # Try top-level first (v5.0), fallback to metadata (legacy)
            score = r.get("confidence_score")
            if score is None:
                score = r.get("metadata", {}).get("confidence_score", 0)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        dist = {i: scores.count(i) for i in range(1, 6)}
        
        return {
            "total": len(valid_records),
            "avg_score": round(avg_score, 2),
            "avg_confidence": round(avg_score, 2),
            "high_quality": sum(1 for s in scores if s >= 4),
            "medium_quality": sum(1 for s in scores if s == 3),
            "low_quality": sum(1 for s in scores if s == 2),
            "distribution": dist
        }

    @staticmethod
    def _get_record_summary(record: Dict) -> str:
        """Helper to get a short summary string for logs (v5.0 Flattened Schema)."""
        enz = record.get("enzyme_name", "Unknown")
        sub = record.get("substrate", "Unknown")
        return f"{enz} + {sub}"
