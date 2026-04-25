"""
Post-processor for merging duplicate enzyme kinetics records.

This module handles the Entity Alignment and Data Fusion problem that occurs
when the same enzyme data is extracted from multiple blocks (text + table).
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


# ========== Schema v6.0 - Complete Field Definition ==========
# All extracted records MUST contain all these fields (null if not extracted)
FULL_SCHEMA = {
    # === Enzyme Identification ===
    "enzyme_name": None,           # REQUIRED: Short name (e.g., "ZEN-rd1")
    "enzyme_full_name": None,      # Full descriptive name
    "enzyme_type": None,           # Enzyme class
    "ec_number": None,             # EC classification (e.g., "EC 1.1.1.x")
    "gene_name": None,             # Gene name
    
    # === Sequence Identifiers (HIGH PRIORITY) ===
    "uniprot_id": None,            # UniProt accession
    "genbank_id": None,            # GenBank accession
    "pdb_id": None,                # PDB structure ID
    "sequence": None,              # Amino acid sequence
    
    # === Biological Context ===
    "organism": None,              # Source organism
    "strain": None,                # Strain/variant
    "is_recombinant": None,        # Boolean: Recombinantly expressed?
    "is_wild_type": None,          # Boolean: Wild-type enzyme?
    "mutations": None,             # Mutation(s) if not wild-type
    
    # === Substrate Information ===
    "substrate": None,             # REQUIRED: Substrate name
    "substrate_smiles": None,      # SMILES structure
    "substrate_concentration": None,  # Concentration used
    
    # === Kinetic Parameters ===
    "Km_value": None,              # Michaelis constant value
    "Km_unit": None,               # Km unit (e.g., "μM")
    "Vmax_value": None,            # Maximum velocity value
    "Vmax_unit": None,             # Vmax unit
    "kcat_value": None,            # Turnover number value
    "kcat_unit": None,             # kcat unit (e.g., "s⁻¹")
    "kcat_Km_value": None,         # Catalytic efficiency value
    "kcat_Km_unit": None,          # kcat/Km unit
    
    # === Degradation/Conversion ===
    "degradation_efficiency": None,  # Degradation percentage
    "reaction_time_value": None,   # Time value for degradation
    "reaction_time_unit": None,    # Time unit (min, h, etc.)
    "products": None,              # Reaction products
    
    # === Experimental Conditions ===
    "temperature_value": None,     # Experimental temperature value
    "temperature_unit": None,      # Temperature unit (°C)
    "ph": None,                    # Experimental pH
    "optimal_ph": None,            # Optimal pH for activity
    "optimal_temperature_value": None,  # Optimal temperature value
    "optimal_temperature_unit": None,   # Optimal temperature unit
    
    # === Metadata ===
    "notes": None,                 # Additional notes
    "confidence_score": None,      # Quality score (1-5)
}

# List of all schema fields
SCHEMA_FIELDS = list(FULL_SCHEMA.keys())


def normalize_record_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a record to ensure it has all schema fields.
    
    Missing fields are filled with None (null in JSON).
    This ensures consistent schema across all extracted records.
    
    Args:
        record: The raw record (may have missing fields)
        
    Returns:
        Record with all schema fields present
    """
    normalized = {}
    
    # First, add all schema fields with default None
    for field in SCHEMA_FIELDS:
        normalized[field] = record.get(field, FULL_SCHEMA[field])
    
    # Preserve any extra fields (like metadata, source, etc.)
    for key, value in record.items():
        if key not in SCHEMA_FIELDS:
            normalized[key] = value
    
    return normalized


def normalize_records_batch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize a batch of records to ensure consistent schema.
    
    Args:
        records: List of raw records
        
    Returns:
        List of records with consistent schema
    """
    return [normalize_record_schema(r) for r in records if isinstance(r, dict)]


class RecordMerger:
    """
    Merges duplicate enzyme kinetics records based on entity alignment.
    
    The algorithm:
    1. Group records by (enzyme_name, substrate, mutations) - the "entity key"
    2. Within each group, check if Km values are similar (if present)
    3. Merge records by keeping the most complete information
    """
    
    # Fields that define entity identity
    IDENTITY_FIELDS = ['enzyme_name', 'substrate', 'mutations']
    
    # Fields to merge (prefer non-null values)
    MERGEABLE_FIELDS = [
        'enzyme_full_name', 'enzyme_type', 'ec_number', 'gene_name',
        'uniprot_id', 'genbank_id', 'pdb_id', 'sequence',
        'organism', 'strain', 'is_recombinant', 'is_wild_type',
        'substrate_smiles', 'substrate_concentration',
        'Km_value', 'Km_unit', 'Vmax_value', 'Vmax_unit',
        'kcat_value', 'kcat_unit', 'kcat_Km_value', 'kcat_Km_unit',
        'degradation_efficiency', 'reaction_time_value', 'reaction_time_unit',
        'products', 'temperature_value', 'temperature_unit',
        'ph', 'optimal_ph', 'optimal_temperature_value', 'optimal_temperature_unit',
        'notes'
    ]
    
    # Numeric fields for similarity comparison
    NUMERIC_FIELDS = ['Km_value', 'Vmax_value', 'kcat_value', 'kcat_Km_value']
    
    # Source type priority (table > text > figure)
    SOURCE_PRIORITY = {'table': 3, 'figure': 2, 'text': 1}
    
    def __init__(self, 
                 km_tolerance: float = 0.1,
                 merge_different_units: bool = False,
                 prefer_table_source: bool = True):
        """
        Initialize the record merger.
        
        Args:
            km_tolerance: Relative tolerance for considering Km values as "same" (0.1 = 10%)
            merge_different_units: Whether to merge records with different units (risky)
            prefer_table_source: Whether to prefer table-sourced values over text
        """
        self.km_tolerance = km_tolerance
        self.merge_different_units = merge_different_units
        self.prefer_table_source = prefer_table_source
    
    # Common mutation patterns (e.g., E186R, A376I, V263N/V336I, etc.)
    MUTATION_PATTERN = r'^[A-Z]\d+[A-Z]$'
    MULTI_MUTATION_PATTERN = r'^[A-Z]\d+[A-Z](/[A-Z]\d+[A-Z])+$'  # V263N/V336I
    
    def is_mutation_name(self, name: str) -> bool:
        """Check if a name looks like a mutation (e.g., E186R, A376I, V263N/V336I)."""
        import re
        if not name:
            return False
        name = name.strip()
        # Single mutation: E186R
        if re.match(self.MUTATION_PATTERN, name):
            return True
        # Multi-mutation: V263N/V336I
        if re.match(self.MULTI_MUTATION_PATTERN, name):
            return True
        return False
    
    def normalize_enzyme_name(self, record: Dict) -> str:
        """
        Get normalized enzyme name, handling cases where mutation is in enzyme_name.
        
        If enzyme_name looks like a mutation (e.g., E186R), use enzyme_full_name 
        or a generic base name.
        """
        enzyme_name = (record.get('enzyme_name') or '').strip()
        
        # If enzyme_name is actually a mutation
        if self.is_mutation_name(enzyme_name):
            # Try to get real enzyme name from enzyme_full_name
            full_name = record.get('enzyme_full_name') or ''
            if full_name:
                # Extract base enzyme name from "CotA-laccase" -> "CotA"
                base = full_name.split('-')[0].strip()
                if base and not self.is_mutation_name(base):
                    return base.lower()
            # Default to generic laccase for this enzyme type
            enzyme_type = record.get('enzyme_type', '')
            if 'laccase' in enzyme_type.lower():
                return 'cota'  # Common laccase in the corpus
            return enzyme_name.lower()
        
        return enzyme_name.lower()
    
    def get_entity_key(self, record: Dict) -> Tuple:
        """
        Generate a key for entity matching.
        
        Returns a tuple of (enzyme_name, substrate, mutations) normalized.
        Handles cases where mutation is incorrectly placed in enzyme_name.
        """
        enzyme_name = self.normalize_enzyme_name(record)
        substrate = (record.get('substrate') or '').strip().lower()
        
        # Get mutations - could be in 'mutations' field or misplaced in 'enzyme_name'
        mutations = (record.get('mutations') or '').strip().lower()
        raw_enzyme_name = (record.get('enzyme_name') or '').strip()
        
        # If enzyme_name looks like a mutation and mutations field is empty or same
        if self.is_mutation_name(raw_enzyme_name):
            if not mutations or mutations == raw_enzyme_name.lower():
                mutations = raw_enzyme_name.lower()
        
        return (enzyme_name, substrate, mutations)
    
    def are_values_similar(self, val1: Optional[float], val2: Optional[float], 
                           tolerance: float = None) -> bool:
        """
        Check if two numeric values are similar within tolerance.
        """
        if val1 is None or val2 is None:
            return True  # Can't compare, assume compatible
        
        if tolerance is None:
            tolerance = self.km_tolerance
            
        if val1 == val2:
            return True
            
        if val1 == 0 and val2 == 0:
            return True
            
        # Relative difference
        avg = (abs(val1) + abs(val2)) / 2
        if avg == 0:
            return True
            
        diff = abs(val1 - val2) / avg
        return diff <= tolerance
    
    def are_units_compatible(self, unit1: Optional[str], unit2: Optional[str]) -> bool:
        """
        Check if two units are compatible (same or convertible).
        """
        if unit1 is None or unit2 is None:
            return True
            
        # Normalize units
        u1 = unit1.strip().lower().replace('⁻¹', '-1').replace('−', '-')
        u2 = unit2.strip().lower().replace('⁻¹', '-1').replace('−', '-')
        
        if u1 == u2:
            return True
        
        # Common equivalent units
        equivalents = [
            {'μm', 'um', 'µm', 'microm'},
            {'mm', 'millim'},
            {'s-1', 's^-1', '/s', 'per s'},
        ]
        
        for eq_set in equivalents:
            if u1 in eq_set and u2 in eq_set:
                return True
        
        return self.merge_different_units
    
    def can_merge(self, record1: Dict, record2: Dict) -> bool:
        """
        Determine if two records can be merged.
        
        Criteria:
        1. Same entity key (enzyme + substrate + mutations)
        2. Km values are similar (if both present)
        3. Units are compatible
        """
        # Must have same entity key
        if self.get_entity_key(record1) != self.get_entity_key(record2):
            return False
        
        # Check Km similarity
        km1 = record1.get('Km_value')
        km2 = record2.get('Km_value')
        unit1 = record1.get('Km_unit')
        unit2 = record2.get('Km_unit')
        
        # If both have Km values with same/similar units, check similarity
        if km1 is not None and km2 is not None:
            if not self.are_units_compatible(unit1, unit2):
                # Different units might mean different measurements
                # Be conservative: don't merge unless values are exactly equal
                if km1 != km2:
                    return False
            else:
                if not self.are_values_similar(km1, km2):
                    return False
        
        return True
    
    def get_source_priority(self, record: Dict) -> int:
        """Get priority score based on source type."""
        source = record.get('source_in_document', {})
        source_type = source.get('source_type', 'text')
        return self.SOURCE_PRIORITY.get(source_type, 1)
    
    def merge_two_records(self, record1: Dict, record2: Dict) -> Dict:
        """
        Merge two records into one, keeping the most complete information.
        
        Strategy:
        1. For identity fields: keep from higher priority source
        2. For mergeable fields: prefer non-null, then higher priority source
        3. For confidence_score: take max
        4. For notes: combine if different
        5. For source: track all sources
        6. Fix enzyme_name if it's a mutation (use real enzyme name from other record)
        """
        # Determine which record has higher priority source
        p1 = self.get_source_priority(record1)
        p2 = self.get_source_priority(record2)
        
        if self.prefer_table_source:
            primary, secondary = (record1, record2) if p1 >= p2 else (record2, record1)
        else:
            # Prefer higher confidence score
            c1 = record1.get('confidence_score', 0)
            c2 = record2.get('confidence_score', 0)
            primary, secondary = (record1, record2) if c1 >= c2 else (record2, record1)
        
        # Start with primary record
        merged = primary.copy()
        
        # Fix enzyme_name if primary has mutation as enzyme_name
        primary_enzyme = (primary.get('enzyme_name') or '').strip()
        secondary_enzyme = (secondary.get('enzyme_name') or '').strip()
        
        if self.is_mutation_name(primary_enzyme) and not self.is_mutation_name(secondary_enzyme):
            # Primary has mutation as enzyme_name, use secondary's real enzyme name
            merged['enzyme_name'] = secondary_enzyme
            # Also ensure mutations field is set
            if not merged.get('mutations'):
                merged['mutations'] = primary_enzyme
        elif self.is_mutation_name(secondary_enzyme) and not self.is_mutation_name(primary_enzyme):
            # Secondary has mutation, ensure mutations field is populated
            if not merged.get('mutations'):
                merged['mutations'] = secondary_enzyme
        
        # Merge fields from secondary if primary is null
        for field in self.MERGEABLE_FIELDS:
            primary_val = primary.get(field)
            secondary_val = secondary.get(field)
            
            # Handle empty lists/strings as null
            if primary_val is None or primary_val == '' or primary_val == []:
                if secondary_val is not None and secondary_val != '' and secondary_val != []:
                    merged[field] = secondary_val
            
            # Special handling for notes: combine if both present
            if field == 'notes':
                if primary_val and secondary_val and primary_val != secondary_val:
                    merged['notes'] = f"{primary_val}; {secondary_val}"
        
        # Take max confidence score
        merged['confidence_score'] = max(
            primary.get('confidence_score', 0),
            secondary.get('confidence_score', 0)
        )
        
        # Keep primary's source info (no need to track merge history)
        # User doesn't want merge tracking overhead
        
        return merged
    
    def merge_records(self, records: List[Dict]) -> List[Dict]:
        """
        Merge a list of records, combining duplicates.
        
        Returns a new list with duplicates merged.
        """
        if not records:
            return []
        
        # Group by entity key
        groups: Dict[Tuple, List[Dict]] = defaultdict(list)
        for record in records:
            key = self.get_entity_key(record)
            groups[key].append(record)
        
        merged_records = []
        merge_stats = {'total': len(records), 'merged_groups': 0, 'records_merged': 0}
        
        for key, group in groups.items():
            if len(group) == 1:
                merged_records.append(group[0])
            else:
                # Try to merge records in this group
                # Use a greedy approach: merge pairs that can be merged
                remaining = group.copy()
                merged_in_group = []
                
                while remaining:
                    current = remaining.pop(0)
                    
                    # Try to merge with other remaining records
                    merged_any = True
                    while merged_any:
                        merged_any = False
                        for i, other in enumerate(remaining):
                            if self.can_merge(current, other):
                                current = self.merge_two_records(current, other)
                                remaining.pop(i)
                                merged_any = True
                                merge_stats['records_merged'] += 1
                                break
                    
                    merged_in_group.append(current)
                
                if len(merged_in_group) < len(group):
                    merge_stats['merged_groups'] += 1
                
                merged_records.extend(merged_in_group)
        
        logger.info(f"Merge stats: {merge_stats['total']} records -> {len(merged_records)} records "
                   f"({merge_stats['records_merged']} merged, {merge_stats['merged_groups']} groups)")
        
        return merged_records


def deduplicate_records(records: List[Dict], 
                        km_tolerance: float = 0.1,
                        prefer_table: bool = True) -> List[Dict]:
    """
    Convenience function to deduplicate a list of records.
    
    Args:
        records: List of enzyme kinetics records
        km_tolerance: Tolerance for Km value comparison (0.1 = 10%)
        prefer_table: Whether to prefer table-sourced data
        
    Returns:
        Deduplicated list of records
    """
    merger = RecordMerger(
        km_tolerance=km_tolerance,
        prefer_table_source=prefer_table
    )
    return merger.merge_records(records)


class ConditionExtractor:
    """
    使用 LLM 从文本中提取 optimal pH 和 temperature，并补全到记录中。
    
    解决的问题：动力学参数 (Km, kcat) 从表格提取，但 optimal pH/temperature 在正文中，
    LLM 提取表格时无法关联这些条件信息。
    """
    
    # 条件补全的 prompt 模板
    CONDITION_PROMPT = """从以下科学论文文本中，提取酶的最优反应条件 (optimal pH 和 optimal temperature)。

文本内容：
{text}

需要补全条件的酶/突变体列表：
{enzymes_list}

请为每个酶/突变体找出：
1. optimal_ph: 最优 pH 值（数字，如 4.0, 7.5）
2. optimal_temperature_value: 最优温度值（数字，单位°C，如 37, 90）

返回 JSON 格式：
```json
{{
  "conditions": [
    {{
      "enzyme_name": "CotA",
      "mutation": null,
      "optimal_ph": 4.0,
      "optimal_temperature_value": 90
    }},
    {{
      "enzyme_name": "CotA", 
      "mutation": "E186A",
      "optimal_ph": 4.0,
      "optimal_temperature_value": 80
    }}
  ]
}}
```

重要规则：
1. WT/wild-type/野生型 对应 mutation=null
2. 文本说 "The optimal pH of the WT and E186A were both pH 4" 意味着：
   - WT (mutation=null) 的 optimal_ph = 4
   - E186A 的 optimal_ph = 4
3. 文本说 "optimal temperatures of the WT and E186A were 90°C and 80°C respectively" 意味着：
   - WT 的 optimal_temperature_value = 90
   - E186A 的 optimal_temperature_value = 80
4. 只返回能从文本确定的值，不确定的字段不要包含
5. 返回纯 JSON，不要其他内容"""

    @classmethod
    def fill_conditions_with_llm(cls, records: List[Dict], source_text: str, llm_client) -> List[Dict]:
        """
        使用 LLM 从文本中提取条件并补全到记录中。
        
        Args:
            records: 需要补全的记录列表
            source_text: 论文原文（用于提取条件）
            llm_client: LLM 客户端（需要有 .chat() 方法）
            
        Returns:
            补全后的记录列表
        """
        import json as json_module
        
        if not records or not source_text or not llm_client:
            return records
        
        # 找出缺少条件的记录
        records_needing_conditions = []
        for i, record in enumerate(records):
            if record.get('optimal_ph') is None or record.get('optimal_temperature_value') is None:
                records_needing_conditions.append({
                    'index': i,
                    'enzyme_name': record.get('enzyme_name', 'Unknown'),
                    'mutation': record.get('mutations')
                })
        
        if not records_needing_conditions:
            return records
        
        # 构建酶列表字符串
        enzymes_list = '\n'.join([
            f"- {r['enzyme_name']}" + (f" ({r['mutation']})" if r['mutation'] else " (WT/wild-type)")
            for r in records_needing_conditions
        ])
        
        # 截取文本（避免太长）
        max_text_len = 8000
        text_for_llm = source_text[:max_text_len] if len(source_text) > max_text_len else source_text
        
        # 构建 prompt
        prompt = cls.CONDITION_PROMPT.format(
            text=text_for_llm,
            enzymes_list=enzymes_list
        )
        
        try:
            # 调用 LLM - 使用封装后的 .chat() 方法
            response_text = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True  # 请求 JSON 格式输出
            )
            
            response_text = response_text.strip()
            
            # 解析 JSON
            # 移除 markdown 代码块标记
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            result = json_module.loads(response_text)
            conditions_list = result.get('conditions', [])
            
            # 将条件填充到对应记录
            filled_count = 0
            for cond in conditions_list:
                cond_enzyme = cond.get('enzyme_name', '').lower()
                cond_mutation = cond.get('mutation')
                
                for record in records:
                    rec_enzyme = record.get('enzyme_name', '').lower()
                    rec_mutation = record.get('mutations')
                    
                    # 匹配酶名和突变体
                    enzyme_match = cond_enzyme in rec_enzyme or rec_enzyme in cond_enzyme
                    mutation_match = (
                        (not cond_mutation and not rec_mutation) or  # 都是 WT
                        (cond_mutation and rec_mutation and cond_mutation.upper() == rec_mutation.upper())
                    )
                    
                    if enzyme_match and mutation_match:
                        # 填充缺失的条件
                        if record.get('optimal_ph') is None and cond.get('optimal_ph') is not None:
                            record['optimal_ph'] = float(cond['optimal_ph'])
                            filled_count += 1
                        if record.get('optimal_temperature_value') is None and cond.get('optimal_temperature_value') is not None:
                            record['optimal_temperature_value'] = float(cond['optimal_temperature_value'])
                            record['optimal_temperature_unit'] = '°C'
                            filled_count += 1
            
            if filled_count > 0:
                logger.info(f"ConditionExtractor (LLM): Filled {filled_count} condition fields")
                
        except Exception as e:
            logger.warning(f"ConditionExtractor LLM call failed: {e}")
        
        return records
    
    @classmethod
    def fill_conditions_to_records(cls, records: List[Dict], source_text: str, llm_client=None) -> List[Dict]:
        """
        补全记录中缺失的条件信息。
        
        如果提供了 llm_client，使用 LLM 智能提取；否则跳过。
        
        Args:
            records: 记录列表
            source_text: 论文原文
            llm_client: LLM 客户端（可选）
            
        Returns:
            补全后的记录列表
        """
        if not records or not source_text:
            return records
        
        if llm_client:
            return cls.fill_conditions_with_llm(records, source_text, llm_client)
        else:
            logger.debug("ConditionExtractor: No LLM client provided, skipping condition filling")
            return records


def process_extraction_results(input_path: Path, output_path: Path = None,
                               km_tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Process extraction results from a directory, merging duplicates.
    
    Args:
        input_path: Path to directory containing JSON files or a single JSON file
        output_path: Path to output directory (optional, defaults to input_path + '_merged')
        km_tolerance: Tolerance for Km value comparison
        
    Returns:
        Statistics about the merging process
    """
    input_path = Path(input_path)
    
    if output_path is None:
        if input_path.is_file():
            output_path = input_path.parent / f"{input_path.stem}_merged.json"
        else:
            output_path = input_path.parent / f"{input_path.name}_merged"
    
    output_path = Path(output_path)
    
    merger = RecordMerger(km_tolerance=km_tolerance)
    stats = {
        'files_processed': 0,
        'total_original': 0,
        'total_merged': 0,
        'per_file': {}
    }
    
    if input_path.is_file():
        # Single file
        with open(input_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        original_count = len(records)
        merged_records = merger.merge_records(records)
        merged_count = len(merged_records)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_records, f, ensure_ascii=False, indent=2)
        
        stats['files_processed'] = 1
        stats['total_original'] = original_count
        stats['total_merged'] = merged_count
        stats['per_file'][input_path.name] = {
            'original': original_count,
            'merged': merged_count
        }
    else:
        # Directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        for json_file in input_path.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            original_count = len(records)
            merged_records = merger.merge_records(records)
            merged_count = len(merged_records)
            
            output_file = output_path / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_records, f, ensure_ascii=False, indent=2)
            
            stats['files_processed'] += 1
            stats['total_original'] += original_count
            stats['total_merged'] += merged_count
            stats['per_file'][json_file.name] = {
                'original': original_count,
                'merged': merged_count
            }
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge duplicate enzyme kinetics records')
    parser.add_argument('input', help='Input JSON file or directory')
    parser.add_argument('-o', '--output', help='Output path (optional)')
    parser.add_argument('-t', '--tolerance', type=float, default=0.1,
                       help='Km value tolerance for matching (default: 0.1 = 10%%)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    stats = process_extraction_results(input_path, output_path, args.tolerance)
    
    print(f"\n{'='*60}")
    print("Merge Statistics")
    print(f"{'='*60}")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total records: {stats['total_original']} -> {stats['total_merged']}")
    print(f"Records reduced: {stats['total_original'] - stats['total_merged']} "
          f"({(1 - stats['total_merged']/stats['total_original'])*100:.1f}%)")
    print(f"\nPer-file breakdown:")
    for filename, file_stats in stats['per_file'].items():
        print(f"  {filename}: {file_stats['original']} -> {file_stats['merged']}")
