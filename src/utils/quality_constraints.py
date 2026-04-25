"""
质量约束过滤器 - 确保提取的数据满足关键质量要求

三大约束：
1. 序列可获取性：必须有数据库ID或序列信息（排除粗提取液/混合物）
2. 霉菌毒素底物：底物必须是已知的霉菌毒素
3. 解毒验证：产物应该是减毒或无毒的（排除生物活化反应）
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityConstraintFilter:
    """质量约束过滤器"""
    
    # 1. 已知的霉菌毒素列表（标准名称）
    MYCOTOXINS = {
        # Aflatoxins
        'aflatoxin b1', 'aflatoxin b2', 'aflatoxin g1', 'aflatoxin g2',
        'aflatoxin m1', 'aflatoxin m2', 'aflatoxin p1',
        'afb1', 'afb2', 'afg1', 'afg2', 'afm1', 'afm2', 'afp1',
        
        # Ochratoxins
        'ochratoxin a', 'ochratoxin b', 'ochratoxin c',
        'ota', 'otb', 'otc',
        
        # Trichothecenes
        'deoxynivalenol', 'nivalenol', 't-2 toxin', 'ht-2 toxin',
        '3-acetyldeoxynivalenol', '15-acetyldeoxynivalenol',
        'diacetoxyscirpenol',
        'don', 'niv', 't-2', 'ht-2', '3-adon', '15-adon', 'das',
        
        # Fumonisins
        'fumonisin b1', 'fumonisin b2', 'fumonisin b3',
        'hydrolyzed fumonisin b1',
        'fb1', 'fb2', 'fb3', 'hfb1',
        
        # Zearalenones
        'zearalenone', 'alpha-zearalenol', 'beta-zearalenol',
        'zen', 'zea', 'α-zel', 'β-zel', 'α-zol', 'β-zol',
        
        # Others
        'patulin', 'citrinin', 'sterigmatocystin',
        'cyclopiazonic acid', 'roquefortine c', 'mycophenolic acid',
        'alternariol', 'alternariol monomethyl ether',
        'tenuazonic acid', 'moniliformin', 'beauvericin',
        'pat', 'cit', 'ste', 'st', 'cpa', 'roq-c', 'mpa',
        'aoh', 'ame', 'tea', 'mon', 'bea',
        
        # Enniatins
        'enniatin', 'enniatin a', 'enniatin b', 'enniatin b1',
        'enn', 'enna', 'ennb', 'ennb1',
        
        # Ergot alkaloids (partial list)
        'ergotamine', 'ergocristine', 'ergocryptine',
    }
    
    # 2. 非霉菌毒素底物（应该被排除的）
    NON_MYCOTOXIN_SUBSTRATES = {
        # Generic enzyme substrates
        'abts', '2,2\'-azino-bis(3-ethylbenzothiazoline-6-sulfonic acid)',
        'syringaldazine', 'sgz',
        '2,6-dimethoxyphenol', 'dmp',
        'guaiacol', 'catechol', 'hydroquinone',
        'veratryl alcohol',
        
        # Dyes
        'remazol brilliant blue r', 'rbbr',
        'reactive black 5', 'rb5',
        'methylene blue', 'malachite green',
        'congo red', 'bromophenol blue',
        
        # Other compounds
        'hydrogen peroxide', 'h2o2', 'h₂o₂',
        'manganese', 'mn2+', 'mn(ii)',
        'phenol', 'aniline', 'benzene',
    }
    
    # 3. 粗提取液/混合物关键词（应该被排除的）
    CRUDE_EXTRACT_KEYWORDS = [
        'crude extract', 'crude enzyme', 'cell lysate', 'lysate',
        'culture supernatant', 'supernatant', 'fermentation broth',
        'partially purified', 'commercial', 'commercial preparation',
        'enzyme cocktail', 'enzyme mixture', 'mixed enzymes',
        'fungal extract', 'bacterial extract', 'microbial extract',
        'sigma', 'sigma-aldrich', 'novozyme', 'novozymes',
    ]
    
    # 4. 生物活化关键词（产物毒性增强，应该被排除的）
    BIOACTIVATION_KEYWORDS = [
        'bioactivation', 'activated', 'activation',
        'more toxic', 'increased toxicity', 'enhanced toxicity',
        'potentiated toxicity', 'toxic metabolite',
        'carcinogenic', 'mutagenic metabolite',
        'epoxide', 'afbo',  # AFB1 → AFBO (highly toxic)
    ]
    
    # 5. 解毒关键词（产物毒性降低，应该被接受的）
    DETOXIFICATION_KEYWORDS = [
        'detoxification', 'detoxified', 'less toxic', 'reduced toxicity',
        'decreased toxicity', 'non-toxic', 'nontoxic', 'non toxic',
        'non-mutagenic', 'decreased cytotoxicity', 'reduced cytotoxicity',
        'loss of toxicity', 'detoxify', 'degradation',
    ]
    
    def __init__(
        self,
        require_sequence: bool = True,
        require_mycotoxin: bool = True,
        check_detoxification: bool = True,
        strict_mode: bool = False
    ):
        """
        Args:
            require_sequence: 是否要求序列可获取性
            require_mycotoxin: 是否要求底物必须是霉菌毒素
            check_detoxification: 是否检查产物毒性
            strict_mode: 严格模式（在不确定时拒绝记录）
        """
        self.require_sequence = require_sequence
        self.require_mycotoxin = require_mycotoxin
        self.check_detoxification = check_detoxification
        self.strict_mode = strict_mode
        
        logger.info("QualityConstraintFilter initialized")
        logger.info(f"  - Require sequence: {require_sequence}")
        logger.info(f"  - Require mycotoxin substrate: {require_mycotoxin}")
        logger.info(f"  - Check detoxification: {check_detoxification}")
        logger.info(f"  - Strict mode: {strict_mode}")
    
    def filter_records(
        self,
        records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        过滤记录列表
        
        Args:
            records: 原始记录列表
            
        Returns:
            (filtered_records, statistics)
        """
        if not records:
            return [], self._empty_stats()
        
        filtered = []
        rejected = []
        stats = {
            'total': len(records),
            'passed': 0,
            'rejected_no_sequence': 0,
            'rejected_non_mycotoxin': 0,
            'rejected_bioactivation': 0,
            'rejected_other': 0,
        }
        
        for record in records:
            passed, reason = self.check_record(record)
            
            if passed:
                filtered.append(record)
                stats['passed'] += 1
            else:
                rejected.append({
                    'record': record,
                    'reason': reason
                })
                
                # 统计拒绝原因
                reason_lower = reason.lower()
                if ('sequence' in reason_lower or 
                    'crude' in reason_lower or 
                    'extract' in reason_lower or 
                    'commercial' in reason_lower or
                    'mixture' in reason_lower or
                    'lysate' in reason_lower):
                    stats['rejected_no_sequence'] += 1
                elif 'mycotoxin' in reason_lower or 'substrate' in reason_lower:
                    stats['rejected_non_mycotoxin'] += 1
                elif 'toxic' in reason_lower or 'activation' in reason_lower:
                    stats['rejected_bioactivation'] += 1
                else:
                    stats['rejected_other'] += 1
        
        stats['rejected'] = len(rejected)
        stats['rejection_rate'] = stats['rejected'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        return filtered, stats
    
    def check_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """
        检查单条记录是否满足质量约束
        
        Returns:
            (passed, reason) - passed=True表示通过，reason为拒绝原因（如果有）
        """
        # 约束1：序列可获取性
        if self.require_sequence:
            passed, reason = self._check_sequence_availability(record)
            if not passed:
                return False, f"REJECT: {reason}"
        
        # 约束2：霉菌毒素底物
        if self.require_mycotoxin:
            passed, reason = self._check_mycotoxin_substrate(record)
            if not passed:
                return False, f"REJECT: {reason}"
        
        # 约束3：解毒验证
        if self.check_detoxification:
            passed, reason = self._check_detoxification(record)
            if not passed:
                return False, f"REJECT: {reason}"
        
        return True, "PASS"
    
    def _check_sequence_availability(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """检查序列可获取性"""
        # 方式1：有数据库ID
        has_uniprot = bool(record.get('uniprot_id'))
        has_genbank = bool(record.get('genbank_id'))
        has_pdb = bool(record.get('pdb_id'))
        has_sequence = bool(record.get('sequence'))
        
        if has_uniprot or has_genbank or has_pdb or has_sequence:
            return True, "Has sequence identifier"
        
        # 方式2：有清晰的基因名称 + 生物体
        has_gene = bool(record.get('gene_name'))
        has_organism = bool(record.get('organism'))
        is_recombinant = record.get('is_recombinant') is True
        
        if has_gene and has_organism:
            # 如果是重组表达，更有信心
            if is_recombinant:
                return True, "Recombinant enzyme with gene name and organism"
            # 否则也接受（但置信度较低）
            return True, "Has gene name and organism"
        
        # 方式3：检查是否是粗提取液/混合物（应该被拒绝）
        enzyme_name = record.get('enzyme_name', '').lower()
        enzyme_full_name = record.get('enzyme_full_name', '').lower()
        notes = record.get('notes', '').lower()
        
        # 组合所有文本进行检查
        combined_text = f"{enzyme_name} {enzyme_full_name} {notes}"
        
        for keyword in self.CRUDE_EXTRACT_KEYWORDS:
            if keyword in combined_text:
                return False, f"Crude extract/mixture detected: '{keyword}'"
        
        # 严格模式：如果没有明确的序列信息，拒绝
        if self.strict_mode:
            return False, "No sequence identifier or gene name (strict mode)"
        
        # 宽松模式：允许通过（假设后续可以人工审核）
        return True, "No clear sequence info but passed in permissive mode"
    
    def _check_mycotoxin_substrate(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """检查底物是否是霉菌毒素"""
        substrate = record.get('substrate', '').lower().strip()
        
        if not substrate:
            return False, "No substrate specified"
        
        # 首先检查是否是已知的非霉菌毒素底物
        for non_mycotoxin in self.NON_MYCOTOXIN_SUBSTRATES:
            if non_mycotoxin in substrate:
                return False, f"Non-mycotoxin substrate: '{substrate}'"
        
        # 然后检查是否是已知的霉菌毒素
        for mycotoxin in self.MYCOTOXINS:
            if mycotoxin in substrate:
                return True, f"Mycotoxin substrate: '{substrate}'"
        
        # 未识别的底物
        if self.strict_mode:
            return False, f"Unknown substrate '{substrate}' (strict mode)"
        else:
            # 宽松模式：假设是霉菌毒素研究中的未知霉菌毒素
            logger.warning(f"Unknown substrate '{substrate}' - passed in permissive mode")
            return True, f"Unknown substrate '{substrate}' - assumed mycotoxin"
    
    def _check_detoxification(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """检查产物是否是解毒产物"""
        # 获取产物信息
        products = record.get('products', [])
        notes = record.get('notes', '').lower()
        
        # 组合所有相关文本
        product_texts = []
        for prod in products:
            if isinstance(prod, dict):
                product_texts.append(prod.get('name', '').lower())
                product_texts.append(str(prod.get('toxicity_change', '')).lower())
        
        combined_text = ' '.join(product_texts) + ' ' + notes
        
        # 检查是否有生物活化关键词（应该被拒绝）
        for keyword in self.BIOACTIVATION_KEYWORDS:
            if keyword in combined_text:
                return False, f"Bioactivation detected: '{keyword}'"
        
        # 检查是否有解毒关键词（应该被接受）
        for keyword in self.DETOXIFICATION_KEYWORDS:
            if keyword in combined_text:
                return True, f"Detoxification confirmed: '{keyword}'"
        
        # 如果没有明确的毒性信息
        if not products or not combined_text.strip():
            # 默认假设：霉菌毒素研究是为了解毒，而非生物活化
            return True, "No toxicity info - assumed detoxification"
        
        # 有产物但没有明确的毒性描述
        if self.strict_mode:
            return False, "No clear detoxification evidence (strict mode)"
        else:
            return True, "Assumed detoxification (permissive mode)"
    
    def _empty_stats(self) -> Dict[str, Any]:
        """返回空统计信息"""
        return {
            'total': 0,
            'passed': 0,
            'rejected': 0,
            'rejected_no_sequence': 0,
            'rejected_non_mycotoxin': 0,
            'rejected_bioactivation': 0,
            'rejected_other': 0,
            'rejection_rate': 0.0,
        }
    
    def print_statistics(self, stats: Dict[str, Any]):
        """打印过滤统计信息"""
        print(f"\n{'='*80}")
        print("🔍 Quality Constraint Filtering Statistics")
        print(f"{'='*80}")
        print(f"Total records:           {stats['total']}")
        print(f"✅ Passed:               {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "✅ Passed: 0")
        print(f"❌ Rejected:             {stats['rejected']} ({stats['rejection_rate']:.1f}%)")
        
        if stats['rejected'] > 0:
            print(f"\nRejection reasons:")
            print(f"  - No sequence:         {stats['rejected_no_sequence']}")
            print(f"  - Non-mycotoxin:       {stats['rejected_non_mycotoxin']}")
            print(f"  - Bioactivation:       {stats['rejected_bioactivation']}")
            print(f"  - Other:               {stats['rejected_other']}")
        
        print(f"{'='*80}\n")


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建过滤器
    filter = QualityConstraintFilter(
        require_sequence=True,
        require_mycotoxin=True,
        check_detoxification=True,
        strict_mode=False
    )
    
    # 测试记录
    test_records = [
        # 好记录：有UniProt ID，霉菌毒素底物，解毒产物
        {
            'enzyme_name': 'CotA',
            'uniprot_id': 'P07788',
            'organism': 'Bacillus subtilis',
            'substrate': 'Aflatoxin B1',
            'products': [{'name': 'less toxic product', 'toxicity_change': 'reduced'}],
            'Km_value': 0.5
        },
        # 差记录：粗提取液
        {
            'enzyme_name': 'Crude laccase extract',
            'organism': 'Pleurotus sp.',
            'substrate': 'Aflatoxin B1',
            'degradation_efficiency': 80.0
        },
        # 差记录：非霉菌毒素底物
        {
            'enzyme_name': 'Laccase',
            'uniprot_id': 'P12345',
            'organism': 'Trametes versicolor',
            'substrate': 'ABTS',
            'Km_value': 2.0
        },
        # 差记录：生物活化
        {
            'enzyme_name': 'CYP450',
            'genbank_id': 'ABC123',
            'organism': 'Homo sapiens',
            'substrate': 'Aflatoxin B1',
            'products': [{'name': 'AFBO', 'toxicity_change': 'increased toxicity'}],
            'notes': 'P450 bioactivation to toxic epoxide'
        },
    ]
    
    # 过滤
    filtered, stats = filter.filter_records(test_records)
    
    # 打印结果
    filter.print_statistics(stats)
    
    print("✅ Passed records:")
    for i, record in enumerate(filtered, 1):
        print(f"  {i}. {record['enzyme_name']} + {record['substrate']}")
