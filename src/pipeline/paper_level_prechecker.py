"""
论文级别预检查器 - 在调用LLM前快速判断论文是否相关

目的：避免为完全不相关的论文调用LLM，节省大量token和时间
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PaperLevelPrechecker:
    """
    论文级别预检查器

    检查论文是否包含：
    1. 霉菌毒素关键词
    2. 酶动力学相关关键词
    3. 降解相关关键词

    如果论文完全不相关，直接跳过，节省API调用
    """

    # 霉菌毒素关键词（必须包含至少一个）
    MYCOTOXIN_KEYWORDS = [
        # 黄曲霉毒素
        'aflatoxin', 'afb1', 'afb2', 'afg1', 'afg2', 'afm1', 'afm2',
        'aflatoxicol',
        # 赭曲霉毒素
        'ochratoxin', 'ota', 'otb', 'otc',
        # 单端孢霉烯族
        'deoxynivalenol', 'don', 'nivalenol', 'niv',
        't-2', 'ht-2', '3-adon', '15-adon', 'das',
        # 伏马毒素
        'fumonisin', 'fb1', 'fb2', 'fb3',
        # 玉米赤霉烯酮
        'zearalenone', 'zen', 'α-zel', 'β-zel', 'alpha-zearalenol', 'beta-zearalenol',
        # 其他
        'patulin', 'citrinin', 'sterigmatocystin',
        'cyclopiazonic acid', 'cpa',
        'alternariol', 'aoh',
    ]

    # 酶动力学关键词（必须包含至少一个）
    ENZYME_KINETICS_KEYWORDS = [
        # 动力学参数
        'km', 'kcat', 'kcat/km', 'vmax', 'ic50', 'ki',
        'km_value', 'kcat_value', 'kcat_km_value',
        'michaelis', 'menten', 'turnover',
        # 酶活性
        'enzyme activity', 'specific activity',
        'enzyme assay', 'enzymatic assay',
        # 降解相关
        'degradation', 'degrade', 'degraded',
        'detoxification', 'detoxify',
        'biotransformation', 'transform',
        # 产物
        'metabolite', 'product', 'degradation product',
    ]

    # 单位关键词（辅助判断）
    UNIT_KEYWORDS = [
        'μm', 'um', 'mm', 'nm', 'µm',  # 浓度单位
        's⁻¹', 's^-1', 'min⁻¹', 'min^-1',  # 速率单位
        'μmol', 'umol', 'nmol',  # 摩尔单位
        'unit/ml', 'u/ml', 'iu',  # 酶活单位
    ]

    REVIEW_TITLE_PATTERNS = [
        r"\bmeta-analysis\b",
        r"\bsystematic review\b",
        r"\boverview\b",
        r"\brecent advances\b",
        r"\bstate of the art\b",
        r"\bcurrent status\b",
        r"\bprogress in\b",
        # 只匹配 "review" 作为独立标题或综述标记，排除 "for a review see" 等引用上下文
        r"(?<!\bfor a )\breview\b(?!\s+(?:see|of|in|ref))",
    ]

    def __init__(
        self,
        min_mycotoxin_hits: int = 1,
        min_kinetics_hits: int = 2,
        enable_unit_check: bool = True
    ):
        """
        Args:
            min_mycotoxin_hits: 最少霉菌毒素关键词命中数
            min_kinetics_hits: 最少动力学关键词命中数
            enable_unit_check: 是否启用单位检查（辅助判断）
        """
        self.min_mycotoxin_hits = min_mycotoxin_hits
        self.min_kinetics_hits = min_kinetics_hits
        self.enable_unit_check = enable_unit_check

        # 编译正则表达式（大小写不敏感）
        self.mycotoxin_patterns = [
            re.compile(rf'\b{kw}\b', re.IGNORECASE)
            for kw in self.MYCOTOXIN_KEYWORDS
        ]
        self.kinetics_patterns = [
            re.compile(rf'\b{kw}\b', re.IGNORECASE)
            for kw in self.ENZYME_KINETICS_KEYWORDS
        ]
        self.unit_patterns = [
            re.compile(rf'{kw}', re.IGNORECASE)
            for kw in self.UNIT_KEYWORDS
        ]

        logger.info("PaperLevelPrechecker initialized")
        logger.info(f"  - Min mycotoxin hits: {min_mycotoxin_hits}")
        logger.info(f"  - Min kinetics hits: {min_kinetics_hits}")

    def should_skip_paper(
        self,
        paper_dir: Path,
        full_md_path: Optional[Path] = None,
        max_chars: int = 30000  # 增加到30000字符，覆盖Abstract+Introduction+Results
    ) -> Dict[str, Any]:
        """
        判断论文是否应该跳过（不相关）

        Args:
            paper_dir: 论文目录路径
            full_md_path: full.md文件路径（如果不提供，则自动查找）
            max_chars: 读取的最大字符数（默认30000，覆盖论文的主要部分）

        Returns:
            {
                "should_skip": bool,  # 是否跳过
                "reason": str,       # 跳过原因
                "mycotoxin_hits": int,  # 霉菌毒素关键词命中数
                "kinetics_hits": int,   # 动力学关键词命中数
                "unit_hits": int,       # 单位命中数
            }
        """
        # 1. 查找full.md文件
        if full_md_path is None:
            full_md_path = paper_dir / "full.md"

        if not full_md_path.exists():
            # 没有full.md，不过滤（保守策略）
            return {
                "should_skip": False,
                "reason": "No full.md found, cannot pre-check",
                "mycotoxin_hits": 0,
                "kinetics_hits": 0,
                "unit_hits": 0
            }

        # 2. 读取full.md内容（读取前30000字符，覆盖Abstract+Introduction+Results）
        try:
            with open(full_md_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(max_chars)
                # 不再使用.lower()，因为正则表达式已经有 re.IGNORECASE
        except Exception as e:
            logger.warning(f"Failed to read {full_md_path}: {e}")
            return {
                "should_skip": False,
                "reason": f"Read error: {e}",
                "mycotoxin_hits": 0,
                "kinetics_hits": 0,
                "unit_hits": 0
            }

        document_type = self.classify_document_type(content)
        if document_type == "review":
            logger.info(f"  ⏭️  Skipping review paper: {paper_dir.name}")
            return {
                "should_skip": True,
                "reason": "Document_Type=review; QC_Status=exclude_review_article",
                "mycotoxin_hits": 0,
                "kinetics_hits": 0,
                "unit_hits": 0,
                "Document_Type": "review",
                "Extraction_Allowed": False,
                "QC_Status": "exclude_review_article",
            }

        # 3. 统计关键词命中数
        mycotoxin_hits = 0
        kinetics_hits = 0
        unit_hits = 0

        for pattern in self.mycotoxin_patterns:
            matches = pattern.findall(content)
            mycotoxin_hits += len(matches)

        for pattern in self.kinetics_patterns:
            matches = pattern.findall(content)
            kinetics_hits += len(matches)

        if self.enable_unit_check:
            for pattern in self.unit_patterns:
                matches = pattern.findall(content)
                unit_hits += len(matches)

        # 4. 判断是否跳过
        should_skip = False
        reason = ""

        # 规则1：必须包含霉菌毒素关键词
        if mycotoxin_hits < self.min_mycotoxin_hits:
            should_skip = True
            reason = f"No mycotoxin keywords found (hits: {mycotoxin_hits}, required: {self.min_mycotoxin_hits})"
        # 规则2：必须包含酶动力学关键词
        elif kinetics_hits < self.min_kinetics_hits:
            should_skip = True
            reason = f"No enzyme kinetics keywords found (hits: {kinetics_hits}, required: {self.min_kinetics_hits})"

        # 日志
        if should_skip:
            logger.info(f"  ⏭️  Skipping paper: {paper_dir.name}")
            logger.info(f"     Reason: {reason}")
        else:
            logger.info(f"  ✓ Paper passed pre-check: {paper_dir.name}")
            logger.info(f"     Mycotoxin hits: {mycotoxin_hits}, Kinetics hits: {kinetics_hits}, Unit hits: {unit_hits}")

        return {
            "should_skip": should_skip,
            "reason": reason,
            "mycotoxin_hits": mycotoxin_hits,
            "kinetics_hits": kinetics_hits,
            "unit_hits": unit_hits,
            "Document_Type": document_type,
            "Extraction_Allowed": not should_skip,
            "QC_Status": "candidate_primary_research" if not should_skip else "precheck_skipped",
        }

    def classify_document_type(self, content: str) -> str:
        """
        Conservative document type classifier for primary dataset filtering.

        Returns one of: primary_research, review, method, database,
        book_chapter, unclear.
        """
        if not content:
            return "unclear"

        head = content[:8000]
        first_lines = "\n".join(head.splitlines()[:30]).lower()
        head_lower = head.lower()

        if re.search(r"\bbook chapter\b", head_lower):
            return "book_chapter"
        if re.search(r"\bdatabase\b|\bweb server\b|\brepository\b", first_lines):
            return "database"
        if re.search(r"\bmethod\b|\bprotocol\b", first_lines) and not re.search(r"\bresults\b", head_lower):
            return "method"

        review_hits = 0
        for pattern in self.REVIEW_TITLE_PATTERNS:
            if re.search(pattern, first_lines, flags=re.IGNORECASE):
                review_hits += 2
            elif re.search(pattern, head_lower, flags=re.IGNORECASE):
                review_hits += 1

        review_context = [
            "summarizes", "literature review", "published studies",
            "previous studies", "reported enzymes", "comparison of reported",
            "summary of", "recent progress"
        ]
        review_hits += sum(1 for phrase in review_context if phrase in head_lower)

        primary_markers = [
            "materials and methods", "materialsandmethods", "experimental",
            "we cloned", "we expressed", "we purified",
            "enzyme assay", "kinetic parameters were determined",
            "experimental procedures", "experimentalprocedures",
        ]
        primary_hits = sum(1 for phrase in primary_markers if phrase in head_lower)

        if review_hits >= 2 and primary_hits == 0:
            return "review"
        if primary_hits > 0:
            return "primary_research"
        if review_hits >= 3:
            return "review"
        return "unclear"

    def batch_check_papers(
        self,
        paper_dirs: list[Path]
    ) -> Dict[str, Any]:
        """
        批量检查论文列表

        Args:
            paper_dirs: 论文目录路径列表

        Returns:
            {
                "total": int,            # 总论文数
                "passed": int,          # 通过检查
                "skipped": int,         # 跳过论文数
                "skip_rate": float,     # 跳过率
                "results": dict,        # 详细结果 {paper_name: result_dict}
            }
        """
        results = {}
        passed = 0
        skipped = 0

        for paper_dir in paper_dirs:
            result = self.should_skip_paper(paper_dir)
            results[paper_dir.name] = result

            if result['should_skip']:
                skipped += 1
            else:
                passed += 1

        total = len(paper_dirs)
        skip_rate = skipped / total if total > 0 else 0

        logger.info(f"\n📊 Pre-check Summary:")
        logger.info(f"  Total papers: {total}")
        logger.info(f"  Passed: {passed} ({(1-skip_rate)*100:.1f}%)")
        logger.info(f"  Skipped: {skipped} ({skip_rate*100:.1f}%)")
        logger.info(f"  💰 Estimated savings: ~{skipped * 5} API calls (5 calls/paper)")

        return {
            "total": total,
            "passed": passed,
            "skipped": skipped,
            "skip_rate": skip_rate,
            "results": results
        }
