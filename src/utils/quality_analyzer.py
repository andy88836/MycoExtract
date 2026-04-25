"""
质量分析器 - 用于分析提取数据质量并生成统计报告

替代HITL逐篇审核，采用统计分析+抽样优化的策略：
1. 按confidence_score分类统计
2. 分析常见失败模式
3. 抽样低质量案例用于prompt优化
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime


class QualityAnalyzer:
    """数据质量分析器"""

    def __init__(self, sample_size: int = 20):
        """
        Args:
            sample_size: 低质量数据抽样数量
        """
        self.sample_size = sample_size

    def analyze_quality(self, all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        分析所有提取数据的质量分布

        Args:
            all_results: {paper_name: [records]}

        Returns:
            质量分析报告
        """
        # 按分数分类（3级：好、中、差）
        quality_buckets = {
            "good": [],    # score 3 (好)
            "medium": [],  # score 2 (中)
            "poor": []     # score 1 (差)
        }

        # 统计缺失字段
        missing_fields = defaultdict(int)
        # 按论文统计
        paper_stats = {}

        total_records = 0

        for paper_name, records in all_results.items():
            paper_record_count = len(records)
            paper_scores = []

            for record in records:
                total_records += 1
                # 获取confidence_score（直接字段或metadata）
                score = record.get("confidence_score") or record.get("metadata", {}).get("confidence_score", 0)

                # 分类存储（3级）
                if score >= 3:
                    quality_buckets["good"].append(record)
                elif score >= 2:
                    quality_buckets["medium"].append(record)
                else:
                    quality_buckets["poor"].append(record)

                paper_scores.append(score)

                # 统计缺失字段（中、差质量数据）
                if score < 3:
                    self._check_missing_fields(record, missing_fields)

            # 论文级别统计
            paper_stats[paper_name] = {
                "total_records": paper_record_count,
                "avg_score": sum(paper_scores) / len(paper_scores) if paper_scores else 0,
                "score_distribution": {
                    "3": paper_scores.count(3),
                    "2": paper_scores.count(2),
                    "1": paper_scores.count(1),
                }
            }

        # 生成报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": total_records,
            "total_papers": len(all_results),
            "quality_distribution": {
                "good (score 3)": len(quality_buckets["good"]),
                "medium (score 2)": len(quality_buckets["medium"]),
                "poor (score 1)": len(quality_buckets["poor"]),
            },
            "quality_percentages": self._calculate_percentages(total_records, quality_buckets),
            "common_missing_fields": dict(sorted(missing_fields.items(), key=lambda x: x[1], reverse=True)[:10]),
            "paper_stats": paper_stats
        }

        return report

    def _check_missing_fields(self, record: Dict, missing_fields: Dict[str, int]):
        """检查记录中缺失的重要字段"""
        # 核心字段
        if not record.get("enzyme_name"):
            missing_fields["enzyme_name"] += 1
        if not record.get("substrate"):
            missing_fields["substrate"] += 1

        # 动力学参数
        if record.get("Km_value") is None:
            missing_fields["Km_value"] += 1
        if record.get("kcat_value") is None:
            missing_fields["kcat_value"] += 1
        if record.get("kcat_Km_value") is None:
            missing_fields["kcat_Km_value"] += 1
        if record.get("degradation_efficiency") is None:
            missing_fields["degradation_efficiency"] += 1

        # 重要字段
        if not record.get("organism") and not record.get("strain"):
            missing_fields["organism_info"] += 1
        if record.get("ph") is None and record.get("optimal_ph") is None:
            missing_fields["ph"] += 1
        if record.get("temperature_value") is None and record.get("optimal_temperature_value") is None:
            missing_fields["temperature"] += 1

        # 加分字段
        if not record.get("ec_number"):
            missing_fields["ec_number"] += 1
        if not record.get("enzyme_full_name"):
            missing_fields["enzyme_full_name"] += 1

    def _calculate_percentages(self, total: int, buckets: Dict) -> Dict[str, float]:
        """计算各质量等级的百分比"""
        percentages = {}
        for level, records in buckets.items():
            percentages[level] = (len(records) / total * 100) if total > 0 else 0
        return percentages

    def sample_low_quality(self, all_results: Dict, min_score: int = 2) -> List[Dict]:
        """
        抽取低质量样本用于分析

        Args:
            all_results: {paper_name: [records]}
            min_score: 最高抽样分数阈值

        Returns:
            低质量样本列表
        """
        low_quality = []

        for paper_name, records in all_results.items():
            for record in records:
                score = record.get("confidence_score") or record.get("metadata", {}).get("confidence_score", 0)
                if score <= min_score:
                    # 添加来源信息
                    record_copy = record.copy()
                    record_copy["_paper_name"] = paper_name
                    low_quality.append(record_copy)

        # 随机抽样
        if len(low_quality) > self.sample_size:
            low_quality = random.sample(low_quality, self.sample_size)

        return low_quality

    def analyze_failure_patterns(self, low_quality_samples: List[Dict]) -> Dict[str, Any]:
        """
        分析失败模式

        Args:
            low_quality_samples: 低质量样本列表

        Returns:
            失败模式分析报告
        """
        patterns = {
            "missing_kinetics_only": [],      # 只有核心字段，缺少动力学
            "missing_organism_only": [],      # 缺少生物体信息
            "missing_conditions_only": [],    # 缺少条件信息
            "missing_core_fields": [],        # 缺少核心字段（酶名/底物）
            "incomplete_data": [],            # 数据不完整
        }

        for sample in low_quality_samples:
            score = sample.get("metadata", {}).get("confidence_score", 0)

            # 检查核心字段
            has_enzyme = bool(sample.get("enzyme_name"))
            has_substrate = bool(sample.get("substrate"))

            # 检查动力学
            has_kinetics = (
                sample.get("Km_value") is not None or
                sample.get("kcat_value") is not None or
                sample.get("kcat_Km_value") is not None or
                sample.get("degradation_efficiency") is not None
            )

            # 检查重要字段
            has_organism = bool(sample.get("organism") or sample.get("strain"))
            has_conditions = (
                sample.get("ph") is not None or
                sample.get("optimal_ph") is not None or
                sample.get("temperature_value") is not None or
                sample.get("optimal_temperature_value") is not None
            )

            # 分类
            if not has_enzyme or not has_substrate:
                patterns["missing_core_fields"].append(sample)
            elif not has_kinetics and has_enzyme and has_substrate:
                patterns["missing_kinetics_only"].append(sample)
            elif not has_organism and has_kinetics:
                patterns["missing_organism_only"].append(sample)
            elif not has_conditions and has_kinetics:
                patterns["missing_conditions_only"].append(sample)
            else:
                patterns["incomplete_data"].append(sample)

        # 统计
        pattern_stats = {}
        for pattern_name, samples in patterns.items():
            pattern_stats[pattern_name] = len(samples)

        return {
            "pattern_distribution": pattern_stats,
            "patterns": patterns,
            "total_analyzed": len(low_quality_samples)
        }

    def generate_prompt_optimization_report(self, low_quality_samples: List[Dict]) -> str:
        """
        生成用于prompt优化的报告

        Args:
            low_quality_samples: 低质量样本列表

        Returns:
            Markdown格式的优化建议报告
        """
        failure_analysis = self.analyze_failure_patterns(low_quality_samples)

        # 质量等级映射
        quality_labels = {3: "好", 2: "中", 1: "差"}

        report = []
        report.append("# 提取质量优化建议报告\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**分析样本数**: {len(low_quality_samples)}\n")

        # 失败模式分布
        report.append("## 失败模式分布\n")
        for pattern, count in failure_analysis["pattern_distribution"].items():
            if count > 0:
                report.append(f"- **{pattern}**: {count} 条")

        # 优化建议
        report.append("\n## Prompt优化建议\n")

        if failure_analysis["pattern_distribution"]["missing_kinetics_only"] > 0:
            report.append("\n### 动力学参数提取优化")
            report.append("- 明确要求提取 Km、kcat、kcat/Km 或降解效率")
            report.append("- 强调这些参数的重要性")
            report.append("- 添加参数识别示例")

        if failure_analysis["pattern_distribution"]["missing_organism_only"] > 0:
            report.append("\n### 生物体信息提取优化")
            report.append("- 明确要求 organism 和 strain 信息")
            report.append("- 强调微生物命名规范")
            report.append("- 添加常见微生物属名示例")

        if failure_analysis["pattern_distribution"]["missing_conditions_only"] > 0:
            report.append("\n### 反应条件提取优化")
            report.append("- 明确要求 pH 和温度条件")
            report.append("- 区分 optimal pH/temperature 和实验条件")
            report.append("- 提取条件范围而不仅仅是单点值")

        # 案例展示
        report.append("\n## 低质量案例展示\n")
        report.append("（随机抽取部分案例用于分析）\n")

        sample_size_display = min(5, len(low_quality_samples))
        for i, sample in enumerate(low_quality_samples[:sample_size_display], 1):
            score = sample.get("confidence_score") or sample.get("metadata", {}).get("confidence_score", 0)
            quality_label = quality_labels.get(score, "未知")
            report.append(f"\n### 案例 {i} (Score: {score}/3 - {quality_label})")
            report.append(f"- **论文**: {sample.get('_paper_name', 'Unknown')}")
            report.append(f"- **酶**: {sample.get('enzyme_name', 'N/A')}")
            report.append(f"- **底物**: {sample.get('substrate', 'N/A')}")

            # 检查缺失字段
            missing = []
            if not sample.get("enzyme_name"):
                missing.append("酶名")
            if not sample.get("substrate"):
                missing.append("底物")
            if sample.get("Km_value") is None:
                missing.append("Km")
            if not (sample.get("organism") or sample.get("strain")):
                missing.append("生物体")

            if missing:
                report.append(f"- **缺失**: {', '.join(missing)}")

        return "\n".join(report)

    def save_reports(self, report: Dict, samples: List[Dict], output_dir: str):
        """
        保存所有报告文件

        Args:
            report: 质量分析报告
            samples: 低质量样本
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 保存质量统计报告
        stats_file = output_path / f"quality_report_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 2. 保存低质量样本
        if samples:
            samples_file = output_path / f"low_quality_samples_{timestamp}.json"
            with open(samples_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

        # 3. 生成优化建议报告（Markdown）
        optimization_report = self.generate_prompt_optimization_report(samples)
        report_file = output_path / f"prompt_optimization_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(optimization_report)

        return {
            "quality_report": stats_file,
            "samples": samples_file if samples else None,
            "optimization_guide": report_file
        }


def analyze_extraction_results(all_results: Dict, output_dir: str, sample_size: int = 20) -> Dict:
    """
    便捷函数：分析提取结果并生成报告

    Args:
        all_results: {paper_name: [records]}
        output_dir: 输出目录
        sample_size: 抽样数量

    Returns:
        包含所有报告文件路径的字典
    """
    analyzer = QualityAnalyzer(sample_size=sample_size)

    # 生成质量报告
    quality_report = analyzer.analyze_quality(all_results)

    # 抽取低质量样本
    low_quality_samples = analyzer.sample_low_quality(all_results, min_score=2)

    # 保存所有报告
    report_files = analyzer.save_reports(quality_report, low_quality_samples, output_dir)

    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 质量分析摘要")
    print("=" * 60)
    print(f"总记录数: {quality_report['total_records']}")
    print(f"质量分布:")
    for level, count in quality_report['quality_distribution'].items():
        pct = quality_report['quality_percentages'].get(level.split()[0], 0)
        print(f"  - {level}: {count} ({pct:.1f}%)")
    print(f"\n低质量样本数: {len(low_quality_samples)}")
    print(f"\n报告文件:")
    for name, path in report_files.items():
        if path:
            print(f"  - {name}: {path}")
    print("=" * 60 + "\n")

    return {
        "quality_report": quality_report,
        "low_quality_samples": low_quality_samples,
        "report_files": report_files
    }
