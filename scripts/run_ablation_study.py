#!/usr/bin/env python3
"""
Run and evaluate MycoExtract ablation experiments.

Outputs:
- ablation_predictions.json: raw records for each paper/configuration
- ablation_metrics_by_field.csv: precision/recall/F1 by field and configuration
- ablation_metrics_overall.csv: micro-averaged metrics by configuration
- ablation_token_usage.json/csv: token usage by provider/model/task
- fig_ablation_micro_f1.png: manuscript-ready comparison plot

Recommended manuscript configurations:
1. kimi_only
2. deepseek_only
3. glm47_only (if enabled in config)
4. students_union
5. majority_vote
6. teacher_aggregation
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ablation_metrics import (  # noqa: E402
    evaluate_predictions_by_doi,
    group_gold_by_doi,
    load_gold_records,
    normalize_doi,
    records_by_doi_from_ablation_payload,
)
from src.extractors.paper_level_extractor import create_paper_level_extractor  # noqa: E402
from src.llm_clients import build_client  # noqa: E402
from src.pipeline.post_processor import RecordMerger, normalize_records_batch  # noqa: E402
from src.utils.token_usage import TokenUsageTracker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ablation")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_clients(config: Dict[str, Any]) -> Dict[str, Any]:
    llm_config = config.get("llm_clients", {})

    def make(name: str, provider_default: str, model_default: str):
        cfg = llm_config.get(name, {})
        return build_client(cfg.get("provider", provider_default), cfg.get("model_name", model_default))

    return {
        "kimi": make("kimi_client", "moonshot", "kimi-k2-0905-preview"),
        "deepseek": make("deepseek_client", "deepseek", "deepseek-chat"),
        "glm47": make("glm47_client", "zhipuai", "glm-4.7"),
        "glm46v": make("glm46v_client", "zhipuai", "glm-4.6v"),
        "aggregation": make("aggregation_client", "openai", "gpt-5.1"),
    }


def load_content_list(paper_dir: Path) -> Optional[List[Dict[str, Any]]]:
    files = list(paper_dir.glob("*_content_list.json"))
    if not files:
        logger.warning("No *_content_list.json found: %s", paper_dir)
        return None
    with files[0].open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_doi(paper_dir: Path, content_list: Optional[List[Dict[str, Any]]] = None) -> str:
    metadata = paper_dir / "metadata.json"
    if metadata.exists():
        try:
            data = json.loads(metadata.read_text(encoding="utf-8"))
            doi = normalize_doi(data.get("doi"))
            if doi:
                return doi
        except Exception:
            pass

    if content_list:
        doi_pattern = r"(?:doi\.org\s*/\s*|DOI\s*:?\s*)(10\.\d{4,}/[^\s,\])\"']+)"
        for block in content_list[:50]:
            if block.get("type") not in ("text", "discarded"):
                continue
            text = block.get("text") or block.get("content") or ""
            text = re.sub(r"\s+", " ", text)
            matches = re.findall(doi_pattern, text, re.IGNORECASE)
            if matches:
                return normalize_doi(matches[0].rstrip(".,;) "))
    return normalize_doi(paper_dir.name)


def normalize_key(record: Dict[str, Any]) -> str:
    enzyme = str(record.get("enzyme_name") or "").strip().lower()
    substrate = str(record.get("substrate") or "").strip().lower()
    mutations = str(record.get("mutations") or "").strip().lower()
    return f"{enzyme}|{substrate}|{mutations}"


def completeness(record: Dict[str, Any]) -> int:
    fields = [
        "enzyme_name",
        "substrate",
        "organism",
        "Km_value",
        "kcat_value",
        "kcat_Km_value",
        "degradation_efficiency",
        "products",
        "temperature_value",
        "ph",
        "uniprot_id",
        "ec_number",
    ]
    return sum(1 for field in fields if record.get(field) not in (None, "", [], {}))


def merge_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    merger = RecordMerger(km_tolerance=0.1, prefer_table_source=True)
    try:
        return normalize_records_batch(merger.merge_records(records))
    except Exception:
        return normalize_records_batch(records)


def majority_vote_records(model_results: Dict[str, List[Dict[str, Any]]], min_votes: int = 2) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    model_votes: Dict[str, set] = defaultdict(set)

    for model_name, records in model_results.items():
        for record in records:
            key = normalize_key(record)
            if key == "||":
                continue
            grouped[key].append(record)
            model_votes[key].add(model_name)

    voted = []
    for key, records in grouped.items():
        if len(model_votes[key]) < min_votes:
            continue
        records = sorted(records, key=completeness, reverse=True)
        merged = dict(records[0])
        for record in records[1:]:
            for field, value in record.items():
                if merged.get(field) in (None, "", [], {}) and value not in (None, "", [], {}):
                    merged[field] = value
        merged["_ablation_votes"] = len(model_votes[key])
        merged["_ablation_models"] = sorted(model_votes[key])
        voted.append(merged)

    return merge_records(voted)


def build_configs(model_results: Dict[str, List[Dict[str, Any]]], aggregated: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    for model_name, records in model_results.items():
        configs[f"{model_name}_only"] = {"records": merge_records(records)}

    union_records = []
    for records in model_results.values():
        union_records.extend(records)
    configs["students_union"] = {"records": merge_records(union_records)}
    configs["majority_vote"] = {"records": majority_vote_records(model_results)}
    configs["teacher_aggregation"] = {"records": normalize_records_batch(aggregated)}
    return configs


def run_live_ablation(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(Path(args.config))
    clients = build_clients(config)
    file_paths = config.get("file_paths", {})

    extractor = create_paper_level_extractor(
        kimi_client=clients["kimi"],
        deepseek_client=clients["deepseek"],
        glm47_client=clients["glm47"] if not args.disable_glm47 else None,
        glm46v_client=clients["glm46v"],
        aggregation_client=clients["aggregation"],
        text_prompt_path=file_paths.get("prompt_text", "prompts/prompts_extract_from_text.txt"),
        table_prompt_path=file_paths.get("prompt_table", "prompts/prompts_extract_from_table.txt"),
        figure_prompt_path=file_paths.get("prompt_figure", "prompts/prompts_extract_from_figure.txt"),
    )

    input_dir = Path(args.input_dir)
    paper_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    if args.limit:
        paper_dirs = paper_dirs[: args.limit]

    TokenUsageTracker.reset()
    payload = {"papers": []}

    for idx, paper_dir in enumerate(paper_dirs, 1):
        content_list = load_content_list(paper_dir)
        if content_list is None:
            continue
        doi = extract_doi(paper_dir, content_list)
        logger.info("[%s/%s] Ablation extraction: %s (%s)", idx, len(paper_dirs), paper_dir.name, doi)

        import asyncio

        result = asyncio.run(extractor.extract_paper(content_list, doi=doi, paper_dir=paper_dir))
        configs = build_configs(result.get("model_results", {}), result.get("aggregated_records", []))
        payload["papers"].append(
            {
                "paper_name": paper_dir.name,
                "doi": doi,
                "configs": configs,
                "model_record_counts": {k: len(v) for k, v in result.get("model_results", {}).items()},
            }
        )

    payload["token_usage"] = TokenUsageTracker.summary()
    return payload


def evaluate_payload(payload: Dict[str, Any], gold_file: str, out_dir: Path) -> None:
    gold_records = load_gold_records(gold_file)
    gold_by_doi = group_gold_by_doi(gold_records)
    config_names = sorted({name for paper in payload.get("papers", []) for name in paper.get("configs", {})})

    field_frames = []
    overall_rows = []
    for config_name in config_names:
        predictions_by_doi = records_by_doi_from_ablation_payload(payload, config_name)
        field_df, micro = evaluate_predictions_by_doi(predictions_by_doi, gold_by_doi)
        field_df.insert(0, "configuration", config_name)
        field_frames.append(field_df)
        overall_rows.append(
            {
                "configuration": config_name,
                **micro,
                "predicted_records": sum(len(v) for v in predictions_by_doi.values()),
                "gold_records": sum(len(v) for v in gold_by_doi.values()),
                "matched_gold_dois": len(set(predictions_by_doi) & set(gold_by_doi)),
            }
        )

    field_metrics = pd.concat(field_frames, ignore_index=True) if field_frames else pd.DataFrame()
    overall = pd.DataFrame(overall_rows).sort_values("f1", ascending=False)
    field_metrics.to_csv(out_dir / "ablation_metrics_by_field.csv", index=False)
    overall.to_csv(out_dir / "ablation_metrics_overall.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    plot_df = overall.sort_values("f1", ascending=True)
    ax.barh(plot_df["configuration"], plot_df["f1"], color="#4E79A7")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Micro-averaged F1")
    ax.set_title("Ablation Study: Extraction Performance by Configuration")
    for i, value in enumerate(plot_df["f1"]):
        ax.text(min(value + 0.01, 0.98), i, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ablation_micro_f1.png", bbox_inches="tight")
    plt.close(fig)


def save_token_tables(payload: Dict[str, Any], out_dir: Path) -> None:
    token_usage = payload.get("token_usage", {})
    with (out_dir / "ablation_token_usage.json").open("w", encoding="utf-8") as f:
        json.dump(token_usage, f, indent=2, ensure_ascii=False)

    records = token_usage.get("records", [])
    if records:
        pd.DataFrame(records).to_csv(out_dir / "ablation_token_usage_records.csv", index=False)

    by_task = Counter()
    for record in records:
        by_task[(record.get("task") or "unspecified", record.get("provider"), record.get("model"))] += record.get("total_tokens", 0)
    if by_task:
        rows = [
            {"task": task, "provider": provider, "model": model, "total_tokens": tokens}
            for (task, provider, model), tokens in by_task.items()
        ]
        task_df = pd.DataFrame(rows).sort_values("total_tokens", ascending=False)
        task_df.to_csv(out_dir / "ablation_token_usage_by_task.csv", index=False)

        plot_df = task_df.groupby("task", as_index=False)["total_tokens"].sum().sort_values("total_tokens")
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
        ax.barh(plot_df["task"], plot_df["total_tokens"], color="#59A14F")
        ax.set_xlabel("Total tokens")
        ax.set_title("Token Usage by Pipeline Task")
        for i, value in enumerate(plot_df["total_tokens"]):
            ax.text(value + max(plot_df["total_tokens"]) * 0.01, i, f"{int(value):,}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_token_usage_by_task.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MycoExtract ablation study and validation metrics.")
    parser.add_argument("--input-dir", help="Parsed paper directory containing one folder per paper.")
    parser.add_argument("--gold-file", default="../gold_dataset.xlsx", help="Manual validation workbook.")
    parser.add_argument("--config", default="config/extraction_config_v7_expanded.yaml")
    parser.add_argument("--out-dir", default="results/ablation_study")
    parser.add_argument("--limit", type=int, help="Limit number of papers for smoke tests.")
    parser.add_argument("--disable-glm47", action="store_true", help="Run 2-student ablation without GLM-4.7.")
    parser.add_argument("--evaluate-only", help="Evaluate an existing ablation_predictions.json file.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        with Path(args.evaluate_only).open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        if not args.input_dir:
            parser.error("--input-dir is required unless --evaluate-only is provided")
        payload = run_live_ablation(args)
        with (out_dir / "ablation_predictions.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    save_token_tables(payload, out_dir)
    evaluate_payload(payload, args.gold_file, out_dir)
    print(f"Ablation outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
