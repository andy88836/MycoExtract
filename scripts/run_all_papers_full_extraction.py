#!/usr/bin/env python3
"""
Run full extraction for parsed papers with input-level de-duplication,
mycotoxin-substrate whitelist gating, and post-extraction eligibility exports.

This script is an orchestration/export layer. v8 schema:
- measurement_type collapsed to {kinetic, degradation}
- kinetic core keeps Km / kcat / kcat_Km only.
- mycotoxin substrate whitelist enforced in apply_eligibility
- zero-record rescue path REMOVED (was a false-positive amplifier;
  see REFACTOR_PROMPT_v8.md §1.1)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_clients import build_client
from src.pipeline.enhanced_pipeline import EnhancedExtractionPipeline
from src.pipeline.paper_level_prechecker import PaperLevelPrechecker
from src.pipeline.post_processor import normalize_records_batch, remove_silver_when_gold_exists
from src.utils.row_level_validator import RowLevelValidator
from src.utils.quality_tier import QualityTierClassifier
from src.utils.table_multiplier import apply_kinetic_unit_multiplier
from src.utils.token_usage import TokenUsageTracker


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "extraction_config_v8.yaml"
ALL_PAPERS_DIR = PROJECT_ROOT / "All_papers"

MYCOTOXIN_RE = re.compile(
    r"\b(aflatoxin|afb1|afb2|afg1|afg2|afm1|ochratoxin|ota|otb|deoxynivalenol|"
    r"\bdon\b|nivalenol|\bniv\b|t-2|ht-2|isotrichodermol|\bisot\b|fumonisin|fb1|fb2|zearalenone|zearalanone|"
    r"zearalanol|zearalenol|\bzea\b|\bzen\b|patulin|citrinin|sterigmatocystin|"
    r"alternariol|trichothecene)\b",
    re.I,
)

ENZYME_TRANSFORM_RE = re.compile(
    r"\b(enzyme|enzymatic|kinetic|km|kcat|kcat/km|vmax|clint|intrinsic clearance|"
    r"degrad|detox|transform|biotransformation|hydrolys|oxidation|reduction|"
    r"acetylation|glucosylation|glucuronidation|conjugation|microsom|subcellular|"
    r"ugt|hsd|tri101|tri201|adh3|lactonase|hydrolase)\b",
    re.I,
)
KINETIC_RE = re.compile(r"\b(km|kcat|kcat/km|vmax|michaelis|clint|intrinsic clearance)\b", re.I)
DEGRADATION_RE = re.compile(r"\b(degrad|detox|conversion|transform|biotransformation|hydrolys|removal)\b", re.I)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(clean(v) for v in value if clean(v))
    text = str(value).strip()
    if text.lower() in {"none", "nan", "null", "[]"}:
        return ""
    return text


def norm_doi(value: str) -> str:
    text = clean(value).lower()
    text = text.replace("\\", "/").split("/")[-1]
    text = re.sub(r"(_origin|-main)?\.pdf$", "", text)
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("_", "/")
    text = re.sub(r"[^a-z0-9./-]+", "", text)
    return text.strip("/")


def doi_from_name(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    if name.endswith("_origin"):
        name = name[:-7]
    if name.endswith("-main"):
        name = name[:-5]
    return norm_doi(name)


def paper_stem(paper_dir: Path) -> str:
    return paper_dir.name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path, limit: Optional[int] = None) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return text[:limit] if limit else text


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_records_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    write_csv(path, records, fields or ["empty"])


def is_parsed_paper_dir(path: Path) -> bool:
    return (
        (path / "full.md").exists()
        or bool(list(path.glob("*_content_list.json")))
        or bool(list(path.glob("*_origin.pdf")))
    )


def find_origin_pdf(paper_dir: Path) -> Optional[Path]:
    pdfs = list(paper_dir.glob("*_origin.pdf")) + list(paper_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def find_paper_dirs(all_papers: Path) -> List[Tuple[Path, str, Optional[Path]]]:
    """Find parsed paper folders in either direct or grouped directory layouts."""
    rows: List[Tuple[Path, str, Optional[Path]]] = []
    seen: set[str] = set()
    for child in sorted([p for p in all_papers.iterdir() if p.is_dir()]):
        if is_parsed_paper_dir(child):
            resolved = str(child.resolve())
            if resolved not in seen:
                rows.append((child, all_papers.name, find_origin_pdf(child)))
                seen.add(resolved)
            continue
        for paper_dir in sorted([p for p in child.iterdir() if p.is_dir()]):
            if not is_parsed_paper_dir(paper_dir):
                continue
            resolved = str(paper_dir.resolve())
            if resolved in seen:
                continue
            rows.append((paper_dir, child.name, find_origin_pdf(paper_dir)))
            seen.add(resolved)
    return rows


def build_inventory(all_papers: Path, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Path], Dict[str, Any]]:
    rows = []
    seen_doi: Dict[str, str] = {}
    seen_hash: Dict[str, str] = {}
    new_dirs: List[Path] = []
    group_id = 0
    for paper_dir, parent, pdf in find_paper_dirs(all_papers):
        parsed_text_only = pdf is None and is_parsed_paper_dir(paper_dir)
        if (pdf is None or not pdf.exists()) and not parsed_text_only:
            row = {
                "pdf_file": "",
                "pdf_path": str(paper_dir),
                "paper_dir": str(paper_dir),
                "parent_subdir": parent,
                "file_size": "",
                "doi_from_filename": paper_dir.name,
                "normalized_doi": norm_doi(paper_dir.name),
                "duplicate_group_id": "",
                "duplicate_reason": "missing_pdf",
                "extraction_status": "invalid_file",
            }
            rows.append(row)
            continue
        doi = norm_doi(paper_dir.name) or (doi_from_name(pdf) if pdf else "")
        file_hash = ""
        if pdf is not None:
            try:
                file_hash = sha256_file(pdf)
            except Exception:
                pass
        status = "new_for_extraction"
        duplicate_reason = ""
        if file_hash and file_hash in seen_hash:
            status = "duplicate_pdf"
            duplicate_reason = f"hash_duplicate_of={seen_hash[file_hash]}"
        elif doi and doi in seen_doi:
            status = "duplicate_pdf"
            duplicate_reason = f"doi_duplicate_of={seen_doi[doi]}"
        if status == "duplicate_pdf":
            group_id += 1
            duplicate_group_id = f"dup_{group_id:04d}"
        else:
            duplicate_group_id = ""
        row = {
            "pdf_file": pdf.name if pdf else f"{paper_dir.name}.pdf",
            "pdf_path": str(pdf) if pdf else "",
            "paper_dir": str(paper_dir),
            "parent_subdir": parent,
            "file_size": pdf.stat().st_size if pdf else "",
            "doi_from_filename": paper_dir.name,
            "normalized_doi": doi,
            "duplicate_group_id": duplicate_group_id,
            "duplicate_reason": duplicate_reason or ("missing_pdf_parsed_text_only" if parsed_text_only else ""),
            "extraction_status": status,
        }
        rows.append(row)
        if file_hash and file_hash not in seen_hash:
            seen_hash[file_hash] = pdf.name if pdf else paper_dir.name
        if doi and doi not in seen_doi:
            seen_doi[doi] = pdf.name if pdf else paper_dir.name
        if status == "new_for_extraction":
            new_dirs.append(paper_dir)

    fields = [
        "pdf_file",
        "pdf_path",
        "paper_dir",
        "parent_subdir",
        "file_size",
        "doi_from_filename",
        "normalized_doi",
        "duplicate_group_id",
        "duplicate_reason",
        "extraction_status",
    ]
    write_csv(out_dir / "all_papers_inventory.csv", rows, fields)
    sub_counts = Counter(row["parent_subdir"] for row in rows)
    status_counts = Counter(row["extraction_status"] for row in rows)
    meta = {
        "subdir_counts": dict(sub_counts),
        "status_counts": dict(status_counts),
        "total_pdf_like_paper_dirs": len(rows),
        "new_for_extraction": len(new_dirs),
    }
    write_json(out_dir / "inventory_summary.json", meta)
    return rows, new_dirs, meta


def init_pipeline(config_path: Path, max_workers: int) -> EnhancedExtractionPipeline:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    llm = config.get("llm_clients", {})

    def client(key: str, default_provider: str, default_model: str):
        conf = llm.get(key, {})
        return build_client(conf.get("provider", default_provider), conf.get("model_name", default_model))

    kimi_client = client("kimi_client", "moonshot", "kimi-k2-0905-preview")
    deepseek_client = client("deepseek_client", "deepseek", "deepseek-chat")
    minimax_client = client("minimax_client", "minimax", "MiniMax-M2.7")
    vision_client = client("mimo_vision_client", "mimo", "mimo-v2.5")
    aggregation_client = client("aggregation_client", "mimo", "mimo-v2.5-pro")

    return EnhancedExtractionPipeline(
        use_paper_level_aggregation=True,
        kimi_client=kimi_client,
        deepseek_client=deepseek_client,
        third_text_client=minimax_client,
        vision_table_client=vision_client,
        aggregation_client=aggregation_client,
        text_prompt_path="prompts/prompts_extract_from_text_v8.txt",
        table_prompt_path="prompts/prompts_extract_from_table_v8.txt",
        max_workers=max_workers,
        use_full_md=True,
        enable_record_merge=True,
        enable_sequence_enrichment=True,
        require_sequence=False,
        save_intermediate=False,
    )


V8_ALLOWED_MEASUREMENT_TYPES = {"kinetic", "degradation"}

# v9: canonical column order for the curated CSV exports.
# Anything not on this list is dropped from CSV output
# (but kept in the JSON debug traces for reproducibility).
V9_SCHEMA_COLUMNS = [
    # === identity ===
    "source_record_id", "pdf_file", "doi", "record_granularity",
    # === primary database scope ===
    "primary_dataset_allowed", "record_scope", "rejection_reason",
    # === enzyme identity ===
    "reported_enzyme_name", "enzyme_name", "enzyme_full_name", "gene_name",
    # === enzyme classification ===
    "organism", "strain", "mutations", "is_recombinant", "is_wild_type",
    "enzyme_state", "enzyme_system_type", "identified_enzyme", "putative_enzyme",
    # === enrichment ===
    "uniprot_id", "genbank_id", "pdb_id", "ec_number", "sequence",
    "enrichment_status",
    # === substrate / product ===
    "raw_substrate", "substrate", "canonical_substrate_name", "substrate_smiles",
    "substrate_concentration", "substrate_concentration_value", "substrate_concentration_unit",
    "products",
    # === measurement ===
    "measurement_type", "matrix", "is_optimum_condition",
    # === kinetic ===
    "Km_value", "Km_unit",
    "kcat_value", "kcat_unit",
    "kcat_Km_value", "kcat_Km_unit",
    "kinetic_temperature_value", "kinetic_temperature_unit",
    "kinetic_ph",
    # === degradation ===
    "degradation_efficiency", "degradation_efficiency_unit",
    "degradation_temperature_value", "degradation_temperature_unit",
    "degradation_ph", "degradation_time_value", "degradation_time_unit",
    # === mediator ===
    "mediator_name", "mediator_concentration", "mediator_concentration_unit",
    # === conditions ===
    "optimum_temperature_value", "optimum_temperature_unit",
    "optimum_ph",
    # === evidence ===
    "evidence_text", "source_section", "notes",
    # === quality tier (replaces eligibility + confidence) ===
    "quality_tier",
    "hard_rule_failures",
    "field_group_count",
    "field_groups_present",
    # === flags (kept for audit) ===
    "error_flags", "human_review_required", "QC_Status",
]


def project_to_v9(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Restrict each record to the V9_SCHEMA_COLUMNS for CSV export."""
    out = []
    for r in records:
        projected = {}
        for col in V9_SCHEMA_COLUMNS:
            v = r.get(col)
            if isinstance(v, list):
                v = ";".join(str(x) for x in v if x is not None)
            projected[col] = "" if v is None else v
        out.append(projected)
    return out




def has_product(row: Dict[str, Any]) -> bool:
    return bool(clean(row.get("products") or row.get("product")))


def has_stability_aux(row: Dict[str, Any]) -> bool:
    return any(clean(row.get(k)) for k in [
        "stability_note", "stability_metric", "stability_value", "stability_unit",
        "stability_temperature_value", "stability_time_value",
        "thermal_stability", "thermal_stability_time",
    ])


def has_optimum_aux(row: Dict[str, Any]) -> bool:
    return any(clean(row.get(k)) for k in [
        "optimum_temperature_value", "optimum_temperature_unit",
        "optimum_ph", "optimum_condition_target", "is_optimum_condition",
        "optimal_temperature_value", "optimal_ph",
    ])


def normalize_auxiliary_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy optimum/stability fields into v8 auxiliary columns."""
    out = dict(row)
    if clean(out.get("degradation_efficiency")) and not clean(out.get("degradation_efficiency_unit")):
        metric_text = " ".join(clean(out.get(k)) for k in ["notes", "evidence_text", "source_section"]).lower()
        if "%" in metric_text or "percent" in metric_text or clean(out.get("measurement_type")).lower() == "degradation":
            out["degradation_efficiency_unit"] = "%"
    if not clean(out.get("optimum_ph")) and clean(out.get("optimal_ph")):
        out["optimum_ph"] = out.get("optimal_ph")
    if not clean(out.get("optimum_temperature_value")) and clean(out.get("optimal_temperature_value")):
        out["optimum_temperature_value"] = out.get("optimal_temperature_value")
    if not clean(out.get("optimum_temperature_unit")) and clean(out.get("optimal_temperature_unit")):
        out["optimum_temperature_unit"] = out.get("optimal_temperature_unit")
    if has_optimum_aux(out) and not clean(out.get("optimum_condition_target")):
        out["optimum_condition_target"] = "reported enzyme/system condition"

    if not clean(out.get("stability_note")) and clean(out.get("thermal_stability")):
        out["stability_note"] = clean(out.get("thermal_stability"))
    if not clean(out.get("stability_time_value")) and clean(out.get("thermal_stability_time")):
        out["stability_time_value"] = out.get("thermal_stability_time")
        out["stability_time_unit"] = out.get("thermal_stability_time_unit")
    return out


def append_note(row: Dict[str, Any], note: str) -> None:
    existing = clean(row.get("notes"))
    if note and note not in existing:
        row["notes"] = f"{existing} | {note}" if existing else note


def add_flag(row: Dict[str, Any], flag: str) -> None:
    flags = split_flags(row.get("error_flags"))
    if flag and flag not in flags:
        flags.append(flag)
    row["error_flags"] = flags


def remove_flags(row: Dict[str, Any], flag_terms: Iterable[str]) -> None:
    terms = [term.lower() for term in flag_terms]
    flags = [flag for flag in split_flags(row.get("error_flags")) if not any(term in flag.lower() for term in terms)]
    row["error_flags"] = flags


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    text = clean(value)
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _scope_blob(record: Dict[str, Any]) -> str:
    return " ".join(clean(record.get(k)) for k in [
        "reported_enzyme_name", "enzyme_name", "enzyme_system_type", "enzyme_state",
        "notes", "evidence_text", "source_section", "table_caption", "measurement_context_id",
        "paper_scope_allowed_context",
    ]).lower()


def _has_non_negated_phrase(text: str, phrase: str) -> bool:
    phrase_re = re.escape(phrase).replace(r"\ ", r"\s+")
    for match in re.finditer(phrase_re, text, flags=re.IGNORECASE):
        prefix = text[max(0, match.start() - 40):match.start()]
        if re.search(r"\b(?:not|non|without|free\s+enzyme,\s+not)\s+(?:\w+\s+){0,4}$", prefix, flags=re.IGNORECASE):
            continue
        return True
    return False


OUT_OF_SCOPE_ENZYME_SYSTEM_TERMS = [
    # crude / lysate / extract
    "crude extract", "crude enzyme", "crude enzymes", "crude enzyme extract", "enzyme extract",
    "cell-free extract", "cell free extract", "cell-free lysate", "cell free lysate",
    "bacterial lysate", "lysate", "intracellular extract", "crude biological material",
    "biomass", "fungal powder", "mushroom powder", "plant powder", "microbial powder",
    "fruiting body material", "mycelium", "mycelial biomass", "powdered mushroom",
    "powdered fungus", "crude powder", "crude biomass", "mushroom extract",
    "fungal extract", "plant extract", "plant juice", "aqueous extract", "leaf extract",
    "root extract", "rhizome extract", "fruit extract", "ginger juice", "botanical extract",
    "digestive matrix", "food matrix material",
    # supernatant / whole cell
    "culture supernatant", "fermentation supernatant", "extracellular supernatant",
    "extracellular fraction", "extracellular proteins", "extracellular enzyme",
    "extracellular enzymes",
    "fermentation broth supernatant", "culture broth", "whole cell", "whole-cell",
    "intact cell", "living cell", "bacterial cells", "yeast cells", "cell suspension",
    "biotransformation by cells", "co-culture", "coculture", "cell line", "l929",
    "mycelial suspension",
    # tissue / host metabolic fractions
    "hepatic microsome", "hepatic microsomes", "liver microsome", "liver microsomes",
    "hepatic cytosol", "hepatic cytosolic", "liver cytosol", "cytosolic fraction",
    "cytosolic fractions", "hepatic extract", "hepatic extracts", "tissue fraction",
    "intestinal content", "rumen fluid", "rumen content",
    # immobilized / material-supported
    "compound enzyme", "compound enzymes", "co-immobilized", "coimmobilized",
    "immobilized", "immobilised", "immobilization", "immobilisation", "nanocomplex",
    "enzyme nanocomplex", "composite", "enzyme-material", "enzyme material",
    "polymer-supported", "supported enzyme", "enzyme support", "carrier",
    "fiber material", "fibre material", "enzyme-loaded", "enzyme-coated",
    "silica gel", "bead", "membrane", "resin", "nanoparticle", "nanocomposite",
    "microsphere", "microspheres", "microbead", "microbeads", "hydrogel",
    "sodium alginate", "alginate microsphere", "montmorillonite",
    "covalently immobilized", "covalent bonding", "cross-linked", "crosslinked",
    "ple50", "ple_50",
    # non-enzymatic material systems
    "adsorbent", "adsorption", "photocatalyst", "photocatalytic", "mof",
    "metal-organic framework", "graphene oxide", "nanomaterial catalyst",
    "nanomaterial", "nano material", "nanozyme", "porous carbon", "carbon material",
    "biocomposite", "magnetic composite", "superparamagnetic", "single-atom catalyst",
    "non-enzymatic catalyst", "non-enzymatic", "non enzymatic", "not a biological enzyme",
    "not biological enzyme",
]


ALLOWED_FREE_PURIFIED_TERMS = [
    "purified enzyme", "purified protein", "purified recombinant protein",
    "free enzyme", "soluble enzyme", "purified laccase", "purified hydrolase",
    "purified oxidase", "purified reductase", "purified transferase",
    "purified mutant", "purified wt", "purified wild-type",
]


ALLOWED_RECOMBINANT_TERMS = [
    "recombinant enzyme", "purified recombinant enzyme", "his-tagged recombinant enzyme",
    "expressed in e. coli", "expressed in escherichia coli", "e. coli bl21",
    "bl21(de3)", "expressed in pichia", "expressed in yeast",
    "recombinant wt", "recombinant mutant",
]


COMMERCIAL_SPECIFIC_ENZYME_TERMS = [
    "commercial laccase", "commercial lipase", "commercial peroxidase",
    "commercial horseradish peroxidase", "commercial porcine pancreatic lipase",
    "amano lipase a", "fumzyme", "commercial fumd", "fumonisin esterase fumd",
    "purchased laccase", "purchased lipase", "purchased peroxidase",
    "purchased horseradish peroxidase", "purchased porcine pancreatic lipase",
]


GENERIC_SYSTEM_NAMES = {
    "commercial enzyme", "commercial enzyme preparation", "enzyme preparation",
    "commercial preparation", "crude enzyme", "crude extract", "cell-free lysate",
    "cell free lysate", "culture supernatant", "fermentation supernatant",
    "whole cell", "whole-cell", "extracellular enzyme", "extracellular enzymes",
    "intracellular enzyme", "intracellular enzymes", "secreted enzyme", "secreted enzymes",
    "unidentified enzyme", "unknown enzyme", "degrading enzyme", "detoxifying enzyme",
    "ota-hydrolytic enzyme", "zen-degrading enzyme", "afb1-degrading enzyme",
    "mycotoxin-degrading enzyme", "enzymatic components", "extracellular proteins",
    "proteinaceous component", "crude enzymes", "enzyme mixture",
}


def is_out_of_scope_enzyme_system(record: Dict[str, Any]) -> bool:
    text = _scope_blob(record)
    if is_commercial_remover_or_mixed_product(record):
        return True
    if any(_has_non_negated_phrase(text, term) for term in OUT_OF_SCOPE_ENZYME_SYSTEM_TERMS):
        return True
    culture_system_patterns = [
        r"\b(?:incubat(?:ed|ion)\s+with|treated\s+with|reaction\s+with|degraded\s+by|with)\s+[a-z0-9.\- ]{0,60}\bculture\b",
        r"\b(?:microbial|bacterial|fungal|yeast)\s+culture\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in culture_system_patterns)


def is_commercial_remover_or_mixed_product(record: Dict[str, Any]) -> bool:
    """Reject commercial remover/blend products whose component enzyme contribution is not isolated."""
    blob = _scope_blob(record)
    mixed_product_markers = [
        "mycotoxin remover", "commercial remover", "commercial mycotoxin remover",
        "commercial product", "remover product", "detoxifier product",
        "enzyme detoxifier", "edr product", "edr1", "edr2", "edr3", "edr4", "edr5",
        "enzyme detoxifying/removing agent",
    ]
    blend_markers = [
        "yeast cell wall", "hscas", "hydrated sodium calcium aluminosilicate",
        "multiple other enzymes", "multiple enzymes", "enzyme blend", "enzyme mixture",
        "compound enzyme", "compound enzymes", "mixed product", "complex product",
        "along with", "cannot be isolated", "specific contribution",
    ]
    remover_metric_markers = ["removability", "removabilities", "removal ability"]
    if any(marker in blob for marker in mixed_product_markers) and (
        any(marker in blob for marker in blend_markers)
        or any(marker in blob for marker in remover_metric_markers)
    ):
        return True
    if any(marker in blob for marker in ["yeast cell wall", "hscas"]) and any(
        enzyme in blob for enzyme in ["lactonase", "de-epoxidase", "hydrolase", "esterase", "peptidase"]
    ):
        return True
    if "specific contribution" in blob and "cannot be isolated" in blob:
        return True
    return False


def _has_clear_enzyme_entity(record: Dict[str, Any]) -> bool:
    enzyme_class_terms = [
        "laccase", "lipase", "peroxidase", "oxidase", "hydrolase", "esterase",
        "reductase", "dehydrogenase", "transferase", "glucosyltransferase",
        "acetyltransferase", "ugt", "udp-glucosyltransferase",
        "fumd", "gsta", "gst", "zph", "zhd", "oph", "afo", "afoth",
    ]
    names = [clean(record.get(k)) for k in ["reported_enzyme_name", "enzyme_name", "gene_name"]]
    names = [name for name in names if name]
    if not names:
        return False
    for name in names:
        lowered = name.lower()
        if lowered in GENERIC_SYSTEM_NAMES or any(term in lowered for term in GENERIC_SYSTEM_NAMES):
            continue
        if any(term in lowered for term in ["preparation", "extract", "lysate", "supernatant", "whole cell", "whole-cell"]):
            continue
        if any(term in lowered for term in enzyme_class_terms):
            return True
        if re.search(r"\b[A-Z]{2,}[A-Za-z0-9_-]*\d+[A-Za-z0-9_-]*\b", name):
            return True
        if re.search(r"\b[A-Za-z]{1,8}\d+[A-Za-z0-9_-]*\b", name):
            return True
        if re.search(r"\bre[A-Z0-9][A-Za-z0-9_-]{2,}\b", name):
            return True
        if re.search(r"\br[A-Z]{2,}[A-Za-z0-9_-]*\b", name):
            return True
    return False


def _is_clearly_identified_commercial_enzyme_record(record: Dict[str, Any]) -> bool:
    """Return True for direct records using a named commercial enzyme.

    This is intentionally record-level. A paper may mention a commercial enzyme
    and also contain strain/lysate measurements; the paper-level context alone
    must not protect every bulk-system record from the inferred-enzyme guard.
    """
    system_type_text = clean(record.get("enzyme_system_type")).lower()
    name = clean(record.get("reported_enzyme_name") or record.get("enzyme_name"))
    context_name = clean(record.get("paper_commercial_enzyme_context"))
    name_blob = " ".join([name, context_name]).lower()
    record_text = " ".join(clean(record.get(k)) for k in [
        "reported_enzyme_name", "enzyme_name", "enzyme_system_name", "enzyme_system_type",
        "enzyme_state", "notes", "evidence_text", "source_section", "table_caption",
        "measurement_context_id",
    ]).lower()
    record_text = re.sub(
        r"enzyme identity is inferred, not directly measured as purified/recombinant/commercial enzyme\.?",
        "",
        record_text,
        flags=re.IGNORECASE,
    )
    # A purchased enzyme mentioned inside a crude/compound/immobilized system
    # does not make that whole measurement a direct commercial-enzyme record.
    if is_out_of_scope_enzyme_system(record):
        return False
    enzyme_class_re = r"\b(?:laccase|lipase|peroxidase|oxidase|hydrolase|esterase|transferase|reductase|fumd|fumonisin esterase)\b"

    if not (re.search(enzyme_class_re, name_blob, flags=re.IGNORECASE) or _has_clear_enzyme_entity(record)):
        return False
    if clean(name).lower() in GENERIC_SYSTEM_NAMES:
        return False

    commercial_signal = (
        system_type_text in {"commercial_enzyme", "clearly_identified_commercial_enzyme"}
        or any(term in record_text for term in [
            "commercial", "purchased", "procured", "obtained from",
            "sigma-aldrich", "sigma aldrich", "sigmaaldrich",
        ])
        or bool(re.search(r"\bfrom\s+[A-Za-z0-9 &.,-]{2,80}", name, flags=re.IGNORECASE) and context_name)
    )
    if not commercial_signal:
        return False
    if "commercial enzyme preparation" in name_blob and not re.search(enzyme_class_re, name_blob, flags=re.IGNORECASE):
        return False
    return True


def is_allowed_primary_enzyme_system(record: Dict[str, Any]) -> bool:
    if is_out_of_scope_enzyme_system(record):
        return False
    text = _scope_blob(record)
    system_type_text = clean(record.get("enzyme_system_type")).lower()
    state_text = clean(record.get("enzyme_state")).lower()
    if system_type_text in {"free_enzyme", "purified_enzyme", "purified_recombinant_enzyme", "commercial_enzyme", "clearly_identified_commercial_enzyme"}:
        return True
    if state_text in {"free", "purified", "soluble"}:
        return True
    if any(term in text for term in ALLOWED_FREE_PURIFIED_TERMS + ALLOWED_RECOMBINANT_TERMS):
        return True
    if _is_clearly_identified_commercial_enzyme_record(record):
        return True
    if any(term in text for term in COMMERCIAL_SPECIFIC_ENZYME_TERMS):
        return True
    if "commercial" in text or "purchased" in text:
        return _has_clear_enzyme_entity(record)
    # Conservative fallback for named catalytic enzyme entities when there is
    # no evidence that the measurement used a crude/whole-cell/material system.
    return _has_clear_enzyme_entity(record)



def infer_paper_allowed_scope_context(text: str) -> str:
    """Return a compact allowed-scope context hint from full text.

    This is only positive context used for records whose table rows contain
    variant labels such as WT/E186A but omit the paper-level phrase saying the
    enzyme was purified/recombinant. Out-of-scope evidence is still evaluated
    from each record's own fields and takes priority.
    """
    lowered = clean(text[:250000]).lower()
    hints: List[str] = []
    if any(term in lowered for term in ALLOWED_FREE_PURIFIED_TERMS):
        hints.append("purified enzyme")
    if (
        "purified" in lowered
        and ("recombinant" in lowered or "expressed in e. coli" in lowered or "expressed in escherichia coli" in lowered or "e. coli bl21" in lowered)
    ):
        hints.append("purified recombinant enzyme")
    if any(term in lowered for term in ALLOWED_RECOMBINANT_TERMS):
        hints.append("purified recombinant enzyme")
    if any(term in lowered for term in COMMERCIAL_SPECIFIC_ENZYME_TERMS):
        hints.append("clearly identified commercial enzyme")
    if re.search(r"\bcommercial\s+(?:[a-z0-9-]+\s+){0,3}(?:laccase|lipase|peroxidase|oxidase|hydrolase|esterase|transferase|reductase|fumd)\b", lowered):
        hints.append("clearly identified commercial enzyme")
    return " | ".join(dict.fromkeys(hints))


INFERRED_ENZYME_EVIDENCE_TERMS = [
    "hgcl2", "pcr", "gene detection", "gene presence",
    "confirmed gene", "detected gene", "pnpp", "p-npp", "paraoxon",
    "auxiliary assay", "may be due to",
    "indicating involvement", "inferred", "responsible is inferred",
    "discussion inference", "conclusion inference",
]


def _bulk_biocatalyst_system_type(record: Dict[str, Any]) -> str:
    text = _scope_blob(record)
    if any(term in text for term in [
        "powder", "biomass", "fungal powder", "mushroom powder", "plant powder",
        "microbial powder", "fruiting body material", "mycelium", "mycelial biomass",
        "crude biological material", "digestive matrix", "food matrix material",
    ]):
        return "crude_biological_system"
    if any(term in text for term in ["cell-free lysate", "cell free lysate", "lysate"]):
        return "cell_free_lysate"
    if any(term in text for term in ["cell-free extract", "cell free extract"]):
        return "cell_free_extract"
    if any(term in text for term in ["crude extract", "crude enzyme", "crude enzyme extract"]):
        return "crude_extract"
    if any(term in text for term in [
        "culture supernatant", "fermentation supernatant", "extracellular supernatant",
        "extracellular fraction", "extracellular proteins", "extracellular enzyme",
        "extracellular enzymes",
    ]):
        return "culture_supernatant"
    if any(term in text for term in ["whole cell", "whole-cell", "cell suspension", "intact cell", "living cell"]):
        return "whole_cell"
    if re.search(r"\b(?:incubat(?:ed|ion)\s+with|treated\s+with|reaction\s+with|degraded\s+by|with)\s+[a-z0-9.\- ]{0,60}\bculture\b", text):
        return "bacterial_culture"
    if any(term in text for term in ["bacterial culture", "microbial culture", "fungal culture", "yeast culture"]):
        return "bacterial_culture"
    if (clean(record.get("organism")) or clean(record.get("strain"))) and _has_inferred_enzyme_evidence(record):
        return "strain_culture"
    return ""


def _has_inferred_enzyme_evidence(record: Dict[str, Any]) -> bool:
    text = _scope_blob(record)
    return any(term in text for term in INFERRED_ENZYME_EVIDENCE_TERMS)


def _has_variant_or_direct_enzyme_context(record: Dict[str, Any]) -> bool:
    """Protect directly assayed variants/recombinant enzymes from inferred guard."""
    text = _scope_blob(record)
    if clean(record.get("mutations")):
        return True
    if re.search(r"\b[A-Z]\d+[A-Z]\b", clean(record.get("reported_enzyme_name") or record.get("enzyme_name"))):
        return True
    return any(term in text for term in [
        "mutant", "variant", "wild-type", "wild type", "recombinant",
        "purified", "purified recombinant", "enzyme variant",
        "kinetic parameters of variants", "expressed and purified",
    ])


def apply_inferred_enzyme_guard(record: Dict[str, Any]) -> Dict[str, Any]:
    """Remove inferred enzyme names from strain/culture/crude-system records.

    Inhibitors, PCR gene detection, and auxiliary non-mycotoxin enzyme assays
    can support a putative mechanism, but they do not identify the catalytic
    entity used in a strain/culture/lysate degradation assay.
    """
    out = dict(record)
    system = _bulk_biocatalyst_system_type(out)
    if not system:
        return out
    if _has_variant_or_direct_enzyme_context(out):
        return out
    if _is_clearly_identified_commercial_enzyme_record(out):
        out["enzyme_system_type"] = clean(out.get("enzyme_system_type")) or "clearly_identified_commercial_enzyme"
        out["enzyme_state"] = clean(out.get("enzyme_state")) or "free"
        out["identified_enzyme"] = "True"
        out["putative_enzyme"] = "False"
        out["_inferred_enzyme_guard_applied"] = ""
        return out
    text = _scope_blob(out)
    direct_enzyme_system = (
        any(term in text for term in ALLOWED_FREE_PURIFIED_TERMS + ALLOWED_RECOMBINANT_TERMS + COMMERCIAL_SPECIFIC_ENZYME_TERMS)
        or _is_clearly_identified_commercial_enzyme_record(out)
        or re.search(r"\bcommercial\s+(?:[a-z0-9-]+\s+){0,3}(?:laccase|lipase|peroxidase|oxidase|hydrolase|esterase|transferase|reductase|fumd)\b", text)
        or ("commercial" in text and clean(out.get("paper_commercial_enzyme_context")))
    )
    has_named_or_inferred_enzyme = any(clean(out.get(k)) for k in ["reported_enzyme_name", "enzyme_name", "gene_name", "uniprot_id"]) or _has_inferred_enzyme_evidence(out)
    if direct_enzyme_system or not has_named_or_inferred_enzyme:
        return out

    for field in [
        "reported_enzyme_name", "enzyme_name", "gene_name", "uniprot_id",
        "candidate_uniprot_id", "candidate_protein_name", "candidate_sequence",
        "sequence",
    ]:
        out[field] = ""
    out["enzyme_system_type"] = clean(out.get("enzyme_system_type")) or system
    out["record_granularity"] = "strain_or_biocatalyst_context"
    out["enrichment_status"] = "blocked_inferred_enzyme"
    out["identified_enzyme"] = "False"
    out["putative_enzyme"] = "True"
    out["_inferred_enzyme_guard_applied"] = "True"
    add_flag(out, "inferred_activity_not_primary_enzyme_name")
    append_note(out, "Enzyme identity is inferred, not directly measured as purified/recombinant/commercial enzyme.")
    return out


def _join_spaced_digits(text: str) -> str:
    return re.sub(r"(?<=\d)\s+(?=\d)", "", text or "")


def _normalize_vendor(value: str) -> str:
    vendor = clean(value).strip(" .;,")
    if re.search(r"sigma[- ]?aldrich", vendor, flags=re.IGNORECASE):
        return "Sigma-Aldrich"
    return vendor


def _commercial_enzyme_definitions(text: str) -> List[Dict[str, str]]:
    """Find clearly identified commercial enzyme definitions in paper text."""
    scan = _join_spaced_digits(text[:250000])
    definitions: List[Dict[str, str]] = []
    enzyme_class = r"(?:laccase|lipase|peroxidase|oxidase|hydrolase|esterase|transferase|reductase|fumd|fumonisin esterase)"
    patterns = [
        rf"(?P<label>{enzyme_class}(?:\s+enzyme)?)\s*\((?P<specific>[^)]*{enzyme_class}[^)]*)\)\s*(?:was\s+)?(?:procured|purchased|obtained)\s+from\s+(?P<vendor>[A-Za-z0-9 &.,-]+)",
        rf"(?P<specific>(?:commercial|purchased)\s+[^.\n]{{0,120}}{enzyme_class}[^.\n]{{0,80}})",
        rf"(?P<specific>{enzyme_class}[^.\n]{{0,120}})\s*(?:was\s+)?(?:procured|purchased|obtained)\s+from\s+(?P<vendor>[A-Za-z0-9 &.,-]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, scan, flags=re.IGNORECASE):
            specific = clean(match.groupdict().get("specific"))
            if not specific:
                continue
            vendor = _normalize_vendor(match.groupdict().get("vendor") or "")
            lower = specific.lower()
            cls_match = re.search(enzyme_class, lower, flags=re.IGNORECASE)
            enzyme_cls = cls_match.group(0).lower() if cls_match else ""
            reported = specific.strip(" .;,")
            if vendor and vendor.lower() not in reported.lower():
                reported = f"{reported}, {vendor}"
            definitions.append({
                "reported_enzyme_name": reported,
                "enzyme_class": enzyme_cls,
                "vendor": vendor,
                "definition_text": match.group(0).strip(),
            })
    unique: List[Dict[str, str]] = []
    seen = set()
    for item in definitions:
        key = item["reported_enzyme_name"].lower()
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def infer_paper_commercial_enzyme_context(text: str) -> str:
    definitions = _commercial_enzyme_definitions(text or "")
    return definitions[0]["reported_enzyme_name"] if definitions else ""


def _substrate_from_text(text: str) -> str:
    lowered = text.lower()
    if "afb1" in lowered or "aflatoxin b1" in lowered:
        return "Aflatoxin B1"
    if "zearalenone" in lowered or re.search(r"\b(?:zen|zea)\b", lowered):
        return "Zearalenone"
    if "ochratoxin a" in lowered or re.search(r"\bota\b", lowered):
        return "Ochratoxin A"
    if "patulin" in lowered:
        return "Patulin"
    if "deoxynivalenol" in lowered or re.search(r"\bdon\b", lowered):
        return "Deoxynivalenol"
    return ""


def _nearest_time(text: str, index: int) -> Tuple[Optional[float], str]:
    after = text[index: index + 160]
    after_match = re.search(r"(?:in|within|after|at)\s+(\d+(?:\.\d+)?)\s*(min|mins|minute|minutes|h|hr|hrs|hour|hours)\b", after, flags=re.IGNORECASE)
    if after_match:
        unit_raw = after_match.group(2).lower()
        return float(after_match.group(1)), "min" if unit_raw.startswith("min") else "h"
    window = text[max(0, index - 140): index + 180]
    matches = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(min|mins|minute|minutes|h|hr|hrs|hour|hours)\b", window, flags=re.IGNORECASE):
        unit_raw = match.group(2).lower()
        unit = "min" if unit_raw.startswith("min") else "h"
        matches.append((abs(match.start() - min(140, index)), float(match.group(1)), unit))
    if not matches:
        return None, ""
    _, value, unit = sorted(matches, key=lambda x: x[0])[0]
    return value, unit


def _temperature_from_text(text: str) -> Tuple[Optional[float], str]:
    normalized = _join_spaced_digits(text)
    patterns = [
        r"(?:at|incubated at|reaction at)\s*(\d{1,3}(?:\.\d+)?)\s*(?:°\s*c|degrees?\s*c|c\b|\\circ)",
        r"(\d{1,3}(?:\.\d+)?)\s*(?:°\s*c|degrees?\s*c|c\b|\\circ)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), "°C"
    return None, ""


def extract_commercial_enzyme_degradation_fallback(text: str, existing_records: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Deterministically recover direct commercial-enzyme mycotoxin degradation rows.

    This fallback is intentionally narrow: it requires a clearly identified
    commercial enzyme definition and a separate direct degradation statement
    for a mycotoxin substrate. It does not extract generic activity assays
    such as pNPP/ABTS kinetics.
    """
    existing_records = existing_records or []
    scan = _join_spaced_digits((text or "").split("\n# References")[0])
    definitions = _commercial_enzyme_definitions(scan)
    if not definitions:
        return []

    records: List[Dict[str, Any]] = []
    for definition in definitions:
        enzyme_cls = definition.get("enzyme_class", "")
        if not enzyme_cls:
            continue
        search_patterns = [
            rf"commercial\s+{re.escape(enzyme_cls)}",
            rf"{re.escape(enzyme_cls)}\s+enzyme",
            re.escape(definition["reported_enzyme_name"].split(",")[0]),
        ]
        candidate_chunks: List[str] = []
        for pattern in search_patterns:
            for match in re.finditer(pattern, scan, flags=re.IGNORECASE):
                chunk = scan[max(0, match.start() - 900): match.end() + 1200]
                lower_chunk = chunk.lower()
                if "degrad" in lower_chunk and _substrate_from_text(chunk):
                    candidate_chunks.append(chunk)
        best: Optional[Tuple[float, str, str, int]] = None
        for chunk in candidate_chunks:
            for pct_match in re.finditer(r"(more than|over|greater than|>|approximately|about)?\s*(\d+(?:\.\d+)?)\s*%", chunk, flags=re.IGNORECASE):
                value = float(pct_match.group(2))
                if value <= 0:
                    continue
                qualifier_raw = clean(pct_match.group(1)).lower()
                qualifier = ">" if qualifier_raw in {"more than", "over", "greater than", ">"} else "approximately" if qualifier_raw in {"approximately", "about"} else ""
                if best is None or value > best[0]:
                    best = (value, qualifier, chunk, pct_match.start())
        if not best:
            continue
        value, qualifier, chunk, pct_index = best
        substrate = _substrate_from_text(chunk)
        if not substrate:
            continue
        if any(
            clean(r.get("measurement_type")).lower() == "degradation"
            and "commercial" in _scope_blob(r)
            and canonical_substrate_name(r.get("substrate")) == canonical_substrate_name(substrate)
            for r in existing_records + records
        ):
            continue
        time_value, time_unit = _nearest_time(chunk, pct_index)
        temp_value, temp_unit = _temperature_from_text(definition.get("definition_text", "") + " " + chunk)
        enzyme_name = definition["reported_enzyme_name"].split(",")[0]
        organism = ""
        organism_match = re.search(r"\bfrom\s+([A-Z][A-Za-z]+(?:\s+sp\.?|\s+[a-z][A-Za-z.-]+)?)", enzyme_name)
        if organism_match:
            organism = organism_match.group(1).strip()
        note = "Deterministic commercial-enzyme fallback from explicit commercial enzyme definition and degradation result."
        if qualifier:
            note += f" Reported as {qualifier}{value:g}%."
        records.append({
            "reported_enzyme_name": definition["reported_enzyme_name"],
            "enzyme_name": enzyme_cls if enzyme_cls != "fumonisin esterase" else "fumonisin esterase FumD",
            "organism": organism,
            "measurement_type": "degradation",
            "substrate": substrate,
            "degradation_efficiency": value,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": time_value,
            "degradation_time_unit": time_unit,
            "degradation_temperature_value": temp_value,
            "degradation_temperature_unit": temp_unit,
            "enzyme_system_type": "clearly_identified_commercial_enzyme",
            "enzyme_state": "free",
            "identified_enzyme": "True",
            "putative_enzyme": "False",
            "enrichment_status": "blocked_due_to_commercial_or_crude_source",
            "human_review_required": False,
            "evidence_text": chunk[:1200],
            "source_section": "commercial_enzyme_degradation_fallback",
            "measurement_context_id": f"commercial|{enzyme_cls}|{canonical_substrate_name(substrate)}|degradation",
            "notes": note,
        })
    return records


# Substrate abbreviation → full canonical name (case-insensitive matching)
_SUBSTRATE_FULL_NAME_MAP: Dict[str, str] = {
    # Zearalenone family
    "zen": "Zearalenone", "zea": "Zearalenone",
    "zel": "Zearalenol",
    "α-zel": "α-Zearalenol", "alpha-zel": "α-Zearalenol", "α-zearalenol": "α-Zearalenol",
    "β-zel": "β-Zearalenol", "beta-zel": "β-Zearalenol", "β-zearalenol": "β-Zearalenol",
    "zearalanone": "Zearalanone",
    "α-zearalanol": "α-Zearalanol", "alpha-zearalanol": "α-Zearalanol",
    "β-zearalanol": "β-Zearalanol", "beta-zearalanol": "β-Zearalanol",
    "zen-14g": "ZEN-14-Glucoside", "zen-14-glucoside": "ZEN-14-Glucoside",
    # Trichothecenes
    "don": "Deoxynivalenol", "deoxynivalenol (don)": "Deoxynivalenol", "niv": "Nivalenol",
    "3-adon": "3-Acetyldeoxynivalenol", "15-adon": "15-Acetyldeoxynivalenol",
    "3 adon": "3-Acetyldeoxynivalenol", "15 adon": "15-Acetyldeoxynivalenol",
    "t2": "T-2 toxin", "t-2": "T-2 toxin", "t2 toxin": "T-2 toxin", "t-2 toxin": "T-2 toxin",
    "ht2": "HT-2 toxin", "ht-2": "HT-2 toxin", "ht2 toxin": "HT-2 toxin", "ht-2 toxin": "HT-2 toxin",
    "das": "Diacetoxyscirpenol", "fus-x": "Fusarenon-X",
    "isot": "Isotrichodermol", "isotrichodermol": "Isotrichodermol",
    "d3g": "DON-3-Glucoside", "don-3-glucoside": "DON-3-Glucoside",
    # Aflatoxins
    "afb1": "Aflatoxin B1", "afb2": "Aflatoxin B2",
    "afg1": "Aflatoxin G1", "afg2": "Aflatoxin G2",
    "afm1": "Aflatoxin M1", "afm2": "Aflatoxin M2",
    "afp1": "Aflatoxin P1", "afq1": "Aflatoxin Q1",
    # Ochratoxins
    "ota": "Ochratoxin A", "ochratoxin a": "Ochratoxin A",
    "otb": "Ochratoxin B", "ochratoxin b": "Ochratoxin B", "otc": "Ochratoxin C",
    # Fumonisins
    "fb1": "Fumonisin B1", "fb2": "Fumonisin B2", "fb3": "Fumonisin B3",
    "hfb1": "Hydrolyzed Fumonisin B1",
    # Others
    "pat": "Patulin", "cit": "Citrinin", "stc": "Sterigmatocystin",
    "cpa": "Cyclopiazonic Acid", "roq-c": "Roquefortine C",
    "mpa": "Mycophenolic Acid",
    "aoh": "Alternariol", "ame": "Alternariol Monomethyl Ether",
    "tea": "Tenuazonic Acid", "mon": "Moniliformin", "bea": "Beauvericin",
}


def normalize_substrate_name(value: Any) -> str:
    """Normalize substrate abbreviation to full canonical name.

    Preserves original if no mapping found. Handles unicode α/β prefixes.
    """
    text = clean(value).strip()
    if not text:
        return text
    lower = text.lower()
    # Direct match
    if lower in _SUBSTRATE_FULL_NAME_MAP:
        return _SUBSTRATE_FULL_NAME_MAP[lower]
    without_parens = re.sub(r"\([^)]*\)", "", lower).strip()
    if without_parens in _SUBSTRATE_FULL_NAME_MAP:
        return _SUBSTRATE_FULL_NAME_MAP[without_parens]
    paren_match = re.search(r"\(([^)]+)\)", lower)
    if paren_match and paren_match.group(1).strip() in _SUBSTRATE_FULL_NAME_MAP:
        return _SUBSTRATE_FULL_NAME_MAP[paren_match.group(1).strip()]
    # Try with unicode α/β normalized to ascii
    normalized = lower.replace("α", "alpha-").replace("β", "beta-")
    if normalized in _SUBSTRATE_FULL_NAME_MAP:
        return _SUBSTRATE_FULL_NAME_MAP[normalized]
    return text


_VALUE_UNIT_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>µg/mL|μg/mL|ug/mL|mg/mL|mg/L|µM|μM|uM|mM|M|ng/mL|g/L|%|U/mL|U/mg)",
    flags=re.IGNORECASE,
)


def parse_value_unit(value: Any) -> Tuple[Optional[float], str]:
    text = clean(value)
    if not text:
        return None, ""
    match = _VALUE_UNIT_RE.search(text)
    if match:
        return float(match.group("value")), match.group("unit")
    number = numeric(text)
    return number, ""


def _is_concentration_unit(unit: Any) -> bool:
    normalized = clean(unit).replace("μ", "µ").lower()
    return normalized in {
        "µg/ml", "ug/ml", "mg/ml", "mg/l", "ng/ml", "g/l",
        "µm", "um", "mm", "m",
    }


MEDIATOR_NAME_RE = (
    r"abts|tempo|hbt|acetosyringone|syringaldehyde|vanillin|vanillic acid|"
    r"syringic acid|p-coumaric acid|ferulic acid|2,6-dimethoxyphenol|dmp"
)
CONCENTRATION_UNIT_RE = r"µg/mL|μg/mL|ug/mL|mg/mL|mg/L|ng/mL|g/L|µM|μM|uM|mM|M"
NUMBER_RE = r"[-+]?\d+(?:\.\d+)?"


def _unit_equivalent(left: Any, right: Any) -> bool:
    return clean(left).replace("μ", "µ").lower() == clean(right).replace("μ", "µ").lower()


def _locally_extract_mediator_concentration(text: str, mediator_name: str) -> Tuple[Optional[float], str]:
    """Return mediator concentration only when value+unit is adjacent to the mediator name."""
    if not text or not mediator_name:
        return None, ""
    mediator = re.escape(clean(mediator_name))
    patterns = [
        rf"(?P<value>{NUMBER_RE})\s*(?P<unit>{CONCENTRATION_UNIT_RE})\s+(?:of\s+)?(?:the\s+)?(?P<mediator>{mediator})\b",
        rf"\b(?P<mediator>{mediator})\b(?:\s+(?:mediator|as\s+mediator))?(?:[^\n]{{0,40}}?)(?P<value>{NUMBER_RE})\s*(?P<unit>{CONCENTRATION_UNIT_RE})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group("value")), match.group("unit")
    return None, ""


def _mediator_concentration_supported_by_source(record: Dict[str, Any]) -> bool:
    """Validate an existing mediator concentration against local source text evidence."""
    mediator = clean(record.get("mediator_name"))
    expected_value = numeric(record.get("mediator_concentration"))
    expected_unit = clean(record.get("mediator_concentration_unit"))
    if not mediator or expected_value is None or not _is_concentration_unit(expected_unit):
        return False
    source_text = " ".join(clean(record.get(k)) for k in [
        "notes", "evidence_text", "source_section", "table_caption",
    ])
    value, unit = _locally_extract_mediator_concentration(source_text, mediator)
    return (
        value is not None
        and _unit_equivalent(unit, expected_unit)
        and abs(value - expected_value) <= max(1e-6, abs(expected_value) * 0.001)
    )


def parse_degradation_time_from_text(text: str) -> Tuple[Optional[float], str]:
    if not text:
        return None, ""
    patterns = [
        r"\bafter\s+(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|minute|minutes|s|sec|seconds)\b",
        r"\bincubat(?:ed|ion)\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|minute|minutes|s|sec|seconds)\b",
        r"\bwithin\s+(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|minute|minutes|s|sec|seconds)\b",
        r"\bfor\s+(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|minute|minutes|s|sec|seconds)\b",
        r"\b(?:gastric|intestinal|oral)\s+phase\s+duration\s+(?:is|was|=)?\s*(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|minute|minutes|s|sec|seconds)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            unit = match.group(2).lower()
            if unit in {"hr", "hrs", "hour", "hours"}:
                unit = "h"
            elif unit in {"minute", "minutes"}:
                unit = "min"
            elif unit in {"sec", "seconds"}:
                unit = "s"
            return float(match.group(1)), unit
    return None, ""


def extract_mediator_from_text(text: str) -> Tuple[str, Optional[float], str]:
    """Extract minimal mediator fields only from likely mycotoxin degradation contexts."""
    lowered = text.lower()
    if not any(term in lowered for term in ["degradation", "degrade", "detox", "conversion", "removal", "reaction mixture"]):
        return "", None, ""
    if any(term in lowered for term in ["activity assay", "enzyme activity", "abts activity", "pnpp", "p-npp"]):
        return "", None, ""
    mediator_match = re.search(
        rf"\b({MEDIATOR_NAME_RE})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not mediator_match:
        return "", None, ""
    mediator = mediator_match.group(1)
    value, unit = _locally_extract_mediator_concentration(text, mediator)
    if value is None:
        return mediator, None, ""
    return mediator, value, unit


def _has_specific_activity_rate_context(text: str) -> bool:
    """Detect enzyme specific-activity/rate units that are not percent conversion."""
    if not text:
        return False
    lowered = text.lower().replace("μ", "µ").replace("−", "-").replace("⁻", "-")
    if any(term in lowered for term in [
        "specific activity",
        "apparent specific activity",
        "converted to degradation efficiency",
        "conversion per minute per mg enzyme",
    ]):
        return True
    rate_unit_patterns = [
        r"\b[µu]?mol\s*(?:min|-1|/min)[^\n;,.]{0,30}\bmg\s*-?1\b",
        r"\b[µu]?mol\s*min\s*-?1\s*mg\s*-?1\b",
        r"\b[µu]?mol\s*/\s*min\s*/\s*mg\b",
        r"\b[µu]?mol\s+min(?:ute)?\s*-?1\s+mg\s*-?1\b",
    ]
    return any(re.search(pattern, lowered, flags=re.I) for pattern in rate_unit_patterns)


def _has_explicit_percent_conversion_evidence(text: str) -> bool:
    """Return True only for local percent degradation/conversion/removal evidence."""
    if not text:
        return False
    lowered = text.lower()
    metric_terms = r"(degrad(?:ation|ed|e)?|conversion|converted|removal|reduction|disappearance|residual\s+toxin)"
    percent_number = r"(?:>|<|≥|≤|about|approximately|more than|less than)?\s*\d+(?:\.\d+)?\s*%"
    return bool(
        re.search(rf"{metric_terms}.{{0,80}}{percent_number}", lowered, flags=re.I)
        or re.search(rf"{percent_number}.{{0,80}}{metric_terms}", lowered, flags=re.I)
    )


def _has_activity_or_stability_context(text: str) -> bool:
    """Detect activity/stability metrics that must not become degradation percent."""
    if not text:
        return False
    lowered = text.lower()
    terms = [
        "specific activity", "specific activities", "apparent specific activity",
        "thermal stability", "thermostability", "remained activity",
        "remained activities", "remaining activity", "residual activity",
        "retained activity", "enzyme activity", "activity after incubation",
        "after incubation at", "optimal temperature", "optimum temperature",
        "optimal ph", "optimum ph", "almost no activity",
    ]
    return any(term in lowered for term in terms)


def _has_true_degradation_metric_context(text: str) -> bool:
    """Detect local degradation/conversion/removal metric labels."""
    if not text:
        return False
    lowered = text.lower()
    terms = [
        "degradation rate", "degradation efficiency", "degradation (%)",
        "hydrolysis efficiency", "hydrolysis efficiencies",
        "hydrolytic efficiency", "conversion efficiency", "conversion rate",
        "removal efficiency", "removal rate", "detoxification rate",
    ]
    return any(term in lowered for term in terms)


def repair_absurd_kcat_km_from_evidence(record: Dict[str, Any]) -> None:
    """Repair kcat/Km values created by mistaking ordinary numbers for 10^n multipliers."""
    value = numeric(record.get("kcat_Km_value"))
    if value is None or abs(value) < 1e12:
        return
    evidence = " ".join(clean(record.get(k)) for k in [
        "evidence_text", "notes", "source_section", "table_caption"
    ])
    if not evidence:
        return
    variant = clean(record.get("mutations") or record.get("reported_enzyme_name"))
    row_text = evidence
    if variant:
        matches = list(re.finditer(rf"\b{re.escape(variant)}\b\s*:?", evidence, flags=re.IGNORECASE))
        if matches:
            row_text = evidence[matches[-1].end():]
    numbers = [
        float(x)
        for x in re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?)?", row_text)
    ]
    # Kinetic rows normally appear as Km, kcat, kcat/Km.  The third value is the
    # safest deterministic repair target when the current value is impossible.
    if len(numbers) >= 3:
        repaired = numbers[2]
        record["kcat_Km_value"] = repaired
        record["kinetic_unit_multiplier"] = None
        add_flag(record, "absurd_kcat_km_repaired_from_evidence")
        append_note(record, "Repaired impossible kcat/Km value from local kinetic table evidence.")


CURRENT_STUDY_REFERENCE_TERMS = [
    "this study", "this work", "current study", "current work",
    "present study", "present work", "our study", "our work",
]


def record_has_current_study_reference(record: Dict[str, Any]) -> bool:
    """Return True when the record itself is explicitly sourced to this paper."""
    blob = " ".join(clean(record.get(k)) for k in [
        "reference", "table_reference", "source_section", "evidence_text", "notes",
    ]).lower()
    return any(term in blob for term in CURRENT_STUDY_REFERENCE_TERMS)


def reference_column_indicates_prior_work(record: Dict[str, Any]) -> bool:
    """Return True for table-comparison rows whose Reference value is prior work."""
    reference_values = [
        clean(value)
        for key, value in record.items()
        if "reference" in str(key).lower() and clean(value)
    ]
    for ref in reference_values:
        lowered_ref = ref.lower()
        if any(term in lowered_ref for term in CURRENT_STUDY_REFERENCE_TERMS):
            continue
        if lowered_ref not in {"none", "null", "-", "n/a"}:
            return True

    blob = " ".join(clean(record.get(k)) for k in [
        "table_caption", "source_section", "evidence_text", "notes",
    ])
    lowered = blob.lower()
    has_reference_column = bool(
        re.search(r"<t[hd][^>]*>\s*reference\s*</t[hd]>", lowered)
        or re.search(r"\breference\s*(?:column|</td>|</th>|:)", lowered)
    )
    if not has_reference_column:
        return False
    if any(term in lowered for term in CURRENT_STUDY_REFERENCE_TERMS):
        return False
    citation_like = bool(
        re.search(r"\b[A-Z][A-Za-z'_-]+\s+et\s+al\.?\s*(?:\(\d{4}\)|\d{4})?", blob)
        or re.search(r"\b[A-Z][A-Za-z'_-]+\s+and\s+[A-Z][A-Za-z'_-]+(?:\s+\(\d{4}\)|\s+\d{4})?", blob)
    )
    return citation_like


def explicit_prior_work_record(record: Dict[str, Any]) -> bool:
    """Return True only when this record itself is a prior-literature row."""
    if reference_column_indicates_prior_work(record):
        return True
    blob = " ".join(clean(record.get(k)) for k in [
        "table_caption", "source_section", "evidence_text", "notes",
    ]).lower()
    if record_has_current_study_reference(record):
        return False
    source_local_patterns = [
        r"\bprevious(?:ly)?\s+(?:reported|published|study|studies)\b",
        r"\bprior\s+(?:reported|published|study|studies|work)\b",
        r"\breported\s+by\b",
        r"\b(?:values|data)\s+from\b",
        r"\bother\s+researchers\b",
        r"\bcomparison\s+(?:with|of)\s+(?:previous|prior|literature)\b",
        r"\bliterature\s+(?:comparison|values|data|row|rows|source)\b",
    ]
    return any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in source_local_patterns)


def _is_generic_or_inferred_enzyme_label(value: Any) -> bool:
    lowered = clean(value).lower()
    if not lowered:
        return False
    if lowered in GENERIC_SYSTEM_NAMES:
        return True
    return any(term in lowered for term in [
        "ota-hydrolytic enzyme", "zen-degrading enzyme", "afb1-degrading enzyme",
        "mycotoxin-degrading enzyme", "degrading enzyme", "detoxifying enzyme",
        "extracellular enzyme", "extracellular enzymes", "secreted enzyme",
        "unidentified enzyme", "unknown enzyme", "enzymatic components",
        "extracellular proteins", "proteinaceous component",
        "glucosyltransferase/reductase system", "glucosyltransferase and reductase activities",
        "reductase activities", "transferase activities", "enzyme activities",
    ])


def _activity_label_without_direct_enzyme_evidence(record: Dict[str, Any]) -> bool:
    """Reject activity/system labels that are not direct enzyme entities.

    This is intentionally generic: an enzyme class plus "activity/activities/system"
    is treated as an inferred biological function unless the local record also
    has purified/recombinant/commercial/direct-enzyme evidence.
    """
    name = " ".join(clean(record.get(field)) for field in ["reported_enzyme_name", "enzyme_name"]).lower()
    if not name:
        return False
    if _is_clearly_identified_commercial_enzyme_record(record):
        return False
    blob = _scope_blob(record)
    has_direct_evidence = any(term in blob for term in PRIMARY_DIRECT_ENZYME_TERMS + ALLOWED_RECOMBINANT_TERMS + ALLOWED_FREE_PURIFIED_TERMS)
    if has_direct_evidence:
        return False
    functional_label = any(term in name for term in [
        "activity", "activities", "system", "extracellular enzymes", "extracellular enzyme",
        "hydrolytic enzyme", "degrading factor",
    ])
    enzyme_class_name = any(term in name for term in [
        "glucosyltransferase", "reductase", "transferase", "laccase", "peroxidase",
        "lipase", "hydrolase", "oxidase", "esterase", "enzyme",
    ])
    inferred_context = any(term in blob for term in [
        "whole-cell", "whole cell", "cell membrane", "membrane-associated",
        "intracellular", "cell extract", "cell fraction", "culture", "strain",
        "product pattern", "inferred", "suggest", "activity assay",
    ])
    return functional_label and enzyme_class_name and inferred_context


def clear_rejected_biocatalyst_enzyme_names(record: Dict[str, Any]) -> Dict[str, Any]:
    """Do not preserve inferred/generic enzyme names for crude biological systems."""
    out = dict(record)
    if _has_concrete_enzyme_identity(out):
        return out
    system = _bulk_biocatalyst_system_type(out)
    generic_label = any(_is_generic_or_inferred_enzyme_label(out.get(field)) for field in ["reported_enzyme_name", "enzyme_name"])
    crude_without_direct_enzyme = bool(system) and not (
        _has_variant_or_direct_enzyme_context(out) or _is_clearly_identified_commercial_enzyme_record(out)
    )
    if not (generic_label or crude_without_direct_enzyme):
        return out

    removed = unique_join([out.get("reported_enzyme_name"), out.get("enzyme_name"), out.get("gene_name")])
    for field in ["reported_enzyme_name", "enzyme_name", "gene_name", "uniprot_id", "sequence"]:
        out[field] = ""
    out["identified_enzyme"] = "False"
    out["putative_enzyme"] = "True"
    out["enrichment_status"] = "blocked_inferred_enzyme"
    if system and not clean(out.get("enzyme_system_type")):
        out["enzyme_system_type"] = system
    if generic_label:
        out["QC_Status"] = clean(out.get("QC_Status")) or "generic_or_unidentified_enzyme_name"
        add_flag(out, "inferred_activity_not_primary_enzyme_name")
    append_note(out, f"Removed inferred/generic enzyme name from rejected biocatalyst context: {removed}.")
    return out


def _has_concrete_enzyme_identity(record: Dict[str, Any]) -> bool:
    """Return True when a record has a non-generic enzyme/gene identifier."""
    for field in ["gene_name", "uniprot_id", "genbank_id", "pdb_id", "ec_number", "sequence"]:
        if clean(record.get(field)):
            return True
    for field in ["enzyme_name", "reported_enzyme_name"]:
        value = clean(record.get(field))
        if value and not _is_generic_or_inferred_enzyme_label(value):
            return True
    return False


def cleanup_mediator_fields(record: Dict[str, Any]) -> None:
    """Keep mediator concentration only with local mediator-name + concentration evidence."""
    mediator = clean(record.get("mediator_name"))
    mediator_value = numeric(record.get("mediator_concentration"))
    blob = " ".join(clean(record.get(k)) for k in ["notes", "evidence_text", "source_section"]).lower()
    if mediator.lower() in {"h2o2", "hydrogen peroxide"}:
        record["mediator_name"] = ""
        record["mediator_concentration"] = None
        record["mediator_concentration_unit"] = None
        add_flag(record, "h2o2_removed_from_mediator_field")
        append_note(record, "H2O2 is a peroxide cofactor/context term, not a mediator field.")
        return
    if "no mediator" in blob or "without mediator" in blob:
        record["mediator_name"] = ""
        record["mediator_concentration"] = None
        record["mediator_concentration_unit"] = None
        add_flag(record, "mediator_context_conflict")
        return

    if (
        mediator
        and clean(record.get("measurement_type")).lower() == "kinetic"
        and not _mediator_concentration_supported_by_source(record)
    ):
        record["mediator_name"] = ""
        record["mediator_concentration"] = None
        record["mediator_concentration_unit"] = None
        add_flag(record, "mediator_removed_not_in_current_reaction_context")
        append_note(record, "Removed mediator field because it was not supported by local kinetic reaction evidence.")
        return

    if mediator_value is not None and not _mediator_concentration_supported_by_source(record):
        record["mediator_concentration"] = None
        record["mediator_concentration_unit"] = None
        add_flag(record, "mediator_concentration_requires_local_evidence")


PRIMARY_DIRECT_ENZYME_TERMS = [
    "purified enzyme", "purified native enzyme", "purified protein",
    "purified recombinant", "purified recombinant enzyme", "purified recombinant protein",
    "recombinant enzyme", "recombinant protein", "his-tagged recombinant",
    "expressed and purified", "expressed in e. coli", "expressed in escherichia coli",
    "e. coli bl21", "bl21(de3)", "expressed in pichia", "expressed in yeast",
    "free enzyme", "soluble enzyme",
]

METABOLIC_SCOPE_TERMS = [
    "hepatic microsome", "hepatic microsomes", "liver microsome", "liver microsomes",
    "hepatic cytosol", "hepatic cytosolic", "liver cytosol", "cytosolic fraction",
    "cytosolic fractions", "hepatic extract", "hepatic extracts", "tissue fraction",
    "intestinal content", "rumen fluid", "cell line", "l929",
    "cytochrome p450", "p450", "cyp1a2", "cyp3a4", "cyp3a5", "cyp3a7",
    "afbo", "aflatoxin b1-8,9-epoxide", "aflatoxin-8,9-epoxide",
    "gst afbo", "afbo-gsh", "gsh conjugation", "glutathione conjugation",
    "apparent km from cytosolic", "no further enzyme purification",
]

INFERRED_ACTIVITY_LABEL_TERMS = [
    "activity", "activities", "extracellular enzymes", "extracellular enzyme",
    "ota-hydrolytic enzyme", "hydrolytic enzyme", "degrading factor",
    "mycelial degrading factor", "likely enzyme", "putative enzyme",
    "inferred enzyme", "suggests", "suggested", "may be due to",
]

RECOVERY_TERMS = ["recovery", "recovered", "extract recovery", "ota recovery", "toxin recovery"]
BINDING_TERMS = ["binding", "bound", "biosorption", "adsorption", "adsorbed", "sorption"]
RELATIVE_ACTIVITY_TERMS = [
    "relative activity", "residual activity", "remaining activity", "retained activity",
    "enzyme activity", "specific activity", "thermal stability", "thermostability",
    "set as 100%", "control was set as 100%", "percent of control",
]
TOXICITY_ENDPOINT_TERMS = [
    "cell viability", "cytotoxicity", "ecotoxicity", "residual bioluminescence",
    "ldh", "ros", "dna damage", "fluorescence endpoint", "toxicity endpoint",
]
DEGRADATION_METRIC_TERMS = [
    "degradation", "degraded", "degradation efficiency", "degradation rate",
    "transformation", "converted", "conversion", "conversion efficiency",
    "residual toxin", "residual mycotoxin", "removal", "reduction",
    "detoxification", "hydrolysis efficiency",
]


def classify_metric_semantic_type(record: Dict[str, Any]) -> str:
    """Classify the local metric meaning without changing the extracted value."""
    measurement_type = clean(record.get("measurement_type")).lower()
    blob = " ".join(clean(record.get(k)) for k in [
        "measurement_type", "notes", "evidence_text", "source_section", "table_caption",
        "products", "QC_Status",
    ]).lower()
    if measurement_type == "kinetic" or any(clean(record.get(k)) for k in ["Km_value", "kcat_value", "kcat_Km_value"]):
        if any(term in blob for term in ["hill equation", "hill coefficient", "s50", " nh", "composite constant k"]):
            return "other"
        return "kinetic"
    if any(term in blob for term in RECOVERY_TERMS):
        return "recovery"
    if any(term in blob for term in BINDING_TERMS):
        return "binding_or_biosorption"
    if any(term in blob for term in RELATIVE_ACTIVITY_TERMS):
        return "relative_activity" if "stability" not in blob else "stability_or_residual_activity"
    if any(term in blob for term in TOXICITY_ENDPOINT_TERMS):
        return "other"
    if any(term in blob for term in ["stability", "residual activity", "retained activity"]):
        return "stability_or_residual_activity"
    if any(term in blob for term in ["transformation", "conversion", "converted", "biotransformation"]):
        return "transformation_efficiency"
    if any(term in blob for term in ["removal", "reduction"]):
        return "removal_efficiency"
    if any(term in blob for term in ["degradation", "degraded", "hydrolysis", "detoxification"]):
        return "degradation_efficiency"
    if measurement_type == "degradation" and clean(record.get("degradation_efficiency")):
        return "degradation_efficiency"
    return "other"


def infer_reaction_mode(record: Dict[str, Any]) -> str:
    blob = _scope_blob(record)
    if any(term in blob for term in ["whole cell", "whole-cell", "strain", "culture", "cell suspension"]):
        return "whole_cell_or_strain"
    if any(term in blob for term in ["hepatic", "microsome", "cytosol", "p450", "cell line", "afbo"]):
        return "metabolic_activity"
    if any(term in blob for term in ["immobilized", "immobilised", "nanocomplex", "supported enzyme", "composite", "carrier"]):
        return "immobilized_enzyme"
    if clean(record.get("matrix")) and clean(record.get("matrix")).lower() not in {"buffer", "phosphate buffer", "reaction buffer"}:
        return "application_matrix"
    if clean(record.get("mediator_name")):
        return "mediator_assisted_reaction"
    if clean(record.get("cofactor")) or any(term in blob for term in ["nadph", "nadh", "gsh", "h2o2", "hydrogen peroxide"]):
        return "cofactor_assisted_reaction"
    if has_primary_enzyme_system_evidence(record):
        return "direct_enzyme_reaction"
    return "unknown"


def infer_oxygen_condition(record: Dict[str, Any]) -> str:
    blob = " ".join(clean(record.get(k)) for k in ["notes", "evidence_text", "source_section", "table_caption"]).lower()
    has_anaerobic = any(term in blob for term in ["anaerobic", "anaerobically", "anoxic"])
    has_aerobic = any(term in blob for term in ["aerobic", "aerobically"])
    if has_anaerobic and not has_aerobic:
        return "anaerobic"
    if has_aerobic and not has_anaerobic:
        return "aerobic"
    return "not_reported"


def is_target_mycotoxin_substrate(value: Any) -> bool:
    text = clean(value).lower()
    if not text:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", text)
    non_targets = {
        "pnpp", "pnitrophenylpalmitate", "abts", "paraoxon", "guaiacol",
        "syringaldazine", "dye", "h2o2", "hydrogenperoxide", "gsh",
    }
    if compact in non_targets:
        return False
    target_compacts = {
        "afb1", "afb2", "afg1", "afg2", "afm1", "aflatoxinb1", "aflatoxinm1",
        "ota", "otb", "ochratoxina", "ochratoxinb",
        "don", "deoxynivalenol", "niv", "nivalenol", "t2", "t2toxin", "ht2",
        "ht2toxin", "das", "3adon", "15adon", "4aniv", "4acetylneosolaniol",
        "isot", "isotrichothecene", "isotrichodermol",
        "fb1", "fb2", "fb3", "hfb1", "fumonisinb1", "fumonisinb2", "fumonisinb3",
        "zen", "zea", "zearalenone", "zearalanone", "alphazel", "betazel",
        "pat", "patulin", "cit", "citrinin", "stc", "sterigmatocystin",
        "aoh", "alternariol", "ame", "tea", "tenuazonicacid",
        "phfb17", "phfb1", "hydrolyzedfumonisinb1",
        "4aniv", "4acetylnivalenol", "fusarenonx", "fusx", "das",
        "diacetoxyscirpenol", "isot", "isotrichothecene", "isotrichodermol", "415dianiv",
        "415diacetylnivalenol",
    }
    if compact in target_compacts:
        return True
    return bool(MYCOTOXIN_RE.search(text))


def has_primary_enzyme_system_evidence(record: Dict[str, Any]) -> bool:
    """Strict primary scope: only direct free/purified/recombinant/commercial enzyme systems."""
    if is_out_of_scope_enzyme_system(record):
        return False
    if _activity_label_without_direct_enzyme_evidence(record):
        return False
    if _is_clearly_identified_commercial_enzyme_record(record):
        return True
    system_type = clean(record.get("enzyme_system_type")).lower()
    if system_type in {
        "free_enzyme", "purified_enzyme", "purified_native_enzyme",
        "purified_recombinant_enzyme", "clearly_identified_commercial_enzyme",
        "commercial_enzyme",
    }:
        return _has_clear_enzyme_entity(record)
    state = clean(record.get("enzyme_state")).lower()
    if state in {"free", "purified", "soluble"} and _has_clear_enzyme_entity(record):
        return True
    blob = _scope_blob(record)
    if any(term in blob for term in PRIMARY_DIRECT_ENZYME_TERMS):
        return _has_clear_enzyme_entity(record)
    if any(term in blob for term in COMMERCIAL_SPECIFIC_ENZYME_TERMS):
        return _has_clear_enzyme_entity(record)
    # A named mutant/construct is primary only when the local/paper context says
    # the protein was recombinant/purified/free, not merely a strain activity.
    if _has_clear_enzyme_entity(record) and re.search(r"\b[A-Z]\d+[A-Z]\b", " ".join([
        clean(record.get("reported_enzyme_name")), clean(record.get("enzyme_name")), clean(record.get("mutations")),
    ])):
        return any(term in blob for term in PRIMARY_DIRECT_ENZYME_TERMS + ALLOWED_RECOMBINANT_TERMS + ALLOWED_FREE_PURIFIED_TERMS)
    # Final conservative fallback: a concrete named enzyme/construct with no
    # crude/strain/material/metabolic evidence is allowed. Out-of-scope and
    # inferred-activity cases have already returned False above.
    return _has_clear_enzyme_entity(record)


def is_inferred_activity_as_enzyme(record: Dict[str, Any]) -> bool:
    """Detect activity labels that should not be treated as an enzyme identity."""
    name = clean(record.get("reported_enzyme_name") or record.get("enzyme_name")).lower()
    if not name:
        return False
    if _is_clearly_identified_commercial_enzyme_record(record):
        return False
    if has_primary_enzyme_system_evidence(record):
        return False
    if _is_generic_or_inferred_enzyme_label(name):
        return True
    if _activity_label_without_direct_enzyme_evidence(record):
        return True
    blob = _scope_blob(record)
    if any(term in blob for term in INFERRED_ACTIVITY_LABEL_TERMS):
        if any(term in name for term in ["activity", "extracellular", "hydrolytic enzyme", "degrading factor", "enzymes"]):
            return True
    return False


def normalize_metric_semantics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Set semantic labels and remove metrics that cannot support primary fields."""
    out = dict(record)
    semantic = classify_metric_semantic_type(out)
    out["metric_semantic_type"] = semantic
    if clean(out.get("measurement_type")).lower() == "degradation":
        if semantic in {"recovery", "binding_or_biosorption", "relative_activity", "stability_or_residual_activity", "other"}:
            add_flag(out, f"metric_semantic_mismatch_{semantic}")
            out["QC_Status"] = clean(out.get("QC_Status")) or f"metric_semantic_mismatch_{semantic}"
            if semantic != "recovery":
                out["degradation_efficiency"] = None
                out["degradation_efficiency_unit"] = None
        if semantic == "recovery":
            append_note(out, "Recovery percentage was not treated as direct degradation efficiency without explicit reduction context.")
    return out


def repair_kcat_km_from_kcat_and_km(record: Dict[str, Any]) -> Dict[str, Any]:
    """Repair OCR-lost scientific notation in kcat/Km using reported Km+kcat.

    If Km is in µM and kcat is in s^-1, kcat/Km in M^-1 s^-1 should be
    kcat / Km * 1e6. We only correct when the reported kcat/Km is an obvious
    power-of-ten truncation of that value, e.g. OCR parsed "2.24 × 10^3" as
    "2.24 × 10".
    """
    out = dict(record)
    km = numeric(out.get("Km_value"))
    kcat = numeric(out.get("kcat_value"))
    current = numeric(out.get("kcat_Km_value"))
    if km is None or kcat is None or current is None or km == 0:
        return out
    km_unit = clean(out.get("Km_unit")).lower().replace("μ", "µ")
    kcat_unit = clean(out.get("kcat_unit")).lower().replace("⁻", "-")
    kcat_km_unit = clean(out.get("kcat_Km_unit")).lower().replace("⁻", "-")
    if "µm" not in km_unit and "um" not in km_unit:
        return out
    if "s" not in kcat_unit:
        return out
    if kcat_km_unit and "m" not in kcat_km_unit:
        return out
    expected = kcat / km * 1_000_000.0
    if expected <= 0:
        return out
    ratio = expected / current if current else 0
    for factor in (10, 100, 1000, 10000, 100000):
        if abs(ratio - factor) / factor <= 0.08:
            # Preserve the reported mantissa when the error is an OCR-lost
            # exponent (e.g. 22.4 should become 2240, not a recomputed 2239).
            repaired = current * factor
            out["kcat_Km_value"] = round(repaired, 3 if abs(repaired) < 100 else 0)
            out["kcat_Km_unit"] = out.get("kcat_Km_unit") or "M^-1 s^-1"
            add_flag(out, "kcat_km_repaired_from_kcat_and_km")
            append_note(out, "Corrected kcat/Km using kcat/Km = kcat / Km with Km in µM converted to M.")
            break
    return out


def clean_inferred_enzyme_names_for_primary_gate(record: Dict[str, Any]) -> Dict[str, Any]:
    """Remove activity-derived enzyme names before final primary scope assignment."""
    out = dict(record)
    if not is_inferred_activity_as_enzyme(out):
        return out
    removed = unique_join([out.get("reported_enzyme_name"), out.get("enzyme_name"), out.get("gene_name")])
    for field in ["reported_enzyme_name", "enzyme_name", "gene_name", "uniprot_id", "sequence"]:
        out[field] = ""
    out["identified_enzyme"] = "False"
    out["putative_enzyme"] = "True"
    out["enrichment_status"] = clean(out.get("enrichment_status")) or "blocked_inferred_enzyme"
    add_flag(out, "inferred_activity_not_primary_enzyme_name")
    append_note(out, f"Removed inferred enzyme/activity label from primary enzyme identity: {removed}.")
    return out


def normalize_time_units_for_export(record: Dict[str, Any]) -> Dict[str, Any]:
    """Use hours as the display unit for reaction/degradation time fields."""
    out = dict(record)
    for prefix in ["degradation", "reaction"]:
        value = numeric(out.get(f"{prefix}_time_value"))
        unit = clean(out.get(f"{prefix}_time_unit")).lower()
        if value is None or not unit:
            continue
        if unit.startswith("h") or "hour" in unit:
            out[f"{prefix}_time_value"] = value
            out[f"{prefix}_time_unit"] = "h"
        elif unit.startswith("min") or unit in {"m"}:
            out[f"{prefix}_time_value"] = round(value / 60.0, 6)
            out[f"{prefix}_time_unit"] = "h"
        elif unit.startswith("s") or "sec" in unit:
            out[f"{prefix}_time_value"] = round(value / 3600.0, 6)
            out[f"{prefix}_time_unit"] = "h"
    return out


def clear_midpoint_values_from_reported_ranges(record: Dict[str, Any]) -> Dict[str, Any]:
    """Do not present midpoint values as exact pH/temperature measurements."""
    out = dict(record)
    blob = " ".join(clean(out.get(k)) for k in ["evidence_text", "notes", "source_section", "table_caption"])

    def midpoint_from_range(pattern: str) -> Optional[Tuple[float, str]]:
        match = re.search(pattern, blob, flags=re.IGNORECASE)
        if not match:
            return None
        left = float(match.group(1))
        right = float(match.group(2))
        raw = match.group(0)
        return ((left + right) / 2.0, raw)

    temp_range = midpoint_from_range(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*°?\s*c")
    temp_value = numeric(out.get("degradation_temperature_value"))
    if temp_range and temp_value is not None and abs(temp_value - temp_range[0]) <= 0.05:
        out["degradation_temperature_value"] = None
        out["degradation_temperature_unit"] = None
        append_note(out, f"Reported temperature is a range ({temp_range[1]}); midpoint was not retained as exact value.")

    ph_range = midpoint_from_range(r"ph\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)")
    ph_value = numeric(out.get("degradation_ph"))
    if ph_range and ph_value is not None and abs(ph_value - ph_range[0]) <= 0.05:
        out["degradation_ph"] = None
        append_note(out, f"Reported pH is a range ({ph_range[1]}); midpoint was not retained as exact value.")
    return out


def _same_entity_label(left: Any, right: Any) -> bool:
    ltxt = re.sub(r"[^a-z0-9]+", "", clean(left).lower())
    rtxt = re.sub(r"[^a-z0-9]+", "", clean(right).lower())
    if not ltxt or not rtxt:
        return False
    return ltxt == rtxt or ltxt in rtxt or rtxt in ltxt


def clean_organism_as_enzyme_name(record: Dict[str, Any]) -> Dict[str, Any]:
    """Clear enzyme_name when the model copied organism/strain into enzyme identity."""
    out = dict(record)
    enzyme_values = [clean(out.get("enzyme_name")), clean(out.get("reported_enzyme_name"))]
    organism_values = [clean(out.get("organism")), clean(out.get("strain"))]
    if not any(enzyme_values) or not any(organism_values):
        return out
    if has_primary_enzyme_system_evidence(out):
        return out
    if not any(_same_entity_label(e, o) for e in enzyme_values for o in organism_values if e and o):
        return out
    removed = unique_join(enzyme_values)
    out["reported_biocatalyst"] = clean(out.get("reported_biocatalyst")) or removed
    for field in ["reported_enzyme_name", "enzyme_name", "gene_name", "uniprot_id", "sequence"]:
        out[field] = ""
    out["identified_enzyme"] = "False"
    out["putative_enzyme"] = "True"
    out["enrichment_status"] = clean(out.get("enrichment_status")) or "blocked_organism_as_enzyme"
    add_flag(out, "organism_or_strain_as_enzyme_name_removed")
    append_note(out, f"Removed organism/strain label from enzyme_name: {removed}.")
    return out


def apply_primary_hard_gate(record: Dict[str, Any]) -> Dict[str, Any]:
    """Final export gate: decide primary scope independently from Gold/Silver/Bronze tier."""
    out = normalize_time_units_for_export(clear_midpoint_values_from_reported_ranges(
        clean_organism_as_enzyme_name(
            clean_inferred_enzyme_names_for_primary_gate(
                repair_kcat_km_from_kcat_and_km(normalize_metric_semantics(dict(record)))
            )
        )
    ))
    # Internal labels may still be useful for debug JSON, but they are no
    # longer exported as schema columns.
    out["oxygen_condition"] = clean(out.get("oxygen_condition")) or infer_oxygen_condition(out)
    out["reaction_mode"] = clean(out.get("reaction_mode")) or infer_reaction_mode(out)

    reasons: List[str] = []
    metric_semantic = clean(out.get("metric_semantic_type"))
    measurement_type = clean(out.get("measurement_type")).lower()
    scope_blob = _scope_blob(out)

    if clean(out.get("quality_tier")) == "Rejected":
        reasons.append(clean(out.get("QC_Status")) or clean(out.get("hard_rule_failures")) or "quality_tier_rejected")
    if is_commercial_remover_or_mixed_product(out):
        reasons.append("commercial_remover_or_mixed_product_not_single_enzyme")
    if any(term in scope_blob for term in METABOLIC_SCOPE_TERMS):
        reasons.append("metabolic_activity_not_primary_degrading_enzyme")
    stale_literature_flag = (
        "literature_comparison_source" in split_flags(out.get("error_flags"))
        and not record_has_current_study_reference(out)
    )
    if explicit_prior_work_record(out) or stale_literature_flag:
        reasons.append("literature_comparison_not_current_experiment")
    if not has_primary_enzyme_system_evidence(out):
        reasons.append("not_identified_primary_enzyme_system")
    if _activity_label_without_direct_enzyme_evidence(out):
        reasons.append("inferred_activity_label_not_primary_enzyme")
    if measurement_type == "kinetic":
        if not is_target_mycotoxin_substrate(out.get("substrate")):
            reasons.append("non_mycotoxin_kinetic_substrate")
    if measurement_type == "degradation" and metric_semantic not in {
        "degradation_efficiency", "transformation_efficiency", "removal_efficiency",
    }:
        reasons.append(f"metric_semantic_mismatch_{metric_semantic or 'unknown'}")
    if any(flag in split_flags(out.get("error_flags")) for flag in [
        "hill_constant_not_michaelis_menten_km",
        "wrong_metric_type_relative_activity_baseline",
        "wrong_metric_type_toxicity_endpoint",
        "wrong_metric_type_specific_activity_rate",
        "wrong_metric_type_activity_or_stability",
    ]):
        reasons.append("metric_semantic_mismatch")
    if "inferred_activity_not_primary_enzyme_name" in split_flags(out.get("error_flags")):
        reasons.append("inferred_activity_label_not_primary_enzyme")
    severe_flags = {
        "missing_time_for_degradation",
        "mixed_matrix_context",
        "matrix_context_incomplete",
        "possible_specific_activity_in_degradation_field",
        "optimum_condition_over_assignment",
    }
    present_severe = sorted(severe_flags & set(split_flags(out.get("error_flags"))))
    if present_severe:
        reasons.append("severe_error_flags:" + ",".join(present_severe))

    if not reasons and clean(out.get("quality_tier")) in {"Gold", "Silver", "Bronze"}:
        out["primary_dataset_allowed"] = "True"
        out["record_scope"] = "primary_enzyme_record"
        out["rejection_reason"] = ""
        return out

    out["primary_dataset_allowed"] = "False"
    out["record_scope"] = "rejected_out_of_scope"
    out["rejection_reason"] = ";".join(dict.fromkeys([r for r in reasons if r])) or "not_primary_database_record"
    out.pop("secondary_reason", None)
    append_note(out, f"Rejected from primary database: {out['rejection_reason']}.")
    return out


def deterministic_record_cleanup(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply hard post-extraction field corrections independent of LLM output."""
    out = repair_kcat_km_from_kcat_and_km(apply_inferred_enzyme_guard(normalize_auxiliary_fields(apply_kinetic_unit_multiplier(dict(row)))))
    # Normalize substrate abbreviation → full name
    sub = clean(out.get("substrate"))
    if sub:
        out.setdefault("raw_substrate", sub)
        out["substrate"] = normalize_substrate_name(sub)
        out["canonical_substrate_name"] = canonical_substrate_name(out["substrate"])
    blob = " ".join(clean(out.get(k)) for k in [
        "notes", "evidence_text", "source_section", "source_table_id",
        "measurement_context_id", "table_caption",
    ]).lower()

    # Minimal concentration split for export; keep substrate_concentration for compatibility.
    if clean(out.get("substrate_concentration")) and not clean(out.get("substrate_concentration_value")):
        conc_value, conc_unit = parse_value_unit(out.get("substrate_concentration"))
        if conc_value is not None:
            out["substrate_concentration_value"] = conc_value
            if conc_unit:
                out["substrate_concentration_unit"] = out.get("substrate_concentration_unit") or conc_unit
            elif not clean(out.get("substrate_concentration_unit")):
                add_flag(out, "unit_context_ambiguity")

    if clean(out.get("measurement_type")).lower() == "degradation":
        if not clean(out.get("degradation_time_value")):
            time_value, time_unit = parse_degradation_time_from_text(blob)
            if time_value is not None:
                out["degradation_time_value"] = time_value
                out["degradation_time_unit"] = time_unit
        if clean(out.get("degradation_time_value")):
            remove_flags(out, ["missing_time_for_degradation"])
        if clean(out.get("degradation_efficiency")) and not clean(out.get("degradation_time_value")):
            add_flag(out, "missing_time_for_degradation")
        if (
            "wet-milling" in blob or "wet milling" in blob
            or "corn steep" in blob or "solid residue" in blob or "solid residues" in blob
        ) and not ("corn steep" in clean(out.get("matrix")).lower() or "solid residue" in clean(out.get("matrix")).lower()):
            add_flag(out, "matrix_context_incomplete")
            append_note(out, "Wet-milling matrix context may be incomplete; no unobserved matrix record was generated.")

    if not clean(out.get("mediator_name")):
        mediator_name, mediator_value, mediator_unit = extract_mediator_from_text(" ".join([
            clean(out.get("notes")), clean(out.get("evidence_text")), clean(out.get("source_section"))
        ]))
        if mediator_name:
            out["mediator_name"] = mediator_name
            if mediator_value is not None:
                out["mediator_concentration"] = mediator_value
            if mediator_unit:
                out["mediator_concentration_unit"] = mediator_unit
    cleanup_mediator_fields(out)

    if record_has_current_study_reference(out):
        remove_flags(out, ["literature_comparison_source"])
        if clean(out.get("QC_Status")) == "literature_comparison_not_current_experiment":
            out["QC_Status"] = ""
    elif explicit_prior_work_record(out):
        out["QC_Status"] = clean(out.get("QC_Status")) or "literature_comparison_not_current_experiment"
        add_flag(out, "literature_comparison_source")

    bioactivation_terms = [
        "bioactivation", "metabolic activation", "afbo", "aflatoxin b1-8,9-epoxide",
        "aflatoxin epoxide", "epoxide formation", "dna adduct", "protein adduct",
        "carcinogenic metabolite", "mutagenic metabolite", "genotoxic metabolite",
        "human liver microsome", "liver microsomes", "cdna-expressed cyp",
        "cyp1a2", "cyp3a4", "cyp3a5", "cyp3a7",
        "p450 1a2", "p450 3a4", "p450 3a5", "p450 3a7",
        "cytochrome p450", "pmol p450", "p450/min",
        "p450-mediated activation", "afq1 formation", "afm1 formation",
    ]
    if "afbo-gsh" not in blob and any(term in blob for term in bioactivation_terms):
        out["QC_Status"] = clean(out.get("QC_Status")) or "metabolic_activation_or_toxic_biotransformation"
        add_flag(out, "metabolic_activation_or_toxic_biotransformation")

    hill_terms = ["hill equation", "hill coefficient", " nh", "s50", "k'", "sigmoidal kinetics", "positive cooperativity", "composite constant k"]
    if clean(out.get("Km_value")) and any(term in blob for term in hill_terms):
        out["Km_value"] = None
        out["Km_unit"] = None
        add_flag(out, "hill_constant_not_michaelis_menten_km")
        append_note(out, "Hill equation K/S50 was not retained as Michaelis-Menten Km.")

    relative_terms = [
        "relative activity", "residual activity", "remaining activity",
        "normalized activity", "set as 100%", "control was set as 100%",
        "untreated control", "percent of control", "reference for other substrates",
    ]
    de_value = numeric(out.get("degradation_efficiency"))
    if de_value is not None and abs(de_value - 100.0) <= 1.0 and any(term in blob for term in relative_terms):
        out["degradation_efficiency"] = None
        out["degradation_efficiency_unit"] = None
        add_flag(out, "wrong_metric_type_relative_activity_baseline")

    toxicity_terms = [
        "residual bioluminescence", "ecotoxicity", "cytotoxicity", "cell viability",
        "ldh", "ros", "dna damage", "inhibition rate", "tissue residue",
        "animal performance",
    ]
    if clean(out.get("degradation_efficiency")) and any(term in blob for term in toxicity_terms):
        out["degradation_efficiency"] = None
        out["degradation_efficiency_unit"] = None
        add_flag(out, "wrong_metric_type_toxicity_endpoint")

    if (
        clean(out.get("measurement_type")).lower() == "degradation"
        and clean(out.get("degradation_efficiency"))
        and _has_specific_activity_rate_context(blob)
        and not _has_explicit_percent_conversion_evidence(clean(out.get("evidence_text")))
    ):
        out["degradation_efficiency"] = None
        out["degradation_efficiency_unit"] = None
        out["QC_Status"] = clean(out.get("QC_Status")) or "specific_activity_rate_not_degradation_efficiency"
        add_flag(out, "wrong_metric_type_specific_activity_rate")
        append_note(
            out,
            "Specific activity/rate units (e.g. µmol min⁻¹ mg⁻¹) were not retained as degradation_efficiency; "
            "conversion/removal percentage requires explicit percent evidence from the paper."
        )

    if (
        clean(out.get("measurement_type")).lower() == "degradation"
        and clean(out.get("degradation_efficiency"))
        and _has_activity_or_stability_context(blob)
        and not _has_true_degradation_metric_context(clean(out.get("evidence_text")))
    ):
        out["degradation_efficiency"] = None
        out["degradation_efficiency_unit"] = None
        out["QC_Status"] = clean(out.get("QC_Status")) or "activity_or_stability_metric_not_degradation_efficiency"
        add_flag(out, "wrong_metric_type_activity_or_stability")
        append_note(out, "Activity/stability percentage was not retained as degradation_efficiency.")

    # Purified recombinant enzymes are not immobilized just because the name contains lac/laccase.
    purified_terms = [
        "purified recombinant", "purified enzyme", "expressed in e. coli",
        "e. coli bl21", "recombinant cota-laccase", "purified cota-laccase",
    ]
    immobilized_terms = [
        "immobilized", "immobilization", "support", "carrier", "fiber",
        "bound", "beads", "nanocomposite",
    ]
    if any(term in blob for term in purified_terms) and not any(term in blob for term in immobilized_terms):
        if clean(out.get("enzyme_system_type")).lower() == "immobilized_enzyme":
            out["enzyme_system_type"] = "purified_recombinant_enzyme"
            append_note(out, "Corrected enzyme system: purified recombinant enzyme, not immobilized.")
        elif not clean(out.get("enzyme_system_type")):
            out["enzyme_system_type"] = "purified_recombinant_enzyme"
        out["enzyme_state"] = "free"

    context_name = clean(out.get("paper_commercial_enzyme_context"))
    commercial_blob = _scope_blob(out)
    if _is_clearly_identified_commercial_enzyme_record(out):
        out["enzyme_system_type"] = "clearly_identified_commercial_enzyme"
        out["enzyme_state"] = clean(out.get("enzyme_state")) or "free"
        out["identified_enzyme"] = "True"
        out["putative_enzyme"] = "False"
        out["_inferred_enzyme_guard_applied"] = ""
        out["notes"] = re.sub(
            r"\s*\|?\s*Enzyme identity is inferred, not directly measured as purified/recombinant/commercial enzyme\.?",
            "",
            clean(out.get("notes")),
            flags=re.IGNORECASE,
        ).strip(" |")
        if context_name:
            current_name = clean(out.get("reported_enzyme_name") or out.get("enzyme_name"))
            context_class = re.search(r"(laccase|lipase|peroxidase|oxidase|hydrolase|esterase|transferase|reductase|fumd)", context_name, flags=re.IGNORECASE)
            if not current_name or (context_class and context_class.group(1).lower() in current_name.lower() and "," not in current_name):
                out["reported_enzyme_name"] = context_name
            if not clean(out.get("enzyme_name")):
                out["enzyme_name"] = context_class.group(1).lower() if context_class else current_name
        if clean(out.get("measurement_type")).lower() == "degradation":
            minute_match = re.search(r"\b(\d+(?:\.\d+)?)\s*min\b", commercial_blob, flags=re.IGNORECASE)
            if minute_match:
                out["degradation_time_value"] = float(minute_match.group(1))
                out["degradation_time_unit"] = "min"

    # Thermodynamic temperature-dependent kcat values are not primary Michaelis-Menten kcat.
    thermo_terms = ["thermodynamic", "arrhenius", "activation energy", "temperature-dependent", "thermodynamic parameters"]
    source_section = clean(out.get("source_section") or out.get("source_table_id")).lower()
    if (
        clean(out.get("measurement_type")).lower() == "kinetic"
        and clean(out.get("kcat_value"))
        and (any(term in blob for term in thermo_terms) or source_section == "table_3" or "table 3" in blob)
        and "independent michaelis" not in blob
        and "own fitted km" not in blob
    ):
        out["kcat_value"] = None
        out["kcat_unit"] = None
        append_note(out, "Thermodynamic temperature-dependent kcat excluded from primary kinetic fields.")
        add_flag(out, "thermodynamic_kcat_removed_from_primary")

    # Measurement-family boundary: degradation rows do not own kinetic fields.
    # If the same paper reports mycotoxin kinetics, those must be a separate
    # kinetic measurement context, not fields on the degradation row.
    if (
        clean(out.get("measurement_type")).lower() == "degradation"
        and any(clean(out.get(k)) for k in ["Km_value", "kcat_value", "kcat_Km_value"])
    ):
        for field in ["Km_value", "Km_unit", "kcat_value", "kcat_unit", "kcat_Km_value", "kcat_Km_unit"]:
            out[field] = None
        append_note(out, "Removed kinetic fields from degradation record; kinetic and degradation measurements must be separate contexts.")

    # Pretreatment/stability temperature is not ordinary degradation reaction temperature.
    pretreatment_terms = [
        "pre-incubated", "pre-incubation", "preincubated", "preincubation",
        "heat-treated", "heat treated", "thermal stability", "retained activity",
        "maintained activity", "before assay",
    ]
    if clean(out.get("measurement_type")).lower() == "degradation" and any(term in blob for term in pretreatment_terms):
        if clean(out.get("degradation_temperature_value")):
            out["stability_temperature_value"] = out.get("stability_temperature_value") or out.get("degradation_temperature_value")
            out["stability_temperature_unit"] = out.get("stability_temperature_unit") or out.get("degradation_temperature_unit")
            out["degradation_temperature_value"] = None
            out["degradation_temperature_unit"] = None
        out["stability_metric"] = out.get("stability_metric") or "residual_AFB1_degradation_activity_after_heat_pretreatment"
        out["human_review_required"] = True
        add_flag(out, "stability_pretreatment_context")
        append_note(out, "Temperature refers to pretreatment/stability context, not ordinary degradation reaction temperature.")

    # Clean multiplier warnings when the value is already correctly scaled.
    multiplier = numeric(out.get("kinetic_unit_multiplier"))
    kcat_km_value = numeric(out.get("kcat_Km_value"))
    if multiplier and kcat_km_value and multiplier != 1:
        raw_value_threshold = max(100.0, abs(multiplier) / 10.0)
        if abs(kcat_km_value) < raw_value_threshold:
            out["kcat_Km_value"] = kcat_km_value * multiplier
            out["_table_multiplier_applied"] = True
            kcat_km_value = numeric(out.get("kcat_Km_value"))
    raw_text = " ".join(clean(out.get(k)) for k in ["evidence_text", "notes", "kinetic_unit_source_text", "kcat_Km_unit"])
    if multiplier and kcat_km_value:
        raw_values = [float(x) for x in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", raw_text)]
        if any(abs((raw * multiplier) - kcat_km_value) <= max(1e-6, abs(kcat_km_value) * 0.01) for raw in raw_values):
            remove_flags(out, ["multiplier", "table_multiplier_scaling_error", "scaling"])
            if "scaling ambiguity" in clean(out.get("notes")).lower():
                out["notes"] = re.sub(
                    r"kcat_Km_unit has (?:potential )?(?:multiplier )?scaling ambiguity:[^|]*(?:\|\s*)?",
                    "",
                    clean(out.get("notes")),
                    flags=re.IGNORECASE,
                ).strip(" |")
            append_note(out, "Multiplier applied from table header.")
            if clean(out.get("error_flags")) in {"[]", ""}:
                out["human_review_required"] = False

    repair_absurd_kcat_km_from_evidence(out)

    # Preserve GST conjugation semantics.
    gst_blob = " ".join(clean(out.get(k)) for k in [
        "reported_enzyme_name", "enzyme_name", "substrate", "notes", "products", "evidence_text"
    ]).lower()
    if "afbo-gsh" in gst_blob or "afbo" in gst_blob or "aflatoxin b1-8,9-epoxide" in gst_blob or "gsta2" in gst_blob:
        append_note(out, "GST-mediated AFBO-GSH conjugation/detoxification; not ordinary hydrolysis.")
    elif "gst" in gst_blob or "glutathione transferase" in gst_blob or "-gsh" in gst_blob:
        out["notes"] = re.sub(
            r"\s*\|?\s*GST-mediated AFBO-GSH conjugation/detoxification; not ordinary hydrolysis\.?",
            "",
            clean(out.get("notes")),
            flags=re.IGNORECASE,
        ).strip(" |")
        append_note(out, "GST-mediated glutathione conjugation; not ordinary hydrolysis.")

    out = clear_rejected_biocatalyst_enzyme_names(out)
    return out


def split_flags(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        raw = [clean(v) for v in value]
    else:
        raw = re.split(r"[;|,]", clean(value))
    return [flag.strip() for flag in raw if flag and flag.strip()]


def system_type(row: Dict[str, Any]) -> str:
    blob = " ".join(clean(row.get(k)) for k in ["enzyme_name", "reported_enzyme_name", "enzyme_system_name", "enzyme_state", "notes"]).lower()
    if _is_clearly_identified_commercial_enzyme_record(row):
        return "clearly_identified_commercial_enzyme"
    if any(x in blob for x in ["microsome", "post-mitochondrial", "subcellular fraction"]):
        return "subcellular_fraction"
    if any(x in blob for x in ["crude extract", "cell lysate"]):
        return "crude_system"
    if "fermentation supernatant" in blob or "culture supernatant" in blob:
        return "fermentation_supernatant"
    if any(x in blob for x in ["immobilized", "composite"]):
        return "immobilized_enzyme"
    return clean(row.get("enzyme_system_type"))


def evidence_modalities(row: Dict[str, Any]) -> List[str]:
    """v8: kinetic/degradation are first-class; stability/optimum are auxiliary."""
    mods = []
    if any(clean(row.get(k)) for k in ["Km_value", "kcat_value", "kcat_Km_value"]):
        mods.append("kinetic")
    if clean(row.get("degradation_efficiency")):
        mods.append("degradation")
    if has_product(row):
        mods.append("product")
    if has_optimum_aux(row):
        mods.append("optimum")
    if has_stability_aux(row):
        mods.append("stability")
    return mods


def canonical_variant(row: Dict[str, Any]) -> str:
    mutation_text = clean(row.get("mutations")).lower()
    enzyme_text = " ".join(clean(row.get(k)) for k in ["reported_enzyme_name", "enzyme_name"]).lower()
    text = " ".join([mutation_text, enzyme_text]).lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    reported_name = clean(row.get("reported_enzyme_name")).lower()
    reported_compact = re.sub(r"[^a-z0-9]+", "", reported_name)
    mutation_match = re.search(r"\b([a-z]\d+[a-z])\b", mutation_text, flags=re.IGNORECASE)
    if mutation_match:
        return mutation_match.group(1).upper()
    if reported_compact in {"wt", "wildtype"} or re.search(r"\b(?:wt|wild[- ]?type)\b", reported_name):
        return "WT"
    if re.search(r"\be186a\b", text) or "cotalaccasee186a" in compact:
        return "E186A"
    if re.search(r"\be186r\b", text) or "cotalaccasee186r" in compact:
        return "E186R"
    enzyme_compact = re.sub(r"[^a-z0-9]+", "", enzyme_text)
    if enzyme_compact in {"wt", "wildtype", "cotalaccasewt", "cotawt"} or re.search(r"\bcota[- ]?laccase\s+(?:wt|wild[- ]?type)\b", enzyme_text):
        return "WT"
    primary_name = clean(row.get("reported_enzyme_name") or row.get("enzyme_name")).lower()
    return re.sub(r"[^a-z0-9]+", "", primary_name)


def canonical_substrate_name(value: Any) -> str:
    text = clean(value).lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    aliases = {
        "afb1": "aflatoxin b1",
        "afm1": "aflatoxin m1",
        "aflatoxinb1": "aflatoxin b1",
        "aflatoxinm1": "aflatoxin m1",
        "zen": "zearalenone",
        "zea": "zearalenone",
        "zearalenone": "zearalenone",
        "zearalanone": "zearalanone",
        "zearalanol": "zearalanol",
        "alphazearalanol": "alpha-zearalanol",
        "betazearalanol": "beta-zearalanol",
        "alphazel": "alpha-zearalenol",
        "betazel": "beta-zearalenol",
        "stc": "sterigmatocystin",
        "sterigmatocystin": "sterigmatocystin",
        "don": "deoxynivalenol",
        "deoxynivalenol": "deoxynivalenol",
        "deoxynivalenoldon": "deoxynivalenol",
        "t2": "t-2 toxin",
        "t2toxin": "t-2 toxin",
        "ht2": "ht-2 toxin",
        "ht2toxin": "ht-2 toxin",
        "ota": "ochratoxin a",
        "ochratoxina": "ochratoxin a",
        "fb1": "fumonisin b1",
        "fumonisinb1": "fumonisin b1",
        "pat": "patulin",
        "patulin": "patulin",
        "phfb17": "phfb1_7",
        "phfb1": "phfb1",
        "hydrolyzedfumonisinb1": "hfb1",
        "hfb1": "hfb1",
        "4aniv": "4-acetylnivalenol",
        "4acetylnivalenol": "4-acetylnivalenol",
        "fusarenonx": "fusarenon-x",
        "fusx": "fusarenon-x",
        "das": "diacetoxyscirpenol",
        "diacetoxyscirpenol": "diacetoxyscirpenol",
        "isot": "isotrichothecene",
        "isotrichothecene": "isotrichothecene",
        "415dianiv": "4,15-diacetylnivalenol",
        "415diacetylnivalenol": "4,15-diacetylnivalenol",
    }
    # Direct match first
    if compact in aliases:
        return aliases[compact]
    # Handle parenthetical aliases: "Deoxynivalenol (DON)" → extract "don" from parens
    paren_match = re.search(r'\(([^)]+)\)', text)
    if paren_match:
        alias_compact = re.sub(r"[^a-z0-9]+", "", paren_match.group(1))
        if alias_compact in aliases:
            return aliases[alias_compact]
    # Strip parenthetical content and retry: "deoxynivalenol(don)" → "deoxynivalenol"
    stripped = re.sub(r'\([^)]*\)', '', text).strip()
    stripped_compact = re.sub(r"[^a-z0-9]+", "", stripped)
    if stripped_compact in aliases:
        return aliases[stripped_compact]
    return compact


def normalized_metric_value(value: Any) -> str:
    number = numeric(value)
    return f"{number:.8g}" if number is not None else clean(value).lower()


def record_completeness_score(row: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    tier_bonus = {
        "gold": 3,
        "silver": 2,
        "bronze": 1,
    }.get(clean(row.get("quality_tier")).lower(), 0)
    source_priority = _record_source_priority(row)
    direct_source = any(clean(row.get(k)) for k in ["evidence_text", "source_section", "source_table_id"])
    note_len = len(clean(row.get("notes")))
    source_bonus = 1 if direct_source else 0
    locked_bonus = 1 if clean(row.get("locked_candidate")).lower() == "true" else 0
    return (tier_bonus, source_priority, locked_bonus, source_bonus, note_len)


def _sample_token(row: Dict[str, Any]) -> str:
    blob = " ".join(clean(row.get(field)) for field in [
        "measurement_context_id", "notes", "evidence_text", "source_section"
    ])
    match = re.search(r"sample\s*#?\s*(\d+)", blob, flags=re.IGNORECASE)
    return f"sample{match.group(1)}" if match else ""


def _mediator_context_token(row: Dict[str, Any]) -> str:
    name = clean(row.get("mediator_name")).lower()
    if name in {"h2o2", "hydrogen peroxide"}:
        name = ""
    value = normalized_metric_value(row.get("mediator_concentration"))
    unit = clean(row.get("mediator_concentration_unit")).replace("μ", "µ").lower()
    blob = " ".join(clean(row.get(field)) for field in ["notes", "evidence_text", "source_section"]).lower()
    if not name and re.search(r"\bno\s+mediator\b|\bwithout\s+mediator\b", blob):
        name = "no_mediator"
    return "|".join(part for part in [name, value, unit] if part)


def _matrix_context_token(row: Dict[str, Any]) -> str:
    text = clean(row.get("matrix")).lower()
    text = text.replace("_", " ").replace("-", " ")
    aliases = {
        "lager beer": "beer",
        "beer": "beer",
        "uht milk": "milk",
        "skim milk": "milk",
        "milk": "milk",
        "phosphate buffer": "buffer",
        "reaction buffer": "buffer",
        "buffer": "buffer",
        "model solution": "buffer",
        "corn steep liquid": "corn steep liquor",
        "corn steep liquor": "corn steep liquor",
        "solid residue": "solid residues",
        "solid residues": "solid residues",
        "corn flour": "corn flour",
        "corn_flour": "corn flour",
    }
    compact_text = re.sub(r"\s+", " ", text).strip()
    if compact_text in aliases:
        return re.sub(r"[^a-z0-9]+", "", aliases[compact_text])
    return re.sub(r"[^a-z0-9]+", "", compact_text)


def _matrix_context_compatible(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_matrix = _matrix_context_token(left)
    right_matrix = _matrix_context_token(right)
    if left_matrix and right_matrix:
        return left_matrix == right_matrix
    # Missing matrix in one duplicate candidate is treated as less complete, not
    # a different context. This lets explicit table rows overwrite text rows
    # that omit "buffer/beer/milk" while all other context and metric fields match.
    return True


def _time_context_hours(row: Dict[str, Any], prefix: str) -> Optional[float]:
    value = numeric(row.get(f"{prefix}_time_value"))
    unit = clean(row.get(f"{prefix}_time_unit")).lower()
    if value is None:
        return None
    if unit.startswith("h") or "hour" in unit:
        return value
    if unit.startswith("min") or unit in {"m"}:
        return value / 60.0
    if unit.startswith("s") or "sec" in unit:
        return value / 3600.0
    return value


def _time_context_matches(left: Dict[str, Any], right: Dict[str, Any], prefix: str) -> bool:
    left_hours = _time_context_hours(left, prefix)
    right_hours = _time_context_hours(right, prefix)
    if left_hours is None or right_hours is None:
        return True
    return abs(left_hours - right_hours) <= max(1e-6, max(abs(left_hours), abs(right_hours), 1.0) * 0.01)


def _enzyme_identity_token(row: Dict[str, Any]) -> str:
    text = " ".join(clean(row.get(k)) for k in [
        "enzyme_name", "reported_enzyme_name", "gene_name", "mutations"
    ]).lower()
    text = text.replace("wild-type", "wt").replace("wild type", "wt")
    text = re.sub(r"\bwild\s*type\b", "wt", text)
    text = re.sub(r"\bwt\s+([a-z0-9_-]+)", r"\1 wt", text)
    class_terms = [
        "peroxidase", "laccase", "lipase", "hydrolase", "oxidase", "reductase",
        "transferase", "glucosyltransferase", "ugt", "esterase", "fumd",
        "os79", "rpod1", "rpod2", "rpod3", "fe1", "fe2", "afo", "afoth", "zhd",
    ]
    hits = [term for term in class_terms if term in text]
    if hits:
        return "|".join(dict.fromkeys(hits))
    return re.sub(r"[^a-z0-9]+", "", text)


def _enzyme_identity_compatible(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    lt = _enzyme_identity_token(left)
    rt = _enzyme_identity_token(right)
    if not lt or not rt:
        return False
    if lt == rt:
        return True
    left_parts = set(lt.split("|"))
    right_parts = set(rt.split("|"))
    if left_parts & right_parts:
        return True
    return lt in rt or rt in lt


def _close_metric(left: Any, right: Any, allow_per_minute: bool = False) -> bool:
    lnum = numeric(left)
    rnum = numeric(right)
    if lnum is None or rnum is None:
        return False
    scale = max(abs(lnum), abs(rnum), 1.0)
    if abs(lnum - rnum) / scale <= 0.025:
        return True
    if allow_per_minute:
        for factor in (60.0, 1.0 / 60.0):
            scaled = rnum * factor
            scale = max(abs(lnum), abs(scaled), 1.0)
            if abs(lnum - scaled) / scale <= 0.025:
                return True
    return False


def _common_metric_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    matched_any = False
    for field, allow_per_minute in [
        ("Km_value", False),
        ("kcat_value", True),
        ("degradation_efficiency", False),
    ]:
        if numeric(left.get(field)) is None or numeric(right.get(field)) is None:
            continue
        if not _close_metric(left.get(field), right.get(field), allow_per_minute=allow_per_minute):
            return False
        matched_any = True
    if _effective_kcat_km(left) is not None and _effective_kcat_km(right) is not None:
        if not _close_metric(_effective_kcat_km(left), _effective_kcat_km(right), allow_per_minute=True):
            return False
        matched_any = True
    return matched_any


def _effective_kcat_km(row: Dict[str, Any]) -> Optional[float]:
    value = numeric(row.get("kcat_Km_value"))
    if value is None:
        return None
    multiplier = numeric(row.get("kinetic_unit_multiplier"))
    if multiplier and multiplier != 1:
        # If a raw table-cell value survived with multiplier metadata, use the
        # effective normalized value for duplicate detection.
        threshold = max(100.0, abs(multiplier) / 10.0)
        if abs(value) < threshold:
            return value * multiplier
    return value


def _same_measurement_context(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    if clean(left.get("_pdf_stem") or left.get("doi") or left.get("pdf_file")).lower() != clean(right.get("_pdf_stem") or right.get("doi") or right.get("pdf_file")).lower():
        return False
    if canonical_substrate_name(left.get("substrate")) != canonical_substrate_name(right.get("substrate")):
        return False
    if clean(left.get("measurement_type")).lower() != clean(right.get("measurement_type")).lower():
        return False
    left_variant = canonical_variant(left)
    right_variant = canonical_variant(right)
    if left_variant != right_variant:
        wt_pair = "WT" in {left_variant, right_variant}
        if not (wt_pair and _enzyme_identity_compatible(left, right)) and not _enzyme_identity_compatible(left, right):
            return False

    left_sample = _sample_token(left)
    right_sample = _sample_token(right)
    if left_sample != right_sample:
        if left_sample and right_sample:
            return False
        left_table = _record_source_priority(left) >= 3
        right_table = _record_source_priority(right) >= 3
        if clean(left.get("measurement_type")).lower() == "kinetic" and (left_table != right_table):
            return True
        return False

    if clean(left.get("measurement_type")).lower() == "degradation":
        if not _matrix_context_compatible(left, right):
            return False
        if not _time_context_matches(left, right, "degradation"):
            return False
        left_mediator = _mediator_context_token(left)
        right_mediator = _mediator_context_token(right)
        if left_mediator or right_mediator:
            if left_mediator != right_mediator:
                return False
    elif clean(left.get("measurement_type")).lower() == "kinetic":
        if not _time_context_matches(left, right, "kinetic"):
            return False

    return _common_metric_match(left, right)


def _record_source_priority(row: Dict[str, Any]) -> int:
    source = clean(row.get("source_channel")).lower().replace("_", "")
    locked = clean(row.get("locked_candidate")).lower() == "true"
    if locked or source in {"parsedtable", "tableimagerescue"}:
        return 3
    if source == "text":
        return 1
    return 2


def _merge_duplicate_record(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(primary)
    for key, value in secondary.items():
        if merged.get(key) in (None, "", []):
            merged[key] = value
    if _has_concrete_enzyme_identity(merged) and clean(merged.get("identified_enzyme")).lower() in {"", "false"}:
        merged["identified_enzyme"] = "True"
        merged["putative_enzyme"] = "False"
        if clean(merged.get("enrichment_status")) == "blocked_inferred_enzyme":
            merged["enrichment_status"] = ""
        if clean(merged.get("QC_Status")) == "generic_or_unidentified_enzyme_name":
            merged["QC_Status"] = ""
    for key in ["enzyme_name", "reported_enzyme_name"]:
        secondary_value = clean(secondary.get(key))
        if secondary_value and not _is_generic_or_inferred_enzyme_label(secondary_value) and _is_generic_or_inferred_enzyme_label(merged.get(key)):
            merged[key] = secondary.get(key)
    if _has_concrete_enzyme_identity(secondary) and not _has_concrete_enzyme_identity(merged):
        for key in [
            "reported_enzyme_name", "enzyme_name", "gene_name", "uniprot_id",
            "genbank_id", "pdb_id", "ec_number", "sequence", "is_recombinant",
            "enzyme_state", "enzyme_system_type",
        ]:
            if clean(secondary.get(key)):
                merged[key] = secondary.get(key)
        merged["identified_enzyme"] = "True"
        merged["putative_enzyme"] = "False"
        if clean(merged.get("enrichment_status")) == "blocked_inferred_enzyme":
            merged["enrichment_status"] = ""
        if clean(merged.get("QC_Status")) == "generic_or_unidentified_enzyme_name":
            merged["QC_Status"] = ""
        append_note(merged, "Concrete enzyme identity restored from duplicate candidate evidence.")
    for field in ["notes", "evidence_text"]:
        left = clean(merged.get(field))
        right = clean(secondary.get(field))
        if right and right not in left:
            merged[field] = f"{left} | {right}" if left else right
    return merged


def deterministic_deduplicate_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate equivalent measurement rows after canonical normalization.

    Conditions are not part of the hard duplicate key because a table row may
    omit assay conditions while a text/teacher row supplies them. Sample labels
    are preserved so independent sample-labeled rows do not collapse.
    """
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        match_idx = None
        for idx, existing in enumerate(deduped):
            if _same_measurement_context(existing, row):
                match_idx = idx
                break
        if match_idx is None:
            deduped.append(row)
            continue
        existing = deduped[match_idx]
        if record_completeness_score(row) > record_completeness_score(existing):
            deduped[match_idx] = _merge_duplicate_record(row, existing)
        else:
            deduped[match_idx] = _merge_duplicate_record(existing, row)
    return deduped


def final_deduplicate_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Final export dedup after primary hard gate.

    Kept as a named wrapper so the final database contract is testable without
    coupling tests to the older deterministic cleanup function name.
    """
    return deterministic_deduplicate_records(rows)


def fill_missing_degradation_time_from_peer_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill missing degradation time only from equivalent same-paper records.

    This is intentionally narrow: same paper, enzyme alias, substrate alias,
    degradation efficiency, and compatible matrix. It repairs duplicate text
    rows that omitted time before the severe flag gate is applied.
    """
    out = [dict(row) for row in rows]
    for row in out:
        if clean(row.get("measurement_type")).lower() != "degradation":
            continue
        if clean(row.get("degradation_time_value")):
            continue
        for peer in out:
            if peer is row or clean(peer.get("measurement_type")).lower() != "degradation":
                continue
            if not clean(peer.get("degradation_time_value")):
                continue
            if (
                clean(row.get("_pdf_stem") or row.get("doi") or row.get("pdf_file")).lower()
                != clean(peer.get("_pdf_stem") or peer.get("doi") or peer.get("pdf_file")).lower()
            ):
                continue
            if canonical_substrate_name(row.get("substrate")) != canonical_substrate_name(peer.get("substrate")):
                continue
            if not _enzyme_identity_compatible(row, peer):
                continue
            if not _matrix_context_compatible(row, peer):
                continue
            if not _close_metric(row.get("degradation_efficiency"), peer.get("degradation_efficiency")):
                continue
            row["degradation_time_value"] = peer.get("degradation_time_value")
            row["degradation_time_unit"] = peer.get("degradation_time_unit")
            remove_flags(row, ["missing_time_for_degradation"])
            append_note(row, "Filled missing degradation time from equivalent same-paper/source context before final gate.")
            break
    return out


def deterministic_final_cleanup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = [deterministic_record_cleanup(row) for row in rows]
    # v9: mycotoxin substrate filtering moved to QualityTierClassifier Rule 1
    return deterministic_deduplicate_records(cleaned)


# NOTE: apply_eligibility() removed — replaced by QualityTierClassifier.classify_record()
# NOTE: validate_rescue_records() and run_rescue_for_paper() were removed in v8.
# The rescue path produced 274 records of which only 6 had any quantitative
# value (Km/kcat/kcat_Km/degradation_efficiency) and 80% were human-flagged false
# positives. See REFACTOR_PROMPT_v8.md §1.1.


def process_paper_level_only(pipeline: EnhancedExtractionPipeline, paper_dir: Path) -> Dict[str, Any]:
    """Run the production paper-level aggregation path without duplicate legacy extraction calls."""
    import asyncio

    full_text = read_text(paper_dir / "full.md", limit=None)
    content_files = list(paper_dir.glob("*_content_list.json"))
    if content_files:
        content_list = json.loads(content_files[0].read_text(encoding="utf-8"))
    else:
        if not full_text:
            return {"success": False, "records": [], "stats": {}, "error": "content_list.json and full.md not found"}
        content_list = [{"type": "text", "content": full_text, "block_id": 1, "source": "full_md_fallback"}]
    doi = pipeline._extract_doi(paper_dir)
    result = asyncio.run(
        pipeline.paper_level_extractor.extract_paper(
            paper_blocks=content_list,
            doi=doi,
            paper_dir=paper_dir,
        )
    )
    records = result.get("aggregated_records", [])
    debug_trace = dict(result.get("debug_trace") or {})
    text_statuses = debug_trace.get("text_model_statuses") or {}
    if not records and text_statuses:
        failed_statuses = {
            clean(v.get("status"))
            for v in text_statuses.values()
            if isinstance(v, dict) and clean(v.get("status")) not in {"success", "empty_success"}
        }
        if failed_statuses and len(failed_statuses) == len(text_statuses):
            return {
                "success": False,
                "records": [],
                "stats": {"model_results": result.get("model_results", {})},
                "debug_trace": debug_trace,
                "error": "all_text_model_calls_failed:" + ";".join(sorted(failed_statuses)),
            }
    paper_commercial_enzyme_context = infer_paper_commercial_enzyme_context(full_text)
    if records:
        if paper_commercial_enzyme_context:
            for record in records:
                record.setdefault("paper_commercial_enzyme_context", paper_commercial_enzyme_context)
        records = pipeline._clean_duplicate_fields(records)
        records = pipeline._filter_primary_dataset_records(records)
        records = [apply_inferred_enzyme_guard(record) for record in records]
    commercial_fallback_records = extract_commercial_enzyme_degradation_fallback(full_text, records)
    if commercial_fallback_records:
        records = list(records or []) + commercial_fallback_records
        debug_trace["commercial_enzyme_fallback_records"] = commercial_fallback_records
    if paper_commercial_enzyme_context and records:
        for record in records:
            record.setdefault("paper_commercial_enzyme_context", paper_commercial_enzyme_context)
    enrichment_stats = None
    if records and pipeline.enable_sequence_enrichment and pipeline.sequence_enricher:
        records, enrichment_stats = pipeline.sequence_enricher.enrich_records(records, auto_fill=True, verbose=False)
    # Re-clear human_review_required for known mycotoxin substrates after enrichment
    # (enrichment may re-set HR=true for ambiguous_candidate even when substrate is a known mycotoxin)
    _MYCOTOXIN_TERMS = (
        "aflatoxin", "afb1", "afm1", "ochratoxin", "ota",
        "deoxynivalenol", "don", "zearalenone", "zearalanone", "zearalanol",
        "zearalenol", "zel", "zen", "zea", "patulin",
        "sterigmatocystin", "citrinin", "fumonisin", "fb1", "fb2",
        "t-2", "ht-2", "nivalenol", "niv", "isot", "isotrichodermol", "mycotoxin",
    )
    if records:
        for rec in records:
            if rec.get("human_review_required"):
                sub_text = str(rec.get("substrate") or "").lower()
                if sub_text and any(t in sub_text for t in _MYCOTOXIN_TERMS):
                    rec["human_review_required"] = False
    if records:
        records = normalize_records_batch(records)
        records = deterministic_final_cleanup(records)
        records = pipeline._validate_row_level_records(records)
    debug_trace["after_validator"] = records
    debug_trace["final_records"] = records
    return {
        "success": True,
        "records": records,
        "stats": {
            "total": len(records),
            "model_results": result.get("model_results", {}),
            "enrichment": enrichment_stats,
        },
        "debug_trace": debug_trace,
        "error": None,
    }


def summary_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    return (
        clean(row.get("_pdf_stem") or row.get("doi")).lower(),
        clean(row.get("reported_enzyme_name") or row.get("canonical_enzyme_name") or row.get("enzyme_name")).lower(),
        clean(row.get("substrate")).lower(),
        clean(row.get("mutations")).lower(),
        clean(row.get("is_wild_type")).lower(),
        clean(row.get("is_recombinant")).lower(),
        clean(row.get("organism")).lower(),
        clean(row.get("strain")).lower(),
        clean(row.get("enzyme_system_type") or row.get("enzyme_state")).lower(),
    )


def unique_join(values: Iterable[Any]) -> str:
    seen = []
    for value in values:
        text = clean(value)
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen)


def build_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[summary_key(row)].append(row)
    summary = []
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    for group_rows in groups.values():
        out = {field: unique_join(r.get(field) for r in group_rows) for field in fields}
        out["record_granularity"] = "enzyme_substrate_summary"
        mods = []
        for mod in ["kinetic", "degradation", "product", "optimum", "stability"]:
            if any(mod in clean(r.get("evidence_modalities")).split(";") for r in group_rows):
                mods.append(mod)
        out["evidence_modalities"] = ";".join(mods)
        out["has_kinetic_data"] = "Yes" if "kinetic" in mods else "No"
        out["has_degradation_data"] = "Yes" if "degradation" in mods else "No"
        out["has_product_data"] = "Yes" if "product" in mods else "No"
        out["has_activity_assay_data"] = "No"
        out["has_optimum_data"] = "Yes" if "optimum" in mods else "No"
        out["has_stability_data"] = "Yes" if "stability" in mods else "No"
        statuses = [clean(r.get("eligibility_status")) for r in group_rows]
        if "primary_eligible" in statuses:
            out["eligibility_status"] = "primary_eligible"
            out["primary_dataset_allowed"] = "True"
        elif "needs_manual_review" in statuses:
            out["eligibility_status"] = "needs_manual_review"
            out["primary_dataset_allowed"] = "False"
        elif "secondary_candidate" in statuses:
            out["eligibility_status"] = "secondary_candidate"
            out["primary_dataset_allowed"] = "False"
        elif statuses:
            out["eligibility_status"] = statuses[0]
            out["primary_dataset_allowed"] = "False"
        summary.append(out)
    return summary


def token_records() -> List[Dict[str, Any]]:
    records = list(getattr(TokenUsageTracker, "_records", []))
    out = []
    for record in records:
        d = record if isinstance(record, dict) else record.__dict__
        out.append(
            {
                "pdf_file": "",
                "stage": clean(d.get("task")) or "unlabeled_llm_call",
                "model_name": clean(d.get("model")),
                "provider": clean(d.get("provider")),
                "input_tokens": d.get("prompt_tokens", 0),
                "output_tokens": d.get("completion_tokens", 0),
                "total_tokens": d.get("total_tokens", 0),
                "token_usage_source": "api_usage" if clean(d.get("source")) == "api" else "tokenizer_estimate",
                "estimated_cost": "",
                "pricing_available": "False",
                "latency_seconds": "",
                "success": "True",
                "error_message": "",
            }
        )
    return out


def keyword_counts(paper_dir: Path) -> Dict[str, int]:
    text = read_text(paper_dir / "full.md", limit=None)
    return {
        "mycotoxin_keyword_hits": len(MYCOTOXIN_RE.findall(text)),
        "enzyme_keyword_hits": len(ENZYME_TRANSFORM_RE.findall(text)),
        "kinetic_keyword_hits": len(KINETIC_RE.findall(text)),
        "degradation_keyword_hits": len(DEGRADATION_RE.findall(text)),
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def print_progress(
    completed: int,
    total: int,
    paper_name: str,
    status: str,
    validated_records: int,
    run_started: float,
    width: int = 28,
) -> None:
    """Print a compact per-paper progress bar without external dependencies."""
    if total <= 0:
        return
    fraction = min(max(completed / total, 0.0), 1.0)
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - run_started
    avg = elapsed / completed if completed else 0
    eta = avg * (total - completed) if completed else 0
    print(
        f"[{bar}] {completed}/{total} ({fraction * 100:5.1f}%) "
        f"status={status} records={validated_records} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"paper={paper_name}",
        flush=True,
    )


def make_clients_and_pipeline(config_path: Path, max_workers: int):
    return init_pipeline(config_path, max_workers=max_workers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-papers-dir", default=str(ALL_PAPERS_DIR))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--output-root")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-workers", type=int, default=1)
    args = ap.parse_args()

    load_dotenv()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or PROJECT_ROOT / "output" / f"all_papers_full_extraction_{timestamp}")
    dirs = {
        "preflight": output_root / "preflight",
        "raw": output_root / "raw_outputs",
        "validated": output_root / "validated_outputs",
        "logs": output_root / "logs",
        "debug": output_root / "debug_traces",
        "review": output_root / "review_filtered",
        "secondary": output_root / "secondary_candidates",
        "rejected": output_root / "rejected_records",
        "exports": output_root / "curated_exports",
        "metrics": output_root / "metrics",
        "reports": output_root / "comparison_reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    classifier = QualityTierClassifier()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(dirs["logs"] / "all_papers_full_extraction.log", encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("all_papers_full_extraction")

    inventory, new_dirs, inventory_meta = build_inventory(Path(args.all_papers_dir), dirs["preflight"])
    if args.limit:
        new_dirs = new_dirs[: args.limit]
    logger.info("Inventory complete: %s", inventory_meta)
    if args.inventory_only:
        print(json.dumps({"output_root": str(output_root), **inventory_meta}, ensure_ascii=False, indent=2))
        return
    if not new_dirs:
        logger.warning("No new PDFs to extract.")
        print(json.dumps({"output_root": str(output_root), "new_for_extraction": 0}, ensure_ascii=False, indent=2))
        return

    TokenUsageTracker.reset()
    pipeline = make_clients_and_pipeline(Path(args.config), args.max_workers)
    prechecker = PaperLevelPrechecker(min_mycotoxin_hits=1, min_kinetics_hits=2, enable_unit_check=True)

    run_rows: List[Dict[str, Any]] = []
    zero_unresolved: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    failed_consecutive = 0
    started = time.time()

    if args.resume:
        run_rows = read_csv(dirs["reports"] / "all_papers_run_summary.csv")
        architecture_mismatch_stems = set()
        for row in run_rows:
            if clean(row.get("status")) != "success":
                continue
            stem = clean(row.get("pdf_file")).removesuffix(".pdf")
            log_path = dirs["logs"] / f"{stem}.log"
            log_text = read_text(log_path) if log_path.exists() else ""
            if "[KIMI] Starting text extraction" not in log_text:
                architecture_mismatch_stems.add(stem)
        if architecture_mismatch_stems:
            logger.warning(
                "Resume detected %s success rows without Kimi logs; they will be rerun with the standard architecture: %s",
                len(architecture_mismatch_stems),
                sorted(architecture_mismatch_stems),
            )
            run_rows = [
                row for row in run_rows
                if clean(row.get("pdf_file")).removesuffix(".pdf") not in architecture_mismatch_stems
            ]
            write_csv(dirs["reports"] / "all_papers_run_summary.csv", run_rows)

        done_stems = {Path(clean(row.get("paper_dir"))).name for row in run_rows if clean(row.get("paper_dir"))}
        done_stems.update(clean(row.get("pdf_file")).removesuffix(".pdf") for row in run_rows if clean(row.get("pdf_file")))
        for path in sorted(dirs["validated"].glob("*_validated.json")):
            if path.name.removesuffix("_validated.json") in architecture_mismatch_stems:
                continue
            data = read_json(path) or {}
            records = data.get("records") if isinstance(data, dict) else data if isinstance(data, list) else []
            for record in records:
                if isinstance(record, dict):
                    all_records.append(record)
        new_dirs = [d for d in new_dirs if d.name not in done_stems]
        logger.info("Resume enabled: loaded %s prior run rows and %s prior records; remaining papers=%s", len(run_rows), len(all_records), len(new_dirs))

    total_to_process = len(run_rows) + len(new_dirs)
    logger.info("Extraction queue: total=%s already_done=%s remaining=%s", total_to_process, len(run_rows), len(new_dirs))
    print_progress(len(run_rows), total_to_process, "queue_start", "starting", 0, started)

    for offset, paper_dir in enumerate(new_dirs, start=1):
        index = len(run_rows) + 1
        start = time.time()
        paper_name = paper_dir.name
        pdf_file = f"{paper_name}.pdf"
        log_path = dirs["logs"] / f"{paper_name}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(fh)
        status = "success"
        notes = []
        normal_records: List[Dict[str, Any]] = []
        precheck = {}
        document_type = ""
        debug_trace: Dict[str, Any] = {}
        try:
            logger.info("[%s/%s] Processing %s", index, total_to_process, paper_name)
            precheck = prechecker.should_skip_paper(paper_dir)
            document_type = clean(precheck.get("Document_Type"))
            if precheck.get("should_skip"):
                status = "skipped_precheck"
                records = []
            else:
                result = process_paper_level_only(pipeline, paper_dir)
                if result.get("success"):
                    normal_records = result.get("records", [])
                    debug_trace = result.get("debug_trace", {}) or {}
                    records = normal_records
                else:
                    status = "failed"
                    records = []
                    notes.append(result.get("error") or "pipeline_failed")

                # v8: zero-record rescue removed. If paper-level extraction
                # returns nothing, that's the signal — do not run a second
                # broadened-prompt pass that would manufacture FPs.
                counts = keyword_counts(paper_dir)
                if (
                    status == "success"
                    and not normal_records
                    and document_type.lower() != "review"
                ):
                    zero_unresolved.append({
                        "pdf_file": pdf_file,
                        "precheck_status": "passed",
                        "document_type": document_type,
                        **counts,
                        "normal_records": 0,
                        "failure_stage": "paper_level_zero",
                        "no_record_reason": "paper_level_extractor_returned_empty",
                        "recommended_action": "manual_review_or_skip",
                    })
            eligible_records = []
            paper_scope_allowed_context = infer_paper_allowed_scope_context(
                read_text(paper_dir / "full.md", limit=250000)
            )
            for idx, record in enumerate(records, start=1):
                record = dict(record)
                record["_pdf_file"] = pdf_file
                record["_pdf_stem"] = paper_name
                record["pdf_file"] = pdf_file
                record["doi"] = paper_name
                if paper_scope_allowed_context and not clean(record.get("paper_scope_allowed_context")):
                    record["paper_scope_allowed_context"] = paper_scope_allowed_context
                record["source_record_id"] = clean(record.get("source_record_id")) or f"{paper_name}:{idx}"
                record = classifier.classify_record(record)
                record = apply_primary_hard_gate(record)
                eligible_records.append(record)
            all_records.extend(eligible_records)
            if debug_trace:
                debug_trace["final_records"] = eligible_records
            write_json(dirs["raw"] / f"{paper_name}.json", {"records": records, "precheck": precheck})
            write_json(dirs["validated"] / f"{paper_name}_validated.json", {"records": eligible_records})
            save_records_csv(dirs["validated"] / f"{paper_name}_validated.csv", eligible_records)
            paper_debug_dir = dirs["debug"] / paper_name
            for debug_name in [
                "text_candidates_raw",
                "text_model_statuses",
                "text_candidates_by_model",
                "parsed_table_candidates_raw",
                "table_image_rescue_candidates_raw",
                "locked_table_candidates",
                "aggregation_input_records",
                "teacher_harmonized_records",
                "aggregation_output_records",
                "final_candidate_pool",
                "after_rescue_protection",
                "after_safety_filters",
                "after_deterministic_validator",
                "text_metric_fallbacks",
                "commercial_enzyme_fallback_records",
                "after_validator",
                "final_records",
            ]:
                write_json(paper_debug_dir / f"{debug_name}.json", debug_trace.get(debug_name, []))
            write_json(
                dirs["debug"] / f"{paper_name}_trace.json",
                {
                    "paper_dir": str(paper_dir),
                    "precheck": precheck,
                    "normal_records": len(normal_records),
                    "validated_records": len(eligible_records),
                    "debug_files_dir": str(paper_debug_dir),
                    "debug_counts": {
                        "text_candidates_raw": sum(len(v) for v in (debug_trace.get("text_candidates_raw") or {}).values())
                        if isinstance(debug_trace.get("text_candidates_raw"), dict) else len(debug_trace.get("text_candidates_raw") or []),
                        "parsed_table_candidates_raw": len(debug_trace.get("parsed_table_candidates_raw") or []),
                        "table_image_rescue_candidates_raw": len(debug_trace.get("table_image_rescue_candidates_raw") or []),
                        "locked_table_candidates": len(debug_trace.get("locked_table_candidates") or []),
                        "teacher_harmonized_records": len(debug_trace.get("teacher_harmonized_records") or []),
                        "aggregation_output_records": len(debug_trace.get("aggregation_output_records") or []),
                        "final_candidate_pool": len(debug_trace.get("final_candidate_pool") or []),
                        "after_rescue_protection": len(debug_trace.get("after_rescue_protection") or []),
                        "after_safety_filters": len(debug_trace.get("after_safety_filters") or []),
                        "after_deterministic_validator": len(debug_trace.get("after_deterministic_validator") or []),
                        "after_validator": len(debug_trace.get("after_validator") or []),
                    },
                    "notes": notes,
                },
            )
            failed_consecutive = 0 if status != "failed" else failed_consecutive + 1
        except Exception as exc:
            status = "failed"
            failed_consecutive += 1
            notes.append(str(exc))
            write_json(dirs["raw"] / f"{paper_name}.json", {"records": [], "error": str(exc)})
            write_json(dirs["validated"] / f"{paper_name}_validated.json", {"records": []})
            save_records_csv(dirs["validated"] / f"{paper_name}_validated.csv", [])
        finally:
            root_logger.removeHandler(fh)
            fh.close()

        row_records = [r for r in all_records if r.get("_pdf_stem") == paper_name]
        tier_counts = Counter(r.get("quality_tier") for r in row_records)
        run_rows.append(
            {
                "pdf_file": pdf_file,
                "paper_dir": str(paper_dir),
                "status": status,
                "precheck_status": "skipped" if precheck.get("should_skip") else "passed",
                "document_type": document_type,
                "normal_records": len(normal_records),
                "validated_records": len(row_records),
                "gold_records": tier_counts.get("Gold", 0),
                "silver_records": tier_counts.get("Silver", 0),
                "bronze_records": tier_counts.get("Bronze", 0),
                "rejected_records": tier_counts.get("Rejected", 0),
                "runtime_seconds": round(time.time() - start, 2),
                "notes": "; ".join(notes),
            }
        )
        write_csv(dirs["reports"] / "all_papers_run_summary.csv", run_rows)
        print_progress(len(run_rows), total_to_process, paper_name, status, len(row_records), started)

        if failed_consecutive >= 10:
            logger.error("Stopping after 10 consecutive failures.")
            break

    # v10: final high-precision primary database gate is independent of
    # Gold/Silver/Bronze quality tier. Tier describes completeness; the gate
    # decides whether the record belongs in the primary enzyme database.
    # Cross-table dedup: remove Silver records when Gold exists for same measurement
    all_records = remove_silver_when_gold_exists(all_records)
    all_records = fill_missing_degradation_time_from_peer_records(all_records)
    all_records = [apply_primary_hard_gate(r) for r in all_records]

    all_extracted = all_records
    primary_records = [r for r in all_records if clean(r.get("primary_dataset_allowed")).lower() == "true" and clean(r.get("record_scope")) == "primary_enzyme_record"]
    primary_records = final_deduplicate_records(primary_records)
    rejected_records = [r for r in all_records if clean(r.get("record_scope")) == "rejected_out_of_scope"]
    # Backward-compatible name, now intentionally high-precision primary only.
    database_records = primary_records
    summary_rows = build_summary_rows(all_extracted)
    gold = [r for r in all_records if r.get("quality_tier") == "Gold"]
    silver = [r for r in all_records if r.get("quality_tier") == "Silver"]
    bronze = [r for r in all_records if r.get("quality_tier") == "Bronze"]

    write_csv(dirs["exports"] / "debug_all_records.csv", project_to_v9(all_extracted))
    write_csv(dirs["exports"] / "all_extracted_records.csv", project_to_v9(all_extracted))
    write_csv(dirs["exports"] / "final_primary_database.csv", project_to_v9(primary_records))
    write_csv(dirs["exports"] / "database_records.csv", project_to_v9(database_records))
    write_csv(dirs["exports"] / "rejected_records.csv", project_to_v9(rejected_records))
    write_csv(dirs["exports"] / "enzyme_substrate_summary.csv", summary_rows)
    write_csv(dirs["reports"] / "zero_record_unresolved_report.csv", zero_unresolved)

    token_rows = token_records()
    write_csv(dirs["metrics"] / "model_usage_summary_all_papers.csv", token_rows)
    write_csv(dirs["metrics"] / "token_usage_audit_all_papers.csv", token_rows)
    write_csv(dirs["metrics"] / "runtime_summary_all_papers.csv", run_rows)
    write_csv(dirs["metrics"] / "precheck_summary_all_papers.csv", run_rows)
    write_csv(dirs["metrics"] / "table_routing_summary_all_papers.csv", [])

    status_counts = Counter(r["status"] for r in run_rows)
    tier_counts = Counter(r.get("quality_tier") for r in all_records)
    scope_counts = Counter(r.get("record_scope") for r in all_records)
    total_runtime = time.time() - started
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in token_rows)
    primary_kinetic = sum(1 for r in primary_records if clean(r.get("measurement_type")).lower() == "kinetic")
    primary_degradation = sum(1 for r in primary_records if clean(r.get("measurement_type")).lower() == "degradation")
    non_enzyme_removed = sum(
        1 for r in all_records
        if clean(r.get("primary_dataset_allowed")).lower() != "true"
        and (
            "not_identified_primary_enzyme" in clean(r.get("rejection_reason"))
            or "metabolic_activity_not_primary" in clean(r.get("rejection_reason"))
        )
    )
    metric_semantic_rejected = sum(
        1 for r in all_records
        if "metric_semantic_mismatch" in clean(r.get("rejection_reason"))
        or "metric_semantic_mismatch" in clean(r.get("error_flags"))
    )
    mixed_context_rejected = sum(
        1 for r in all_records
        if "mixed_measurement_context" in clean(r.get("error_flags"))
        or "context_conflict" in clean(r.get("error_flags"))
    )

    report = [
        "# All Papers Full Extraction Report",
        "",
        "## Corpus Summary",
        f"- output directory: `{output_root}`",
        f"- total PDFs in All_papers inventory: {inventory_meta['total_pdf_like_paper_dirs']}",
        f"- new_for_extraction: {inventory_meta['status_counts'].get('new_for_extraction', 0)}",
        f"- duplicate_pdf: {inventory_meta['status_counts'].get('duplicate_pdf', 0)}",
        f"- processed this run: {len(run_rows)}",
        f"- status counts: {dict(status_counts)}",
        "",
        "## Record Summary",
        f"- all extracted records: {len(all_extracted)}",
        f"- final primary database records: {len(primary_records)}",
        f"- rejected records: {len(rejected_records)}",
        f"- enzyme-substrate summary records: {len(summary_rows)}",
        f"- primary kinetic records: {primary_kinetic}",
        f"- primary degradation records: {primary_degradation}",
        "",
        "## Quality Tier Summary",
        f"- Gold: {tier_counts.get('Gold', 0)}",
        f"- Silver: {tier_counts.get('Silver', 0)}",
        f"- Bronze: {tier_counts.get('Bronze', 0)}",
        f"- Rejected: {tier_counts.get('Rejected', 0)}",
        "",
        "## Primary Scope Gate Summary",
        f"- scope counts: {dict(scope_counts)}",
        f"- non-enzyme records removed from primary: {non_enzyme_removed}",
        f"- metric semantic mismatches rejected from primary: {metric_semantic_rejected}",
        f"- mixed/context-conflict records flagged: {mixed_context_rejected}",
        "",
        "## Zero-Record Papers (paper-level extraction returned empty)",
        f"- zero-record papers: {len(zero_unresolved)}",
        "",
        "## Token / Runtime",
        f"- total token records: {len(token_rows)}",
        f"- total tokens: {total_tokens}",
        f"- total runtime seconds: {round(total_runtime, 2)}",
    ]
    (dirs["reports"] / "all_papers_final_report.md").write_text("\n".join(report), encoding="utf-8")
    shutil.copyfile(dirs["reports"] / "all_papers_final_report.md", dirs["reports"] / "all_papers_eligibility_report.md")
    shutil.copyfile(dirs["reports"] / "all_papers_final_report.md", dirs["reports"] / "all_papers_error_cluster_report.md")
    shutil.copyfile(dirs["reports"] / "all_papers_final_report.md", dirs["reports"] / "all_papers_review_filter_report.md")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "all_papers_total": inventory_meta["total_pdf_like_paper_dirs"],
                "new_for_extraction": inventory_meta["status_counts"].get("new_for_extraction", 0),
                "processed": len(run_rows),
                "status_counts": dict(status_counts),
                "all_extracted_records": len(all_extracted),
                "final_primary_database_records": len(primary_records),
                "database_records": len(database_records),
                "gold_records": len(gold),
                "silver_records": len(silver),
                "bronze_records": len(bronze),
                "rejected_records": len(rejected_records),
                "primary_kinetic_records": primary_kinetic,
                "primary_degradation_records": primary_degradation,
                "non_enzyme_records_removed_from_primary": non_enzyme_removed,
                "metric_semantic_mismatch_records": metric_semantic_rejected,
                "mixed_context_flagged_records": mixed_context_rejected,
                "zero_record_papers": len(zero_unresolved),
                "total_tokens": total_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
