#!/usr/bin/env python3
"""Deterministic parsed-PDF preflight before MycoExtract LLM extraction.

This script does not call any LLM API. It scans parsed paper directories,
checks file completeness, PDF/DOI duplicates, parsing quality, document type,
and biological enzyme/mycotoxin assay signals. It then writes a paper-level
triage table and a conservative primary extraction queue.

Supported layouts:

1. One-level parsed layout:
   parsed_pdf/<paper_dir>/full.md
   parsed_pdf/<paper_dir>/*_content_list.json
   parsed_pdf/<paper_dir>/*_origin.pdf

2. Two-level layout:
   all_papers/<parent>/<paper_dir>/full.md
   all_papers/<parent>/<paper_dir>/*_content_list.json
   all_papers/<parent>/<paper_dir>/*_origin.pdf
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Positive eligibility signals
# ---------------------------------------------------------------------------

MYCOTOXIN_RE = re.compile(
    r"\b(aflatoxin|aflatoxins|afb1|afb2|afg1|afg2|afm1|afm2|"
    r"ochratoxin|ochratoxins|ota|otb|otc|deoxynivalenol|\bdon\b|"
    r"nivalenol|\bniv\b|t-2|ht-2|fumonisin|fumonisins|fb1|fb2|fb3|"
    r"zearalenone|\bzea\b|\bzen\b|patulin|citrinin|sterigmatocystin|"
    r"alternariol|beauvericin|tenuazonic|trichothecene|trichothecenes|"
    r"enniatin|ergot alkaloid|mycotoxin|mycotoxins)\b",
    re.I,
)

BIOLOGICAL_ENZYME_SYSTEM_RE = re.compile(
    r"\b(purified enzyme|recombinant enzyme|enzyme|enzymatic|laccase|"
    r"peroxidase|oxidase|hydrolase|esterase|lactonase|epimerase|"
    r"reductase|dehydrogenase|transferase|glucosyltransferase|\bugt\b|"
    r"acetyltransferase|crude enzyme|crude extract|extracellular extract|"
    r"culture supernatant|cell-free extract|cell free extract|"
    r"fermentation supernatant|whole cell|whole-cell|microbial culture|"
    r"bacteria|bacterial|fungus|fungal|yeast|strain|recombinant|purified|"
    r"commercial enzyme)\b",
    re.I,
)

DIRECT_ASSAY_RE = re.compile(
    r"\b(assay|reaction mixture|incubated|incubation|treatment|treated with|"
    r"degradation rate|transformation rate|residual toxin|hplc|lc-ms|lcms|"
    r"uplc|uhplc|tlc|product|metabolite|kinetic parameters|\bkm\b|kcat|"
    r"vmax|kcat/km|residual activity|heat treatment|thermostable|"
    r"storage stability)\b",
    re.I,
)

TRANSFORMATION_RE = re.compile(
    r"\b(degradation|degraded|detoxification|detoxified|biotransformation|"
    r"transformation|conversion|converted|hydrolysis|hydrolyzed|oxidation|"
    r"reduction|glycosylation|glucosylation|acetylation|deacetylation|"
    r"epimerization|conjugation|removal|removed)\b",
    re.I,
)

PRIMARY_SECTION_RE = re.compile(
    r"\b(materials and methods|methods|experimental|results|"
    r"results and discussion|enzymatic assay|degradation assay|"
    r"kinetic assay|product identification)\b",
    re.I,
)

KINETIC_RE = re.compile(
    r"\b(kinetic|kinetic parameters|\bkm\b|kcat|kcat/km|vmax|michaelis|"
    r"catalytic efficiency|intrinsic clearance|\bclint\b)\b",
    re.I,
)

DEGRADATION_RE = re.compile(
    r"\b(degrad|detox|conversion|removal|transform|biotransformation|"
    r"hydrolys|oxidation|reduction|decomposition)\b",
    re.I,
)

REVIEW_STRONG_RE = re.compile(
    r"\b(review|systematic review|meta-analysis|meta analysis|mini-review|"
    r"mini review|scoping review|bibliometric|perspective|opinion|overview)\b",
    re.I,
)

REVIEW_ABSTRACT_PHRASE_RE = re.compile(
    r"\b(this review|we review|this article reviews|we summarize|"
    r"this paper summarizes recent advances|reviewed here)\b",
    re.I,
)

REVIEW_POSSIBLE_RE = re.compile(
    r"\b(recent advances|progress|current status|future perspectives|"
    r"state of the art|summarizes|overview of)\b",
    re.I,
)

IN_SILICO_RE = re.compile(
    r"\b(molecular docking|docking|in silico|molecular dynamics|simulation|"
    r"homology modeling|binding energy)\b",
    re.I,
)

VERY_LIGHT_NONBIOLOGICAL_WARNING_RE = re.compile(
    r"\b(mof|metal-organic framework|photocatalyst|photocatalytic|"
    r"chemical catalyst|adsorbent|pms/|persulfate|fenton|ribosome binding)\b",
    re.I,
)

JOURNAL_HEADER_RE = re.compile(
    r"\b(journal|volume|issue|copyright|received|accepted|available online|"
    r"contents lists available|elsevier|springer|wiley|mdpi|article info|"
    r"keywords|doi:|https?://)\b",
    re.I,
)


TRIAGE_FIELDS = [
    "paper_dir",
    "paper_dir_path",
    "parent_label",
    "pdf_file",
    "pdf_path",
    "normalized_doi_or_id",
    "title",
    "abstract_snippet",
    "triage_status",
    "triage_reason",
    "review_status",
    "review_reason",
    "has_direct_assay_signal",
    "has_biological_enzyme_system_signal",
    "has_transformation_signal",
    "allow_zero_record_rescue",
    "identity_level_note",
    "warning_flags",
    "mycotoxin_keyword_hits",
    "biological_enzyme_system_hits",
    "direct_assay_hits",
    "transformation_hits",
    "primary_section_hits",
    "kinetic_keyword_hits",
    "degradation_keyword_hits",
    "in_silico_hits",
    "text_length",
    "has_pdf",
    "has_full_md",
    "has_content_list",
    "table_count",
    "figure_count",
    "duplicate_reason",
    "duplicate_of",
]


INVENTORY_FIELDS = [
    "paper_dir",
    "paper_dir_path",
    "parent_label",
    "pdf_file",
    "pdf_path",
    "pdf_size",
    "normalized_doi_or_id",
    "title",
    "abstract_snippet",
    "has_pdf",
    "has_full_md",
    "has_content_list",
    "has_layout_json",
    "content_list_file",
    "text_length",
    "block_count",
    "table_count",
    "figure_count",
    "mycotoxin_keyword_hits",
    "biological_enzyme_system_hits",
    "direct_assay_hits",
    "transformation_hits",
    "primary_section_hits",
    "kinetic_keyword_hits",
    "degradation_keyword_hits",
    "in_silico_hits",
    "review_keyword_hits",
    "review_status",
    "review_reason",
    "triage_status",
    "triage_reason",
    "extraction_recommendation",
    "allow_zero_record_rescue",
    "identity_level_note",
    "warning_flags",
    "duplicate_reason",
    "duplicate_of",
    "pdf_sha256",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def clean_id(value: str) -> str:
    text = clean(value).lower().replace("\\", "/").split("/")[-1]
    text = re.sub(r"\.(pdf|html?|xml|json|md)$", "", text)
    text = re.sub(r"(_origin|-main)$", "", text)
    text = re.sub(r"\s+\(\d+\)$", "", text)
    text = re.sub(r"[^a-z0-9._/-]+", "", text)
    return text.strip("._-/")


def norm_doi(value: str) -> str:
    """Normalize DOI-like strings without dropping the DOI prefix.

    Examples:
    - 10.1021_acs.biochem.7b01007 -> 10.1021/acs.biochem.7b01007
    - https://doi.org/10.xxxx/xxx -> 10.xxxx/xxx
    - doi:10.xxxx/xxx -> 10.xxxx/xxx

    If no DOI can be found, returns a cleaned paper id.
    """
    raw = clean(value).lower().strip()
    if not raw:
        return ""
    text = raw.replace("\\", "/")
    text = re.sub(r"^\s*doi\s*:\s*", "", text)
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"\.(pdf|html?|xml|json|md)$", "", text)
    text = re.sub(r"(_origin|-main)$", "", text)
    text = re.sub(r"\s+\(\d+\)$", "", text)

    doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:a-z0-9]+)", text, flags=re.I)
    if doi_match:
        return doi_match.group(1).strip(" .")

    underscore_match = re.search(r"(10\.\d{4,9})_([a-z0-9][-_./;():a-z0-9]+)", text, flags=re.I)
    if underscore_match:
        prefix = underscore_match.group(1)
        suffix = underscore_match.group(2).replace("_", ".")
        candidate = f"{prefix}/{suffix}"
        doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:a-z0-9]+)", candidate, flags=re.I)
        if doi_match:
            return doi_match.group(1).strip(" .")

    return clean_id(text)


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_paper_dirs(root: Path) -> List[Tuple[Path, str]]:
    """Return candidate paper dirs with a parent label.

    A directory is treated as a paper dir if it directly contains a full.md,
    a PDF, a content_list JSON, or a layout.json. If top-level dirs are only
    containers, their direct children are scanned.
    """
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    paper_dirs: List[Tuple[Path, str]] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        direct_markers = (
            (child / "full.md").exists()
            or (child / "layout.json").exists()
            or any(child.glob("*.pdf"))
            or any(child.glob("*_content_list.json"))
        )
        if direct_markers:
            paper_dirs.append((child, root.name))
            continue

        for grandchild in sorted(p for p in child.iterdir() if p.is_dir()):
            nested_markers = (
                (grandchild / "full.md").exists()
                or (grandchild / "layout.json").exists()
                or any(grandchild.glob("*.pdf"))
                or any(grandchild.glob("*_content_list.json"))
            )
            if nested_markers:
                paper_dirs.append((grandchild, child.name))
    return paper_dirs


def first_file(paths: Iterable[Path]) -> Optional[Path]:
    for path in sorted(paths):
        return path
    return None


def content_list_stats(path: Optional[Path]) -> Tuple[int, int, int]:
    if not path or not path.exists():
        return 0, 0, 0
    data = read_json(path)
    if isinstance(data, dict):
        data = data.get("blocks") or data.get("content") or data.get("items") or []
    if not isinstance(data, list):
        return 0, 0, 0
    table_count = 0
    figure_count = 0
    for block in data:
        if not isinstance(block, dict):
            continue
        block_type = clean(block.get("type") or block.get("block_type")).lower()
        if block_type == "table":
            table_count += 1
        elif block_type in {"figure", "image"}:
            figure_count += 1
    return len(data), table_count, figure_count


def keyword_count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text or ""))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text)).strip()


def is_bad_title_candidate(line: str) -> bool:
    text = normalize_space(line)
    if len(text) < 12 or len(text) > 320:
        return True
    if JOURNAL_HEADER_RE.search(text):
        return True
    if re.match(r"^(abstract|keywords|introduction|references)\b", text, flags=re.I):
        return True
    if len(re.findall(r"[A-Za-z]", text)) < 8:
        return True
    return False


def extract_title_abstract(text: str, paper_dir_name: str) -> Tuple[str, str]:
    """Extract a best-effort title and abstract from full.md.

    The title is first taken from a markdown heading in the first 15k chars.
    If no heading is available, the function chooses the longest plausible
    early line. The abstract is taken from the "Abstract" section up to
    Keywords or Introduction. Failures return paper_dir_name and empty abstract.
    """
    head = text[:15000] if text else ""
    title = ""

    for line in head.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,3}\s+\S", stripped):
            candidate = re.sub(r"^#{1,3}\s+", "", stripped).strip()
            if not is_bad_title_candidate(candidate):
                title = candidate
                break

    if not title:
        candidates = []
        for line in head.splitlines()[:80]:
            stripped = re.sub(r"^[#*\-\s]+", "", line).strip()
            if not is_bad_title_candidate(stripped):
                candidates.append(stripped)
        if candidates:
            title = max(candidates, key=len)

    abstract = ""
    abstract_match = re.search(
        r"(?ims)^\s*#{0,6}\s*abstract\s*[:.\-]?\s*(.+?)(?=^\s*#{0,6}\s*(keywords?|introduction|1\.?\s+introduction)\b)",
        head,
    )
    if abstract_match:
        abstract = normalize_space(abstract_match.group(1))
    else:
        inline_match = re.search(
            r"(?is)\babstract\b\s*[:.\-]?\s*(.{100,2500}?)(?=\b(keywords?|introduction)\b)",
            head,
        )
        if inline_match:
            abstract = normalize_space(inline_match.group(1))

    return title or paper_dir_name, abstract


def assess_review_status(title: str, abstract: str, text_head: str) -> Tuple[str, str, int]:
    """Classify review likelihood using title/abstract-first signals."""
    title_abstract = f"{title}\n{abstract}"
    review_hits = keyword_count(REVIEW_STRONG_RE, title_abstract)
    if REVIEW_STRONG_RE.search(title_abstract):
        return "REVIEW_CONFIRMED", "strong review phrase in title_or_abstract", review_hits
    if REVIEW_ABSTRACT_PHRASE_RE.search(abstract):
        return "REVIEW_CONFIRMED", "review self-description in abstract", review_hits

    head_blob = f"{title_abstract}\n{text_head[:30000]}"
    possible_hits = keyword_count(REVIEW_POSSIBLE_RE, head_blob)
    strong_head_hits = keyword_count(REVIEW_STRONG_RE, text_head[:30000])
    if possible_hits > 0 or strong_head_hits >= 2:
        return "REVIEW_POSSIBLE", "possible review/overview language outside strong title-abstract signal", strong_head_hits
    return "NOT_REVIEW_LIKE", "no review-like signal", review_hits


def biological_identity_note(text: str) -> str:
    blob = text[:60000]
    if re.search(r"\b(whole cell|whole-cell|culture supernatant|fermentation supernatant|crude extract|crude enzyme|cell-free extract|cell free extract|extracellular extract)\b", blob, re.I):
        return "unidentified biological activity possible"
    if re.search(r"\b(purified|recombinant|commercial enzyme)\b", blob, re.I):
        return "identified or semi-identified enzyme system possible"
    return ""


def warning_flags(text: str) -> str:
    flags = []
    if VERY_LIGHT_NONBIOLOGICAL_WARNING_RE.search(text[:60000]):
        flags.append("nonbiological_catalyst_warning")
    if IN_SILICO_RE.search(text[:60000]):
        flags.append("in_silico_signal")
    return ";".join(flags)


def load_exclusion_keys(paths: List[Path], hash_pdfs: bool) -> Tuple[set[str], set[str]]:
    doi_keys: set[str] = set()
    hash_keys: set[str] = set()
    for root in paths:
        if not root.exists():
            continue
        for pdf in root.rglob("*.pdf"):
            doi_keys.add(norm_doi(pdf.stem))
            if hash_pdfs:
                try:
                    hash_keys.add(sha256_file(pdf))
                except Exception:
                    pass
        try:
            for paper_dir, _parent in iter_paper_dirs(root):
                doi_keys.add(norm_doi(paper_dir.name))
        except Exception:
            continue
    return doi_keys, hash_keys


def classify_row(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return triage_status, triage_reason, and extraction recommendation."""
    if row["duplicate_reason"]:
        return "DUPLICATE", row["duplicate_reason"], "skip_duplicate"
    if row["has_full_md"] != "True":
        return "MISSING_PARSED_TEXT", "full.md missing", "manual_check_or_reparse"
    if int(row["text_length"] or 0) < 1500:
        return "PARSING_LOW_QUALITY", "full.md shorter than 1500 characters", "manual_check_or_reparse"
    if row["review_status"] == "REVIEW_CONFIRMED":
        return "SECONDARY_ONLY_REVIEW", row["review_reason"], "send_to_secondary_review_pool"

    myco = int(row["mycotoxin_keyword_hits"] or 0)
    bio = int(row["biological_enzyme_system_hits"] or 0)
    transform = int(row["transformation_hits"] or 0)
    direct = int(row["direct_assay_hits"] or 0)
    kinetic = int(row["kinetic_keyword_hits"] or 0)
    primary_section = int(row["primary_section_hits"] or 0)
    in_silico = int(row["in_silico_hits"] or 0)

    if myco == 0:
        return "OUT_OF_SCOPE_NO_MYCOTOXIN", "no mycotoxin signal", "skip_out_of_scope"
    if bio == 0:
        return "OUT_OF_SCOPE_NO_BIOLOGICAL_ENZYME_SYSTEM", "mycotoxin signal but no biological enzyme/system signal", "skip_out_of_scope"
    if row["review_status"] == "REVIEW_POSSIBLE":
        return "UNCERTAIN_NEEDS_LLM_TRIAGE", row["review_reason"], "manual_or_llm_triage_before_extraction"

    if myco > 0 and bio > 0 and transform > 0 and direct > 0:
        return "PRIMARY_ELIGIBLE", "mycotoxin + biological enzyme/system + transformation + direct assay signals", "run_full_pipeline"

    if myco > 0 and bio > 0 and (transform > 0 or kinetic > 0 or in_silico > 0):
        if direct == 0:
            return "UNCERTAIN_NEEDS_LLM_TRIAGE", "enzyme/mycotoxin signal but direct assay evidence unclear", "manual_or_llm_triage_before_extraction"
        if primary_section == 0:
            return "UNCERTAIN_NEEDS_LLM_TRIAGE", "direct assay signal present but primary section evidence weak", "manual_or_llm_triage_before_extraction"

    return "ZERO_RECORD_CANDIDATE", "insufficient positive eligibility signals for full extraction", "do_not_extract_without_manual_triage"


def build_row(
    paper_dir: Path,
    parent_label: str,
    exclusion_dois: set[str],
    exclusion_hashes: set[str],
    seen_doi: Dict[str, str],
    seen_hash: Dict[str, str],
    hash_pdfs: bool,
    triage_text_limit: int,
) -> Dict[str, Any]:
    """Build one inventory/triage row without raising on malformed files."""
    pdf = first_file(list(paper_dir.glob("*.pdf")) + list(paper_dir.glob("*_origin.pdf")))
    full_md = paper_dir / "full.md"
    content_list = first_file(paper_dir.glob("*_content_list.json"))
    layout = paper_dir / "layout.json"

    doi_key = norm_doi(paper_dir.name) or (norm_doi(pdf.stem) if pdf else "")
    pdf_hash = ""
    if pdf and hash_pdfs:
        try:
            pdf_hash = sha256_file(pdf)
        except Exception:
            pdf_hash = ""

    duplicate_reason = ""
    duplicate_of = ""
    if doi_key and doi_key in exclusion_dois:
        duplicate_reason = "matches_exclusion_doi"
        duplicate_of = doi_key
    elif pdf_hash and pdf_hash in exclusion_hashes:
        duplicate_reason = "matches_exclusion_pdf_hash"
        duplicate_of = pdf_hash[:16]
    elif pdf_hash and pdf_hash in seen_hash:
        duplicate_reason = "hash_duplicate"
        duplicate_of = seen_hash[pdf_hash]
    elif doi_key and doi_key in seen_doi:
        duplicate_reason = "doi_or_folder_duplicate"
        duplicate_of = seen_doi[doi_key]

    if pdf_hash and pdf_hash not in seen_hash:
        seen_hash[pdf_hash] = paper_dir.name
    if doi_key and doi_key not in seen_doi:
        seen_doi[doi_key] = paper_dir.name

    triage_text = read_text(full_md, limit=triage_text_limit) if full_md.exists() else ""
    full_text_length = len(read_text(full_md)) if full_md.exists() else 0
    title, abstract = extract_title_abstract(triage_text, paper_dir.name)
    abstract_snippet = abstract[:800]
    block_count, table_count, figure_count = content_list_stats(content_list)
    review_status, review_reason, review_hits = assess_review_status(title, abstract, triage_text)

    row: Dict[str, Any] = {
        "paper_dir": paper_dir.name,
        "paper_dir_path": str(paper_dir),
        "parent_label": parent_label,
        "pdf_file": pdf.name if pdf else "",
        "pdf_path": str(pdf) if pdf else "",
        "pdf_size": pdf.stat().st_size if pdf and pdf.exists() else "",
        "normalized_doi_or_id": doi_key,
        "title": title,
        "abstract_snippet": abstract_snippet,
        "has_pdf": str(bool(pdf)),
        "has_full_md": str(full_md.exists()),
        "has_content_list": str(bool(content_list)),
        "has_layout_json": str(layout.exists()),
        "content_list_file": content_list.name if content_list else "",
        "text_length": full_text_length,
        "block_count": block_count,
        "table_count": table_count,
        "figure_count": figure_count,
        "mycotoxin_keyword_hits": keyword_count(MYCOTOXIN_RE, triage_text),
        "biological_enzyme_system_hits": keyword_count(BIOLOGICAL_ENZYME_SYSTEM_RE, triage_text),
        "direct_assay_hits": keyword_count(DIRECT_ASSAY_RE, triage_text),
        "transformation_hits": keyword_count(TRANSFORMATION_RE, triage_text),
        "primary_section_hits": keyword_count(PRIMARY_SECTION_RE, triage_text),
        "kinetic_keyword_hits": keyword_count(KINETIC_RE, triage_text),
        "degradation_keyword_hits": keyword_count(DEGRADATION_RE, triage_text),
        "in_silico_hits": keyword_count(IN_SILICO_RE, triage_text),
        "review_keyword_hits": review_hits,
        "review_status": review_status,
        "review_reason": review_reason,
        "has_direct_assay_signal": "True" if keyword_count(DIRECT_ASSAY_RE, triage_text) > 0 else "False",
        "has_biological_enzyme_system_signal": "True" if keyword_count(BIOLOGICAL_ENZYME_SYSTEM_RE, triage_text) > 0 else "False",
        "has_transformation_signal": "True" if keyword_count(TRANSFORMATION_RE, triage_text) > 0 else "False",
        "identity_level_note": biological_identity_note(triage_text),
        "warning_flags": warning_flags(triage_text),
        "duplicate_reason": duplicate_reason,
        "duplicate_of": duplicate_of,
        "pdf_sha256": pdf_hash,
    }
    triage_status, triage_reason, recommendation = classify_row(row)
    row["triage_status"] = triage_status
    row["triage_reason"] = triage_reason
    row["extraction_status"] = triage_status
    row["extraction_recommendation"] = recommendation
    row["allow_zero_record_rescue"] = "True" if triage_status == "PRIMARY_ELIGIBLE" else "False"
    return row


def build_inventory(
    input_dir: Path,
    output_dir: Path,
    exclude_dirs: List[Path],
    hash_pdfs: bool,
    text_scan_limit: int,
    triage_text_limit: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_text_limit = triage_text_limit or text_scan_limit
    exclusion_dois, exclusion_hashes = load_exclusion_keys(exclude_dirs, hash_pdfs)
    seen_doi: Dict[str, str] = {}
    seen_hash: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []

    for paper_dir, parent_label in iter_paper_dirs(input_dir):
        try:
            row = build_row(
                paper_dir=paper_dir,
                parent_label=parent_label,
                exclusion_dois=exclusion_dois,
                exclusion_hashes=exclusion_hashes,
                seen_doi=seen_doi,
                seen_hash=seen_hash,
                hash_pdfs=hash_pdfs,
                triage_text_limit=effective_text_limit,
            )
        except Exception as exc:
            row = {
                "paper_dir": paper_dir.name,
                "paper_dir_path": str(paper_dir),
                "parent_label": parent_label,
                "triage_status": "PARSING_LOW_QUALITY",
                "triage_reason": f"exception during preflight: {exc}",
                "extraction_status": "PARSING_LOW_QUALITY",
                "extraction_recommendation": "manual_check_or_reparse",
                "review_status": "NOT_REVIEW_LIKE",
                "allow_zero_record_rescue": "False",
            }
        rows.append(row)

    primary = [r for r in rows if r.get("triage_status") == "PRIMARY_ELIGIBLE"]
    uncertain = [
        r for r in rows
        if r.get("triage_status") == "UNCERTAIN_NEEDS_LLM_TRIAGE"
        or r.get("review_status") == "REVIEW_POSSIBLE"
    ]
    reviews = [r for r in rows if r.get("triage_status") == "SECONDARY_ONLY_REVIEW"]
    zero_candidates = [
        r for r in rows
        if r.get("triage_status") in {
            "ZERO_RECORD_CANDIDATE",
            "OUT_OF_SCOPE_NO_MYCOTOXIN",
            "OUT_OF_SCOPE_NO_BIOLOGICAL_ENZYME_SYSTEM",
        }
    ]
    duplicates = [r for r in rows if r.get("triage_status") == "DUPLICATE"]
    parsing_qc = [
        r for r in rows
        if r.get("triage_status") in {"MISSING_PARSED_TEXT", "PARSING_LOW_QUALITY"}
        or r.get("has_content_list") != "True"
        or r.get("has_pdf") != "True"
    ]

    write_csv(output_dir / "parsed_pdf_inventory.csv", rows, INVENTORY_FIELDS)
    write_csv(output_dir / "paper_triage.csv", rows, TRIAGE_FIELDS)
    write_csv(output_dir / "primary_extraction_queue.csv", primary, TRIAGE_FIELDS)
    write_csv(output_dir / "extraction_candidates.csv", primary, TRIAGE_FIELDS)
    write_csv(output_dir / "uncertain_needs_llm_triage.csv", uncertain, TRIAGE_FIELDS)
    write_csv(output_dir / "secondary_review_pool.csv", reviews, TRIAGE_FIELDS)
    write_csv(output_dir / "zero_record_candidates.csv", zero_candidates, TRIAGE_FIELDS)
    write_csv(output_dir / "skipped_duplicates.csv", duplicates, INVENTORY_FIELDS)
    write_csv(output_dir / "parsing_qc_report.csv", parsing_qc, INVENTORY_FIELDS)

    triage_counts = Counter(r.get("triage_status", "") for r in rows)
    review_counts = Counter(r.get("review_status", "") for r in rows)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_paper_dirs": len(rows),
        "triage_counts": dict(triage_counts),
        "review_status_counts": dict(review_counts),
        "primary_extraction_queue_count": len(primary),
        "uncertain_needs_llm_triage_count": len(uncertain),
        "secondary_review_pool_count": len(reviews),
        "zero_record_candidate_count": len(zero_candidates),
        "duplicate_count": len(duplicates),
        "parsing_qc_count": len(parsing_qc),
        "hash_pdfs": hash_pdfs,
        "exclude_dirs": [str(p) for p in exclude_dirs],
        "zero_record_rescue_policy": (
            "Only PRIMARY_ELIGIBLE papers may enter rescue_queue after a later extraction returns 0 records. "
            "ZERO_RECORD_CANDIDATE, review, uncertain, duplicate, and out-of-scope papers must not be rescued directly."
        ),
    }
    write_json(output_dir / "preflight_summary.json", summary)
    write_summary_md(output_dir / "preflight_summary.md", summary)
    return summary


def write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Parsed PDF Preflight Summary",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- total_paper_dirs: {summary['total_paper_dirs']}",
        f"- primary_extraction_queue_count: {summary['primary_extraction_queue_count']}",
        f"- uncertain_needs_llm_triage_count: {summary['uncertain_needs_llm_triage_count']}",
        f"- secondary_review_pool_count: {summary['secondary_review_pool_count']}",
        f"- zero_record_candidate_count: {summary['zero_record_candidate_count']}",
        f"- hash_pdfs: {summary['hash_pdfs']}",
        "",
        "## Triage Counts",
    ]
    for key, value in sorted(summary["triage_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Review Status Counts"])
    for key, value in sorted(summary["review_status_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Zero-Record Rescue Policy",
        summary["zero_record_rescue_policy"],
        "",
        "## Output Files",
    ])
    for name in [
        "parsed_pdf_inventory.csv",
        "paper_triage.csv",
        "primary_extraction_queue.csv",
        "extraction_candidates.csv",
        "uncertain_needs_llm_triage.csv",
        "secondary_review_pool.csv",
        "zero_record_candidates.csv",
        "skipped_duplicates.csv",
        "parsing_qc_report.csv",
    ]:
        lines.append(f"- `{name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify_text_for_test(title: str, abstract: str, body: str) -> str:
    """Small helper for deterministic unit-like checks; not run by default."""
    text = f"# {title}\n\nAbstract\n{abstract}\n\n{body}"
    review_status, review_reason, review_hits = assess_review_status(title, abstract, text)
    row = {
        "duplicate_reason": "",
        "has_full_md": "True",
        "text_length": max(len(text), 2000),
        "review_status": review_status,
        "review_reason": review_reason,
        "mycotoxin_keyword_hits": keyword_count(MYCOTOXIN_RE, text),
        "biological_enzyme_system_hits": keyword_count(BIOLOGICAL_ENZYME_SYSTEM_RE, text),
        "direct_assay_hits": keyword_count(DIRECT_ASSAY_RE, text),
        "transformation_hits": keyword_count(TRANSFORMATION_RE, text),
        "primary_section_hits": keyword_count(PRIMARY_SECTION_RE, text),
        "kinetic_keyword_hits": keyword_count(KINETIC_RE, text),
        "in_silico_hits": keyword_count(IN_SILICO_RE, text),
    }
    status, _reason, _recommendation = classify_row(row)
    return status


def run_self_tests() -> None:
    """Run lightweight triage checks. Call manually; not executed by default."""
    cases = [
        (
            "A review of enzymatic degradation of mycotoxins",
            "",
            "This paper discusses mycotoxins and enzymes.",
            "SECONDARY_ONLY_REVIEW",
        ),
        (
            "Mycotoxin detoxification",
            "This systematic review summarizes recent advances in enzymatic degradation.",
            "Methods and Results describe search criteria.",
            "SECONDARY_ONLY_REVIEW",
        ),
        (
            "AFB1 degradation by crude enzyme extract",
            "Crude enzyme extract degraded AFB1 by HPLC assay.",
            "Materials and Methods. The reaction mixture was incubated and residual toxin was quantified.",
            "PRIMARY_ELIGIBLE",
        ),
        (
            "Catalytic degradation of AFB1",
            "PMS/MF@CRHHT catalytic degradation of AFB1 was investigated.",
            "The photocatalytic material removed AFB1.",
            "OUT_OF_SCOPE_NO_BIOLOGICAL_ENZYME_SYSTEM",
        ),
        (
            "Docking of enzyme with DON",
            "Molecular docking of enzyme with DON was performed.",
            "The study predicted binding energy and in silico interactions.",
            "UNCERTAIN_NEEDS_LLM_TRIAGE",
        ),
    ]
    for title, abstract, body, expected in cases:
        observed = classify_text_for_test(title, abstract, body)
        if observed != expected:
            raise AssertionError(f"{title!r}: expected {expected}, observed {observed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight parsed PDF folders before MycoExtract LLM extraction.")
    parser.add_argument("--input-dir", default=r"F:\Mycotoxin\parsed_pdf", help="Parsed PDF root directory.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to analysis_outputs/parsed_pdf_preflight_YYYYMMDD_HHMMSS/preflight.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory containing already-processed PDFs/paper dirs to exclude. Can be repeated.",
    )
    parser.add_argument("--no-hash", action="store_true", help="Skip PDF SHA256 hashing for faster inventory.")
    parser.add_argument("--text-scan-limit", type=int, default=120000, help="Legacy max full.md chars scanned for keywords.")
    parser.add_argument("--triage-text-limit", type=int, default=200000, help="Max full.md chars scanned for paper triage.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("analysis_outputs") / f"parsed_pdf_preflight_{timestamp}" / "preflight"
    )
    summary = build_inventory(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        exclude_dirs=[Path(p) for p in args.exclude_dir],
        hash_pdfs=not args.no_hash,
        text_scan_limit=args.text_scan_limit,
        triage_text_limit=args.triage_text_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
