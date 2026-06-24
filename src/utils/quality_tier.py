"""
Quality Tier Classifier - 5 Hard Rules + 3 Quality Tiers

Replaces the legacy eligibility routing (8 rejection categories), dual confidence
scoring systems, and multi-layer filtering with a single, clear classification:

    5 Hard Rules → pass/fail
    8 Field Groups → Gold / Silver / Bronze / Rejected

Design principles:
- Extraction stage only extracts, never deletes records
- Missing units are flagged, not removed
- All records get classified (no silent drops)
"""

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    """Normalize a value to a clean string. Empty/None → ''."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(_clean(v) for v in value if _clean(v))
    text = str(value).strip()
    if text.lower() in {"none", "nan", "null", "[]"}:
        return ""
    return text


class QualityTierClassifier:
    """Classifies extracted records into Gold/Silver/Bronze/Rejected."""

    # =========================================================================
    # Rule 1: Mycotoxin substrate whitelist
    # Consolidated from quality_constraints.py + is_target_mycotoxin()
    # =========================================================================

    MYCOTOXIN_WHITELIST = {
        # Aflatoxins
        "aflatoxin b1", "aflatoxin b2", "aflatoxin g1", "aflatoxin g2",
        "aflatoxin m1", "aflatoxin m2", "aflatoxin p1",
        "afb1", "afb2", "afg1", "afg2", "afm1", "afm2", "afp1",
        # Ochratoxins
        "ochratoxin a", "ochratoxin b", "ochratoxin c",
        "ota", "otb", "otc",
        # Trichothecenes
        "deoxynivalenol", "nivalenol", "t-2 toxin", "ht-2 toxin",
        "3-acetyldeoxynivalenol", "15-acetyldeoxynivalenol",
        "diacetoxyscirpenol", "4-acetylneosolaniol", "isotrichothecene",
        "isotrichodermol",
        "don", "niv", "t-2", "ht-2", "3-adon", "15-adon", "das",
        "4-aniv", "4aniv", "isot",
        # Fumonisins
        "fumonisin b1", "fumonisin b2", "fumonisin b3",
        "hydrolyzed fumonisin b1",
        "fb1", "fb2", "fb3", "hfb1",
        # Zearalenones
        "zearalenone", "alpha-zearalenol", "beta-zearalenol",
        "zearalanone", "alpha-zearalanol", "beta-zearalanol",
        "zen", "zea",
        # Others
        "patulin", "citrinin", "sterigmatocystin",
        "cyclopiazonic acid", "roquefortine c", "mycophenolic acid",
        "alternariol", "alternariol monomethyl ether",
        "tenuazonic acid", "moniliformin", "beauvericin",
        "pat", "cit", "ste", "cpa", "roq-c", "mpa",
        "aoh", "ame", "tea", "mon", "bea",
        # Enniatins
        "enniatin", "enniatin a", "enniatin b", "enniatin b1",
        "enn", "enna", "ennb", "ennb1",
        # Ergot alkaloids
        "ergotamine", "ergocristine", "ergocryptine", "ergot alkaloids",
        # Metabolites / derivatives
        "aflatoxicol", "aflatoxin q1", "afq1",
        "ochratoxin alpha", "fusarenon-x", "fus-x",
        "don-3-glucoside", "d3g",
        "zearalenone-14-glucoside", "zen-14g",
        # Generic family names
        "mycotoxin", "mycotoxins",
        "aflatoxin", "ochratoxin", "fumonisin", "trichothecene",
        "ergot alkaloid",
    }

    # Compact aliases (no spaces/special chars) for fast matching
    MYCOTOXIN_COMPACT_ALIASES = {
        "afb1", "afb2", "afg1", "afg2", "afm1", "aflatoxinb1",
        "aflatoxinb2", "aflatoxing1", "aflatoxing2", "aflatoxinm1",
        "ota", "otb", "ochratoxina", "ochratoxinb",
        "don", "deoxynivalenol", "niv", "nivalenol", "t2", "ht2", "das",
        "3adon", "15adon", "3acetyldeoxynivalenol", "15acetyldeoxynivalenol",
        "4aniv", "4acetylneosolaniol", "isotrichothecene", "isotrichodermol", "isot",
        "fb1", "fb2", "fb3", "hfb1", "fumonisinb1", "fumonisinb2", "fumonisinb3",
        "hydrolyzedfumonisinb1",
        "zen", "zea", "zearalenone", "alphazel", "betazel", "zel",
        "zearalanone", "alphazearalanol", "betazearalanol",
        "zearalanol", "alphazearalanone", "betazearalanone",
        "pat", "patulin", "citrinin", "stc", "sterigmatocystin",
        "alternariol", "aoh", "ame", "alternariolmonomethylether",
        "tea", "tenuazonicacid",
    }

    MYCOTOXIN_PATTERNS = [
        r"\baflatoxin\s*(?:b1|b2|g1|g2|m1)\b",
        r"\bochratoxin\s*(?:a|b)\b",
        r"\bfumonisin\s*(?:b1|b2|b3)\b",
        r"\bzearalenone\b|\bzearalanone\b|\bzearalanol\b|\bzen\b|\bzea\b|\bzea[- ]?p\b",
        r"\balpha[- ]?zearalenol\b|\bbeta[- ]?zearalenol\b|\balpha[- ]?zearalanol\b|\bbeta[- ]?zearalanol\b",
        r"\bdeoxynivalenol\b|\bdon\b|\bnivalenol\b|\bniv\b",
        r"\bt[- ]?2\b|\bht[- ]?2\b|\b3[- ]?adon\b|\b15[- ]?adon\b|\bisotrichodermol\b|\bisot\b",
        r"\bpatulin\b|\bcitrinin\b|\bsterigmatocystin\b",
        r"\balternariol\b|\btenuazonic acid\b",
        r"\bmycotoxin\b",
    ]

    NON_MYCOTOXIN_SUBSTRATES = {
        "abts", "syringaldazine", "sgz",
        "2,6-dimethoxyphenol", "dmp",
        "guaiacol", "catechol", "hydroquinone",
        "veratryl alcohol",
        "p-nitrophenol", "4-nitrophenol", "p-np", "pnp",
        "p-nitrophenyl phosphate", "4-nitrophenyl phosphate", "p-npp",
        "hippuryl-l-phenylalanine", "hlp",
        "7-ethoxyresorufin", "7-pentoxyresorufin", "7-methoxyresorufin",
        "pentoxyresorufin", "ethoxyresorufin", "methoxyresorufin",
        "remazol brilliant blue r", "rbbr",
        "reactive black 5", "rb5",
        "methylene blue", "malachite green",
        "congo red", "bromophenol blue",
        "benzo[a]pyrene", "benzo(a)pyrene", "bap",
        "styrene oxide", "aniline", "pentachlorophenol",
        "dichloromethane", "dibromomethane",
        "docetaxel", "taxotere", "nifedipine", "terfenadine",
        "coumarin", "d-cysteine", "l-cysteine",
        "5-hydroxymethylfurfural", "hmf",
        "gentisyl alcohol", "toluquinol",
        "brefeldin a",
        "hydrogen peroxide", "h2o2",
        "glutathione", "gsh", "gssg", "oxidized glutathione",
        "monodehydroascorbate", "dehydroascorbate",
        "sphingosine-1-phosphate", "s1p",
        "reactive oxygen species", "ros", "superoxide",
        "synbiotic a", "synbiotic b", "synbiotic c",
        "bioplus 2b", "cylactin", "inulin",
        "turkey tissue", "intestinal content", "fecal content",
    }

    # =========================================================================
    # Rule 2: Non-enzymatic markers
    # From is_nonenzymatic() in run_all_papers_full_extraction.py
    # =========================================================================

    NON_ENZYMATIC_MARKERS = [
        "mof", "metal-organic framework", "pcn(ptpd)", "photocatalyst", "photocatalytic",
        "chemical catalyst", "adsorbent", "adsorption", "ribosome binding", "toxin binding",
        "magnetic graphene oxide", "graphene oxide", "nanocomposite", "magnetic beads",
        "sulfhydryl-terminated magnetic beads", "membrane adsorber", "dialysis", "uv light",
        "photolysis", "ozone", "plasma", "pms/", "pds/", "persulfate", "advanced oxidation",
        "nanomaterial", "nano material", "nanozyme", "porous carbon", "carbon material",
        "biocomposite", "magnetic composite", "superparamagnetic", "metal catalyst",
        "single-atom catalyst", "fenton", "pms ", "pds ", "peroxymonosulfate",
        "peroxydisulfate", "not a biological enzyme", "not biological enzyme",
    ]

    HYBRID_ENZYME_MARKERS = [
        "-lac", "laccase", "peroxidase", "oxidase", "hydrolase", "reductase", "dehydrogenase",
        "transferase", "commercial enzyme", "crude enzyme", "immobilized enzyme", "enzyme-metal",
        "enzyme metal", "culture supernatant", "cell-free extract", "crude extract",
    ]

    BIOACTIVATION_MARKERS = [
        "bioactivation", "metabolic activation", "afbo",
        "aflatoxin b1-8,9-epoxide", "aflatoxin-8,9-epoxide",
        "aflatoxin epoxide", "epoxide formation", "dna adduct",
        "protein adduct", "carcinogenic metabolite", "mutagenic metabolite",
        "genotoxic metabolite", "human liver microsome", "human liver microsomes",
        "liver microsome", "liver microsomes", "cdna-expressed cyp",
        "cyp1a2", "cyp3a4", "cyp3a5", "cyp3a7",
        "p450 1a2", "p450 3a4", "p450 3a5", "p450 3a7",
        "cytochrome p450", "pmol p450", "p450/min",
        "p450-mediated activation", "afq1 formation", "afm1 formation",
    ]

    GENERIC_ENZYME_NAMES = {
        "extracellular enzyme", "extracellular enzymes",
        "intracellular enzyme", "intracellular enzymes",
        "secreted enzyme", "secreted enzymes",
        "unidentified enzyme", "unknown enzyme",
        "degrading enzyme", "detoxifying enzyme",
        "ota-hydrolytic enzyme", "zen-degrading enzyme",
        "afb1-degrading enzyme", "mycotoxin-degrading enzyme",
        "enzymatic components", "extracellular proteins",
        "proteinaceous component", "crude enzyme", "crude enzymes",
        "enzyme preparation", "enzyme mixture",
        "commercial enzyme", "commercial enzyme preparation",
        "commercial preparation", "crude extract", "cell-free lysate",
        "cell free lysate", "culture supernatant", "fermentation supernatant",
        "whole cell", "whole-cell",
    }

    UNIDENTIFIED_BIOLOGICAL_SYSTEM_MARKERS = [
        "crude biological material", "biomass", "powder", "fungal powder",
        "mushroom powder", "plant powder", "microbial powder",
        "fruiting body material", "mycelium", "mycelial biomass",
        "whole cell", "whole-cell", "intact cell", "cell suspension",
        "bacterial culture", "fungal culture", "yeast culture",
        "culture supernatant", "fermentation supernatant",
        "extracellular fraction", "extracellular proteins",
        "crude extract", "crude enzyme extract", "cell-free extract",
        "cell free extract", "cell-free lysate", "cell free lysate",
        "tissue fraction", "microsomal fraction", "digestive matrix",
        "food matrix material",
    ]

    DIRECT_ENZYME_SCOPE_BLOCKERS = [
        "crude enzyme", "crude enzymes", "crude extract", "crude enzyme extract",
        "cell-free extract", "cell free extract", "cell-free lysate", "cell free lysate",
        "lysate", "culture supernatant", "fermentation supernatant",
        "extracellular fraction", "extracellular proteins", "whole cell", "whole-cell",
        "cell suspension", "biomass", "powder", "compound enzyme", "compound enzymes",
        "immobilized", "immobilised", "immobilization", "immobilisation",
        "co-immobilized", "coimmobilized", "composite", "nanocomplex",
        "supported enzyme", "enzyme support", "carrier", "enzyme-loaded", "enzyme-coated",
        "microsphere", "microspheres", "microbead", "microbeads", "hydrogel",
        "sodium alginate", "alginate microsphere", "montmorillonite",
        "covalently immobilized", "covalent bonding", "cross-linked", "crosslinked",
        "zif-8", "msn-nh2",
    ]

    DIRECT_ENZYME_RECORD_MARKERS = [
        "purified enzyme", "purified protein", "purified recombinant",
        "recombinant enzyme", "recombinant protein", "enzyme variant",
        "commercial laccase", "commercial lipase", "commercial peroxidase",
        "commercial horseradish peroxidase", "commercial porcine pancreatic lipase",
        "amano lipase a", "fumzyme", "commercial fumd",
    ]

    # =========================================================================
    # Rule 5: Non-experimental markers
    # =========================================================================

    NON_EXPERIMENTAL_REVIEW_MARKERS = [
        "review", "meta-analysis", "recent advances", "comprehensive review",
        "literature review", "summarizes", "published studies",
    ]

    NON_EXPERIMENTAL_COMPUTATIONAL_MARKERS = [
        "molecular docking", "molecular dynamics", "dft calculation",
        "density functional", "in silico", "computational prediction",
    ]

    LITERATURE_COMPARISON_MARKERS = [
        "previous study", "prior study", "literature", "reported by",
        "reported in", "values from", "data from", "comparison with",
        "comparative study", "other researchers", "previously reported",
    ]

    # =========================================================================
    # Field Groups (8 groups for tier calculation)
    # =========================================================================

    FIELD_GROUPS = {
        "enzyme_identity": ["enzyme_name", "enzyme_full_name", "gene_name"],
        "substrate": ["substrate"],
        "quantitative_metric": ["Km_value", "kcat_value", "kcat_Km_value", "degradation_efficiency"],
        "metric_unit": ["Km_unit", "kcat_unit", "kcat_Km_unit", "degradation_efficiency_unit"],
        "reaction_condition": [
            "ph", "temperature_value",
            "kinetic_ph", "kinetic_temperature_value",
            "degradation_ph", "degradation_temperature_value",
            "buffer", "cofactor",
        ],
        "biological_source": ["organism", "strain", "gene_name"],
        "provenance": ["source_section", "evidence_type", "measurement_type"],
        "external_identifier": ["uniprot_id", "genbank_id", "pdb_id", "ec_number", "sequence"],
    }

    # =========================================================================
    # Rule 1: Mycotoxin substrate check
    # =========================================================================

    def _is_mycotoxin_substrate(self, substrate: str) -> bool:
        """Check if substrate is a known mycotoxin or derivative."""
        text = _clean(substrate).lower().strip()
        if not text:
            return False

        # Blacklist check first
        for non_mycotoxin in self.NON_MYCOTOXIN_SUBSTRATES:
            if non_mycotoxin in text:
                return False

        # Whitelist check (substring match)
        for mycotoxin in self.MYCOTOXIN_WHITELIST:
            if mycotoxin in text:
                return True

        # Compact alias check
        compact = re.sub(r"[^a-z0-9]+", "", text)
        if compact in self.MYCOTOXIN_COMPACT_ALIASES:
            return True

        # Pattern check
        for pattern in self.MYCOTOXIN_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True

        return False

    # =========================================================================
    # Rule 2: Enzymatic process check
    # =========================================================================

    def _is_nonenzymatic(self, record: Dict[str, Any]) -> bool:
        """Return True if the record describes a non-enzymatic process."""
        blob = " ".join(_clean(record.get(k)) for k in [
            "enzyme_name", "reported_enzyme_name", "canonical_enzyme_name",
            "enzyme_full_name", "enzyme_type", "enzyme_system_type", "notes",
            "evidence_text", "source_section", "table_caption",
        ]).lower()

        hard_nonenzyme_phrases = [
            "not a biological enzyme", "not biological enzyme", "not an enzyme",
            "non-enzymatic", "non enzymatic",
        ]
        if any(phrase in blob for phrase in hard_nonenzyme_phrases):
            return True

        # Hybrid markers override: if enzyme terms present, not non-enzymatic
        if any(h in blob for h in self.HYBRID_ENZYME_MARKERS):
            return False

        return any(m in blob for m in self.NON_ENZYMATIC_MARKERS)

    def _is_bioactivation_or_toxic_metabolism(self, record: Dict[str, Any]) -> bool:
        """Reject host metabolism / toxic bioactivation records from the enzyme database."""
        blob = " ".join(_clean(record.get(k)) for k in [
            "notes", "evidence_text", "products", "source_section",
            "enzyme_name", "reported_enzyme_name",
        ]).lower()
        if "afbo-gsh" in blob or "gst-mediated" in blob:
            return False
        return any(marker in blob for marker in self.BIOACTIVATION_MARKERS)

    def _has_direct_enzyme_record_evidence(self, record: Dict[str, Any]) -> bool:
        blob = " ".join(_clean(record.get(k)) for k in [
            "notes", "evidence_text", "source_section", "enzyme_system_type",
            "enzyme_state", "reported_enzyme_name", "enzyme_name",
        ]).lower()
        if any(marker in blob for marker in self.DIRECT_ENZYME_SCOPE_BLOCKERS):
            return False
        if any(marker in blob for marker in self.DIRECT_ENZYME_RECORD_MARKERS):
            return True
        system_type = _clean(record.get("enzyme_system_type")).lower()
        if system_type in {
            "purified_enzyme", "free_enzyme", "purified_recombinant_enzyme",
            "commercial_enzyme", "clearly_identified_commercial_enzyme",
        }:
            return True
        if _clean(record.get("mutations")) or re.search(r"\b[A-Z]\d+[A-Z]\b", _clean(record.get("reported_enzyme_name"))):
            if "purified" in blob or "recombinant" in blob or "kinetic parameters" in blob:
                return True
        return False

    def _is_unidentified_biological_material(self, record: Dict[str, Any]) -> bool:
        """Reject crude/whole-cell/powder systems unless a direct enzyme assay is evident."""
        blob = " ".join(_clean(record.get(k)) for k in [
            "notes", "evidence_text", "source_section", "enzyme_system_type",
            "enzyme_state", "reported_enzyme_name", "enzyme_name",
        ]).lower()
        if any(marker in blob for marker in self.DIRECT_ENZYME_SCOPE_BLOCKERS):
            return True
        if not any(marker in blob for marker in self.UNIDENTIFIED_BIOLOGICAL_SYSTEM_MARKERS):
            return False
        return not self._has_direct_enzyme_record_evidence(record)

    def _reference_column_indicates_prior_work(self, record: Dict[str, Any]) -> bool:
        """Detect table rows sourced from a Reference column rather than this paper."""
        current_terms = [
            "this study", "this work", "current study", "current work",
            "present study", "present work", "our study", "our work",
        ]
        for key, value in record.items():
            if "reference" not in str(key).lower():
                continue
            ref = _clean(value).lower()
            if not ref or ref in {"none", "null", "-", "n/a"}:
                continue
            if any(term in ref for term in current_terms):
                continue
            return True

        blob = " ".join(_clean(record.get(k)) for k in [
            "table_caption", "source_section", "evidence_text", "notes",
        ])
        lowered = blob.lower()
        has_reference_column = bool(
            re.search(r"<t[hd][^>]*>\s*reference\s*</t[hd]>", lowered)
            or re.search(r"\breference\s*(?:column|</td>|</th>|:)", lowered)
        )
        if not has_reference_column or any(term in lowered for term in current_terms):
            return False
        return bool(
            re.search(r"\b[A-Z][A-Za-z'_-]+\s+et\s+al\.?\s*(?:\(\d{4}\)|\d{4})?", blob)
            or re.search(r"\b[A-Z][A-Za-z'_-]+\s+and\s+[A-Z][A-Za-z'_-]+(?:\s+\(\d{4}\)|\s+\d{4})?", blob)
        )

    def _is_prior_work_record(self, record: Dict[str, Any]) -> bool:
        """Detect only source-local prior literature rows, not current data near a comparison table."""
        if self._reference_column_indicates_prior_work(record):
            return True
        blob = " ".join(_clean(record.get(k)) for k in [
            "table_caption", "source_section", "evidence_text", "notes",
        ]).lower()
        current_terms = [
            "this study", "this work", "current study", "current work",
            "present study", "present work", "our study", "our work",
        ]
        if any(term in blob for term in current_terms):
            return False
        patterns = [
            r"\bprevious(?:ly)?\s+(?:reported|published|study|studies)\b",
            r"\bprior\s+(?:reported|published|study|studies|work)\b",
            r"\breported\s+(?:by|in)\b",
            r"\b(?:values|data)\s+from\b",
            r"\bother\s+researchers\b",
            r"\bcomparison\s+(?:with|of)\s+(?:previous|prior|literature)\b",
            r"\bliterature\s+(?:comparison|values|data|row|rows|source)\b",
        ]
        return any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in patterns)

    # =========================================================================
    # Rule 3: Quantitative metric check
    # =========================================================================

    def _has_quantitative_metric(self, record: Dict[str, Any]) -> bool:
        """Check if record has at least one quantitative metric with a value."""
        mt = _clean(record.get("measurement_type")).lower()
        if mt == "kinetic":
            return any(_clean(record.get(k)) for k in ["Km_value", "kcat_value", "kcat_Km_value"])
        elif mt == "degradation":
            return bool(_clean(record.get("degradation_efficiency")))
        else:
            # No measurement_type set — check all metric fields
            return any(_clean(record.get(k)) for k in [
                "Km_value", "kcat_value", "kcat_Km_value", "degradation_efficiency",
            ])

    # =========================================================================
    # Rule 4: Enzyme identity check
    # =========================================================================

    def _has_enzyme_identity(self, record: Dict[str, Any]) -> bool:
        """Check if record has an identifiable enzyme entity."""
        # If inferred enzyme guard explicitly marked this as unidentified, fail
        if _clean(record.get("identified_enzyme")).lower() == "false":
            return False

        # Direct enzyme name fields
        name = _clean(record.get("reported_enzyme_name") or record.get("enzyme_name") or record.get("gene_name"))
        if name:
            lowered = name.lower()
            enzyme_class_terms = [
                "laccase", "lipase", "peroxidase", "oxidase", "hydrolase", "esterase",
                "reductase", "dehydrogenase", "transferase", "glucosyltransferase",
                "acetyltransferase", "fumd", "gsta", "gst", "zph", "oph",
            ]
            generic_hit = lowered in self.GENERIC_ENZYME_NAMES or any(g in lowered for g in self.GENERIC_ENZYME_NAMES)
            concrete_enzyme_name = any(term in lowered for term in enzyme_class_terms)
            # Reject generic system names only when they are not attached to a concrete enzyme class.
            if not generic_hit or concrete_enzyme_name:
                if not any(term in lowered for term in [
                    "preparation", "extract", "lysate", "supernatant", "whole cell", "whole-cell",
                ]):
                    return True

        # Explicit identifiers
        identifier_fields = [
            "gene_name", "uniprot_id", "genbank_id", "pdb_id", "ec_number", "sequence",
        ]
        if any(_clean(record.get(k)) for k in identifier_fields):
            return True

        # Commercial enzyme detection
        if name and any(term in name.lower() for term in enzyme_class_terms):
            return True

        # Broader fallback: organism + enzyme_system_type
        if _clean(record.get("organism")) and _clean(record.get("enzyme_system_type")):
            return not self._is_unidentified_biological_material(record)

        return False

    # =========================================================================
    # Rule 5: Original experiment check
    # =========================================================================

    def _is_original_experiment(self, record: Dict[str, Any]) -> bool:
        """Check if record is from the current paper's original experiment."""
        doc_type = _clean(record.get("Document_Type") or record.get("document_type")).lower()
        source_type = _clean(record.get("source_type")).lower()
        qc = _clean(record.get("QC_Status")).lower()
        source_section = _clean(record.get("source_section")).lower()

        # Review markers
        if doc_type == "review" or source_type == "review" or qc == "exclude_review_article":
            return False

        # Computational prediction markers
        combined = f"{doc_type} {source_type} {source_section}"
        if any(marker in combined for marker in self.NON_EXPERIMENTAL_COMPUTATIONAL_MARKERS):
            return False
        if self._is_prior_work_record(record):
            return False

        return True

    # =========================================================================
    # Hard Rules (all 5 must pass)
    # =========================================================================

    def check_hard_rules(self, record: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check all 5 hard rules.

        Returns:
            (passed, details) where details contains rule_results and failures list.
        """
        substrate_is_mycotoxin = self._is_mycotoxin_substrate(record.get("substrate", ""))
        substrate_compact = re.sub(r"[^a-z0-9]+", "", _clean(record.get("substrate", "")).lower())
        if substrate_compact == "afbo":
            substrate_is_mycotoxin = False

        results = {
            "rule_1_mycotoxin_substrate": substrate_is_mycotoxin,
            "rule_2_enzymatic_process": (
                not self._is_nonenzymatic(record)
                and not self._is_bioactivation_or_toxic_metabolism(record)
                and not self._is_unidentified_biological_material(record)
            ),
            "rule_3_quantitative_metric": self._has_quantitative_metric(record),
            "rule_4_enzyme_identity": self._has_enzyme_identity(record),
            "rule_5_original_experiment": self._is_original_experiment(record),
        }

        failures = [name for name, passed in results.items() if not passed]

        return len(failures) == 0, {
            "passed": len(failures) == 0,
            "failures": failures,
            "rule_results": results,
        }

    # =========================================================================
    # Field Group Counting
    # =========================================================================

    def count_field_groups(self, record: Dict[str, Any]) -> Tuple[int, List[str]]:
        """
        Count how many of the 8 field groups have at least one non-empty value.

        Special handling for metric_unit: only counts if at least one metric
        value AND its corresponding unit are both present.
        """
        present = []

        for group_name, fields in self.FIELD_GROUPS.items():
            if group_name == "metric_unit":
                # Special: unit only counts if corresponding value exists
                value_unit_pairs = [
                    ("Km_value", "Km_unit"),
                    ("kcat_value", "kcat_unit"),
                    ("kcat_Km_value", "kcat_Km_unit"),
                    ("degradation_efficiency", "degradation_efficiency_unit"),
                ]
                has_paired_unit = any(
                    _clean(record.get(v)) and _clean(record.get(u))
                    for v, u in value_unit_pairs
                )
                if has_paired_unit:
                    present.append(group_name)
            else:
                if any(_clean(record.get(f)) for f in fields):
                    present.append(group_name)

        return len(present), present

    # =========================================================================
    # Tier Calculation
    # =========================================================================

    def calculate_tier(self, record: Dict[str, Any]) -> str:
        """
        Calculate quality tier.

        Returns:
            "Gold" (>=7 groups), "Silver" (5-6), "Bronze" (3-4), or "Rejected"
        """
        passed, _ = self.check_hard_rules(record)
        if not passed:
            return "Rejected"

        group_count, _ = self.count_field_groups(record)

        if group_count >= 7:
            return "Gold"
        elif group_count >= 5:
            return "Silver"
        elif group_count >= 3:
            return "Bronze"
        else:
            return "Rejected"

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def classify_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a record in-place. Sets:
        - quality_tier: "Gold"/"Silver"/"Bronze"/"Rejected"
        - hard_rule_failures: semicolon-joined failure names (empty if pass)
        - field_group_count: int (0-8)
        - field_groups_present: semicolon-joined group names
        - eligibility_status: mapped from tier (backward compat)

        Returns the mutated record.
        """
        passed, details = self.check_hard_rules(record)
        group_count, groups_present = self.count_field_groups(record)
        tier = self.calculate_tier(record)

        record["quality_tier"] = tier
        record["hard_rule_failures"] = ";".join(details["failures"])
        record["field_group_count"] = group_count
        record["field_groups_present"] = ";".join(groups_present)

        if tier == "Rejected" and not _clean(record.get("QC_Status")):
            if self._is_bioactivation_or_toxic_metabolism(record):
                record["QC_Status"] = "metabolic_activation_or_toxic_biotransformation"
            elif self._is_unidentified_biological_material(record):
                record["QC_Status"] = "unidentified_biological_system_not_enzyme_record"
            elif not self._is_original_experiment(record):
                record["QC_Status"] = "literature_comparison_not_current_experiment"
            elif not self._has_enzyme_identity(record):
                record["QC_Status"] = "generic_or_unidentified_enzyme_name"

        # Backward compatibility mapping
        if tier == "Rejected":
            if not passed:
                record["eligibility_status"] = "rejected_" + details["failures"][0].replace("rule_", "").replace("_", " ", 1)
            else:
                record["eligibility_status"] = "rejected_incomplete_fields"
        else:
            record["eligibility_status"] = f"tier_{tier.lower()}"

        return record
