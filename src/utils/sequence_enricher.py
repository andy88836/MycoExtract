"""
Sequence and Substrate Enrichment Module

Automatically query UniProt/PubChem APIs to fill:
1. Enzyme sequence identifiers and amino acid sequences (UniProt)
2. Substrate SMILES structures (PubChem)
"""
import requests
import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class SequenceCandidate:
    """Candidate sequence from database query."""
    uniprot_id: str
    protein_name: str
    organism: str
    gene_name: Optional[str]
    sequence_length: int
    reviewed: bool  # Swiss-Prot (reviewed) vs TrEMBL (unreviewed)
    score: float = 0.0  # Matching score


class SequenceEnricher:
    """
    Enrich extracted records with sequence identifiers and sequences from UniProt,
    and substrate SMILES from PubChem.
    
    Strategy for enzyme sequences:
    1. If record already has uniprot_id/genbank_id → Skip ID lookup
    2. Query UniProt REST API with: enzyme_name + organism
    3. If single match → Auto-fill
    4. If multiple matches → Save candidates for manual review
    5. If no match → Log warning, leave null
    6. Fetch actual amino acid sequence for records with uniprot_id
    
    Strategy for substrate SMILES:
    1. If record already has substrate_smiles → Skip
    2. Query PubChem with substrate name
    3. Return canonical SMILES
    
    Fusion protein tag handling:
    - Automatically removes common purification/detection tags (His6, GST, GFP, etc.)
    - Searches UniProt with core enzyme name
    - Returns native sequence without tags
    """
    
    # Common organism abbreviations mapping
    ORGANISM_ABBREVIATIONS = {
        "b.": "Bacillus",
        "e.": "Escherichia",
        "s.": "Saccharomyces",
        "p.": "Pseudomonas",
        "a.": "Aspergillus",
        "t.": "Trametes",
        "c.": "Candida",
        "k.": "Kluyveromyces",
        "l.": "Lactobacillus",
        "m.": "Mycobacterium",
        "r.": "Rhizopus",
        "f.": "Fusarium",
        "n.": "Neurospora",
        "y.": "Yarrowia",
    }
    
    # Protein tag patterns (affinity tags, fusion tags, detection tags)
    TAG_PATTERNS = [
        # N-terminal His tags: His6-, His8-, His10-, 6xHis-, etc.
        (r'^(His\d+)-', 'N-terminal His tag'),
        (r'^(\d+xHis)-', 'N-terminal His tag'),
        (r'^(His)-', 'N-terminal His tag'),
        
        # N-terminal fusion/solubility tags
        (r'^(GST|MBP|SUMO|Trx|NusA|DsbA)-', 'N-terminal fusion tag'),
        (r'^(Thioredoxin|Maltose[_\s]?binding)-', 'N-terminal fusion tag'),
        
        # N-terminal affinity tags
        (r'^(FLAG|Strep|StrepII|StrepTag|SBP)-', 'N-terminal affinity tag'),
        
        # N-terminal fluorescent tags
        (r'^(GFP|EGFP|mCherry|RFP|YFP|CFP|BFP)-', 'N-terminal fluorescent tag'),
        
        # N-terminal epitope tags
        (r'^(HA|Myc|V5|T7|S)-', 'N-terminal epitope tag'),
        
        # C-terminal His tags: -His6, -His8, -6xHis, etc.
        (r'-(His\d+)$', 'C-terminal His tag'),
        (r'-(\d+xHis)$', 'C-terminal His tag'),
        (r'-(His)$', 'C-terminal His tag'),
        
        # C-terminal fluorescent tags
        (r'-(GFP|EGFP|mCherry|RFP|YFP|CFP)$', 'C-terminal fluorescent tag'),
        
        # C-terminal epitope tags
        (r'-(HA|Myc|V5|FLAG)$', 'C-terminal epitope tag'),
    ]
    
    def __init__(self, auto_fill_threshold: float = 0.9, fetch_sequences: bool = True, fetch_smiles: bool = True):
        """
        Args:
            auto_fill_threshold: Confidence threshold for auto-filling (0-1)
                0.9 = only fill if 90%+ confident (single Swiss-Prot match)
            fetch_sequences: If True, fetch actual amino acid sequences from UniProt
            fetch_smiles: If True, fetch substrate SMILES from PubChem
        """
        self.uniprot_api = "https://rest.uniprot.org/uniprotkb/search"
        self.pubchem_api = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.auto_fill_threshold = auto_fill_threshold
        self.fetch_sequences = fetch_sequences
        self.fetch_smiles = fetch_smiles
        self.request_delay = 0.2  # 200ms between requests (rate limiting)
        
        # Cache for SMILES lookups (substrate name -> SMILES)
        self._smiles_cache: Dict[str, Optional[str]] = {}
        # Cache for sequence lookups (uniprot_id -> sequence)
        self._sequence_cache: Dict[str, Optional[str]] = {}
        
    def enrich_records(
        self, 
        records: List[Dict[str, Any]], 
        auto_fill: bool = True,
        verbose: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Batch enrich records with sequence identifiers, sequences, and substrate SMILES.
        
        Args:
            records: List of extracted records
            auto_fill: If True, automatically fill single high-confidence matches
            verbose: If True, print detailed matching info
            
        Returns:
            (enriched_records, stats_dict)
        """
        stats = {
            "total": len(records),
            "already_has_id": 0,
            "auto_filled": 0,
            "multiple_candidates": 0,
            "no_match": 0,
            "errors": 0,
            # New stats for sequence and SMILES
            "sequences_fetched": 0,
            "sequences_failed": 0,
            "smiles_fetched": 0,
            "smiles_failed": 0,
            "smiles_already_has": 0,
        }
        
        enriched = []
        
        for i, record in enumerate(records):
            if verbose and (i % 10 == 0 or i == len(records) - 1):
                logger.info(f"Processing record {i+1}/{len(records)}...")
            
            # ===== Part 1: Enzyme UniProt ID Enrichment =====
            already_has_uniprot = bool(record.get("uniprot_id") or record.get("genbank_id"))
            
            if already_has_uniprot:
                stats["already_has_id"] += 1
            else:
                # Try to find UniProt ID
                enzyme_name = record.get("enzyme_name") or record.get("enzyme_full_name")
                organism = record.get("organism")
                gene_name = record.get("gene_name")
                
                if enzyme_name:
                    try:
                        candidates = self._query_uniprot(enzyme_name, organism, gene_name)
                        
                        if len(candidates) == 0:
                            stats["no_match"] += 1
                            if verbose:
                                logger.info(f"  No UniProt match: {enzyme_name} ({organism})")
                        
                        elif len(candidates) == 1:
                            candidate = candidates[0]
                            if auto_fill and candidate.score >= self.auto_fill_threshold:
                                record["uniprot_id"] = candidate.uniprot_id
                                # Always use UniProt protein name as full name (even if enzyme_name has tags)
                                # enzyme_name keeps original literature name (e.g., "His6-OPH")
                                # enzyme_full_name gets core enzyme name from UniProt (e.g., "Organophosphorus hydrolase")
                                record["enzyme_full_name"] = candidate.protein_name
                                record["gene_name"] = record.get("gene_name") or candidate.gene_name
                                
                                # Record tag removal info if applicable
                                if hasattr(candidate, 'metadata') and candidate.metadata.get('tags_removed'):
                                    if "_enrichment" not in record:
                                        record["_enrichment"] = {}
                                    record["_enrichment"]["tags_removed"] = [
                                        f"{t['tag']} ({t['position']})" for t in candidate.metadata['tags_removed']
                                    ]
                                    record["_enrichment"]["core_enzyme_searched"] = candidate.metadata.get('core_enzyme_searched')
                                
                                stats["auto_filled"] += 1
                                if verbose:
                                    logger.info(f"  ✅ Auto-filled: {enzyme_name} → {candidate.uniprot_id} ({candidate.protein_name})")
                        
                        else:
                            # 总是选择top-1候选，无需人工介入
                            top_candidate = candidates[0]
                            record["uniprot_id"] = top_candidate.uniprot_id
                            # Always use UniProt protein name as full name (even if enzyme_name has tags)
                            # enzyme_name keeps original literature name (e.g., "His6-OPH")
                            # enzyme_full_name gets core enzyme name from UniProt (e.g., "Organophosphorus hydrolase")
                            record["enzyme_full_name"] = top_candidate.protein_name
                            record["gene_name"] = record.get("gene_name") or top_candidate.gene_name
                            record["organism"] = top_candidate.organism  # 补充organism字段
                            
                            # 添加enrichment置信度信息
                            if "_enrichment" not in record:
                                record["_enrichment"] = {}
                            record["_enrichment"]["uniprot_match_score"] = round(top_candidate.score, 3)
                            record["_enrichment"]["uniprot_reviewed"] = top_candidate.reviewed
                            record["_enrichment"]["match_method"] = "auto_top1"
                            
                            # Record tag removal info if applicable
                            if hasattr(top_candidate, 'metadata') and top_candidate.metadata.get('tags_removed'):
                                record["_enrichment"]["tags_removed"] = [
                                    f"{t['tag']} ({t['position']})" for t in top_candidate.metadata['tags_removed']
                                ]
                                record["_enrichment"]["core_enzyme_searched"] = top_candidate.metadata.get('core_enzyme_searched')
                            
                            stats["auto_filled"] += 1
                            if verbose:
                                confidence_emoji = "🟢" if top_candidate.score >= 0.9 else "🟡" if top_candidate.score >= 0.7 else "🟠"
                                reviewed_mark = "✓" if top_candidate.reviewed else ""
                                logger.info(f"  {confidence_emoji} Auto-filled (top-1): {enzyme_name} → {top_candidate.uniprot_id} {reviewed_mark} (score={top_candidate.score:.3f})")
                        
                        time.sleep(self.request_delay)
                        
                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Error finding UniProt ID: {e}")
            
            # ===== Part 2: Fetch Amino Acid Sequence =====
            if self.fetch_sequences and record.get("uniprot_id") and not record.get("sequence"):
                uniprot_id = record["uniprot_id"]
                try:
                    sequence = self.get_sequence_from_uniprot(uniprot_id)
                    if sequence:
                        record["sequence"] = sequence
                        stats["sequences_fetched"] += 1
                        if verbose:
                            logger.info(f"  🧬 Fetched sequence: {uniprot_id} ({len(sequence)} aa)")
                    else:
                        stats["sequences_failed"] += 1
                except Exception as e:
                    stats["sequences_failed"] += 1
                    logger.error(f"Error fetching sequence for {uniprot_id}: {e}")
            
            # ===== Part 3: Fetch Substrate SMILES =====
            if self.fetch_smiles:
                substrate = record.get("substrate")
                if substrate and not record.get("substrate_smiles"):
                    try:
                        smiles = self.get_smiles_from_pubchem(substrate)
                        if smiles:
                            record["substrate_smiles"] = smiles
                            stats["smiles_fetched"] += 1
                            if verbose:
                                logger.info(f"  🔬 Fetched SMILES: {substrate} → {smiles[:50]}...")
                        else:
                            stats["smiles_failed"] += 1
                            if verbose:
                                logger.warning(f"  ⚠️ No SMILES found: {substrate}")
                    except Exception as e:
                        stats["smiles_failed"] += 1
                        logger.error(f"Error fetching SMILES for {substrate}: {e}")
                elif record.get("substrate_smiles"):
                    stats["smiles_already_has"] += 1
            
            enriched.append(record)
        
        return enriched, stats
    
    def _remove_protein_tags(self, enzyme_name: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Remove common fusion protein tags from enzyme name.
        
        This handles purification tags (His6, GST), detection tags (GFP), and 
        epitope tags (FLAG, HA) that are added for experimental purposes but
        are not part of the native enzyme sequence in UniProt.
        
        Args:
            enzyme_name: Original enzyme name (e.g., "His6-OPH", "GST-Laccase")
        
        Returns:
            (core_enzyme_name, removed_tags_list)
            
            removed_tags_list format:
            [
                {"tag": "His6", "position": "N-terminal", "type": "His tag"},
                {"tag": "GFP", "position": "C-terminal", "type": "fluorescent tag"}
            ]
        
        Examples:
            "His6-OPH" → ("OPH", [{"tag": "His6", ...}])
            "GST-Laccase" → ("Laccase", [{"tag": "GST", ...}])
            "GFP-His6-CotA" → ("CotA", [{"tag": "GFP", ...}, {"tag": "His6", ...}])
            "Laccase" → ("Laccase", [])  # No tags
        """
        clean_name = enzyme_name
        removed_tags = []
        
        # Iteratively apply tag patterns until no more tags are found
        # This handles cases like "GFP-His6-CotA" where multiple tags exist
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            found_tag = False
            
            for pattern, tag_type in self.TAG_PATTERNS:
                match = re.search(pattern, clean_name, re.IGNORECASE)
                if match:
                    tag = match.group(1)
                    position = "N-terminal" if pattern.startswith('^') else "C-terminal"
                    
                    removed_tags.append({
                        "tag": tag,
                        "position": position,
                        "type": tag_type
                    })
                    
                    # Remove the matched tag from the name
                    clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
                    clean_name = clean_name.strip('-').strip()
                    
                    found_tag = True
                    break  # Start from beginning after each removal
            
            if not found_tag:
                break  # No more tags found
        
        return clean_name, removed_tags
    
    def _expand_organism_name(self, organism: str) -> str:
        """
        Expand abbreviated organism names to full genus names.
        
        Examples:
            "B. subtilis" → "Bacillus subtilis"
            "E. coli" → "Escherichia coli"
            "Bacillus sp." → "Bacillus"
        """
        if not organism:
            return organism
        
        org_clean = organism.strip()
        
        # Handle "sp." suffix (unspecified species)
        if org_clean.endswith(" sp.") or org_clean.endswith(" sp"):
            org_clean = org_clean.replace(" sp.", "").replace(" sp", "").strip()
        
        # Check if starts with an abbreviation like "B."
        parts = org_clean.split()
        if len(parts) >= 1:
            first_part = parts[0].lower()
            
            # Check against known abbreviations
            for abbrev, full_name in self.ORGANISM_ABBREVIATIONS.items():
                if first_part == abbrev or first_part == abbrev.rstrip('.'):
                    # Replace abbreviation with full genus name
                    parts[0] = full_name
                    return " ".join(parts)
        
        return org_clean
    
    def _query_uniprot(
        self, 
        enzyme_name: str, 
        organism: Optional[str] = None,
        gene_name: Optional[str] = None
    ) -> List[SequenceCandidate]:
        """
        Query UniProt REST API for enzyme sequence with automatic tag removal.
        
        Strategy:
        1. Try original enzyme name first
        2. If no results, automatically remove common tags (His6, GST, GFP, etc.)
        3. Retry with core enzyme name
        4. If successful, annotate candidates with tag removal info
        
        Args:
            enzyme_name: Enzyme name (can include tags like "His6-OPH")
            organism: Source organism (e.g., "Bacillus subtilis")
            gene_name: Gene symbol (e.g., "cotA")
            
        Returns:
            List of SequenceCandidate objects, sorted by confidence
        """
        # Expand organism abbreviations
        organism_expanded = self._expand_organism_name(organism) if organism else None
        
        # Try with original name first
        candidates = self._try_uniprot_strategies(enzyme_name, organism_expanded, gene_name)
        
        if candidates:
            logger.debug(f"  ✅ Found {len(candidates)} candidates with original name: {enzyme_name}")
            return candidates
        
        # If no results, try removing protein tags
        clean_enzyme_name, removed_tags = self._remove_protein_tags(enzyme_name)
        
        if removed_tags and clean_enzyme_name != enzyme_name:
            tag_names = [t['tag'] for t in removed_tags]
            logger.info(f"  🏷️ No match for '{enzyme_name}', retrying without tags: '{clean_enzyme_name}'")
            logger.info(f"     Removed tags: {', '.join(tag_names)}")
            
            # Retry with cleaned name
            candidates = self._try_uniprot_strategies(clean_enzyme_name, organism_expanded, gene_name)
            
            if candidates:
                # Annotate all candidates with tag removal info
                for cand in candidates:
                    if not hasattr(cand, 'metadata'):
                        cand.metadata = {}
                    cand.metadata['tags_removed'] = removed_tags
                    cand.metadata['original_enzyme_name'] = enzyme_name
                    cand.metadata['core_enzyme_searched'] = clean_enzyme_name
                
                logger.info(f"  ✅ Found {len(candidates)} candidates after removing tags")
        
        return candidates
    
    def _try_uniprot_strategies(
        self,
        enzyme_name: str,
        organism_expanded: Optional[str],
        gene_name: Optional[str]
    ) -> List[SequenceCandidate]:
        """
        Try multiple UniProt query strategies with a given enzyme name.
        
        Returns:
            List of candidates (empty if no matches found)
        """
        candidates = []
        
        # Strategy 1: Use gene name if available (most specific)
        if gene_name:
            candidates = self._do_uniprot_query(
                f"gene:{gene_name}",
                organism_expanded,
                enzyme_name, organism_expanded, gene_name
            )
        
        # Strategy 2: If no results, try protein name with enzyme_name
        if not candidates:
            candidates = self._do_uniprot_query(
                f"protein_name:{enzyme_name}",
                organism_expanded,
                enzyme_name, organism_expanded, gene_name
            )
        
        # Strategy 3: If still no results and enzyme_name looks like a gene (short, alphanumeric)
        if not candidates and enzyme_name and len(enzyme_name) <= 10 and enzyme_name.replace('-', '').isalnum():
            candidates = self._do_uniprot_query(
                f"gene:{enzyme_name}",
                organism_expanded,
                enzyme_name, organism_expanded, gene_name
            )
        
        # Strategy 4: Broaden search - try without strict organism filter
        if not candidates and organism_expanded:
            # Extract just the genus
            genus = organism_expanded.split()[0] if organism_expanded else None
            if genus:
                candidates = self._do_uniprot_query(
                    f"protein_name:{enzyme_name}",
                    genus,
                    enzyme_name, organism_expanded, gene_name
                )
        
        return candidates
    
    def _do_uniprot_query(
        self,
        name_query: str,
        organism_filter: Optional[str],
        orig_enzyme: str,
        orig_organism: Optional[str],
        orig_gene: Optional[str]
    ) -> List[SequenceCandidate]:
        """Execute a single UniProt query and parse results."""
        query_parts = [name_query]
        
        if organism_filter:
            org_clean = organism_filter.replace(".", "").strip()
            query_parts.append(f"organism_name:{org_clean}")
        
        query = " AND ".join(query_parts)
        
        params = {
            "query": query,
            "format": "json",
            "size": 10,
            "fields": "accession,protein_name,organism_name,gene_names,sequence,reviewed,length"
        }
        
        try:
            response = requests.get(self.uniprot_api, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            candidates = []
            for entry in data.get("results", []):
                uniprot_id = entry.get("primaryAccession", "")
                
                protein_desc = entry.get("proteinDescription", {})
                recommended = protein_desc.get("recommendedName", {})
                protein_name = recommended.get("fullName", {}).get("value", "Unknown")
                
                organism_info = entry.get("organism", {})
                organism_name = organism_info.get("scientificName", "Unknown")
                
                genes = entry.get("genes", [])
                gene_primary = genes[0].get("geneName", {}).get("value") if genes else None
                
                seq_info = entry.get("sequence", {})
                seq_length = seq_info.get("length", 0)
                
                reviewed = entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"
                
                score = self._calculate_match_score(
                    orig_enzyme, orig_organism, orig_gene,
                    protein_name, organism_name, gene_primary, reviewed
                )
                
                candidates.append(SequenceCandidate(
                    uniprot_id=uniprot_id,
                    protein_name=protein_name,
                    organism=organism_name,
                    gene_name=gene_primary,
                    sequence_length=seq_length,
                    reviewed=reviewed,
                    score=score
                ))
            
            candidates.sort(key=lambda c: c.score, reverse=True)
            logger.debug(f"UniProt query: '{query}' → {len(candidates)} results")
            return candidates
            
        except Exception as e:
            logger.debug(f"UniProt query failed: '{query}': {e}")
            return []
    
    def _calculate_match_score(
        self,
        query_enzyme: str,
        query_organism: Optional[str],
        query_gene: Optional[str],
        result_protein: str,
        result_organism: str,
        result_gene: Optional[str],
        reviewed: bool
    ) -> float:
        """
        Calculate matching confidence score (0-1).
        
        Scoring factors:
        - Exact gene name match: +0.5 (highest priority)
        - Reviewed (Swiss-Prot) status: +0.2
        - Exact organism match: +0.3
        - Fuzzy organism match: +0.15
        - Protein name contains enzyme name: +0.1
        """
        score = 0.0
        
        # Factor 1: Gene name match (HIGHEST PRIORITY)
        if query_gene and result_gene:
            if query_gene.lower() == result_gene.lower():
                score += 0.5  # Strong signal for correct enzyme
        
        # Factor 2: Organism match
        if query_organism and result_organism:
            query_org_lower = query_organism.lower().replace(".", "").strip()
            result_org_lower = result_organism.lower().replace(".", "").strip()
            
            # Extract genus and species
            query_parts = query_org_lower.split()
            result_parts = result_org_lower.split()
            
            if query_org_lower == result_org_lower:
                score += 0.3  # Exact match
            elif len(query_parts) >= 2 and len(result_parts) >= 2:
                # Check genus + species match
                if query_parts[0] == result_parts[0] and query_parts[1] == result_parts[1]:
                    score += 0.25  # Species match (ignore strain)
                elif query_parts[0] == result_parts[0]:
                    score += 0.15  # Genus match only
            elif query_org_lower in result_org_lower or result_org_lower in query_org_lower:
                score += 0.15  # Partial match
        
        # Factor 3: Reviewed status (Swiss-Prot is curated)
        if reviewed:
            score += 0.2
        
        # Factor 4: Protein name similarity
        query_enzyme_lower = query_enzyme.lower()
        result_protein_lower = result_protein.lower()
        
        # Check if enzyme name appears in protein description
        if query_enzyme_lower in result_protein_lower:
            score += 0.1
        elif len(query_enzyme_lower) > 3:  # Avoid matching short abbreviations
            # Check word-level match
            query_words = set(query_enzyme_lower.split())
            result_words = set(result_protein_lower.split())
            overlap = query_words & result_words
            if overlap:
                score += 0.05 * min(1.0, len(overlap) / len(query_words))
        
        return min(1.0, score)
    
    def get_sequence_from_uniprot(self, uniprot_id: str) -> Optional[str]:
        """
        Fetch amino acid sequence from UniProt by accession.
        
        Args:
            uniprot_id: UniProt accession (e.g., "Q8X1Z5")
            
        Returns:
            Amino acid sequence string, or None if error
        """
        # Check cache first
        if uniprot_id in self._sequence_cache:
            return self._sequence_cache[uniprot_id]
        
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse FASTA format
            lines = response.text.strip().split("\n")
            sequence = "".join(lines[1:])  # Skip header line
            
            # Cache the result
            self._sequence_cache[uniprot_id] = sequence
            time.sleep(self.request_delay)
            
            return sequence
            
        except Exception as e:
            logger.error(f"Failed to fetch sequence for {uniprot_id}: {e}")
            self._sequence_cache[uniprot_id] = None
            return None
    
    def get_smiles_from_pubchem(self, compound_name: str) -> Optional[str]:
        """
        Fetch canonical SMILES from PubChem by compound name.
        
        Args:
            compound_name: Name of the compound (e.g., "AFB1", "Aflatoxin B1", "ABTS")
            
        Returns:
            Canonical SMILES string, or None if not found
        """
        # Normalize compound name for cache lookup
        cache_key = compound_name.lower().strip()
        
        # Check cache first
        if cache_key in self._smiles_cache:
            return self._smiles_cache[cache_key]
        
        # Comprehensive compound name mappings (abbreviations to PubChem-recognized names)
        # This serves as a fallback if LLM extraction didn't use standard names
        compound_aliases = {
            # Aflatoxins
            "afb1": "Aflatoxin B1", "afb₁": "Aflatoxin B1",
            "afb2": "Aflatoxin B2", "afb₂": "Aflatoxin B2",
            "afg1": "Aflatoxin G1", "afg₁": "Aflatoxin G1",
            "afg2": "Aflatoxin G2", "afg₂": "Aflatoxin G2",
            "afm1": "Aflatoxin M1", "afm₁": "Aflatoxin M1",
            "afm2": "Aflatoxin M2",
            "afp1": "Aflatoxin P1",
            # Ochratoxins
            "ota": "Ochratoxin A",
            "otb": "Ochratoxin B",
            # Trichothecenes
            "don": "Deoxynivalenol",
            "3-adon": "3-Acetyldeoxynivalenol",
            "15-adon": "15-Acetyldeoxynivalenol",
            "niv": "Nivalenol",
            "t-2": "T-2 toxin", "t2": "T-2 toxin",
            "ht-2": "HT-2 toxin", "ht2": "HT-2 toxin",
            "das": "Diacetoxyscirpenol",
            # Zearalenones
            "zea": "Zearalenone", "zen": "Zearalenone",
            "α-zol": "alpha-Zearalenol", "alpha-zol": "alpha-Zearalenol",
            "β-zol": "beta-Zearalenol", "beta-zol": "beta-Zearalenol",
            "α-zel": "alpha-Zearalenol", "β-zel": "beta-Zearalenol",
            # Fumonisins
            "fb1": "Fumonisin B1", "fb₁": "Fumonisin B1",
            "fb2": "Fumonisin B2", "fb₂": "Fumonisin B2",
            "fb3": "Fumonisin B3",
            "hfb1": "Hydrolyzed fumonisin B1",
            # Other mycotoxins
            "pat": "Patulin",
            "cit": "Citrinin",
            "ste": "Sterigmatocystin", "st": "Sterigmatocystin",
            "cpa": "Cyclopiazonic acid",
            "roq-c": "Roquefortine C",
            "mpa": "Mycophenolic acid",
            "aoh": "Alternariol",
            "ame": "Alternariol monomethyl ether",
            "tea": "Tenuazonic acid",
            "mon": "Moniliformin",
            "bea": "Beauvericin",
            "enna": "Enniatin A", "ennb": "Enniatin B",
            # Common enzyme substrates
            "abts": "2,2'-azino-bis(3-ethylbenzothiazoline-6-sulfonic acid)",
            "sgz": "Syringaldazine",
            "dmp": "2,6-Dimethoxyphenol",
            "guaiacol": "Guaiacol",
            "catechol": "Catechol",
            "hydroquinone": "Hydroquinone",
            "veratryl alcohol": "Veratryl alcohol",
            "rbbr": "Remazol Brilliant Blue R",
            "rb5": "Reactive Black 5",
            "p-np": "4-Nitrophenol", "pnp": "4-Nitrophenol",
            "p-npp": "4-Nitrophenyl phosphate",
        }
        
        # Try to resolve alias
        query_name = compound_aliases.get(cache_key, compound_name)
        
        # Try multiple search strategies
        smiles = None
        
        # Strategy 1: Direct name search
        smiles = self._pubchem_name_search(query_name)
        
        # Strategy 2: Try original name if alias didn't work
        if not smiles and query_name != compound_name:
            smiles = self._pubchem_name_search(compound_name)
        
        # Strategy 3: Try partial name matching for complex names
        if not smiles and " " in compound_name:
            # Try first word (often the compound class)
            first_word = compound_name.split()[0]
            if len(first_word) > 3:
                smiles = self._pubchem_name_search(first_word)
        
        # Cache the result (even if None, to avoid repeated failed queries)
        self._smiles_cache[cache_key] = smiles
        
        return smiles
    
    def _pubchem_name_search(self, compound_name: str) -> Optional[str]:
        """
        Search PubChem by compound name and return canonical SMILES.
        """
        try:
            # URL encode the compound name
            encoded_name = quote(compound_name)
            
            # First, get the CID (Compound ID) from name
            cid_url = f"{self.pubchem_api}/compound/name/{encoded_name}/cids/JSON"
            
            response = requests.get(cid_url, timeout=10)
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            cids = data.get("IdentifierList", {}).get("CID", [])
            if not cids:
                return None
            
            # Get the first (most relevant) CID
            cid = cids[0]
            
            # Fetch SMILES for this CID
            smiles_url = f"{self.pubchem_api}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            
            time.sleep(self.request_delay)
            
            smiles_response = requests.get(smiles_url, timeout=10)
            smiles_response.raise_for_status()
            smiles_data = smiles_response.json()
            
            properties = smiles_data.get("PropertyTable", {}).get("Properties", [])
            if properties:
                # PubChem may return CanonicalSMILES or ConnectivitySMILES
                smiles = properties[0].get("CanonicalSMILES") or properties[0].get("ConnectivitySMILES")
                if smiles:
                    logger.debug(f"PubChem: {compound_name} → CID:{cid} → {smiles[:50]}...")
                    return smiles
            
            return None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"PubChem: {compound_name} not found")
            else:
                logger.warning(f"PubChem HTTP error for {compound_name}: {e}")
            return None
        except Exception as e:
            logger.warning(f"PubChem error for {compound_name}: {e}")
            return None
    
    def generate_enrichment_report(self, stats: Dict[str, Any]) -> str:
        """Generate human-readable enrichment report."""
        total = stats["total"]
        already = stats.get("already_has_id", 0)
        filled = stats.get("auto_filled", 0)
        multiple = stats.get("multiple_candidates", 0)
        no_match = stats.get("no_match", 0)
        errors = stats.get("errors", 0)
        
        # New stats
        seq_fetched = stats.get("sequences_fetched", 0)
        seq_failed = stats.get("sequences_failed", 0)
        smiles_fetched = stats.get("smiles_fetched", 0)
        smiles_failed = stats.get("smiles_failed", 0)
        smiles_already = stats.get("smiles_already_has", 0)
        
        coverage = ((already + filled) / total * 100) if total > 0 else 0
        seq_coverage = (seq_fetched / total * 100) if total > 0 else 0
        smiles_coverage = ((smiles_fetched + smiles_already) / total * 100) if total > 0 else 0
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║        Sequence & Substrate Enrichment Report                ║
╚══════════════════════════════════════════════════════════════╝

📊 UniProt ID Enrichment:
────────────────────────────────────────────────────────────────
  Total Records:              {total}
  Already Has ID:             {already} ({already/total*100:.1f}%)
  Auto-Filled (Top-1):        {filled} ({filled/total*100:.1f}%)
  No Match Found:             {no_match} ({no_match/total*100:.1f}%)
  Errors:                     {errors}
  
  → ID Coverage:              {already + filled}/{total} ({coverage:.1f}%)

🧬 Amino Acid Sequence Fetching:
────────────────────────────────────────────────────────────────
  Sequences Fetched:          {seq_fetched}
  Fetch Failed:               {seq_failed}
  
  → Sequence Coverage:        {seq_fetched}/{total} ({seq_coverage:.1f}%)

🔬 Substrate SMILES Enrichment:
────────────────────────────────────────────────────────────────
  SMILES Fetched:             {smiles_fetched}
  Already Has SMILES:         {smiles_already}
  Not Found in PubChem:       {smiles_failed}
  
  → SMILES Coverage:          {smiles_fetched + smiles_already}/{total} ({smiles_coverage:.1f}%)

═══════════════════════════════════════════════════════════════

💡 Recommendations:
"""
        
        # 不再有multiple candidates的推荐
        
        if no_match > 0:
            report += f"  • {no_match} records had no UniProt match\n"
            report += f"    → Check if enzyme names are standard nomenclature\n"
        
        if seq_failed > 0:
            report += f"  • {seq_failed} sequence fetches failed\n"
            report += f"    → Verify UniProt IDs are valid\n"
        
        if smiles_failed > 0:
            report += f"  • {smiles_failed} substrates not found in PubChem\n"
            report += f"    → Consider adding manual SMILES or using ChEMBL\n"
        
        if coverage < 50:
            report += f"  • Low ID coverage ({coverage:.1f}%)\n"
            report += f"    → Improve enzyme_full_name extraction in LLM prompts\n"
        
        return report


# ========== Convenience Functions ==========

def enrich_json_file(
    input_json_path: str, 
    output_json_path: str,
    auto_fill: bool = True,
    fetch_sequences: bool = True,
    fetch_smiles: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Enrich a JSON file with sequence identifiers, sequences, and substrate SMILES.
    
    Args:
        input_json_path: Path to input JSON (extracted records)
        output_json_path: Path to save enriched JSON
        auto_fill: Auto-fill high-confidence matches
        fetch_sequences: Fetch amino acid sequences from UniProt
        fetch_smiles: Fetch substrate SMILES from PubChem
        verbose: Print progress
        
    Returns:
        Enrichment statistics
    """
    import json
    
    # Load records
    with open(input_json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    logger.info(f"Loaded {len(records)} records from {input_json_path}")
    
    # Enrich
    enricher = SequenceEnricher(
        auto_fill_threshold=0.9,
        fetch_sequences=fetch_sequences,
        fetch_smiles=fetch_smiles
    )
    enriched_records, stats = enricher.enrich_records(records, auto_fill=auto_fill, verbose=verbose)
    
    # Save
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_records, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(enriched_records)} enriched records to {output_json_path}")
    
    # Print report
    report = enricher.generate_enrichment_report(stats)
    print(report)
    
    return stats


if __name__ == "__main__":
    # Example usage
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Enrich enzyme records with sequences and SMILES")
    parser.add_argument("--input", "-i", help="Input JSON file path")
    parser.add_argument("--output", "-o", help="Output JSON file path (default: input with _enriched suffix)")
    parser.add_argument("--no-sequences", action="store_true", help="Skip fetching amino acid sequences")
    parser.add_argument("--no-smiles", action="store_true", help="Skip fetching substrate SMILES")
    parser.add_argument("--test", action="store_true", help="Run API tests")
    
    args = parser.parse_args()
    
    if args.test or not args.input:
        # Run API tests
        enricher = SequenceEnricher()
        
        print("\n" + "="*70)
        print("🧬 Testing UniProt API Queries")
        print("="*70)
        
        test_enzymes = [
            ("CotA", "Bacillus subtilis", "cotA"),
            ("laccase", "Trametes versicolor", None),
        ]
        
        for enzyme, organism, gene in test_enzymes:
            print(f"\nQuery: {enzyme} from {organism} (gene: {gene})")
            candidates = enricher._query_uniprot(enzyme, organism, gene)
            
            if candidates:
                print(f"  Found {len(candidates)} candidates:")
                for i, c in enumerate(candidates[:3], 1):
                    print(f"    {i}. {c.uniprot_id} | {c.protein_name} | {c.organism}")
                    print(f"       Gene: {c.gene_name} | Reviewed: {c.reviewed} | Score: {c.score:.2f}")
                
                # Test sequence fetching
                if candidates[0].score >= 0.5:
                    print(f"\n  Fetching sequence for {candidates[0].uniprot_id}...")
                    seq = enricher.get_sequence_from_uniprot(candidates[0].uniprot_id)
                    if seq:
                        print(f"  ✅ Got sequence: {len(seq)} amino acids")
                        print(f"     First 60 aa: {seq[:60]}...")
            else:
                print("  ❌ No matches found")
            
            time.sleep(0.3)
        
        print("\n" + "="*70)
        print("🔬 Testing PubChem SMILES Queries")
        print("="*70)
        
        test_substrates = [
            "AFB1",
            "Aflatoxin B1",
            "ABTS",
            "Guaiacol",
            "Zearalenone",
        ]
        
        for substrate in test_substrates:
            print(f"\nQuery: {substrate}")
            smiles = enricher.get_smiles_from_pubchem(substrate)
            if smiles:
                print(f"  ✅ SMILES: {smiles[:60]}{'...' if len(smiles) > 60 else ''}")
            else:
                print(f"  ❌ Not found")
            time.sleep(0.3)
        
        print("\n" + "="*70)
        print("✅ API Tests Complete")
        print("="*70)
    
    else:
        # Process input file
        input_path = args.input
        output_path = args.output or input_path.replace(".json", "_enriched.json")
        
        enrich_json_file(
            input_path,
            output_path,
            fetch_sequences=not args.no_sequences,
            fetch_smiles=not args.no_smiles,
            verbose=True
        )
