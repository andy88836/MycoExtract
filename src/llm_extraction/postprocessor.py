"""
Data Postprocessor

Flattens nested JSON knowledge fragments into a tabular CSV format
for easier analysis and database import.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class DataPostprocessor:
    """
    Postprocessor to flatten JSON fragments and export to CSV.
    """
    
    def __init__(self):
        """Initialize the postprocessor."""
        logger.info("Initialized DataPostprocessor")
    
    def flatten_and_save(self, fragments: List[Dict[str, Any]], output_csv: str) -> pd.DataFrame:
        """
        Flatten nested JSON fragments and save to CSV.
        
        Args:
            fragments: List of JSON knowledge fragments
            output_csv: Path to output CSV file
            
        Returns:
            DataFrame containing flattened data
        """
        if not fragments:
            logger.warning("No fragments to process")
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=self._get_column_names())
            df.to_csv(output_csv, index=False)
            return df
        
        logger.info(f"Flattening {len(fragments)} fragments to CSV")
        
        # Flatten each fragment
        flattened_records = []
        for fragment in fragments:
            flat_record = self._flatten_fragment(fragment)
            flattened_records.append(flat_record)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_records)
        
        # Reorder columns for better readability
        df = self._reorder_columns(df)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(df)} records to: {output_csv}")
        
        return df
    
    def _flatten_fragment(self, fragment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten a single JSON fragment (v4.2 schema).
        
        The v4.2 schema is already flat, so this mainly handles:
        1. Renaming fields to match CSV column names
        2. Converting products array to string
        3. Extracting source metadata
        
        Args:
            fragment: JSON object (v4.2 schema - already flat)
            
        Returns:
            Dictionary with CSV-friendly column names
        """
        flat = {}
        
        # v4.2 schema fields (already flat!)
        # Enzyme information
        flat["enzyme_name"] = fragment.get("enzyme_name")
        flat["enzyme_ec_number"] = fragment.get("ec_number")
        flat["source_organism_name"] = fragment.get("organism")
        flat["source_organism_strain"] = fragment.get("strain")
        flat["is_recombinant"] = fragment.get("is_recombinant")
        flat["is_wild_type"] = fragment.get("is_wild_type")  # NEW in v4.2
        flat["mutations"] = fragment.get("mutations")
        
        # Legacy fields (set to None for v4.2)
        flat["preparation_level"] = None
        flat["uniprot_id"] = None
        flat["pdb_id"] = None
        flat["ncbi_protein_id"] = None
        
        # Substrate information
        flat["substrate_name"] = fragment.get("substrate")
        flat["substrate_initial_concentration_value"] = None  # Parse from substrate_concentration if needed
        flat["substrate_initial_concentration_unit"] = None
        
        # Parse substrate_concentration if it's a string (e.g., "10 μM")
        substrate_conc = fragment.get("substrate_concentration")
        if substrate_conc and isinstance(substrate_conc, str):
            # Try to split value and unit (e.g., "10 μM" -> value=10, unit="μM")
            parts = substrate_conc.strip().split(maxsplit=1)
            if len(parts) == 2:
                try:
                    flat["substrate_initial_concentration_value"] = float(parts[0])
                    flat["substrate_initial_concentration_unit"] = parts[1]
                except ValueError:
                    pass  # Keep as None if not parseable
        
        flat["substrate_matrix"] = None
        
        # Kinetic parameters (v4.2 uses separate value/unit fields)
        flat["km_value"] = fragment.get("Km_value")
        flat["km_unit"] = fragment.get("Km_unit")
        flat["vmax_value"] = fragment.get("Vmax_value")
        flat["vmax_unit"] = fragment.get("Vmax_unit")
        flat["kcat_value"] = fragment.get("kcat_value")
        flat["kcat_unit"] = fragment.get("kcat_unit")
        flat["kcat_km_value"] = fragment.get("kcat_Km_value")
        flat["kcat_km_unit"] = fragment.get("kcat_Km_unit")
        
        # Degradation assay
        flat["degradation_efficiency"] = fragment.get("degradation_efficiency")
        flat["reaction_time_value"] = fragment.get("reaction_time_value")
        flat["reaction_time_unit"] = fragment.get("reaction_time_unit")
        flat["detection_method"] = None
        
        # Reaction conditions
        flat["temperature_value"] = fragment.get("temperature_value")
        flat["temperature_unit"] = fragment.get("temperature_unit")
        flat["ph_value"] = fragment.get("ph")
        flat["buffer_system"] = None
        flat["enzyme_concentration_value"] = None
        flat["enzyme_concentration_unit"] = None
        flat["cofactors"] = None
        
        # Reaction products (convert array to JSON string)
        products = fragment.get("products", [])
        if products and isinstance(products, list):
            flat["reaction_products"] = json.dumps(products, ensure_ascii=False)
        else:
            flat["reaction_products"] = None
        
        # Optimal conditions
        flat["optimal_ph"] = fragment.get("optimal_ph")
        flat["optimal_temperature_value"] = fragment.get("optimal_temperature_value")
        flat["optimal_temperature_unit"] = fragment.get("optimal_temperature_unit")
        
        # Notes
        flat["notes"] = fragment.get("notes")
        
        # Source metadata
        source = fragment.get("source_in_document", {})
        flat["doi"] = source.get("doi", "unknown")
        flat["source_type"] = source.get("source_type", "unknown")
        
        return flat
    
    def _get_column_names(self) -> List[str]:
        """
        Get the standard column names for the flattened CSV (v4.2 schema).
        
        Returns:
            List of column names
        """
        return [
            # Enzyme info
            "enzyme_name", "enzyme_ec_number",
            "source_organism_name", "source_organism_strain",
            "is_recombinant", "is_wild_type", "preparation_level",
            "uniprot_id", "pdb_id", "ncbi_protein_id", "mutations",
            
            # Substrate info
            "substrate_name",
            "substrate_initial_concentration_value", "substrate_initial_concentration_unit",
            "substrate_matrix",
            
            # Kinetics
            "km_value", "km_unit",
            "vmax_value", "vmax_unit",
            "kcat_value", "kcat_unit",
            "kcat_km_value", "kcat_km_unit",
            
            # Degradation
            "degradation_efficiency",
            "reaction_time_value", "reaction_time_unit",
            "detection_method",
            
            # Conditions
            "temperature_value", "temperature_unit",
            "ph_value", "buffer_system",
            "enzyme_concentration_value", "enzyme_concentration_unit",
            "cofactors",
            
            # Products
            "reaction_products",
            
            # Additional
            "optimal_ph",
            "optimal_temperature_value", "optimal_temperature_unit",
            "notes",
            
            # Source
            "doi", "source_type"
        ]
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder DataFrame columns for better readability.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with reordered columns
        """
        standard_cols = self._get_column_names()
        
        # Get columns that exist in both
        existing_cols = [col for col in standard_cols if col in df.columns]
        
        # Add any extra columns not in standard list
        extra_cols = [col for col in df.columns if col not in standard_cols]
        
        return df[existing_cols + extra_cols]


if __name__ == "__main__":
    """
    Test the postprocessor.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    test_fragments = [
        {
            "enzyme_info": {
                "name": "Aflatoxin B1 aldehyde reductase",
                "ec_number": "1.1.1.2",
                "source_organism": {"name": "Bacillus subtilis", "strain": "ATCC 6633"},
                "is_recombinant": True,
                "preparation_level": "purified",
                "sequence_identifiers": {"uniprot_id": "P12345"},
                "mutations": None
            },
            "substrate_info": {
                "name": "Aflatoxin B1",
                "initial_concentration": {"value": 10, "unit": "μM"},
                "matrix": "phosphate buffer"
            },
            "kinetic_assay": {
                "km_value": 15.5,
                "km_unit": "μM",
                "vmax_value": 45.2,
                "vmax_unit": "nmol/min/mg",
                "kcat_value": None,
                "kcat_unit": None,
                "kcat_km_value": None,
                "kcat_km_unit": None
            },
            "degradation_assay": {
                "degradation_efficiency": 85.5,
                "reaction_time": {"value": 60, "unit": "min"},
                "detection_method": "HPLC"
            },
            "reaction_conditions": {
                "temperature": {"value": 37, "unit": "°C"},
                "ph_value": 7.0,
                "buffer_system": "50 mM phosphate buffer",
                "enzyme_concentration": {"value": 0.5, "unit": "mg/mL"},
                "cofactors": ["NADPH"]
            },
            "reaction_products": [
                {"name": "Aflatoxicol", "toxicity_change": "reduced"}
            ],
            "additional_info": {
                "optimal_conditions": {
                    "ph": 7.0,
                    "temperature": {"value": 37, "unit": "°C"}
                },
                "notes": "High activity observed"
            },
            "source_in_document": {
                "doi": "10.1234/test.2024",
                "source_type": "table"
            }
        }
    ]
    
    # Create postprocessor
    processor = DataPostprocessor()
    
    # Flatten and save
    df = processor.flatten_and_save(test_fragments, "test_output.csv")
    
    print(f"✓ Processed {len(df)} records")
    print(f"✓ Columns: {len(df.columns)}")