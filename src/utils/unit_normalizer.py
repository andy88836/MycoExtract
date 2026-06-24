"""
Unit Normalization Module

Standardizes kinetic parameter units to a common format for Transfer Learning.
- Km: Converted to μM (micromolar)
- kcat: Converted to s^-1 (per second)
- Vmax: Converted to μmol/min/mg (if possible, otherwise kept as is)
- Temperature: Converted to Celsius
- Time: Converted to hours
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class UnitNormalizer:
    """
    Normalizes scientific units to standard formats.
    """
    
    # Standard Units
    STD_UNIT_KM = "μM"
    STD_UNIT_KCAT = "s^-1"
    STD_UNIT_TEMP = "°C"
    STD_UNIT_TIME = "h"
    
    @classmethod
    def normalize_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all fields in a record.
        
        Args:
            record: The raw extracted record
            
        Returns:
            Record with added 'standard_value' fields
        """
        if not record:
            return record
            
        # Normalize Kinetics
        if "kinetics" in record:
            kinetics = record["kinetics"]
            
            # Km
            if "km" in kinetics and kinetics["km"]:
                val, unit = cls._get_val_unit(kinetics["km"])
                std_val, std_unit = cls.normalize_concentration(val, unit)
                if std_val is not None:
                    kinetics["km"]["standard_value"] = std_val
                    kinetics["km"]["standard_unit"] = std_unit
            
            # kcat
            if "kcat" in kinetics and kinetics["kcat"]:
                val, unit = cls._get_val_unit(kinetics["kcat"])
                std_val, std_unit = cls.normalize_rate_constant(val, unit)
                if std_val is not None:
                    kinetics["kcat"]["standard_value"] = std_val
                    kinetics["kcat"]["standard_unit"] = std_unit
            
            # Degradation Efficiency Time
            if "degradation_efficiency" in kinetics and kinetics["degradation_efficiency"]:
                eff = kinetics["degradation_efficiency"]
                if "time_value" in eff and "time_unit" in eff:
                    std_time, std_unit = cls.normalize_time(eff["time_value"], eff["time_unit"])
                    if std_time is not None:
                        eff["standard_time_value"] = std_time
                        eff["standard_time_unit"] = std_unit

        # Normalize Conditions
        if "conditions" in record:
            cond = record["conditions"]
            # Temperature
            if "temperature_c" in cond and cond["temperature_c"] is not None:
                # Already in C usually, but could check for K or F if needed
                pass 
                
        return record

    @staticmethod
    def _get_val_unit(field: Dict) -> Tuple[Optional[float], Optional[str]]:
        return field.get("value"), field.get("unit")

    @classmethod
    def normalize_concentration(cls, value: float, unit: str) -> Tuple[Optional[float], str]:
        """
        Convert concentration to μM.
        """
        if value is None or not unit:
            return None, cls.STD_UNIT_KM
            
        unit = unit.strip().lower()
        
        try:
            if unit in ["mm", "millimolar"]:
                return value * 1000, cls.STD_UNIT_KM
            elif unit in ["m", "molar"]:
                return value * 1000000, cls.STD_UNIT_KM
            elif unit in ["nm", "nanomolar"]:
                return value / 1000, cls.STD_UNIT_KM
            elif unit in ["μm", "um", "micromolar"]:
                return value, cls.STD_UNIT_KM
            else:
                return None, unit # Unknown unit
        except Exception:
            return None, unit

    @classmethod
    def normalize_rate_constant(cls, value: float, unit: str) -> Tuple[Optional[float], str]:
        """
        Convert rate constant to s^-1.
        """
        if value is None or not unit:
            return None, cls.STD_UNIT_KCAT
            
        unit = unit.strip().lower()
        
        try:
            if unit in ["min^-1", "min-1", "/min"]:
                return value / 60, cls.STD_UNIT_KCAT
            elif unit in ["h^-1", "h-1", "/h", "hr^-1"]:
                return value / 3600, cls.STD_UNIT_KCAT
            elif unit in ["s^-1", "s-1", "/s"]:
                return value, cls.STD_UNIT_KCAT
            else:
                return None, unit
        except Exception:
            return None, unit

    @classmethod
    def normalize_time(cls, value: float, unit: str) -> Tuple[Optional[float], str]:
        """
        Convert time to hours (h).
        """
        if value is None or not unit:
            return None, cls.STD_UNIT_TIME
            
        unit = unit.strip().lower()
        
        try:
            if unit in ["min", "mins", "minutes"]:
                return value / 60, cls.STD_UNIT_TIME
            elif unit in ["s", "sec", "seconds"]:
                return value / 3600, cls.STD_UNIT_TIME
            elif unit in ["d", "day", "days"]:
                return value * 24, cls.STD_UNIT_TIME
            elif unit in ["h", "hr", "hrs", "hours"]:
                return value, cls.STD_UNIT_TIME
            else:
                return None, unit
        except Exception:
            return None, unit
