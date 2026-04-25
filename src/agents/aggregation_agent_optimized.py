"""
优化版聚合Agent Prompt - 减少Token消耗
移除冗余的模型对比输出，只保留核心数据
"""

# 简化版聚合prompt
OPTIMIZED_AGgregation_PROMPT = """
## YOUR TASK

Aggregate enzyme kinetics data from multiple AI assistants, keeping ONLY verified correct values.

## INPUT

### Original Text (for verification):
```
{original_text}
```

### Model Outputs:
{model_outputs}

---

## EXTRACTION RULES

1. **USE ONLY values explicitly stated in the article**
2. **NEVER infer or fill missing values**
3. **If uncertain, leave as null**

---

## OUTPUT FORMAT

Return ONLY the JSON array below (no explanations, no comparisons):

```json
[
  {{
    "enzyme_name": "string",
    "enzyme_full_name": "string or null",
    "enzyme_type": "string or null",
    "ec_number": "string or null",
    "gene_name": "string or null",
    "uniprot_id": "string or null",
    "genbank_id": "string or null",
    "pdb_id": "string or null",
    "sequence": "string or null",
    "organism": "string or null",
    "strain": "string or null",
    "is_recombinant": "boolean or null",
    "is_wild_type": "boolean or null",
    "mutations": "string or null",
    "substrate": "string",
    "substrate_smiles": "string or null",
    "substrate_concentration": "string or null",
    "Km_value": "number or null",
    "Km_unit": "string or null",
    "Vmax_value": "number or null",
    "Vmax_unit": "string or null",
    "kcat_value": "number or null",
    "kcat_unit": "string or null",
    "kcat_Km_value": "number or null",
    "kcat_Km_unit": "string or null",
    "degradation_efficiency": "number or null",
    "reaction_time_value": "number or null",
    "reaction_time_unit": "string or null",
    "products": [
      {{"name": "string", "toxicity_change": "string or null"}}
    ],
    "temperature_value": "number or null",
    "temperature_unit": "string or null",
    "ph": "number or null",
    "optimal_ph": "string or null",
    "optimal_temperature_value": "number or null",
    "optimal_temperature_unit": "string or null",
    "thermal_stability": "number or null",
    "thermal_stability_unit": "string or null",
    "thermal_stability_time": "number or null",
    "thermal_stability_time_unit": "string or null",
    "notes": "string or null"
  }}
]
```

## CRITICAL RULES:

1. **No explanatory text** - Return ONLY the JSON array
2. **No model comparisons** - Don't list which model extracted what
3. **No confidence scores** - Just the final verified data
4. **Null for missing** - Don't fill in uncertain values

OUTPUT YOUR JSON NOW:
"""


class OptimizedAggregationPrompt:
    """优化版聚合prompt生成器"""

    @staticmethod
    def build_prompt(
        original_text: str,
        model_outputs: str,
        extraction_prompt: str
    ) -> str:
        """构建简化版聚合prompt"""

        return f"""
## ORIGINAL TEXT
```
{original_text[:10000]}
```
{...(text_truncated)} if len(original_text) > 10000 else ''}

## MODEL OUTPUTS TO AGGREGATE
{model_outputs}

---

## YOUR TASK

Extract and verify enzyme kinetics data for mycotoxin-degrading enzymes.

### REQUIREMENTS:
1. Keep ONLY values found in the original text
2. Merge complementary information from different models
3. Resolve conflicts by checking the original text
4. Remove duplicates
5. Return ONLY valid JSON (no explanations)

### OUTPUT FORMAT:
```json
[
  {{
    "enzyme_name": "required",
    "substrate": "required",
    "Km_value": "number or null",
    "Km_unit": "string or null",
    "kcat_value": "number or null",
    "kcat_unit": "string or null",
    "kcat_Km_value": "number or null",
    "kcat_Km_unit": "string or null",
    "organism": "string or null",
    "gene_name": "string or null",
    "uniprot_id": "string or null",
    "products": [{{"name": "string", "toxicity_change": "string or null"}}],
    "notes": "string or null"
  }}
]
```

Return ONLY the JSON array. No explanations.
"""

# 数据库格式优化
DATABASE_SCHEMA = {
    "tables": {
        "enzymes": """
        CREATE TABLE enzymes (
            enzyme_id INTEGER PRIMARY KEY AUTOINCREMENT,
            enzyme_name VARCHAR(100) NOT NULL,
            enzyme_full_name VARCHAR(255),
            enzyme_type VARCHAR(50),
            ec_number VARCHAR(20),
            gene_name VARCHAR(50),
            uniprot_id VARCHAR(20),
            genbank_id VARCHAR(50),
            pdb_id VARCHAR(20),
            sequence TEXT,
            organism VARCHAR(255) NOT NULL,
            strain VARCHAR(100),
            is_recombinant BOOLEAN,
            is_wild_type BOOLEAN,
            mutations VARCHAR(100),
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,

        "substrates": """
        CREATE TABLE substrates (
            substrate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            substrate_name VARCHAR(255) NOT NULL UNIQUE,
            smiles TEXT,
            substrate_category VARCHAR(50), -- mycotoxin, generic, other
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,

        "kinetic_parameters": """
        CREATE TABLE kinetic_parameters (
            parameter_id INTEGER PRIMARY KEY AUTOINCREMENT,
            enzyme_id INTEGER,
            substrate_id INTEGER,
            Km_value REAL,
            Km_unit VARCHAR(20),
            kcat_value REAL,
            kcat_unit VARCHAR(20),
            kcat_Km_value REAL,
            kcat_Km_unit VARCHAR(20),
            Vmax_value REAL,
            Vmax_unit VARCHAR(50),
            temperature_value REAL,
            temperature_unit VARCHAR(20),
            ph_value REAL,
            optimal_ph VARCHAR(20),
            optimal_temperature_value REAL,
            optimal_temperature_unit VARCHAR(20),
            degradation_efficiency REAL,
            reaction_time_value REAL,
            reaction_time_unit VARCHAR(20),
            FOREIGN KEY (enzyme_id) REFERENCES enzymes(enzyme_id),
            FOREIGN KEY (substrate_id) REFERENCES substrates(substrate_id),
            UNIQUE(enzyme_id, substrate_id)
        );
        """,

        "products": """
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            parameter_id INTEGER,
            product_name VARCHAR(255) NOT NULL,
            toxicity_change VARCHAR(50),
            FOREIGN KEY (parameter_id) REFERENCES kinetic_parameters(parameter_id)
        );
        """
    },

    "json_to_sql_template": """
-- 数据库转换模板
-- 每个JSON记录对应一条kinetic_parameters记录

INSERT INTO enzymes (enzyme_name, enzyme_full_name, organism, gene_name)
VALUES (
    '{enzyme_name}',
    '{enzyme_full_name}',
    '{organism}',
    '{gene_name}'
)
ON CONFLICT(enzyme_name, organism) DO NOTHING;

-- 获取enzyme_id
SET @enzyme_id = last_insert_id();

-- 插入或获取substrate
INSERT INTO substrates (substrate_name, smiles)
VALUES ('{substrate}', '{substrate_smiles}')
ON CONFLICT(substrate_name) DO UPDATE SET smiles = EXCLUDED.smiles;
SET @substrate_id = (SELECT substrate_id FROM substrates WHERE substrate_name = '{substrate}');

-- 插入动力学参数
INSERT INTO kinetic_parameters (
    enzyme_id, substrate_id,
    Km_value, Km_unit,
    kcat_value, kcat_unit,
    kcat_Km_value, kcat_Km_unit,
    temperature_value, temperature_unit,
    ph_value
) VALUES (
    @enzyme_id, @substrate_id,
    {Km_value}, '{Km_unit}',
    {kcat_value}, '{kcat_unit}',
    {kcat_Km_value}, '{kcat_Km_unit}',
    {temperature_value}, '{temperature_unit}',
    {ph_value}
);
"""
}


def get_optimized_aggregation_prompt():
    """返回优化后的聚合prompt"""
    return OPTIMIZED_AGgregation_PROMPT
