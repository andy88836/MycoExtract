"""
蛋白质序列侦探 (Bio-Sequence Detective) Agent

功能：
1. 从论文中提取关键生物信息（菌株号、引物序列、分子量等）
2. 调用UniProt API进行智能搜索
3. 通过逻辑推理和交叉验证找到正确的UniProt ID
4. 输出置信度和推理过程

工作流：
Step 1: 扫描指纹 - 提取Organism, Strain, Accession Numbers, Primers
Step 2: 初步检索 - UniProt搜索
Step 3: 缩小范围 - 根据GenBank ID、引物、分子量等过滤
Step 4: 交叉验证 - 验证分子量、基因长度等
Step 5: 输出结论 - 最佳匹配及推理过程

使用GLM-4.6模型进行文本分析
"""

import re
import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BioFingerprint:
    """从论文中提取的生物指纹信息"""
    organism: str = ""
    strain: str = ""
    reference_strain: str = ""
    enzyme_name: str = ""
    gene_name: str = ""
    genbank_id: str = ""
    uniprot_id: str = ""  # 如果论文直接给出
    pdb_id: str = ""
    ec_number: str = ""
    molecular_weight_kda: Optional[float] = None
    gene_length_bp: Optional[int] = None
    protein_length_aa: Optional[int] = None
    primer_forward: str = ""
    primer_reverse: str = ""
    taxonomy_id: str = ""
    additional_info: Dict[str, str] = field(default_factory=dict)


@dataclass
class UniProtCandidate:
    """UniProt候选条目"""
    entry_id: str
    entry_name: str
    protein_name: str
    organism: str
    gene_names: List[str]
    length: int
    mass: float  # Da
    reviewed: bool  # Swiss-Prot (True) vs TrEMBL (False)
    sequence: str = ""
    ec_numbers: List[str] = field(default_factory=list)
    score: float = 0.0  # 匹配分数
    match_reasons: List[str] = field(default_factory=list)


@dataclass 
class DetectiveResult:
    """侦探结果"""
    best_match: Optional[UniProtCandidate]
    candidates: List[UniProtCandidate]
    confidence: str  # "High", "Medium", "Low", "None"
    reasoning: str
    fingerprint: BioFingerprint
    search_log: List[str] = field(default_factory=list)


class UniProtAPI:
    """UniProt REST API 封装"""
    
    BASE_URL = "https://rest.uniprot.org"
    
    def __init__(self, timeout: int = 30):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def search(
        self, 
        query: str, 
        size: int = 25,
        fields: List[str] = None
    ) -> List[Dict]:
        """搜索UniProt"""
        if fields is None:
            fields = [
                "accession", "id", "protein_name", "organism_name", 
                "gene_names", "length", "mass", "reviewed", "sequence",
                "ec"
            ]
        
        params = {
            "query": query,
            "format": "json",
            "size": size,
            "fields": ",".join(fields)
        }
        
        url = f"{self.BASE_URL}/uniprotkb/search"
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    else:
                        logger.error(f"UniProt API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"UniProt search failed: {e}")
            return []
    
    async def get_entry(self, entry_id: str) -> Optional[Dict]:
        """获取单个条目详情"""
        url = f"{self.BASE_URL}/uniprotkb/{entry_id}.json"
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Failed to get entry {entry_id}: {e}")
            return None
    
    async def id_mapping(
        self, 
        from_db: str, 
        to_db: str, 
        ids: List[str]
    ) -> Dict[str, str]:
        """ID映射（如GenBank -> UniProt）"""
        # UniProt ID Mapping API
        url = f"{self.BASE_URL}/idmapping/run"
        
        data = {
            "from": from_db,
            "to": to_db,
            "ids": ",".join(ids)
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # 提交任务
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        return {}
                    result = await response.json()
                    job_id = result.get("jobId")
                
                if not job_id:
                    return {}
                
                # 轮询结果
                status_url = f"{self.BASE_URL}/idmapping/status/{job_id}"
                for _ in range(30):  # 最多等30秒
                    async with session.get(status_url) as response:
                        status = await response.json()
                        if "results" in status:
                            # 获取结果
                            results_url = f"{self.BASE_URL}/idmapping/results/{job_id}"
                            async with session.get(results_url) as res_response:
                                results = await res_response.json()
                                mapping = {}
                                for item in results.get("results", []):
                                    mapping[item["from"]] = item["to"]["primaryAccession"]
                                return mapping
                        elif status.get("jobStatus") == "FINISHED":
                            break
                    await asyncio.sleep(1)
                
                return {}
        except Exception as e:
            logger.error(f"ID mapping failed: {e}")
            return {}


class SequenceDetectiveAgent:
    """蛋白质序列侦探Agent"""
    
    # 从文件加载System Prompt
    @staticmethod
    def _load_system_prompt() -> str:
        """从prompts文件夹加载系统提示词"""
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "prompts_sequence_detective.txt"
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load sequence detective prompt: {e}")
            return SequenceDetectiveAgent._get_fallback_prompt()
    
    @staticmethod
    def _get_fallback_prompt() -> str:
        """Fallback提示词"""
        return """你是一位专业的生物信息学侦探，擅长从科学论文中提取关键信息并找到对应的UniProt蛋白质条目。

请从提供的论文文本中提取以下信息：
1. Organism (生物名称)
2. Strain (菌株号)
3. Reference Strain (参考菌株，如果有)
4. Enzyme/Protein Name (酶/蛋白名称)
5. Gene Name (基因名)
6. GenBank/NCBI ID (如NC_..., NP_..., WP_...)
7. UniProt ID (如果直接给出)
8. EC Number
9. Molecular Weight (分子量，kDa)
10. Gene Length (基因长度，bp)
11. Primer Sequences (引物序列)

请以JSON格式输出，字段名使用英文。"""
    
    SYSTEM_PROMPT = None  # 将在首次使用时加载
    
    def __init__(self, llm_client, config: Dict = None):
        """
        Args:
            llm_client: LLM客户端
            config: 配置
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.uniprot_api = UniProtAPI()
        
        # 加载System Prompt
        if SequenceDetectiveAgent.SYSTEM_PROMPT is None:
            SequenceDetectiveAgent.SYSTEM_PROMPT = SequenceDetectiveAgent._load_system_prompt()
    
    # ========== 智能段落筛选（Token优化）==========
    
    # 生物指纹关键词模式
    BIO_PATTERNS = [
        # 菌株号模式
        r'(?:ATCC|MTCC|NRRL|DSM|JCM|KCTC|CGMCC|CECT|NBRC|IFO|CBS|NCIMB|LMG)\s*\d+',
        # GenBank/RefSeq ID
        r'(?:NC_|NZ_|NP_|WP_|YP_|XP_)\d+(?:\.\d+)?',
        # GenBank 蛋白 accession
        r'\b[A-Z]{3}\d{5,7}(?:\.\d+)?\b',
        # UniProt ID (6字符)
        r'\b[A-Z][0-9][A-Z0-9]{3}[0-9]\b',
        r'\b[A-Z][0-9][A-Z0-9]{3}[0-9]_[A-Z0-9]+\b',  # UniProt entry name
        # PDB ID
        r'\b\d[A-Z0-9]{3}\b',
        # EC 编号
        r'EC\s*[\d\.\-]+',
        r'\b\d+\.\d+\.\d+\.\d+\b',
        # 分子量
        r'\b\d+(?:\.\d+)?\s*k?Da\b',
        # 基因长度
        r'\b\d+\s*bp\b',
        r'\b\d+\s*aa\b',
        # 引物序列
        r"5['\u2019]-[ATCG]{15,}-3['\u2019]",
        r'(?:forward|reverse|primer)[^.]{0,50}[ATCG]{15,}',
        # 基因/蛋白名
        r'\b(?:gene|protein|enzyme)\s+\w+',
        # 表达系统关键词
        r'(?:pET|pGEX|pMAL|pQE|expression\s+vector)',
        # 菌株保藏关键词
        r'(?:strain|isolate)\s+\w+',
        r'(?:obtained|purchased)\s+from',
        # 序列分析关键词
        r'(?:sequence|cloning|PCR|amplif)',
        r'(?:SDS-PAGE|gel\s+electrophoresis)',
    ]
    
    def _extract_relevant_paragraphs(
        self, 
        paper_text: str, 
        enzyme_name: str = None,
        organism: str = None
    ) -> str:
        """
        智能提取包含生物指纹信息的段落（节省90%+ token）
        
        策略：
        1. 按段落分割
        2. 用正则匹配包含关键模式的段落
        3. 加入酶名/生物名匹配的段落
        4. 返回去重后的相关段落
        """
        # 按段落分割（支持多种分隔符）
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', paper_text)
        
        # 如果段落太少，按句子分割
        if len(paragraphs) < 5:
            paragraphs = re.split(r'(?<=[.!?])\s+', paper_text)
        
        relevant_paragraphs = []
        seen_content = set()  # 去重
        
        # 编译正则模式
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.BIO_PATTERNS]
        
        # 添加酶名和生物名到搜索模式
        custom_patterns = []
        if enzyme_name:
            # 处理酶名的各种变体
            enzyme_words = re.split(r'[\s\-_]+', enzyme_name)
            for word in enzyme_words:
                if len(word) > 2:
                    custom_patterns.append(re.compile(re.escape(word), re.IGNORECASE))
        if organism:
            # 处理生物名（属名 + 种名）
            org_words = organism.split()
            for word in org_words:
                if len(word) > 2:
                    custom_patterns.append(re.compile(re.escape(word), re.IGNORECASE))
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 30:  # 跳过太短的段落
                continue
            
            # 内容去重（用前100字符作为key）
            content_key = para[:100].lower()
            if content_key in seen_content:
                continue
            
            # 检查是否匹配任何生物指纹模式
            is_relevant = False
            
            # 检查预定义模式
            for pattern in compiled_patterns:
                if pattern.search(para):
                    is_relevant = True
                    break
            
            # 检查自定义模式（酶名/生物名）
            if not is_relevant:
                match_count = sum(1 for p in custom_patterns if p.search(para))
                if match_count >= 2:  # 至少匹配2个关键词
                    is_relevant = True
            
            # 检查章节标题（Methods, Materials, Results等）
            if not is_relevant:
                section_keywords = [
                    'materials and methods', 'methods', 'materials',
                    'experimental', 'strains', 'cloning', 'expression',
                    'purification', 'characterization', 'sequence'
                ]
                para_lower = para.lower()
                for kw in section_keywords:
                    if kw in para_lower and len(para) < 200:  # 可能是标题
                        is_relevant = True
                        break
            
            if is_relevant:
                seen_content.add(content_key)
                relevant_paragraphs.append(para)
        
        # 如果筛选结果太少，放宽条件
        if len(relevant_paragraphs) < 3:
            logger.warning("  Too few relevant paragraphs found, using broader selection")
            # 返回前5000字符的文本
            return paper_text[:5000]
        
        # 合并段落
        result = '\n\n'.join(relevant_paragraphs)
        
        # 如果结果太长，截断
        if len(result) > 8000:
            result = result[:8000] + "\n\n[... 更多内容已截断 ...]"
        
        return result

    async def investigate(
        self, 
        paper_text: str,
        enzyme_name: str = None,
        organism: str = None,
        existing_data: Dict = None
    ) -> DetectiveResult:
        """
        调查并找到正确的UniProt ID
        
        Args:
            paper_text: 论文文本（最好是Methods和Results部分）
            enzyme_name: 已知的酶名（可选）
            organism: 已知的生物名（可选）
            existing_data: 已提取的数据（可选）
            
        Returns:
            DetectiveResult
        """
        search_log = []
        search_log.append("🔍 开始蛋白质序列调查...")
        
        # Step 1: 提取生物指纹
        search_log.append("\n📋 Step 1: 扫描生物指纹...")
        fingerprint = await self._extract_fingerprint(paper_text, enzyme_name, organism, existing_data)
        search_log.append(f"  - Organism: {fingerprint.organism}")
        search_log.append(f"  - Strain: {fingerprint.strain}")
        search_log.append(f"  - Reference Strain: {fingerprint.reference_strain}")
        search_log.append(f"  - Enzyme: {fingerprint.enzyme_name}")
        search_log.append(f"  - Gene: {fingerprint.gene_name}")
        search_log.append(f"  - GenBank ID: {fingerprint.genbank_id}")
        search_log.append(f"  - UniProt ID: {fingerprint.uniprot_id}")
        search_log.append(f"  - EC: {fingerprint.ec_number}")
        search_log.append(f"  - MW: {fingerprint.molecular_weight_kda} kDa")
        search_log.append(f"  - Gene Length: {fingerprint.gene_length_bp} bp")
        
        # 如果论文直接给出了UniProt ID
        if fingerprint.uniprot_id:
            search_log.append(f"\n✅ 论文直接提供了UniProt ID: {fingerprint.uniprot_id}")
            entry = await self.uniprot_api.get_entry(fingerprint.uniprot_id)
            if entry:
                candidate = self._parse_uniprot_entry(entry)
                candidate.score = 1.0
                candidate.match_reasons = ["论文直接提供UniProt ID"]
                return DetectiveResult(
                    best_match=candidate,
                    candidates=[candidate],
                    confidence="High",
                    reasoning=f"论文明确给出UniProt ID: {fingerprint.uniprot_id}",
                    fingerprint=fingerprint,
                    search_log=search_log
                )
        
        # Step 2: 尝试通过GenBank ID映射
        candidates = []
        if fingerprint.genbank_id:
            search_log.append(f"\n🔗 Step 2: 尝试GenBank ID映射 ({fingerprint.genbank_id})...")
            mapped = await self._try_genbank_mapping(fingerprint.genbank_id)
            if mapped:
                search_log.append(f"  ✅ 找到映射: {mapped}")
                entry = await self.uniprot_api.get_entry(mapped)
                if entry:
                    candidate = self._parse_uniprot_entry(entry)
                    candidate.score = 0.95
                    candidate.match_reasons = [f"通过GenBank ID ({fingerprint.genbank_id}) 映射"]
                    candidates.append(candidate)
        
        # Step 3: UniProt搜索
        search_log.append("\n🔎 Step 3: UniProt数据库搜索...")
        search_candidates = await self._search_uniprot(fingerprint)
        search_log.append(f"  找到 {len(search_candidates)} 个候选")
        
        # 合并候选（去重）
        seen_ids = {c.entry_id for c in candidates}
        for c in search_candidates:
            if c.entry_id not in seen_ids:
                candidates.append(c)
                seen_ids.add(c.entry_id)
        
        if not candidates:
            search_log.append("\n❌ 未找到任何候选")
            return DetectiveResult(
                best_match=None,
                candidates=[],
                confidence="None",
                reasoning="未能在UniProt中找到匹配的蛋白质条目",
                fingerprint=fingerprint,
                search_log=search_log
            )
        
        # Step 4: 评分和排序
        search_log.append(f"\n📊 Step 4: 评估 {len(candidates)} 个候选...")
        scored_candidates = await self._score_candidates(candidates, fingerprint, search_log)
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Step 5: 确定最佳匹配
        best = scored_candidates[0] if scored_candidates else None
        
        # 确定置信度
        if best:
            if best.score >= 0.9:
                confidence = "High"
            elif best.score >= 0.7:
                confidence = "Medium"
            elif best.score >= 0.5:
                confidence = "Low"
            else:
                confidence = "Very Low"
        else:
            confidence = "None"
        
        # 生成推理说明
        reasoning = self._generate_reasoning(best, scored_candidates, fingerprint)
        search_log.append(f"\n🎯 结论: {confidence} 置信度")
        search_log.append(f"  最佳匹配: {best.entry_id if best else 'None'}")
        
        return DetectiveResult(
            best_match=best,
            candidates=scored_candidates[:10],  # 返回前10个
            confidence=confidence,
            reasoning=reasoning,
            fingerprint=fingerprint,
            search_log=search_log
        )
    
    async def _extract_fingerprint(
        self, 
        paper_text: str,
        enzyme_name: str = None,
        organism: str = None,
        existing_data: Dict = None
    ) -> BioFingerprint:
        """使用LLM从论文中提取生物指纹（带智能段落筛选）"""
        
        # ===== 智能段落筛选（节省90%+ token）=====
        relevant_text = self._extract_relevant_paragraphs(paper_text, enzyme_name, organism)
        logger.info(f"  [Token Optimization] Original: {len(paper_text)} chars → Filtered: {len(relevant_text)} chars ({100-len(relevant_text)*100//max(len(paper_text),1)}% saved)")
        
        # 构建提示
        prompt = f"""请从以下论文文本中提取蛋白质/酶的关键信息。

## 已知信息（如果有）
- 酶名: {enzyme_name or '未知'}
- 生物来源: {organism or '未知'}
- 已提取数据: {json.dumps(existing_data, ensure_ascii=False) if existing_data else '无'}

## 论文相关段落（已筛选关键信息）
{relevant_text[:8000]}

## 请提取以下信息并以JSON格式输出：

```json
{{
    "organism": "完整的生物学名（如 Pseudomonas putida）",
    "strain": "实验使用的菌株号（如 MTCC 2445）",
    "reference_strain": "参考菌株（如果引物/序列基于其他菌株设计）",
    "enzyme_name": "酶的完整名称",
    "gene_name": "基因名（如 lipA, estA）",
    "genbank_id": "GenBank/NCBI ID（如 NC_002947, NP_xxx, WP_xxx）",
    "uniprot_id": "UniProt ID（如果论文直接给出，如 Q88Q20）",
    "pdb_id": "PDB ID（如果有）",
    "ec_number": "EC编号（如 3.1.1.3）",
    "molecular_weight_kda": 分子量（数字，单位kDa）,
    "gene_length_bp": 基因长度（数字，单位bp）,
    "protein_length_aa": 蛋白质长度（数字，单位aa）,
    "primer_forward": "正向引物序列",
    "primer_reverse": "反向引物序列",
    "taxonomy_id": "NCBI Taxonomy ID（如果知道）",
    "additional_info": {{
        "key": "其他有助于鉴定的信息"
    }}
}}
```

请仔细阅读，特别关注：
1. 菌株来源和编号
2. PCR引物设计的参考序列
3. 基因克隆的模板来源
4. SDS-PAGE显示的分子量
5. 任何提到的数据库ID

如果某项信息在文中未找到，请填写空字符串或null。"""

        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # 解析JSON
            content = response.get("content", "")
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # 尝试直接解析
                data = json.loads(content)
            
            return BioFingerprint(
                organism=data.get("organism", "") or "",
                strain=data.get("strain", "") or "",
                reference_strain=data.get("reference_strain", "") or "",
                enzyme_name=data.get("enzyme_name", enzyme_name or "") or "",
                gene_name=data.get("gene_name", "") or "",
                genbank_id=data.get("genbank_id", "") or "",
                uniprot_id=data.get("uniprot_id", "") or "",
                pdb_id=data.get("pdb_id", "") or "",
                ec_number=data.get("ec_number", "") or "",
                molecular_weight_kda=data.get("molecular_weight_kda"),
                gene_length_bp=data.get("gene_length_bp"),
                protein_length_aa=data.get("protein_length_aa"),
                primer_forward=data.get("primer_forward", "") or "",
                primer_reverse=data.get("primer_reverse", "") or "",
                taxonomy_id=data.get("taxonomy_id", "") or "",
                additional_info=data.get("additional_info", {}) or {}
            )
            
        except Exception as e:
            logger.error(f"Failed to extract fingerprint: {e}")
            # 返回基于已知信息的指纹
            return BioFingerprint(
                organism=organism or "",
                enzyme_name=enzyme_name or ""
            )
    
    async def _try_genbank_mapping(self, genbank_id: str) -> Optional[str]:
        """尝试将GenBank ID映射到UniProt"""
        # 清理ID
        genbank_id = genbank_id.strip()
        
        # 判断ID类型
        if genbank_id.startswith("NC_") or genbank_id.startswith("NZ_"):
            # 这是基因组ID，不能直接映射
            # 需要搜索该基因组的蛋白质
            return None
        elif genbank_id.startswith("NP_") or genbank_id.startswith("WP_") or genbank_id.startswith("YP_"):
            # RefSeq蛋白ID
            mapping = await self.uniprot_api.id_mapping("RefSeq_Protein", "UniProtKB", [genbank_id])
            return mapping.get(genbank_id)
        elif re.match(r'^[A-Z]{3}\d{5}', genbank_id):
            # GenBank蛋白ID
            mapping = await self.uniprot_api.id_mapping("EMBL-GenBank-DDBJ", "UniProtKB", [genbank_id])
            return mapping.get(genbank_id)
        
        return None
    
    async def _search_uniprot(self, fingerprint: BioFingerprint) -> List[UniProtCandidate]:
        """在UniProt中搜索候选"""
        candidates = []
        
        # 构建搜索查询
        queries = []
        
        # 策略1：精确搜索（如果有足够信息）
        if fingerprint.organism and fingerprint.enzyme_name:
            # 优先搜索参考菌株（如果有）
            org = fingerprint.reference_strain or fingerprint.organism
            queries.append(f'(organism_name:"{org}") AND (protein_name:"{fingerprint.enzyme_name}")')
        
        # 策略2：基因名搜索
        if fingerprint.gene_name and fingerprint.organism:
            org = fingerprint.reference_strain or fingerprint.organism
            queries.append(f'(organism_name:"{org}") AND (gene:{fingerprint.gene_name})')
        
        # 策略3：EC号搜索
        if fingerprint.ec_number and fingerprint.organism:
            org = fingerprint.reference_strain or fingerprint.organism
            queries.append(f'(organism_name:"{org}") AND (ec:{fingerprint.ec_number})')
        
        # 策略4：宽松搜索
        if fingerprint.enzyme_name:
            queries.append(f'(protein_name:"{fingerprint.enzyme_name}") AND (reviewed:true)')
        
        # 执行搜索
        seen_ids = set()
        for query in queries:
            try:
                results = await self.uniprot_api.search(query, size=20)
                for entry in results:
                    candidate = self._parse_uniprot_entry(entry)
                    if candidate.entry_id not in seen_ids:
                        candidates.append(candidate)
                        seen_ids.add(candidate.entry_id)
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
        
        return candidates
    
    def _parse_uniprot_entry(self, entry: Dict) -> UniProtCandidate:
        """解析UniProt条目"""
        # 提取蛋白名
        protein_name = ""
        if "proteinDescription" in entry:
            rec_name = entry["proteinDescription"].get("recommendedName", {})
            if rec_name:
                protein_name = rec_name.get("fullName", {}).get("value", "")
            if not protein_name:
                sub_names = entry["proteinDescription"].get("submissionNames", [])
                if sub_names:
                    protein_name = sub_names[0].get("fullName", {}).get("value", "")
        
        # 提取基因名
        gene_names = []
        for gene in entry.get("genes", []):
            if "geneName" in gene:
                gene_names.append(gene["geneName"].get("value", ""))
            for syn in gene.get("synonyms", []):
                gene_names.append(syn.get("value", ""))
        
        # 提取EC号
        ec_numbers = []
        if "proteinDescription" in entry:
            rec_name = entry["proteinDescription"].get("recommendedName", {})
            for ec in rec_name.get("ecNumbers", []):
                ec_numbers.append(ec.get("value", ""))
        
        # 提取序列
        sequence = ""
        if "sequence" in entry:
            sequence = entry["sequence"].get("value", "")
        
        return UniProtCandidate(
            entry_id=entry.get("primaryAccession", ""),
            entry_name=entry.get("uniProtkbId", ""),
            protein_name=protein_name,
            organism=entry.get("organism", {}).get("scientificName", ""),
            gene_names=gene_names,
            length=entry.get("sequence", {}).get("length", 0),
            mass=entry.get("sequence", {}).get("molWeight", 0),
            reviewed=entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)",
            sequence=sequence,
            ec_numbers=ec_numbers
        )
    
    async def _score_candidates(
        self, 
        candidates: List[UniProtCandidate], 
        fingerprint: BioFingerprint,
        search_log: List[str]
    ) -> List[UniProtCandidate]:
        """为候选评分"""
        
        for candidate in candidates:
            score = 0.0
            reasons = []
            
            # 1. Swiss-Prot (reviewed) 加分
            if candidate.reviewed:
                score += 0.15
                reasons.append("✅ Swiss-Prot (人工审核)")
            
            # 2. 生物名匹配
            org_lower = fingerprint.organism.lower()
            cand_org_lower = candidate.organism.lower()
            if org_lower and org_lower in cand_org_lower:
                score += 0.2
                reasons.append(f"✅ 生物来源匹配: {candidate.organism}")
            
            # 检查参考菌株
            if fingerprint.reference_strain:
                ref_lower = fingerprint.reference_strain.lower()
                if ref_lower in cand_org_lower:
                    score += 0.1
                    reasons.append(f"✅ 参考菌株匹配: {fingerprint.reference_strain}")
            
            # 3. 基因名匹配
            if fingerprint.gene_name:
                gene_lower = fingerprint.gene_name.lower()
                for gene in candidate.gene_names:
                    if gene_lower == gene.lower():
                        score += 0.2
                        reasons.append(f"✅ 基因名精确匹配: {gene}")
                        break
                    elif gene_lower in gene.lower() or gene.lower() in gene_lower:
                        score += 0.1
                        reasons.append(f"✅ 基因名部分匹配: {gene}")
                        break
            
            # 4. EC号匹配
            if fingerprint.ec_number:
                for ec in candidate.ec_numbers:
                    if fingerprint.ec_number == ec:
                        score += 0.15
                        reasons.append(f"✅ EC号匹配: {ec}")
                        break
            
            # 5. 分子量验证
            if fingerprint.molecular_weight_kda and candidate.mass:
                expected_da = fingerprint.molecular_weight_kda * 1000
                actual_da = candidate.mass
                # 允许10%误差
                if abs(expected_da - actual_da) / expected_da < 0.1:
                    score += 0.15
                    reasons.append(f"✅ 分子量匹配: {actual_da/1000:.1f} kDa (预期 {fingerprint.molecular_weight_kda} kDa)")
                elif abs(expected_da - actual_da) / expected_da < 0.2:
                    score += 0.08
                    reasons.append(f"⚠️ 分子量接近: {actual_da/1000:.1f} kDa (预期 {fingerprint.molecular_weight_kda} kDa)")
            
            # 6. 蛋白长度验证（基于基因长度）
            if fingerprint.gene_length_bp and candidate.length:
                expected_aa = fingerprint.gene_length_bp / 3
                if abs(expected_aa - candidate.length) / expected_aa < 0.1:
                    score += 0.1
                    reasons.append(f"✅ 长度匹配: {candidate.length} aa (基于基因 {fingerprint.gene_length_bp} bp)")
            
            # 7. 直接蛋白长度验证
            if fingerprint.protein_length_aa and candidate.length:
                if abs(fingerprint.protein_length_aa - candidate.length) / fingerprint.protein_length_aa < 0.05:
                    score += 0.15
                    reasons.append(f"✅ 蛋白长度精确匹配: {candidate.length} aa")
            
            # 更新候选
            candidate.score = min(score, 1.0)  # 最高1.0
            candidate.match_reasons = reasons
            
            # 记录日志
            search_log.append(f"  [{candidate.entry_id}] Score: {candidate.score:.2f}")
            for reason in reasons:
                search_log.append(f"    {reason}")
        
        return candidates
    
    def _generate_reasoning(
        self, 
        best: Optional[UniProtCandidate],
        candidates: List[UniProtCandidate],
        fingerprint: BioFingerprint
    ) -> str:
        """生成推理说明"""
        if not best:
            return "未能找到匹配的UniProt条目。建议手动搜索UniProt数据库。"
        
        reasoning_parts = []
        
        # 基本信息
        reasoning_parts.append(f"**最佳匹配：** {best.entry_id} ({best.entry_name})")
        reasoning_parts.append(f"**蛋白名称：** {best.protein_name}")
        reasoning_parts.append(f"**生物来源：** {best.organism}")
        reasoning_parts.append(f"**匹配分数：** {best.score:.2f}")
        
        # 匹配原因
        if best.match_reasons:
            reasoning_parts.append("\n**匹配依据：**")
            for reason in best.match_reasons:
                reasoning_parts.append(f"- {reason}")
        
        # 关键推理
        reasoning_parts.append("\n**推理过程：**")
        
        if fingerprint.reference_strain and fingerprint.strain:
            if fingerprint.reference_strain != fingerprint.strain:
                reasoning_parts.append(
                    f"虽然实验使用菌株 {fingerprint.strain}，"
                    f"但引物/序列设计基于参考菌株 {fingerprint.reference_strain}，"
                    f"因此目标序列应锚定到参考菌株的UniProt条目。"
                )
        
        if fingerprint.molecular_weight_kda and best.mass:
            mw_match = abs(fingerprint.molecular_weight_kda * 1000 - best.mass) / (fingerprint.molecular_weight_kda * 1000) < 0.1
            if mw_match:
                reasoning_parts.append(
                    f"分子量验证通过：论文报告 {fingerprint.molecular_weight_kda} kDa，"
                    f"UniProt记录 {best.mass/1000:.1f} kDa。"
                )
        
        # 其他候选提示
        if len(candidates) > 1:
            reasoning_parts.append(f"\n**其他候选 ({len(candidates)-1}个)：**")
            for c in candidates[1:5]:  # 显示前4个其他候选
                reasoning_parts.append(f"- {c.entry_id}: {c.protein_name} (Score: {c.score:.2f})")
        
        return "\n".join(reasoning_parts)


# ============== 便捷函数 ==============

async def investigate_sequence(
    llm_client,
    paper_text: str,
    enzyme_name: str = None,
    organism: str = None,
    existing_data: Dict = None
) -> DetectiveResult:
    """便捷函数：调查蛋白质序列"""
    agent = SequenceDetectiveAgent(llm_client)
    return await agent.investigate(paper_text, enzyme_name, organism, existing_data)


def format_detective_result(result: DetectiveResult) -> str:
    """格式化侦探结果为人类可读的文本"""
    lines = []
    
    # 标题
    lines.append("=" * 60)
    lines.append("🔬 蛋白质序列侦探报告")
    lines.append("=" * 60)
    
    # 置信度
    confidence_emoji = {
        "High": "🟢",
        "Medium": "🟡", 
        "Low": "🟠",
        "Very Low": "🔴",
        "None": "⚫"
    }
    emoji = confidence_emoji.get(result.confidence, "❓")
    lines.append(f"\n{emoji} **置信度: {result.confidence}**")
    
    # 最佳匹配
    if result.best_match:
        best = result.best_match
        lines.append(f"\n### 🎯 最佳匹配")
        lines.append(f"- **UniProt ID:** {best.entry_id}")
        lines.append(f"- **Entry Name:** {best.entry_name}")
        lines.append(f"- **蛋白名称:** {best.protein_name}")
        lines.append(f"- **生物来源:** {best.organism}")
        lines.append(f"- **基因名:** {', '.join(best.gene_names) if best.gene_names else 'N/A'}")
        lines.append(f"- **长度:** {best.length} aa")
        lines.append(f"- **分子量:** {best.mass/1000:.1f} kDa")
        lines.append(f"- **审核状态:** {'Swiss-Prot ✅' if best.reviewed else 'TrEMBL'}")
        lines.append(f"- **匹配分数:** {best.score:.2f}")
        
        if best.sequence:
            lines.append(f"\n**序列 (前100aa):**")
            lines.append(f"```")
            lines.append(best.sequence[:100] + "..." if len(best.sequence) > 100 else best.sequence)
            lines.append(f"```")
    else:
        lines.append("\n❌ 未找到匹配的UniProt条目")
    
    # 推理
    lines.append(f"\n### 📝 推理说明")
    lines.append(result.reasoning)
    
    # 提取的指纹
    fp = result.fingerprint
    lines.append(f"\n### 🧬 从论文提取的信息")
    lines.append(f"- Organism: {fp.organism or 'N/A'}")
    lines.append(f"- Strain: {fp.strain or 'N/A'}")
    lines.append(f"- Reference Strain: {fp.reference_strain or 'N/A'}")
    lines.append(f"- Enzyme: {fp.enzyme_name or 'N/A'}")
    lines.append(f"- Gene: {fp.gene_name or 'N/A'}")
    lines.append(f"- GenBank ID: {fp.genbank_id or 'N/A'}")
    lines.append(f"- EC: {fp.ec_number or 'N/A'}")
    lines.append(f"- MW: {fp.molecular_weight_kda or 'N/A'} kDa")
    
    # 调查日志
    if result.search_log:
        lines.append(f"\n### 📋 调查日志")
        for log in result.search_log:
            lines.append(log)
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)
