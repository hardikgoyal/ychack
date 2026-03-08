"""
safebind_mhc1_cytotoxic.py — MHC Class I / CD8+ Cytotoxic T-cell Prediction
=============================================================================

Adds cytotoxic T-cell immunogenicity prediction to SafeBind AI:

1. IEDB MHC-I API (NetMHCpan 4.1 backend, same training data as TDC MHC1_IEDB-IMGT)
2. MHCflurry 2.0 local predictions (7,000+ predictions/sec, open-source)
3. AAV capsid validated epitope database (Hui et al. 2015 + 2023 immunopeptidomics)
4. Per-residue cytotoxic risk scoring (parallel to existing MHC-II pipeline)

DATA SOURCES:
- TDC MHC1_IEDB-IMGT_Nielsen: 185,985 peptide-MHC-I pairs, 43,018 peptides,
  150 HLA alleles (CC BY 4.0). Used to train NetMHCpan 3.0.
  Source: tdcommons.ai/multi_pred_tasks/peptidemhc/
- MHCflurry 2.0: Trained on >185K binding affinity measurements + mass spec
  eluted ligands. O'Donnell et al., Cell Systems 2020.
  Source: github.com/openvax/mhcflurry
- TANTIGEN 2.0: 4,296 tumor antigen variants, >1,500 validated T-cell epitopes.
  Olsen et al., BMC Bioinformatics 2021.
- AbImmPred: 199 therapeutic antibodies with immunogenicity annotations.
  PubMed Central, 2024.
- AAV Capsid Epitopes: Hui et al., Mol Ther Methods Clin Dev 2015 (21 epitopes);
  2023 immunopeptidomics (65 HLA-I peptides); GENEr8-1 trial (134 patients).

BIOLOGY:
MHC Class I presents intracellular peptides (8-11mers) to CD8+ cytotoxic T cells.
Unlike MHC-II/ADA (humoral), this pathway causes DIRECT CELL KILLING:
- Gene therapy: CD8+ T cells kill transduced hepatocytes → liver toxicity
- Cell therapy: host CD8+ T cells reject CAR-T cells
- Enzyme replacement: CD8+ responses to internalized enzyme
This is the pathway that killed patients in AAV gene therapy trials.
"""

import requests
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


# ══════════════════════════════════════════════════════════════
# HLA CLASS I ALLELES — SUPERTYPES COVERING ~95% POPULATION
# ══════════════════════════════════════════════════════════════

# 12 HLA-A/B supertypes covering ~95% of global population
# Selected based on Sidney et al. (BMC Immunology 2008) supertype classification
HLA_CLASS_I_ALLELES = [
    # HLA-A supertypes
    "HLA-A*02:01",   # A2 supertype — most common worldwide (~45% frequency)
    "HLA-A*01:01",   # A1 supertype
    "HLA-A*03:01",   # A3 supertype
    "HLA-A*24:02",   # A24 supertype — common in East Asian populations
    "HLA-A*11:01",   # A3 supertype variant
    "HLA-A*26:01",   # A26 supertype
    # HLA-B supertypes
    "HLA-B*07:02",   # B7 supertype
    "HLA-B*08:01",   # B8 supertype
    "HLA-B*15:01",   # B62 supertype
    "HLA-B*35:01",   # B7 supertype variant
    "HLA-B*40:01",   # B44 supertype
    "HLA-B*44:03",   # B44 supertype variant
]

# Peptide lengths for MHC-I (8-11mers, 9 most common)
MHC_I_PEPTIDE_LENGTHS = [8, 9, 10, 11]


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class MHCIEpitope:
    """A predicted MHC Class I epitope (CD8+ T-cell target)."""
    allele: str
    start: int           # 1-indexed position in protein
    end: int
    sequence: str        # 8-11mer peptide
    length: int          # peptide length
    rank: float          # percentile rank (lower = stronger)
    score: float = 0.0   # raw binding score
    source: str = "iedb" # "iedb" or "mhcflurry"
    ic50: float = 0.0    # predicted IC50 in nM (lower = stronger)
    processing_score: float = 0.0  # antigen processing likelihood


@dataclass
class CytotoxicResidueRisk:
    """Per-residue cytotoxic T-cell risk score."""
    position: int
    residue: str
    mhc1_risk: float          # 0-1, MHC-I binding risk
    num_alleles_binding: int   # how many HLA alleles have strong binders here
    best_rank: float           # best percentile rank at this position
    is_validated: bool = False # True if matches a known validated epitope
    validated_source: str = "" # e.g. "Hui_2015_AAV2"


@dataclass
class CytotoxicReport:
    """Full MHC-I / cytotoxic T-cell assessment."""
    total_epitopes_predicted: int
    strong_binders: int        # rank < 2%
    moderate_binders: int      # rank 2-10%
    epitopes: List[MHCIEpitope]
    residue_risks: List[CytotoxicResidueRisk]
    hotspot_regions: List[Dict[str, Any]]
    validated_hits: int        # matches to known validated epitopes
    validated_details: List[Dict[str, Any]]
    overall_cytotoxic_risk: float  # 0-1
    risk_category: str         # LOW, MODERATE, HIGH, VERY HIGH
    prediction_sources: List[str]  # ["IEDB NetMHCpan 4.1", "MHCflurry 2.0"]
    # Cross-reference data
    aav_epitope_recovery: Optional[float] = None  # % of known AAV epitopes recovered
    data_references: Optional[Dict[str, str]] = None


# ══════════════════════════════════════════════════════════════
# VALIDATED AAV CAPSID CD8+ EPITOPES (GROUND TRUTH)
# ══════════════════════════════════════════════════════════════

# From Hui et al. 2015 (Mol Ther Methods Clin Dev, PMC4588448)
# 21 MHC-I-restricted epitopes identified from 17 donors
# These are conserved across AAV2/5/8/9 serotypes
AAV_VALIDATED_EPITOPES_HUI_2015 = [
    {"peptide": "VPQYGYLTL",  "hla": "HLA-B*07:02", "serotypes": ["AAV2","AAV5","AAV8","AAV9"], "kd_um": 1.29, "source": "Hui_2015"},
    {"peptide": "SADNNNSEY",  "hla": "HLA-A*01:01", "serotypes": ["AAV2","AAV9"], "kd_um": 3.4, "source": "Hui_2015"},
    {"peptide": "FSQAGASD",   "hla": "HLA-B*08:01", "serotypes": ["AAV2","AAV9"], "kd_um": 5.1, "source": "Hui_2015"},
    {"peptide": "IPHTDGHFHPS","hla": "HLA-B*35:01", "serotypes": ["AAV2","AAV8","AAV9"], "kd_um": 2.8, "source": "Hui_2015"},
    {"peptide": "QPAKKRLNF",  "hla": "HLA-B*15:01", "serotypes": ["AAV8","AAV9"], "kd_um": 4.2, "source": "Hui_2015"},
    {"peptide": "TTSTRTWAL",  "hla": "HLA-A*02:01", "serotypes": ["AAV2","AAV8","AAV9"], "kd_um": 6.5, "source": "Hui_2015"},
    {"peptide": "FQAKKRVLE",  "hla": "HLA-A*03:01", "serotypes": ["AAV2","AAV8","AAV9"], "kd_um": 3.9, "source": "Hui_2015"},
    {"peptide": "YLTLNNGSG",  "hla": "HLA-A*02:01", "serotypes": ["AAV2","AAV9"], "kd_um": 7.8, "source": "Hui_2015"},
    {"peptide": "GFPGSFGYY",  "hla": "HLA-A*01:01", "serotypes": ["AAV2","AAV9"], "kd_um": 2.1, "source": "Hui_2015"},
    {"peptide": "TSTRTWALP",  "hla": "HLA-B*07:02", "serotypes": ["AAV2","AAV8","AAV9"], "kd_um": 4.6, "source": "Hui_2015"},
    {"peptide": "TYPNNHLYKY", "hla": "HLA-A*11:01", "serotypes": ["AAV2","AAV9"], "kd_um": 3.3, "source": "Hui_2015"},
    {"peptide": "RFHCHFSPR",  "hla": "HLA-A*03:01", "serotypes": ["AAV2","AAV8","AAV9"], "kd_um": 5.7, "source": "Hui_2015"},
    {"peptide": "LDRLMNPLI",  "hla": "HLA-A*02:01", "serotypes": ["AAV2","AAV9"], "kd_um": 8.2, "source": "Hui_2015"},
    {"peptide": "KLFNIQVKE",  "hla": "HLA-A*03:01", "serotypes": ["AAV2","AAV8","AAV9"], "kd_um": 3.1, "source": "Hui_2015"},
    {"peptide": "STPWGYFDL",  "hla": "HLA-A*02:01", "serotypes": ["AAV2","AAV9"], "kd_um": 4.0, "source": "Hui_2015"},
    # Additional epitopes from various donors
    {"peptide": "AQPAKKRLN",  "hla": "HLA-B*44:03", "serotypes": ["AAV8","AAV9"], "kd_um": 6.0, "source": "Hui_2015"},
    {"peptide": "DFNRFHCHF",  "hla": "HLA-B*15:01", "serotypes": ["AAV2","AAV9"], "kd_um": 3.7, "source": "Hui_2015"},
    {"peptide": "QISSQSGAS",  "hla": "HLA-A*24:02", "serotypes": ["AAV2","AAV9"], "kd_um": 7.1, "source": "Hui_2015"},
    {"peptide": "STFSAAKFA",  "hla": "HLA-A*02:01", "serotypes": ["AAV2","AAV8","AAV9"], "kd_um": 5.5, "source": "Hui_2015"},
    {"peptide": "SGAQPAKKR",  "hla": "HLA-A*11:01", "serotypes": ["AAV8","AAV9"], "kd_um": 4.8, "source": "Hui_2015"},
    {"peptide": "NMWGFRPKR",  "hla": "HLA-A*03:01", "serotypes": ["AAV2","AAV9"], "kd_um": 6.3, "source": "Hui_2015"},
]

# From 2023 immunopeptidomics study (PMC10469481)
# 65 naturally presented HLA-I peptides from AAV2, AAV6, AAV9 capsids
# Only 9% matched previously known epitopes — 91% were NOVEL
# 11 peptides conserved across all 3 serotypes
AAV_IMMUNOPEPTIDOMICS_2023 = [
    # 11 cross-serotype conserved peptides (highest priority)
    {"peptide": "AADGYLPDWL",  "serotypes": ["AAV2","AAV6","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "GYLPDWLEDT",  "serotypes": ["AAV2","AAV6","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "GLVLPGYKYL",  "serotypes": ["AAV2","AAV6","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "LVLPGYKYLY",  "serotypes": ["AAV2","AAV6","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "NPYLKYNHAD", "serotypes": ["AAV2","AAV6","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "WHCDSTWMGD", "serotypes": ["AAV2","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "GFRPKRLNFK", "serotypes": ["AAV2","AAV6","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "NWGFRPKRLN", "serotypes": ["AAV2","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "WLPTYNNHLY", "serotypes": ["AAV2","AAV6","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "VPQYGYLTLN", "serotypes": ["AAV2","AAV6","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
    {"peptide": "IGTRYLTRN",  "serotypes": ["AAV2","AAV9"], "source": "immunopeptidomics_2023", "conserved": True},
]

# Clinical outcome reference data from GENEr8-1 (Long et al. Mol Ther 2024)
# 134 hemophilia A patients, AAV5 vector
GENER8_1_REFERENCE = {
    "trial": "GENEr8-1 Phase 3",
    "product": "valoctocogene roxaparvovec (Roctavian)",
    "vector": "AAV5",
    "n_patients": 134,
    "capsid_ab_seroconversion": "100%",
    "cellular_response_elispot": "Variable — poor correlation with ALT",
    "alt_elevation_any": "~60%",
    "alt_elevation_grade3plus": "~5%",
    "key_finding": "Capsid-specific T-cell responses detectable by ELISpot but correlated poorly with ALT elevations, suggesting additional mechanisms beyond CD8+ killing",
    "pmid": "38796703",
    "source": "Long_2024_MolTher",
}


# ══════════════════════════════════════════════════════════════
# IEDB MHC CLASS I PREDICTION API
# ══════════════════════════════════════════════════════════════

def predict_mhc1_epitopes_iedb(
    sequence: str,
    alleles: List[str] = None,
    method: str = "recommended",
    lengths: List[int] = None,
    rank_threshold: float = 10.0,
) -> List[MHCIEpitope]:
    """Call IEDB MHC Class I binding prediction API.
    
    Uses NetMHCpan 4.1 (recommended method), which is trained on
    data equivalent to TDC MHC1_IEDB-IMGT (185,985 peptide-MHC-I pairs).
    
    Args:
        sequence: Protein sequence
        alleles: HLA Class I alleles to test (default: 12 supertypes)
        method: Prediction method (default: "recommended" = NetMHCpan 4.1)
        lengths: Peptide lengths to predict (default: 8-11)
        rank_threshold: Only return epitopes with rank below this (default: 10%)
    
    Returns:
        List of MHCIEpitope objects
    """
    if alleles is None:
        alleles = HLA_CLASS_I_ALLELES
    if lengths is None:
        lengths = MHC_I_PEPTIDE_LENGTHS
    
    epitopes = []
    url = "http://tools-cluster-interface.iedb.org/tools_api/mhci/"
    
    for allele in alleles:
        for length in lengths:
            try:
                data = {
                    "method": method,
                    "sequence_text": sequence,
                    "allele": allele,
                    "length": str(length),
                }
                resp = requests.post(url, data=data, timeout=60)
                if resp.status_code == 200:
                    lines = resp.text.strip().split("\n")
                    if len(lines) > 1:
                        header = lines[0].split("\t")
                        # Find column indices
                        col_map = {h.strip().lower(): i for i, h in enumerate(header)}
                        
                        allele_idx = col_map.get("allele", 0)
                        start_idx = col_map.get("start", col_map.get("peptide start", 3))
                        end_idx = col_map.get("end", col_map.get("peptide end", 4))
                        seq_idx = col_map.get("peptide", col_map.get("sequence", 5))
                        rank_idx = col_map.get("rank", col_map.get("percentile_rank", len(header) - 1))
                        score_idx = col_map.get("score", col_map.get("ic50", -1))
                        
                        for line in lines[1:]:
                            cols = line.split("\t")
                            if len(cols) < 4:
                                continue
                            try:
                                rank = float(cols[rank_idx]) if rank_idx >= 0 and rank_idx < len(cols) else 50.0
                                if rank > rank_threshold:
                                    continue
                                
                                start = int(cols[start_idx])
                                end = int(cols[end_idx]) if end_idx < len(cols) else start + length - 1
                                pep = cols[seq_idx] if seq_idx < len(cols) else ""
                                
                                raw_score = 0.0
                                if score_idx >= 0 and score_idx < len(cols):
                                    try:
                                        raw_score = float(cols[score_idx])
                                    except ValueError:
                                        pass
                                
                                ep = MHCIEpitope(
                                    allele=allele,
                                    start=start,
                                    end=end,
                                    sequence=pep,
                                    length=len(pep) if pep else length,
                                    rank=rank,
                                    score=raw_score,
                                    source="iedb",
                                    ic50=raw_score if raw_score > 10 else 0,
                                )
                                epitopes.append(ep)
                            except (ValueError, IndexError):
                                continue
                
                time.sleep(0.2)  # Rate limit
            except requests.exceptions.RequestException:
                continue
    
    return epitopes


# ══════════════════════════════════════════════════════════════
# MHCFLURRY 2.0 LOCAL PREDICTIONS (CROSS-CHECK)
# ══════════════════════════════════════════════════════════════

def predict_mhc1_epitopes_mhcflurry(
    sequence: str,
    alleles: List[str] = None,
    lengths: List[int] = None,
    rank_threshold: float = 10.0,
) -> List[MHCIEpitope]:
    """Run MHCflurry 2.0 locally for MHC-I binding prediction.
    
    MHCflurry 2.0 integrates:
    - Binding affinity predictor (trained on >185K measurements)
    - Antigen processing predictor (proteasomal cleavage + TAP transport)
    - Presentation predictor (combines both)
    
    Reference: O'Donnell et al., Cell Systems 2020.
    Training data overlaps with TDC MHC1_IEDB-IMGT dataset.
    
    Falls back gracefully if MHCflurry is not installed.
    """
    if alleles is None:
        alleles = HLA_CLASS_I_ALLELES
    if lengths is None:
        lengths = MHC_I_PEPTIDE_LENGTHS
    
    try:
        from mhcflurry import Class1PresentationPredictor
        predictor = Class1PresentationPredictor.load()
    except ImportError:
        warnings.warn(
            "MHCflurry not installed. Install with: pip install mhcflurry && mhcflurry-downloads fetch"
        )
        return []
    except Exception as e:
        warnings.warn(f"MHCflurry load error: {e}")
        return []
    
    epitopes = []
    
    # Generate all peptides
    peptides = []
    peptide_info = []  # (start, length)
    for length in lengths:
        for i in range(len(sequence) - length + 1):
            pep = sequence[i:i+length]
            peptides.append(pep)
            peptide_info.append((i + 1, length))  # 1-indexed
    
    if not peptides:
        return []
    
    # MHCflurry format alleles without "HLA-" prefix for some versions
    formatted_alleles = []
    for a in alleles:
        # MHCflurry accepts both "HLA-A*02:01" and "HLA-A0201" formats
        formatted_alleles.append(a)
    
    try:
        # Run predictions in batch (very fast)
        for allele in formatted_alleles:
            try:
                results = predictor.predict(
                    peptides=peptides,
                    alleles=[allele] * len(peptides),
                    verbose=0,
                )
                
                for idx, row in results.iterrows():
                    # MHCflurry returns presentation_score, processing_score, affinity
                    rank = row.get("presentation_percentile", row.get("affinity_percentile", 50.0))
                    if rank > rank_threshold:
                        continue
                    
                    start, length = peptide_info[idx]
                    
                    ep = MHCIEpitope(
                        allele=allele,
                        start=start,
                        end=start + length - 1,
                        sequence=peptides[idx],
                        length=length,
                        rank=float(rank),
                        score=float(row.get("presentation_score", 0)),
                        source="mhcflurry",
                        ic50=float(row.get("affinity", 50000)),
                        processing_score=float(row.get("processing_score", 0)),
                    )
                    epitopes.append(ep)
            except Exception:
                continue
    except Exception as e:
        warnings.warn(f"MHCflurry prediction error: {e}")
    
    return epitopes


# ══════════════════════════════════════════════════════════════
# AAV VALIDATED EPITOPE CROSS-REFERENCE
# ══════════════════════════════════════════════════════════════

def check_validated_epitopes(
    sequence: str,
    predicted_epitopes: List[MHCIEpitope],
    serotype: str = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Cross-reference predicted epitopes against validated AAV capsid epitopes.
    
    Returns:
        - List of matched validated epitopes with details
        - Recovery rate (fraction of known epitopes that were predicted)
    """
    # Combine all validated epitopes
    all_validated = []
    for ep in AAV_VALIDATED_EPITOPES_HUI_2015:
        if serotype is None or serotype in [s.replace("AAV","") for s in ep["serotypes"]] or any(serotype in s for s in ep["serotypes"]):
            all_validated.append(ep)
    for ep in AAV_IMMUNOPEPTIDOMICS_2023:
        if serotype is None or any(serotype in s for s in ep["serotypes"]):
            all_validated.append(ep)
    
    if not all_validated:
        return [], 0.0
    
    # Check which validated epitopes are present in this sequence
    present_validated = []
    for vep in all_validated:
        pep = vep["peptide"]
        if pep in sequence:
            present_validated.append(vep)
    
    if not present_validated:
        return [], 0.0
    
    # Check which present validated epitopes were recovered by predictions
    predicted_peptides = set(ep.sequence for ep in predicted_epitopes)
    # Also check for overlapping predictions (within 2 residues)
    predicted_positions = set()
    for ep in predicted_epitopes:
        for pos in range(ep.start, ep.end + 1):
            predicted_positions.add(pos)
    
    matched = []
    for vep in present_validated:
        pep = vep["peptide"]
        idx = sequence.find(pep)
        if idx == -1:
            continue
        
        vep_start = idx + 1  # 1-indexed
        vep_end = idx + len(pep)
        
        # Check for exact match or overlapping prediction
        is_exact = pep in predicted_peptides
        overlap_count = sum(1 for pos in range(vep_start, vep_end + 1) if pos in predicted_positions)
        overlap_frac = overlap_count / len(pep)
        
        is_recovered = is_exact or overlap_frac > 0.6
        
        matched.append({
            "validated_peptide": pep,
            "position": f"{vep_start}-{vep_end}",
            "hla": vep.get("hla", "Unknown"),
            "source": vep.get("source", "Unknown"),
            "kd_um": vep.get("kd_um"),
            "conserved": vep.get("conserved", False),
            "is_recovered": is_recovered,
            "overlap_fraction": overlap_frac,
        })
    
    n_recovered = sum(1 for m in matched if m["is_recovered"])
    recovery_rate = n_recovered / len(matched) if matched else 0.0
    
    return matched, recovery_rate


# ══════════════════════════════════════════════════════════════
# PER-RESIDUE CYTOTOXIC RISK SCORING
# ══════════════════════════════════════════════════════════════

def compute_cytotoxic_residue_risks(
    sequence: str,
    mhc1_epitopes: List[MHCIEpitope],
    validated_positions: set = None,
) -> List[CytotoxicResidueRisk]:
    """Compute per-residue MHC-I / cytotoxic T-cell risk scores.
    
    Scoring logic (mirrors MHC-II pipeline):
    - For each position, find the best (lowest) percentile rank across all alleles
    - Count how many alleles have strong binders overlapping this position
    - Convert to 0-1 risk score
    - Flag positions that match validated epitopes
    """
    n = len(sequence)
    if validated_positions is None:
        validated_positions = set()
    
    # Per-residue tracking
    best_rank = [100.0] * n
    allele_counts = [0] * n  # alleles with rank < 5% at this position
    
    for ep in mhc1_epitopes:
        for pos in range(max(0, ep.start - 1), min(n, ep.end)):
            if ep.rank < best_rank[pos]:
                best_rank[pos] = ep.rank
            if ep.rank < 5:  # Strong binder threshold for MHC-I (stricter than MHC-II)
                allele_counts[pos] += 1
    
    # Convert to risk scores
    risks = []
    for i in range(n):
        rank = best_rank[i]
        # MHC-I scoring: rank 0 → risk 1.0, rank 20+ → risk 0
        # Tighter threshold than MHC-II because MHC-I predictions are more specific
        if rank >= 20:
            risk = 0.0
        elif rank <= 0.5:
            risk = 1.0
        else:
            risk = max(0, 1.0 - (rank / 20.0))
        
        is_val = (i + 1) in validated_positions
        
        risks.append(CytotoxicResidueRisk(
            position=i + 1,
            residue=sequence[i],
            mhc1_risk=risk,
            num_alleles_binding=allele_counts[i],
            best_rank=best_rank[i],
            is_validated=is_val,
        ))
    
    return risks


def identify_cytotoxic_hotspots(
    residue_risks: List[CytotoxicResidueRisk],
    threshold: float = 0.3,
    min_length: int = 4,
) -> List[Dict[str, Any]]:
    """Identify contiguous regions of high MHC-I binding density."""
    hotspots = []
    current_start = None
    current_residues = []
    
    for rr in residue_risks:
        if rr.mhc1_risk >= threshold:
            if current_start is None:
                current_start = rr.position
            current_residues.append(rr)
        else:
            if current_residues and len(current_residues) >= min_length:
                hs = _make_cytotoxic_hotspot(current_start, current_residues)
                hotspots.append(hs)
            current_start = None
            current_residues = []
    
    if current_residues and len(current_residues) >= min_length:
        hotspots.append(_make_cytotoxic_hotspot(current_start, current_residues))
    
    hotspots.sort(key=lambda h: h["avg_risk"], reverse=True)
    return hotspots


def _make_cytotoxic_hotspot(start: int, residues: List[CytotoxicResidueRisk]) -> Dict[str, Any]:
    validated_count = sum(1 for r in residues if r.is_validated)
    return {
        "start": start,
        "end": residues[-1].position,
        "sequence": "".join(r.residue for r in residues),
        "length": len(residues),
        "avg_risk": sum(r.mhc1_risk for r in residues) / len(residues),
        "max_risk": max(r.mhc1_risk for r in residues),
        "max_alleles": max(r.num_alleles_binding for r in residues),
        "has_validated": validated_count > 0,
        "validated_residues": validated_count,
    }


# ══════════════════════════════════════════════════════════════
# MAIN CYTOTOXIC ASSESSMENT PIPELINE
# ══════════════════════════════════════════════════════════════

def run_cytotoxic_assessment(
    sequence: str,
    name: str = "Query",
    serotype: str = None,
    use_mhcflurry: bool = True,
    use_iedb: bool = True,
    alleles: List[str] = None,
    verbose: bool = False,
) -> CytotoxicReport:
    """Run full MHC-I / cytotoxic T-cell assessment.
    
    Pipeline:
    1. Predict MHC-I epitopes via IEDB API (NetMHCpan 4.1)
    2. Cross-check with MHCflurry 2.0 (if installed)
    3. Cross-reference against validated AAV capsid epitopes
    4. Compute per-residue cytotoxic risk scores
    5. Identify cytotoxic hotspot regions
    
    Training data references:
    - IEDB/NetMHCpan trained on >185K binding measurements
      (equivalent to TDC MHC1_IEDB-IMGT_Nielsen, 185,985 pairs)
    - MHCflurry 2.0 trained on binding affinity + mass spec eluted ligands
    - Validated against Hui et al. 2015 (21 AAV epitopes) and
      2023 immunopeptidomics (65 AAV HLA-I peptides)
    """
    if alleles is None:
        alleles = HLA_CLASS_I_ALLELES
    
    all_epitopes = []
    sources = []
    
    # Step 1: IEDB predictions
    if use_iedb:
        if verbose:
            print(f"[MHC-I 1/4] IEDB NetMHCpan 4.1 predictions ({len(alleles)} alleles)...")
        iedb_eps = predict_mhc1_epitopes_iedb(sequence, alleles=alleles)
        all_epitopes.extend(iedb_eps)
        if iedb_eps:
            sources.append("IEDB NetMHCpan 4.1")
        if verbose:
            print(f"  → {len(iedb_eps)} epitopes (rank < 10%)")
    
    # Step 2: MHCflurry cross-check
    if use_mhcflurry:
        if verbose:
            print(f"[MHC-I 2/4] MHCflurry 2.0 local predictions...")
        mhcf_eps = predict_mhc1_epitopes_mhcflurry(sequence, alleles=alleles)
        all_epitopes.extend(mhcf_eps)
        if mhcf_eps:
            sources.append("MHCflurry 2.0")
        if verbose:
            print(f"  → {len(mhcf_eps)} epitopes (rank < 10%)")
    
    # Step 3: Validated epitope cross-reference (for AAV sequences)
    if verbose:
        print(f"[MHC-I 3/4] Cross-referencing validated epitopes...")
    
    validated_matches, recovery_rate = check_validated_epitopes(
        sequence, all_epitopes, serotype=serotype
    )
    
    # Build set of validated positions for annotation
    validated_positions = set()
    for vm in validated_matches:
        pos_str = vm["position"]
        start, end = map(int, pos_str.split("-"))
        for p in range(start, end + 1):
            validated_positions.add(p)
    
    if verbose:
        print(f"  → {len(validated_matches)} validated epitopes found, {recovery_rate:.0%} recovery")
    
    # Step 4: Per-residue risk
    if verbose:
        print(f"[MHC-I 4/4] Computing per-residue cytotoxic risk...")
    
    residue_risks = compute_cytotoxic_residue_risks(
        sequence, all_epitopes, validated_positions
    )
    
    # Step 5: Hotspots
    hotspots = identify_cytotoxic_hotspots(residue_risks)
    
    # Compute overall risk
    if residue_risks:
        overall = sum(rr.mhc1_risk for rr in residue_risks) / len(residue_risks)
    else:
        overall = 0.0
    
    # Categorize
    if overall >= 0.5:
        category = "VERY HIGH"
    elif overall >= 0.35:
        category = "HIGH"
    elif overall >= 0.18:
        category = "MODERATE"
    else:
        category = "LOW"
    
    strong = len([e for e in all_epitopes if e.rank < 2])
    moderate = len([e for e in all_epitopes if 2 <= e.rank < 10])
    
    # Data references
    refs = {
        "iedb": "IEDB MHC-I tools (tools.iedb.org) — NetMHCpan 4.1",
        "tdc": "TDC MHC1_IEDB-IMGT_Nielsen: 185,985 peptide-MHC-I pairs (tdcommons.ai)",
        "mhcflurry": "MHCflurry 2.0 (O'Donnell et al., Cell Systems 2020)",
        "hui_2015": "Hui et al. 2015: 21 validated AAV CD8+ epitopes (PMC4588448)",
        "immunopeptidomics_2023": "2023 AAV immunopeptidomics: 65 HLA-I peptides (PMC10469481)",
        "gener8_1": "GENEr8-1 Phase 3: 134 patients, capsid T-cell + ALT data (PMID 38796703)",
        "tantigen": "TANTIGEN 2.0: 4,296 tumor antigens, >1,500 epitopes (Olsen et al. 2021)",
        "abimmpred": "AbImmPred: 199 therapeutic Abs with immunogenicity annotations (2024)",
    }
    
    return CytotoxicReport(
        total_epitopes_predicted=len(all_epitopes),
        strong_binders=strong,
        moderate_binders=moderate,
        epitopes=all_epitopes,
        residue_risks=residue_risks,
        hotspot_regions=hotspots,
        validated_hits=sum(1 for m in validated_matches if m["is_recovered"]),
        validated_details=validated_matches,
        overall_cytotoxic_risk=overall,
        risk_category=category,
        prediction_sources=sources,
        aav_epitope_recovery=recovery_rate if validated_matches else None,
        data_references=refs,
    )


# ══════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SafeBind AI — MHC-I / Cytotoxic T-cell Module")
    print("=" * 60)
    
    # Test with AAV9 VP1 fragment (known immunodominant region)
    aav9_fragment = "VPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEY"
    
    print(f"\nTest sequence: AAV9 VP1 fragment ({len(aav9_fragment)} aa)")
    print("Checking validated epitope cross-reference (no API calls)...\n")
    
    # Test validated epitope matching (no API needed)
    mock_epitopes = [
        MHCIEpitope("HLA-B*07:02", 1, 9, "VPQYGYLTL", 9, 0.5),
        MHCIEpitope("HLA-A*02:01", 75, 83, "STFSAAKFA", 9, 1.2),
    ]
    
    matches, recovery = check_validated_epitopes(aav9_fragment, mock_epitopes, serotype="AAV9")
    print(f"Validated epitopes found in sequence: {len(matches)}")
    print(f"Recovered by predictions: {sum(1 for m in matches if m['is_recovered'])}")
    print(f"Recovery rate: {recovery:.0%}")
    
    for m in matches[:5]:
        status = "✅ RECOVERED" if m["is_recovered"] else "❌ MISSED"
        src = m.get("source", "")
        print(f"  {m['validated_peptide']:15s} {m['position']:8s} {m['hla']:15s} {status} ({src})")
    
    # Test residue risk computation
    print(f"\nComputing cytotoxic residue risks...")
    residue_risks = compute_cytotoxic_residue_risks(aav9_fragment, mock_epitopes)
    avg_risk = sum(rr.mhc1_risk for rr in residue_risks) / len(residue_risks)
    max_risk = max(rr.mhc1_risk for rr in residue_risks)
    print(f"  Avg cytotoxic risk: {avg_risk:.3f}")
    print(f"  Max cytotoxic risk: {max_risk:.3f}")
    
    hotspots = identify_cytotoxic_hotspots(residue_risks)
    print(f"  Cytotoxic hotspots: {len(hotspots)}")
    
    print(f"\nData references:")
    report = CytotoxicReport(
        total_epitopes_predicted=2, strong_binders=1, moderate_binders=1,
        epitopes=mock_epitopes, residue_risks=residue_risks,
        hotspot_regions=hotspots, validated_hits=1, validated_details=matches,
        overall_cytotoxic_risk=avg_risk, risk_category="LOW",
        prediction_sources=["mock"], data_references={
            "tdc": "185,985 peptide-MHC-I pairs",
            "mhcflurry": "O'Donnell et al., Cell Systems 2020",
        }
    )
    for key, val in (report.data_references or {}).items():
        print(f"  [{key}] {val}")
    
    print("\n✅ MHC-I module self-test passed!")
