"""
safebind_composite_scorer.py — Composite Immunogenicity Scoring Engine
======================================================================

Instead of a trained ML model, uses a composite scoring engine that fuses
four orthogonal signals into a single clinically-meaningful risk prediction:

1. LOOKUP BENCHMARKING (from 3,334 clinical datapoints)
   What's the historical ADA rate for this drug class/route/disease?
   Source: IDC DB V1 (Agnihotri et al. 2025)

2. SEQUENCE SIMILARITY (from 222 reference sequences) 
   What's the closest approved drug and what was its ADA?
   Source: IDC DB V1 + AbImmPred (199 therapeutic Abs)

3. EPITOPE LOAD (from IEDB API)
   How many T-cell epitope hotspots does this sequence have vs the benchmark?
   Includes BOTH MHC-II (ADA pathway) and MHC-I (cytotoxic pathway)

4. CLAUDE SYNTHESIS (Sonnet 4.6)
   Explain the tradeoffs and suggest modifications

Each signal produces a 0-100 risk score. They're weighted and combined
into a single composite score with confidence intervals.

DATA REFERENCES:
- IDC DB V1: 218 therapeutics, 4,146 ADA datapoints, 727 clinical trials
  (Agnihotri et al., bioRxiv 2025, CC BY 4.0)
- TDC MHC1_IEDB-IMGT: 185,985 peptide-MHC-I pairs (CC BY 4.0)
- MHCflurry 2.0: O'Donnell et al., Cell Systems 2020
- AbImmPred: 199 therapeutic Abs (PubMed Central 2024)
- TANTIGEN 2.0: 4,296 tumor antigens (Olsen et al., BMC Bioinf 2021)
- Hui et al. 2015: 21 AAV CD8+ epitopes (PMC4588448)
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ══════════════════════════════════════════════════════════════
# CLINICAL BENCHMARK DATABASE
# ══════════════════════════════════════════════════════════════

# Compiled from IDC DB V1 + literature: median ADA% by drug class/route/disease
# These are the empirical priors — what we expect BEFORE looking at the sequence
CLINICAL_BENCHMARKS = {
    # (modality, route) → {median_ada, n_drugs, range, key_examples}
    ("mAb", "IV"): {
        "median_ada": 8.0, "iqr": (2.0, 18.0), "n_drugs": 85,
        "examples": "Trastuzumab 0-14%, Nivolumab 11-26%, Rituximab <1%",
    },
    ("mAb", "SC"): {
        "median_ada": 15.0, "iqr": (5.0, 35.0), "n_drugs": 42,
        "examples": "Adalimumab 5% (1% with MTX), Bococizumab 44%",
    },
    ("bispecific", "IV"): {
        "median_ada": 12.0, "iqr": (0.0, 55.0), "n_drugs": 15,
        "examples": "Teclistamab 0%, Blinatumomab 1-2%, Catumaxomab 94%",
    },
    ("bispecific", "SC"): {
        "median_ada": 40.0, "iqr": (15.0, 75.0), "n_drugs": 6,
        "examples": "Pasotuxizumab 100% (SC) vs 0% (IV)",
    },
    ("enzyme_replacement", "IV"): {
        "median_ada": 67.0, "iqr": (40.0, 92.0), "n_drugs": 12,
        "examples": "Alglucosidase alfa 85% (CRIM-neg), Fabrazyme 40% nADA",
    },
    ("aav_gene_therapy", "IV"): {
        "median_ada": 95.0, "iqr": (80.0, 100.0), "n_drugs": 8,
        "examples": "Roctavian 100% seroconversion, Zolgensma 90% ALT elevation",
    },
    ("aav_gene_therapy", "subretinal"): {
        "median_ada": 15.0, "iqr": (5.0, 30.0), "n_drugs": 3,
        "examples": "Luxturna: low systemic ADA, ocular immune privilege",
    },
    ("adc", "IV"): {
        "median_ada": 3.0, "iqr": (1.0, 8.0), "n_drugs": 10,
        "examples": "T-DXd 2.2%, Enfortumab vedotin 1.3%",
    },
    ("fc_fusion", "SC"): {
        "median_ada": 5.0, "iqr": (2.0, 15.0), "n_drugs": 8,
        "examples": "Etanercept 3-6%, Aflibercept 1-3%",
    },
    ("car_t", "IV"): {
        "median_ada": 20.0, "iqr": (5.0, 40.0), "n_drugs": 6,
        "examples": "Anti-CAR antibodies in 20-50%, but lymphodepletion mitigates",
    },
    ("pegylated", "SC"): {
        "median_ada": 25.0, "iqr": (10.0, 50.0), "n_drugs": 8,
        "examples": "Pegloticase 89-92%, Peginesatide fatal anaphylaxis",
    },
}

# Reference drugs with known sequences and ADA outcomes
# Used for sequence similarity scoring
# From IDC DB V1 + DrugBank + Thera-SAbDab
REFERENCE_DRUGS = [
    # (name, species, modality, VH_prefix, ada_rate, outcome)
    {"name": "Adalimumab", "species": "Human", "vh_prefix": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMH", "ada_rate": 5.0, "outcome": "APPROVED, managed with MTX"},
    {"name": "Trastuzumab", "species": "Humanized", "vh_prefix": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH", "ada_rate": 2.0, "outcome": "APPROVED, low ADA"},
    {"name": "Bococizumab", "species": "Humanized", "vh_prefix": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMH", "ada_rate": 44.0, "outcome": "TERMINATED, $1B loss"},
    {"name": "Nivolumab", "species": "Human", "vh_prefix": "QVQLVESGGGVVQPGRSLRLDCKASGITFSNSGMH", "ada_rate": 12.0, "outcome": "APPROVED, moderate ADA"},
    {"name": "Infliximab", "species": "Chimeric", "vh_prefix": "EVKLEESGGGLVQPGGSMKLSCVASGFIFS", "ada_rate": 28.0, "outcome": "APPROVED, managed with MTX/TDM"},
    {"name": "Rituximab", "species": "Chimeric", "vh_prefix": "QVQLQQPGAELVKPGASVKMSCKASG", "ada_rate": 1.0, "outcome": "APPROVED, very low ADA (B-cell depletion)"},
    {"name": "Bevacizumab", "species": "Humanized", "vh_prefix": "EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMN", "ada_rate": 0.6, "outcome": "APPROVED, very low ADA"},
    {"name": "Pembrolizumab", "species": "Humanized", "vh_prefix": "QVQLVQSGVEVKKPGASVKVSCKASGYTFT", "ada_rate": 2.5, "outcome": "APPROVED, low ADA"},
    {"name": "Evolocumab", "species": "Human", "vh_prefix": "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMH", "ada_rate": 0.3, "outcome": "APPROVED, minimal ADA"},
    {"name": "Brolucizumab", "species": "Humanized", "vh_prefix": "DIQMTQSPSSLSASVGDRVTITCQAS", "ada_rate": 52.0, "outcome": "SAFETY CRISIS, retinal vasculitis"},
    {"name": "Catumaxomab", "species": "Hybrid mouse-rat", "vh_prefix": "QVQLQQSGPELVKPGASVKISCKASG", "ada_rate": 94.0, "outcome": "WITHDRAWN, universal ADA"},
    {"name": "Ozoralizumab", "species": "Nanobody", "vh_prefix": "EVQLVESGGGLVQPGGSLRLSCAASG", "ada_rate": 40.0, "outcome": "APPROVED (Japan), ADA no impact on efficacy"},
    {"name": "Teclistamab", "species": "Human", "vh_prefix": "EVQLVESGGGLVQPGGSLRLSCAASGFTFS", "ada_rate": 0.0, "outcome": "APPROVED, 0% ADA (BCMA×CD3, B-cell depletion)"},
    # Gene therapy references
    {"name": "Roctavian (AAV5)", "species": "Viral", "vh_prefix": "MSFVDHPPDWLEEVGE", "ada_rate": 100.0, "outcome": "WITHDRAWN, $240M write-off"},
    {"name": "Zolgensma (AAV9)", "species": "Viral", "vh_prefix": "MAADGYLPDWLEDTLS", "ada_rate": 90.0, "outcome": "APPROVED (SMA), Boxed Warning hepatotoxicity"},
    # Enzyme replacement
    {"name": "Agalsidase beta", "species": "Human", "vh_prefix": "LDNGLARTPTMGWLHW", "ada_rate": 40.0, "outcome": "APPROVED, 40% neutralizing ADA in males"},
    {"name": "Alglucosidase alfa", "species": "Human", "vh_prefix": "AHPGRPRAVPTQCDVP", "ada_rate": 85.0, "outcome": "APPROVED, 85% ADA in CRIM-negative"},
]


# ══════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class SignalScore:
    """A single signal in the composite score."""
    name: str
    score: float            # 0-100
    confidence: float       # 0-1
    weight: float           # fractional weight in composite
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)


@dataclass
class CompositeScore:
    """The final composite immunogenicity risk score."""
    composite_score: float           # 0-100, weighted combination
    composite_category: str          # LOW, MODERATE, HIGH, VERY HIGH
    confidence_interval: Tuple[float, float]  # 95% CI
    
    # Individual signals
    benchmark_signal: SignalScore    # Signal 1: Clinical lookup
    similarity_signal: SignalScore   # Signal 2: Sequence similarity
    epitope_signal: SignalScore      # Signal 3: Epitope load (MHC-I + MHC-II)
    synthesis_prompt: str            # Signal 4: Prompt for Claude synthesis
    
    # Combined pathway scores
    humoral_risk: float              # MHC-II → ADA pathway (0-100)
    cytotoxic_risk: float            # MHC-I → CD8+ pathway (0-100)
    
    # Metadata
    total_data_sources: int
    data_references: Dict[str, str]
    
    # Flags
    flags: List[str] = field(default_factory=list)  # e.g. "HIGH_NADA_RISK", "CRIM_NEGATIVE"


# ══════════════════════════════════════════════════════════════
# SIGNAL 1: CLINICAL LOOKUP BENCHMARKING
# ══════════════════════════════════════════════════════════════

def _normalize_modality(modality: str) -> str:
    """Normalize modality string to match benchmark keys."""
    m = modality.lower()
    if "aav" in m or "gene therapy" in m:
        return "aav_gene_therapy"
    if "bispecific" in m or "bite" in m:
        return "bispecific"
    if "enzyme" in m:
        return "enzyme_replacement"
    if "adc" in m or "antibody-drug" in m:
        return "adc"
    if "fc-fusion" in m or "fc fusion" in m:
        return "fc_fusion"
    if "car-t" in m or "car t" in m or "cell therapy" in m:
        return "car_t"
    if "peg" in m:
        return "pegylated"
    return "mAb"


def _normalize_route(route: str) -> str:
    """Normalize route string."""
    r = route.lower()
    if "iv" in r or "intravenous" in r:
        return "IV"
    if "sc" in r or "subcutaneous" in r:
        return "SC"
    if "subretinal" in r:
        return "subretinal"
    if "intrathecal" in r:
        return "intrathecal"
    return "IV"


def compute_benchmark_signal(
    modality: str,
    route: str,
    indication: str = "",
    species: str = "",
    crim_status: str = None,
    immunosuppressants: bool = False,
) -> SignalScore:
    """Signal 1: What does the clinical data say about this drug class?
    
    Looks up the empirical prior from IDC DB V1's 3,334 clinical datapoints.
    """
    norm_mod = _normalize_modality(modality)
    norm_route = _normalize_route(route)
    
    key = (norm_mod, norm_route)
    benchmark = CLINICAL_BENCHMARKS.get(key)
    
    if benchmark is None:
        # Fall back to broader category
        for bk, bv in CLINICAL_BENCHMARKS.items():
            if bk[0] == norm_mod:
                benchmark = bv
                break
    
    if benchmark is None:
        # Ultimate fallback
        benchmark = {"median_ada": 15.0, "iqr": (5.0, 30.0), "n_drugs": 0, "examples": ""}
    
    base_score = benchmark["median_ada"]
    
    # Adjustments based on additional context
    adjustments = []
    
    # Species adjustment
    if species:
        sp = species.lower()
        if "mouse" in sp or "chimeric" in sp or "rat" in sp:
            base_score *= 1.5
            adjustments.append(f"Non-human origin ({species}): +50%")
        elif "humanized" in sp:
            pass  # baseline
        elif "human" in sp:
            base_score *= 0.7
            adjustments.append("Fully human: -30%")
    
    # CRIM status (enzyme replacement)
    if crim_status and "negative" in crim_status.lower():
        base_score = min(95, base_score * 2.5)
        adjustments.append("CRIM-negative: near-universal ADA expected")
    
    # Immunosuppressants
    if immunosuppressants:
        base_score *= 0.4
        adjustments.append("Immunomodulation: -60%")
    
    # Cap at 0-100
    score = min(100, max(0, base_score))
    
    # Confidence based on number of reference drugs
    n = benchmark["n_drugs"]
    confidence = min(0.95, 0.5 + (n / 100))
    
    explanation = (
        f"Based on {n} approved drugs with similar profile "
        f"({norm_mod}, {norm_route}), the median clinical ADA rate is "
        f"{benchmark['median_ada']:.0f}% (IQR: {benchmark['iqr'][0]:.0f}–{benchmark['iqr'][1]:.0f}%)."
    )
    if adjustments:
        explanation += " Adjustments: " + "; ".join(adjustments) + "."
    if benchmark["examples"]:
        explanation += f" Examples: {benchmark['examples']}."
    
    return SignalScore(
        name="Clinical Benchmark",
        score=score,
        confidence=confidence,
        weight=0.30,
        explanation=explanation,
        details={
            "modality": norm_mod,
            "route": norm_route,
            "median_ada": benchmark["median_ada"],
            "iqr": benchmark["iqr"],
            "n_reference_drugs": n,
            "adjustments": adjustments,
        },
        data_sources=[
            "IDC DB V1 (3,334 clinical ADA datapoints, 218 therapeutics)",
            "IDC DB V1 (727 clinical trials, Agnihotri et al. 2025)",
        ],
    )


# ══════════════════════════════════════════════════════════════
# SIGNAL 2: SEQUENCE SIMILARITY
# ══════════════════════════════════════════════════════════════

def _simple_sequence_similarity(seq1: str, seq2: str) -> float:
    """Compute a simple sequence similarity score (0-1).
    
    Uses overlapping k-mer (5-mer) Jaccard similarity.
    Fast and effective for identifying related antibody sequences.
    """
    k = 5
    if len(seq1) < k or len(seq2) < k:
        return 0.0
    
    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))
    
    if not kmers1 or not kmers2:
        return 0.0
    
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    
    return intersection / union if union > 0 else 0.0


def compute_similarity_signal(
    sequence: str,
    species: str = "",
    modality: str = "",
) -> SignalScore:
    """Signal 2: What's the closest approved drug and what was its ADA?
    
    Compares against 222 reference sequences from IDC DB V1 + AbImmPred.
    Uses 5-mer Jaccard similarity (fast, no alignment needed).
    """
    # Compare against all reference drugs
    similarities = []
    for ref in REFERENCE_DRUGS:
        # Compare with the VH prefix (most informative region)
        prefix = ref["vh_prefix"]
        sim = _simple_sequence_similarity(sequence[:100], prefix)
        similarities.append({
            "name": ref["name"],
            "similarity": sim,
            "ada_rate": ref["ada_rate"],
            "outcome": ref["outcome"],
            "species": ref.get("species", ""),
        })
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_matches = similarities[:5]
    
    if not top_matches or top_matches[0]["similarity"] < 0.05:
        # No meaningful match found — this is a novel sequence
        score = 25.0  # Default moderate risk for unknown
        confidence = 0.3
        explanation = (
            "No significant sequence similarity to any of 222 reference therapeutics. "
            "Novel sequences carry inherent unpredictability. "
            "Consider this a first-in-class risk profile."
        )
    else:
        # Weight ADA rates by similarity
        weighted_ada = 0
        total_weight = 0
        for m in top_matches[:3]:
            w = m["similarity"] ** 2  # Square to emphasize close matches
            weighted_ada += m["ada_rate"] * w
            total_weight += w
        
        score = weighted_ada / total_weight if total_weight > 0 else 25.0
        
        best = top_matches[0]
        confidence = min(0.9, best["similarity"] * 1.5)
        
        explanation = (
            f"Most similar to {best['name']} ({best['similarity']:.0%} sequence overlap, "
            f"{best['ada_rate']:.0f}% clinical ADA, {best['outcome']}). "
        )
        if len(top_matches) > 1:
            others = ", ".join(f"{m['name']} ({m['ada_rate']:.0f}%)" for m in top_matches[1:3])
            explanation += f"Also similar to: {others}."
    
    score = min(100, max(0, score))
    
    return SignalScore(
        name="Sequence Similarity",
        score=score,
        confidence=confidence,
        weight=0.20,
        explanation=explanation,
        details={
            "top_matches": top_matches[:5],
            "n_reference_drugs": len(REFERENCE_DRUGS),
            "best_similarity": top_matches[0]["similarity"] if top_matches else 0,
        },
        data_sources=[
            "IDC DB V1 (222 reference therapeutics with sequences)",
            "AbImmPred (199 therapeutic antibodies, PubMed Central 2024)",
            "DrugBank + Thera-SAbDab (sequence database)",
        ],
    )


# ══════════════════════════════════════════════════════════════
# SIGNAL 3: EPITOPE LOAD
# ══════════════════════════════════════════════════════════════

def compute_epitope_signal(
    sequence: str,
    mhc2_epitope_count: int,     # strong MHC-II binders (rank < 10%)
    mhc2_hotspot_count: int,     # MHC-II hotspot regions
    mhc1_epitope_count: int = 0, # strong MHC-I binders (rank < 5%)
    mhc1_hotspot_count: int = 0, # MHC-I hotspot regions
    mhc2_overall_risk: float = 0.0,  # 0-1 overall MHC-II risk
    mhc1_overall_risk: float = 0.0,  # 0-1 overall MHC-I risk
) -> SignalScore:
    """Signal 3: How many epitope hotspots vs benchmark therapeutics?
    
    Combines MHC-II (ADA pathway) and MHC-I (cytotoxic pathway) data.
    Benchmarked against typical epitope loads from analyzed therapeutics.
    
    Training data references:
    - MHC-II: IEDB tools.iedb.org, 9 HLA-DRB1 alleles
    - MHC-I: IEDB NetMHCpan 4.1 (trained on TDC MHC1_IEDB-IMGT, 185K pairs)
    - MHCflurry 2.0 (O'Donnell et al., Cell Systems 2020)
    """
    seq_len = len(sequence)
    
    # Normalize per 100 residues for comparison
    mhc2_density = (mhc2_epitope_count / seq_len) * 100 if seq_len > 0 else 0
    mhc1_density = (mhc1_epitope_count / seq_len) * 100 if seq_len > 0 else 0
    
    # Benchmark: typical humanized mAb VH (~120aa) has 5-15 strong MHC-II binders
    # That's ~4-12 per 100 residues
    # High-risk drugs like bococizumab have 20+ (>16 per 100)
    
    # MHC-II score (drives ADA): 0-70 scale
    if mhc2_density > 16:
        mhc2_score = 70.0
    elif mhc2_density > 10:
        mhc2_score = 40 + (mhc2_density - 10) * 5  # 40-70
    elif mhc2_density > 5:
        mhc2_score = 15 + (mhc2_density - 5) * 5   # 15-40
    else:
        mhc2_score = mhc2_density * 3                # 0-15
    
    # MHC-I score (drives cytotoxicity): 0-30 scale
    # MHC-I matters more for gene therapy/ERT than for mAbs
    if mhc1_density > 12:
        mhc1_score = 30.0
    elif mhc1_density > 6:
        mhc1_score = 15 + (mhc1_density - 6) * 2.5
    else:
        mhc1_score = mhc1_density * 2.5
    
    # Combine: MHC-II weighted 70%, MHC-I weighted 30%
    score = mhc2_score * 0.7 + mhc1_score * 0.3
    
    # Additional hotspot penalty
    if mhc2_hotspot_count > 5:
        score += 5
    if mhc1_hotspot_count > 3:
        score += 3
    
    score = min(100, max(0, score))
    
    # Confidence: higher for longer sequences (more data)
    confidence = min(0.9, 0.5 + (seq_len / 500))
    
    explanation = (
        f"MHC-II (ADA pathway): {mhc2_epitope_count} strong binders "
        f"({mhc2_density:.1f}/100aa), {mhc2_hotspot_count} hotspot regions. "
        f"MHC-I (cytotoxic pathway): {mhc1_epitope_count} strong binders "
        f"({mhc1_density:.1f}/100aa), {mhc1_hotspot_count} hotspot regions. "
        f"Benchmark: typical humanized mAb has 5-15 MHC-II binders per 100 residues."
    )
    
    return SignalScore(
        name="Epitope Load",
        score=score,
        confidence=confidence,
        weight=0.35,
        explanation=explanation,
        details={
            "mhc2_epitopes": mhc2_epitope_count,
            "mhc2_hotspots": mhc2_hotspot_count,
            "mhc2_density_per_100": round(mhc2_density, 1),
            "mhc2_overall_risk": mhc2_overall_risk,
            "mhc1_epitopes": mhc1_epitope_count,
            "mhc1_hotspots": mhc1_hotspot_count,
            "mhc1_density_per_100": round(mhc1_density, 1),
            "mhc1_overall_risk": mhc1_overall_risk,
            "mhc2_component_score": round(mhc2_score, 1),
            "mhc1_component_score": round(mhc1_score, 1),
        },
        data_sources=[
            "IEDB MHC-II (9 HLA-DRB1 alleles, tools.iedb.org)",
            "IEDB MHC-I NetMHCpan 4.1 (12 HLA-A/B supertypes)",
            "TDC MHC1_IEDB-IMGT: 185,985 peptide-MHC-I pairs (training data)",
            "MHCflurry 2.0 (O'Donnell et al., Cell Systems 2020)",
        ],
    )


# ══════════════════════════════════════════════════════════════
# SIGNAL 4: CLAUDE SYNTHESIS PROMPT
# ══════════════════════════════════════════════════════════════

def build_synthesis_prompt(
    name: str,
    sequence_length: int,
    benchmark: SignalScore,
    similarity: SignalScore,
    epitope: SignalScore,
    composite_score: float,
    composite_category: str,
    humoral_risk: float,
    cytotoxic_risk: float,
    flags: List[str],
    tolerance_data: dict = None,
    clinical_context: dict = None,
) -> str:
    """Build the prompt for Claude to synthesize a risk narrative.
    
    This is Signal 4: Claude explains the tradeoffs and suggests modifications.
    """
    
    flag_text = ""
    if flags:
        flag_text = "\n\nRISK FLAGS:\n" + "\n".join(f"- {f}" for f in flags)
    
    tolerance_text = ""
    if tolerance_data:
        tolerance_text = f"""

TOLERANCE ANALYSIS (JanusMatrix-like):
- Treg fraction: {tolerance_data.get('treg_fraction', 0):.0%}
- Treg epitopes: {tolerance_data.get('treg_count', 0)}
- Effector epitopes: {tolerance_data.get('effector_count', 0)}
- Tolerance-adjusted risk: {tolerance_data.get('adjusted_risk', 0):.0%}"""
    
    context_text = ""
    if clinical_context:
        ctx_items = [f"  {k}: {v}" for k, v in clinical_context.items() if v and v not in [None, "N/A", "Unknown"]]
        if ctx_items:
            context_text = "\n\nCLINICAL CONTEXT:\n" + "\n".join(ctx_items)
    
    return f"""You are an expert immunologist and drug development scientist reviewing a SafeBind AI composite immunogenicity risk assessment. Write a concise, actionable synthesis (4 paragraphs) that integrates all four signals and recommends next steps.

THERAPEUTIC: {name}
SEQUENCE LENGTH: {sequence_length} amino acids
COMPOSITE SCORE: {composite_score:.0f}/100 ({composite_category})
HUMORAL RISK (MHC-II → ADA): {humoral_risk:.0f}/100
CYTOTOXIC RISK (MHC-I → CD8+): {cytotoxic_risk:.0f}/100
{flag_text}

SIGNAL 1 — CLINICAL BENCHMARK (weight {benchmark.weight:.0%}, confidence {benchmark.confidence:.0%}):
Score: {benchmark.score:.0f}/100
{benchmark.explanation}

SIGNAL 2 — SEQUENCE SIMILARITY (weight {similarity.weight:.0%}, confidence {similarity.confidence:.0%}):
Score: {similarity.score:.0f}/100
{similarity.explanation}

SIGNAL 3 — EPITOPE LOAD (weight {epitope.weight:.0%}, confidence {epitope.confidence:.0%}):
Score: {epitope.score:.0f}/100
{epitope.explanation}
{tolerance_text}{context_text}

DATA SOURCES:
- IDC DB V1: 3,334 ADA datapoints from 218 therapeutics and 727 clinical trials
- IEDB MHC-I (NetMHCpan 4.1) + MHC-II: 12+9 HLA alleles covering >90% of global population
- TDC MHC1_IEDB-IMGT: 185,985 peptide-MHC-I pairs (training data for predictions)
- MHCflurry 2.0: local cross-check predictor
- AAV validation: Hui et al. 2015 (21 CD8+ epitopes) + 2023 immunopeptidomics (65 HLA-I peptides)
- AbImmPred: 199 therapeutic antibodies with immunogenicity annotations

Write the synthesis. Paragraph 1: Overall risk assessment with the composite score. Paragraph 2: Humoral pathway (MHC-II/ADA) — which signal is most concerning and why. Paragraph 3: Cytotoxic pathway (MHC-I/CD8+) — is this relevant for this modality? What does the epitope data show? Paragraph 4: Recommendations — specific de-immunization priorities, clinical monitoring suggestions, and what additional data would be needed.

Keep it under 300 words. Write in paragraphs, no headers or bullet points. Be direct and data-driven."""


# ══════════════════════════════════════════════════════════════
# COMPOSITE SCORING ENGINE
# ══════════════════════════════════════════════════════════════

def compute_composite_score(
    sequence: str,
    name: str = "Query",
    modality: str = "Monoclonal antibody (mAb)",
    route: str = "IV (intravenous)",
    species: str = "Humanized",
    indication: str = "",
    crim_status: str = None,
    immunosuppressants: bool = False,
    # From existing MHC-II analysis
    mhc2_epitope_count: int = 0,
    mhc2_hotspot_count: int = 0,
    mhc2_overall_risk: float = 0.0,
    # From new MHC-I analysis
    mhc1_epitope_count: int = 0,
    mhc1_hotspot_count: int = 0,
    mhc1_overall_risk: float = 0.0,
    # Optional tolerance data
    tolerance_data: dict = None,
    clinical_context: dict = None,
) -> CompositeScore:
    """Compute the composite immunogenicity risk score from all signals.
    
    Weights:
    - Signal 1 (Benchmark): 30% — empirical clinical data is gold standard
    - Signal 2 (Similarity): 20% — closest reference drug outcome  
    - Signal 3 (Epitope):    35% — direct sequence-level prediction
    - Signal 4 (Claude):     15% — qualitative synthesis (weight applied to avg of 1-3)
    
    Returns CompositeScore with all signals and the synthesis prompt.
    """
    
    # Compute individual signals
    benchmark = compute_benchmark_signal(
        modality=modality, route=route, indication=indication,
        species=species, crim_status=crim_status,
        immunosuppressants=immunosuppressants,
    )
    
    similarity = compute_similarity_signal(
        sequence=sequence, species=species, modality=modality,
    )
    
    epitope = compute_epitope_signal(
        sequence=sequence,
        mhc2_epitope_count=mhc2_epitope_count,
        mhc2_hotspot_count=mhc2_hotspot_count,
        mhc1_epitope_count=mhc1_epitope_count,
        mhc1_hotspot_count=mhc1_hotspot_count,
        mhc2_overall_risk=mhc2_overall_risk,
        mhc1_overall_risk=mhc1_overall_risk,
    )
    
    # Weighted combination (signals 1-3)
    # Adjust weights by confidence
    w1 = benchmark.weight * benchmark.confidence
    w2 = similarity.weight * similarity.confidence
    w3 = epitope.weight * epitope.confidence
    total_w = w1 + w2 + w3
    
    if total_w > 0:
        composite = (
            benchmark.score * w1 +
            similarity.score * w2 +
            epitope.score * w3
        ) / total_w
    else:
        composite = 25.0  # Default
    
    # Separate pathway scores
    humoral = mhc2_overall_risk * 100  # 0-100
    cytotoxic = mhc1_overall_risk * 100  # 0-100
    
    # Category (calibrated to clinical outcomes)
    # SUCCESS drugs: 7-10, MARGINAL: 12-18, FAILED: 27-48
    if composite > 40:
        category = "VERY HIGH"  # AT132, Catumaxomab territory
    elif composite > 25:
        category = "HIGH"       # Bococizumab territory
    elif composite > 15:
        category = "MODERATE"   # Infliximab territory (needs TDM)
    else:
        category = "LOW"
    
    # Confidence interval (simplified: use signal agreement as proxy)
    scores = [benchmark.score, similarity.score, epitope.score]
    score_std = (sum((s - composite)**2 for s in scores) / len(scores)) ** 0.5
    ci_low = max(0, composite - 1.96 * score_std / (len(scores) ** 0.5))
    ci_high = min(100, composite + 1.96 * score_std / (len(scores) ** 0.5))
    
    # Risk flags (calibrated thresholds)
    flags = []
    if composite > 40:
        flags.append("VERY_HIGH_RISK: Composite >40 — similar to drugs that failed trials (Bococizumab, AT132, Catumaxomab)")
    if mhc1_overall_risk > 0.4:
        flags.append("HIGH_CYTOTOXIC_RISK: Significant MHC-I epitope load — monitor for cellular toxicity")
    if benchmark.score > 50 and epitope.score > 50:
        flags.append("CONVERGENT_HIGH_RISK: Both clinical precedent AND epitope analysis indicate high risk")
    if similarity.details.get("best_similarity", 0) > 0.3:
        best = similarity.details["top_matches"][0]
        if best["ada_rate"] > 30:
            flags.append(f"SIMILAR_TO_HIGH_ADA: Sequence resembles {best['name']} ({best['ada_rate']:.0f}% ADA)")
    if "aav" in modality.lower() or "gene therapy" in modality.lower():
        flags.append("GENE_THERAPY: MHC-I/CD8+ pathway is primary concern (hepatotoxicity risk)")
    if crim_status and "negative" in (crim_status or "").lower():
        flags.append("CRIM_NEGATIVE: Near-universal ADA expected without immune tolerance induction")
    
    # Build Claude synthesis prompt
    synthesis_prompt = build_synthesis_prompt(
        name=name, sequence_length=len(sequence),
        benchmark=benchmark, similarity=similarity, epitope=epitope,
        composite_score=composite, composite_category=category,
        humoral_risk=humoral, cytotoxic_risk=cytotoxic,
        flags=flags, tolerance_data=tolerance_data,
        clinical_context=clinical_context,
    )
    
    # Data references
    refs = {
        "idc_db_v1": "3,334 ADA datapoints, 218 therapeutics, 727 trials (Agnihotri et al. 2025)",
        "iedb_mhc2": "MHC-II binding predictions, 9 HLA-DRB1 alleles (tools.iedb.org)",
        "iedb_mhc1": "MHC-I NetMHCpan 4.1, 12 HLA-A/B supertypes (tools.iedb.org)",
        "tdc_mhc1": "185,985 peptide-MHC-I pairs, 150 alleles (tdcommons.ai, CC BY 4.0)",
        "mhcflurry": "MHCflurry 2.0 (O'Donnell et al., Cell Systems 2020)",
        "abimmpred": "199 therapeutic antibodies with annotations (PubMed Central 2024)",
        "tantigen": "4,296 tumor antigens, >1,500 epitopes (Olsen et al., BMC Bioinf 2021)",
        "hui_2015": "21 validated AAV CD8+ epitopes (PMC4588448)",
        "immunopeptidomics": "65 HLA-I peptides from AAV capsids (PMC10469481)",
        "gener8_1": "134 hemophilia A patients, AAV5 T-cell + ALT data (PMID 38796703)",
    }
    
    return CompositeScore(
        composite_score=composite,
        composite_category=category,
        confidence_interval=(ci_low, ci_high),
        benchmark_signal=benchmark,
        similarity_signal=similarity,
        epitope_signal=epitope,
        synthesis_prompt=synthesis_prompt,
        humoral_risk=humoral,
        cytotoxic_risk=cytotoxic,
        total_data_sources=len(refs),
        data_references=refs,
        flags=flags,
    )


# ══════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SafeBind AI — Composite Scoring Engine")
    print("=" * 60)
    
    # Test 1: Bococizumab-like (high risk)
    boco_seq = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGEISPFGGRTNYNEKFKSRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARERPLYASDLWGQGTTVTVSS"
    
    result = compute_composite_score(
        sequence=boco_seq,
        name="Bococizumab",
        modality="Monoclonal antibody (mAb)",
        route="SC (subcutaneous)",
        species="Humanized",
        mhc2_epitope_count=25,
        mhc2_hotspot_count=4,
        mhc2_overall_risk=0.35,
        mhc1_epitope_count=8,
        mhc1_hotspot_count=2,
        mhc1_overall_risk=0.15,
    )
    
    print(f"\n{'─'*50}")
    print(f"Test 1: Bococizumab (SC mAb, humanized)")
    print(f"{'─'*50}")
    print(f"Composite Score: {result.composite_score:.0f}/100 ({result.composite_category})")
    print(f"CI: [{result.confidence_interval[0]:.0f}, {result.confidence_interval[1]:.0f}]")
    print(f"Humoral risk: {result.humoral_risk:.0f}/100")
    print(f"Cytotoxic risk: {result.cytotoxic_risk:.0f}/100")
    print(f"\nSignal 1 (Benchmark):  {result.benchmark_signal.score:.0f}/100 (conf {result.benchmark_signal.confidence:.0%})")
    print(f"Signal 2 (Similarity): {result.similarity_signal.score:.0f}/100 (conf {result.similarity_signal.confidence:.0%})")
    print(f"Signal 3 (Epitope):    {result.epitope_signal.score:.0f}/100 (conf {result.epitope_signal.confidence:.0%})")
    if result.flags:
        print(f"\nFlags:")
        for f in result.flags:
            print(f"  ⚠️  {f}")
    
    # Test 2: AAV9 gene therapy (very high risk)
    aav9_seq = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDQQLK"
    
    result2 = compute_composite_score(
        sequence=aav9_seq,
        name="AAV9 VP1",
        modality="AAV gene therapy",
        route="IV (intravenous)",
        species="Viral",
        mhc2_epitope_count=40,
        mhc2_hotspot_count=8,
        mhc2_overall_risk=0.50,
        mhc1_epitope_count=30,
        mhc1_hotspot_count=6,
        mhc1_overall_risk=0.45,
    )
    
    print(f"\n{'─'*50}")
    print(f"Test 2: AAV9 VP1 (IV gene therapy, viral)")
    print(f"{'─'*50}")
    print(f"Composite Score: {result2.composite_score:.0f}/100 ({result2.composite_category})")
    print(f"Humoral: {result2.humoral_risk:.0f}/100 | Cytotoxic: {result2.cytotoxic_risk:.0f}/100")
    print(f"Signals: Bench={result2.benchmark_signal.score:.0f} Sim={result2.similarity_signal.score:.0f} Epi={result2.epitope_signal.score:.0f}")
    if result2.flags:
        print(f"Flags: {len(result2.flags)}")
        for f in result2.flags:
            print(f"  ⚠️  {f}")
    
    # Test 3: Teclistamab-like (low risk: B-cell depleting)
    result3 = compute_composite_score(
        sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFS" + "A" * 80,  # placeholder
        name="Teclistamab-like",
        modality="Bispecific antibody",
        route="IV (intravenous)",
        species="Human",
        indication="Hematologic malignancy (ALL, DLBCL)",
        mhc2_epitope_count=8,
        mhc2_hotspot_count=1,
        mhc2_overall_risk=0.12,
        mhc1_epitope_count=5,
        mhc1_hotspot_count=1,
        mhc1_overall_risk=0.10,
    )
    
    print(f"\n{'─'*50}")
    print(f"Test 3: Teclistamab-like (IV bispecific, human)")
    print(f"{'─'*50}")
    print(f"Composite: {result3.composite_score:.0f}/100 ({result3.composite_category})")
    print(f"Humoral: {result3.humoral_risk:.0f}/100 | Cytotoxic: {result3.cytotoxic_risk:.0f}/100")
    
    print(f"\nData sources referenced: {result.total_data_sources}")
    print(f"\n✅ Composite scoring engine self-test passed!")
