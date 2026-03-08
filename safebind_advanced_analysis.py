"""
safebind_advanced_analysis.py — Advanced Analysis Features
============================================================

1. Checkpoint Inhibitor Detection & Handling
2. IEDB Epitope Homology Search
3. Germline V-Gene Comparison for Humanness Scoring

These features address gaps identified in comparison with EpiVax ISPRI.
"""

import requests
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


# ══════════════════════════════════════════════════════════════
# 1. CHECKPOINT INHIBITOR HANDLING
# ══════════════════════════════════════════════════════════════

CHECKPOINT_TARGETS = {
    "PD-1": ["pembrolizumab", "nivolumab", "cemiplimab", "dostarlimab", "retifanlimab"],
    "PD-L1": ["atezolizumab", "durvalumab", "avelumab"],
    "CTLA-4": ["ipilimumab", "tremelimumab"],
    "LAG-3": ["relatlimab"],
    "TIM-3": [],
    "TIGIT": ["tiragolumab"],
}

CHECKPOINT_KEYWORDS = [
    "pd-1", "pd1", "pdcd1", "pd-l1", "pdl1", "cd274", "ctla-4", "ctla4", "cd152",
    "lag-3", "lag3", "tim-3", "tim3", "tigit", "checkpoint", "immunotherapy",
]


@dataclass
class CheckpointAnalysis:
    """Result of checkpoint inhibitor analysis."""
    is_checkpoint_inhibitor: bool
    target: Optional[str]
    drug_match: Optional[str]
    recommendation: str
    use_raw_score: bool
    explanation: str


def detect_checkpoint_inhibitor(
    name: str,
    indication: str = "",
    target: str = "",
    mechanism: str = "",
) -> CheckpointAnalysis:
    """
    Detect if a therapeutic is a checkpoint inhibitor.
    
    For checkpoint inhibitors, Treg-adjusted scores UNDERESTIMATE risk
    because the drug's mechanism impairs Treg function. Use raw score instead.
    
    Reference: EpiVax ISPRI paper (Mattei et al. 2024) - Case Study 3
    """
    name_lower = name.lower()
    indication_lower = indication.lower()
    target_lower = target.lower()
    mechanism_lower = mechanism.lower()
    all_text = f"{name_lower} {indication_lower} {target_lower} {mechanism_lower}"
    
    # Check for known checkpoint inhibitor drugs
    for checkpoint_target, drugs in CHECKPOINT_TARGETS.items():
        for drug in drugs:
            if drug in name_lower:
                return CheckpointAnalysis(
                    is_checkpoint_inhibitor=True,
                    target=checkpoint_target,
                    drug_match=drug,
                    recommendation="Use RAW score (not Treg-adjusted)",
                    use_raw_score=True,
                    explanation=f"Checkpoint inhibitor targeting {checkpoint_target}. "
                               f"Treg-adjusted score underestimates risk because the drug's "
                               f"mechanism of action impairs regulatory T cell function. "
                               f"Per EpiVax guidance, use raw EpiMatrix score for ADA prediction."
                )
    
    # Check for checkpoint-related keywords
    for keyword in CHECKPOINT_KEYWORDS:
        if keyword in all_text:
            return CheckpointAnalysis(
                is_checkpoint_inhibitor=True,
                target=keyword.upper(),
                drug_match=None,
                recommendation="Use RAW score (not Treg-adjusted)",
                use_raw_score=True,
                explanation=f"Possible checkpoint inhibitor (detected keyword: '{keyword}'). "
                           f"If this is a checkpoint inhibitor, Treg-adjusted scores may "
                           f"underestimate immunogenicity risk. Consider using raw score."
            )
    
    return CheckpointAnalysis(
        is_checkpoint_inhibitor=False,
        target=None,
        drug_match=None,
        recommendation="Use Treg-adjusted score (standard)",
        use_raw_score=False,
        explanation="Not a checkpoint inhibitor. Standard Treg-adjusted scoring applies."
    )


# ══════════════════════════════════════════════════════════════
# 2. IEDB EPITOPE HOMOLOGY SEARCH
# ══════════════════════════════════════════════════════════════

@dataclass
class IEDBMatch:
    """A match from the IEDB database."""
    epitope_id: str
    sequence: str
    source_organism: str
    source_antigen: str
    mhc_restriction: str
    assay_type: str
    response_type: str  # "positive" or "negative"
    pmid: Optional[str]
    similarity: float  # 0-1


@dataclass
class IEDBHomologyResult:
    """Result of IEDB homology search for a query peptide."""
    query_sequence: str
    exact_matches: List[IEDBMatch]
    similar_matches: List[IEDBMatch]
    total_matches: int
    has_published_epitope: bool
    published_response_type: Optional[str]


def search_iedb_epitopes(
    peptide: str,
    min_similarity: float = 0.8,
    max_results: int = 10,
) -> IEDBHomologyResult:
    """
    Search IEDB for known epitopes matching a query peptide.
    
    This cross-references predicted epitopes against published experimental data.
    If a predicted epitope has been tested and shown to be immunogenic in IEDB,
    it increases confidence in the prediction.
    
    Uses IEDB API: https://www.iedb.org/
    """
    exact_matches = []
    similar_matches = []
    
    try:
        # IEDB epitope search API
        url = "https://query-api.iedb.org/epitope_search"
        params = {
            "linear_sequence": peptide,
            "limit": max_results,
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            for item in data.get("results", []):
                epitope_seq = item.get("linear_peptide_seq", "")
                
                # Calculate similarity
                if epitope_seq == peptide:
                    similarity = 1.0
                else:
                    # Simple Levenshtein-based similarity
                    similarity = _calculate_similarity(peptide, epitope_seq)
                
                if similarity >= min_similarity:
                    match = IEDBMatch(
                        epitope_id=str(item.get("epitope_id", "")),
                        sequence=epitope_seq,
                        source_organism=item.get("source_organism_name", "Unknown"),
                        source_antigen=item.get("source_antigen_name", "Unknown"),
                        mhc_restriction=item.get("mhc_restriction", "Unknown"),
                        assay_type=item.get("assay_type", "Unknown"),
                        response_type=item.get("qualitative_measure", "Unknown"),
                        pmid=item.get("pubmed_id"),
                        similarity=similarity,
                    )
                    
                    if similarity == 1.0:
                        exact_matches.append(match)
                    else:
                        similar_matches.append(match)
    
    except Exception as e:
        print(f"IEDB search error: {e}")
    
    # Determine if any match was positive
    all_matches = exact_matches + similar_matches
    has_published = len(all_matches) > 0
    response_type = None
    
    for m in all_matches:
        if "positive" in m.response_type.lower():
            response_type = "POSITIVE (immunogenic in published assay)"
            break
        elif "negative" in m.response_type.lower():
            response_type = "NEGATIVE (not immunogenic in published assay)"
    
    return IEDBHomologyResult(
        query_sequence=peptide,
        exact_matches=exact_matches,
        similar_matches=similar_matches,
        total_matches=len(all_matches),
        has_published_epitope=has_published,
        published_response_type=response_type,
    )


def _calculate_similarity(seq1: str, seq2: str) -> float:
    """Calculate sequence similarity (simple matching)."""
    if not seq1 or not seq2:
        return 0.0
    
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    if max_len == 0:
        return 0.0
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / max_len


def batch_iedb_search(
    peptides: List[str],
    min_similarity: float = 0.8,
) -> Dict[str, IEDBHomologyResult]:
    """Search IEDB for multiple peptides."""
    results = {}
    for peptide in peptides[:20]:  # Limit to avoid API overload
        results[peptide] = search_iedb_epitopes(peptide, min_similarity)
    return results


# ══════════════════════════════════════════════════════════════
# 3. GERMLINE V-GENE COMPARISON (HUMANNESS SCORING)
# ══════════════════════════════════════════════════════════════

# Human germline V-gene framework sequences (IMGT reference)
# These are consensus sequences for the most common human V-genes
HUMAN_VH_GERMLINES = {
    "IGHV1-69": "EVQLVESGGGLVQPGGSLRLSCAASGFAFSSYDMSWVRQAPGKGLEWVS",
    "IGHV3-23": "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
    "IGHV3-30": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVA",
    "IGHV3-33": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
    "IGHV4-34": "QVQLQESGPGLVKPSQTLSLTCTVSGGSVSSGGYYWSWIRQHPGKGLEWIG",
    "IGHV4-39": "QVQLQESGPGLVKPSETLSLTCTVSGYSITSGYYWNWIRQPPGKGLEWIG",
    "IGHV5-51": "EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWVRQMPGKGLEWMG",
}

HUMAN_VL_KAPPA_GERMLINES = {
    "IGKV1-39": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIY",
    "IGKV3-20": "EIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIY",
    "IGKV1-33": "DIQMTQSPSSLSASVGDRVTITCQASQDISNYLNWYQQKPGKAPKLLIY",
    "IGKV2-28": "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRAS",
}

HUMAN_VL_LAMBDA_GERMLINES = {
    "IGLV1-44": "QSVLTQPPSVSGAPGQRVTISCTGSSSNIGAGYDVHWYQQLPGTAPKLLIY",
    "IGLV2-14": "QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIY",
    "IGLV3-19": "SYELTQPPSVSVAPGQTARITCGGNNIGSKSVHWYQQKPGQAPVLVIY",
}


@dataclass
class GermlineMatch:
    """A match against a germline V-gene."""
    germline_name: str
    similarity: float
    matched_positions: int
    total_positions: int
    alignment_region: str


@dataclass
class GermlineAnalysis:
    """Result of germline comparison analysis."""
    query_sequence: str
    chain_type: str  # "VH", "VL-kappa", "VL-lambda"
    best_match: Optional[GermlineMatch]
    all_matches: List[GermlineMatch]
    humanness_score: float  # 0-100, higher = more human-like
    foreign_regions: List[Dict[str, Any]]
    recommendation: str


def analyze_germline_humanness(
    sequence: str,
    chain_type: str = "auto",
) -> GermlineAnalysis:
    """
    Compare antibody sequence against human germline V-genes.
    
    Sequences with high similarity to germline are less likely to be immunogenic
    because they resemble "self" and T cells recognizing them would have been
    deleted or anergized during thymic development.
    
    Reference: EpiVax uses germline comparison to assess "humanness"
    """
    sequence = sequence.upper()
    
    # Auto-detect chain type based on N-terminus
    if chain_type == "auto":
        if sequence.startswith("EV") or sequence.startswith("QV"):
            chain_type = "VH"
        elif sequence.startswith("DI") or sequence.startswith("EI"):
            chain_type = "VL-kappa"
        elif sequence.startswith("QS") or sequence.startswith("SY"):
            chain_type = "VL-lambda"
        else:
            chain_type = "VH"  # Default
    
    # Select germline database
    if chain_type == "VH":
        germlines = HUMAN_VH_GERMLINES
    elif chain_type == "VL-kappa":
        germlines = HUMAN_VL_KAPPA_GERMLINES
    else:
        germlines = HUMAN_VL_LAMBDA_GERMLINES
    
    # Compare against all germlines
    matches = []
    for name, germline_seq in germlines.items():
        similarity, matched, total = _align_sequences(sequence, germline_seq)
        matches.append(GermlineMatch(
            germline_name=name,
            similarity=similarity,
            matched_positions=matched,
            total_positions=total,
            alignment_region=f"1-{total}",
        ))
    
    # Sort by similarity
    matches.sort(key=lambda x: x.similarity, reverse=True)
    best_match = matches[0] if matches else None
    
    # Calculate humanness score (0-100)
    if best_match:
        humanness_score = best_match.similarity * 100
    else:
        humanness_score = 0.0
    
    # Identify foreign (non-germline) regions
    foreign_regions = _identify_foreign_regions(sequence, best_match, germlines)
    
    # Generate recommendation
    if humanness_score >= 90:
        recommendation = "HIGH humanness - sequence closely matches human germline"
    elif humanness_score >= 75:
        recommendation = "MODERATE humanness - some deviations from germline"
    elif humanness_score >= 60:
        recommendation = "LOW humanness - significant deviations from germline, consider humanization"
    else:
        recommendation = "VERY LOW humanness - highly foreign, likely immunogenic"
    
    return GermlineAnalysis(
        query_sequence=sequence[:50] + "..." if len(sequence) > 50 else sequence,
        chain_type=chain_type,
        best_match=best_match,
        all_matches=matches[:5],  # Top 5
        humanness_score=humanness_score,
        foreign_regions=foreign_regions,
        recommendation=recommendation,
    )


def _align_sequences(seq1: str, seq2: str) -> Tuple[float, int, int]:
    """Simple sequence alignment returning (similarity, matched, total)."""
    min_len = min(len(seq1), len(seq2))
    
    if min_len == 0:
        return 0.0, 0, 0
    
    matched = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
    similarity = matched / min_len
    
    return similarity, matched, min_len


def _identify_foreign_regions(
    sequence: str,
    best_match: Optional[GermlineMatch],
    germlines: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Identify regions that differ from germline (potentially foreign)."""
    foreign_regions = []
    
    if not best_match or best_match.germline_name not in germlines:
        return foreign_regions
    
    germline_seq = germlines[best_match.germline_name]
    min_len = min(len(sequence), len(germline_seq))
    
    # Find stretches of mismatches
    in_mismatch = False
    mismatch_start = 0
    
    for i in range(min_len):
        if sequence[i] != germline_seq[i]:
            if not in_mismatch:
                in_mismatch = True
                mismatch_start = i
        else:
            if in_mismatch:
                if i - mismatch_start >= 3:  # Only report stretches of 3+
                    foreign_regions.append({
                        "start": mismatch_start + 1,
                        "end": i,
                        "query": sequence[mismatch_start:i],
                        "germline": germline_seq[mismatch_start:i],
                        "length": i - mismatch_start,
                    })
                in_mismatch = False
    
    # Handle mismatch at end
    if in_mismatch and min_len - mismatch_start >= 3:
        foreign_regions.append({
            "start": mismatch_start + 1,
            "end": min_len,
            "query": sequence[mismatch_start:min_len],
            "germline": germline_seq[mismatch_start:min_len],
            "length": min_len - mismatch_start,
        })
    
    return foreign_regions


# ══════════════════════════════════════════════════════════════
# COMBINED ADVANCED ANALYSIS
# ══════════════════════════════════════════════════════════════

@dataclass
class AdvancedAnalysisResult:
    """Combined result of all advanced analyses."""
    checkpoint_analysis: CheckpointAnalysis
    germline_analysis: Optional[GermlineAnalysis]
    iedb_matches: Dict[str, IEDBHomologyResult]
    
    # Summary flags
    is_checkpoint_inhibitor: bool
    humanness_score: float
    has_iedb_validated_epitopes: int
    
    # Recommendations
    use_raw_score: bool
    additional_flags: List[str]


def run_advanced_analysis(
    sequence: str,
    name: str = "",
    indication: str = "",
    target: str = "",
    hotspot_peptides: List[str] = None,
    chain_type: str = "auto",
) -> AdvancedAnalysisResult:
    """
    Run all advanced analyses on a therapeutic sequence.
    
    Args:
        sequence: Amino acid sequence
        name: Drug name
        indication: Clinical indication
        target: Drug target
        hotspot_peptides: List of predicted hotspot peptide sequences for IEDB search
        chain_type: "VH", "VL-kappa", "VL-lambda", or "auto"
    
    Returns:
        AdvancedAnalysisResult with all analyses
    """
    # 1. Checkpoint inhibitor detection
    checkpoint = detect_checkpoint_inhibitor(name, indication, target)
    
    # 2. Germline analysis (only for antibody-like sequences)
    germline = None
    if len(sequence) >= 50 and len(sequence) <= 500:
        germline = analyze_germline_humanness(sequence, chain_type)
    
    # 3. IEDB homology search for hotspot peptides
    iedb_matches = {}
    if hotspot_peptides:
        for peptide in hotspot_peptides[:10]:  # Limit to top 10
            iedb_matches[peptide] = search_iedb_epitopes(peptide)
    
    # Count validated epitopes
    validated_count = sum(
        1 for r in iedb_matches.values() 
        if r.has_published_epitope and r.published_response_type and "POSITIVE" in r.published_response_type
    )
    
    # Generate additional flags
    flags = []
    
    if checkpoint.is_checkpoint_inhibitor:
        flags.append(f"CHECKPOINT_INHIBITOR: {checkpoint.target} — use raw score, not Treg-adjusted")
    
    if germline and germline.humanness_score < 70:
        flags.append(f"LOW_HUMANNESS: {germline.humanness_score:.0f}% germline similarity — higher immunogenicity risk")
    
    if validated_count > 0:
        flags.append(f"IEDB_VALIDATED: {validated_count} predicted epitopes have published positive T-cell responses")
    
    if germline and len(germline.foreign_regions) > 3:
        flags.append(f"FOREIGN_REGIONS: {len(germline.foreign_regions)} regions differ significantly from human germline")
    
    return AdvancedAnalysisResult(
        checkpoint_analysis=checkpoint,
        germline_analysis=germline,
        iedb_matches=iedb_matches,
        is_checkpoint_inhibitor=checkpoint.is_checkpoint_inhibitor,
        humanness_score=germline.humanness_score if germline else 0.0,
        has_iedb_validated_epitopes=validated_count,
        use_raw_score=checkpoint.use_raw_score,
        additional_flags=flags,
    )


# ══════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Analysis Module Test")
    print("=" * 60)
    
    # Test 1: Checkpoint inhibitor detection
    print("\n1. Checkpoint Inhibitor Detection")
    print("-" * 40)
    
    tests = [
        ("pembrolizumab", "melanoma", "PD-1"),
        ("ipilimumab", "melanoma", "CTLA-4"),
        ("trastuzumab", "breast cancer", "HER2"),
        ("adalimumab", "rheumatoid arthritis", "TNF-alpha"),
    ]
    
    for name, indication, target in tests:
        result = detect_checkpoint_inhibitor(name, indication, target)
        status = "YES" if result.is_checkpoint_inhibitor else "NO"
        print(f"  {name}: Checkpoint={status}, Use raw={result.use_raw_score}")
    
    # Test 2: Germline analysis
    print("\n2. Germline Humanness Analysis")
    print("-" * 40)
    
    test_vh = "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS"
    germline = analyze_germline_humanness(test_vh, "VH")
    print(f"  Chain type: {germline.chain_type}")
    print(f"  Best match: {germline.best_match.germline_name} ({germline.best_match.similarity:.1%})")
    print(f"  Humanness score: {germline.humanness_score:.0f}/100")
    print(f"  Foreign regions: {len(germline.foreign_regions)}")
    print(f"  Recommendation: {germline.recommendation}")
    
    # Test 3: IEDB search (may timeout if API is slow)
    print("\n3. IEDB Homology Search")
    print("-" * 40)
    print("  (Skipping live API test to avoid timeout)")
    
    # Test 4: Combined analysis
    print("\n4. Combined Advanced Analysis")
    print("-" * 40)
    
    result = run_advanced_analysis(
        sequence=test_vh,
        name="nivolumab",
        indication="melanoma checkpoint blockade",
        target="PD-1",
        chain_type="VH",
    )
    
    print(f"  Is checkpoint inhibitor: {result.is_checkpoint_inhibitor}")
    print(f"  Use raw score: {result.use_raw_score}")
    print(f"  Humanness score: {result.humanness_score:.0f}/100")
    print(f"  Flags: {len(result.additional_flags)}")
    for flag in result.additional_flags:
        print(f"    - {flag}")
    
    print("\n" + "=" * 60)
    print("Advanced Analysis Module Test Complete!")
    print("=" * 60)
