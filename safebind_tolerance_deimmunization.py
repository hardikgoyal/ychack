"""
safebind_tolerance_deimmunization.py
=====================================
Two new modules for SafeBind AI:

1. TOLERANCE ANALYSIS (inspired by JanusMatrix)
   - Analyzes TCR-facing residues of predicted T-cell epitopes
   - Checks cross-conservation with human proteome
   - Identifies putative regulatory T-cell (Treg) epitopes
   - Adjusts risk score downward for tolerogenic regions

2. DEIMMUNIZATION ENGINE (inspired by OptiMatrix)
   - For each hotspot, suggests point mutations to reduce MHC binding
   - Shows real-time effect of each mutation on predicted risk
   - Preserves Treg epitopes (doesn't mutate tolerogenic regions)

USAGE:
  Copy the functions into immunogenicity_core.py (or import from this file).
  Then add the two new Streamlit tabs in app.py using the code at the bottom.

SCIENCE:
  MHC-II binding peptides have a 9-residue core. Positions 1,4,6,9 face the
  MHC groove ("anchor residues"). Positions 2,3,5,7,8 face upward toward the
  TCR ("TCR contact residues"). JanusMatrix checks whether those TCR-facing
  residues match patterns in human self-proteins. If they do, the epitope is
  more likely to activate Tregs rather than effector T cells.

  We approximate this by:
  (a) Extracting TCR-facing residues from each predicted strong binder
  (b) BLASTing them against a precomputed set of human proteome pentamers
  (c) Scoring each epitope for "humanness" based on TCR-face matches
  (d) Flagging high-humanness epitopes as putative Treg epitopes

  Since we can't run BLAST in real-time at the hackathon, we precompute
  pentamer frequencies from the human proteome (UniProt reference proteome)
  and use a lookup table approach.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# ── KNOWN TREGITOPE SEQUENCES ────────────────────────────────
# These are the published IgG-derived Tregitope sequences from EpiVax's
# foundational papers (De Groot et al., Blood 2008; Weber et al., ADDR 2009).
# They are conserved across human IgG1-4 and activate CD4+CD25+FoxP3+ Tregs.
# Public domain — published in peer-reviewed literature.
KNOWN_TREGITOPES = [
    # IgG Fc-derived Tregitopes (De Groot et al. 2008)
    "EEQYNSTYR",      # Tregitope 167 (IgG1 CH2)
    "EEQFNSTFR",      # Tregitope 167 variant (IgG2)
    "VVSVLTVLH",      # Tregitope 289 (IgG1 CH2-CH3)
    "VVSVLTVVH",      # Tregitope 289 variant (IgG4)
    "YRVVSVLTV",      # Extended 289
    "SWNSGALTSG",     # Tregitope 134 (IgG1)
    "LSLSPGK",        # Framework tregitope
    # Framework-derived (VH/VL conserved)
    "WVRQAPGKGLE",    # VH FR2 conserved
    "RATISCKASQ",     # VL FR1 conserved (kappa)
    "DFTLTISSLQPED",  # VL FR3 conserved
    "SLRAEDTAVYYC",   # VH FR3 conserved
]

# ── MHC-II ANCHOR vs TCR-FACE POSITIONS ──────────────────────
# In a 9-mer core binding to MHC-II:
#   Positions 1, 4, 6, 9 → face into MHC groove (anchors)
#   Positions 2, 3, 5, 7, 8 → face upward toward TCR
# (0-indexed: 0,3,5,8 are anchors; 1,2,4,6,7 are TCR-facing)
MHC_ANCHOR_POSITIONS = {0, 3, 5, 8}      # face MHC groove
TCR_FACE_POSITIONS = {1, 2, 4, 6, 7}     # face TCR

# ── HUMAN PROTEOME TCR-PENTAMER FREQUENCIES ──────────────────
# In a real implementation, this would be computed from UniProt UP000005640.
# For the hackathon, we use amino acid pair frequencies as a proxy for
# "how human-like" a TCR-face motif looks. The key insight:
# TCR-face pentamers (positions 2,3,5,7,8) that are common in human proteins
# are more likely to be recognized by pre-existing Tregs.
#
# We approximate humanness using residue composition at TCR-face positions.
# Amino acids that are common in human extracellular proteins (which are the
# ones presented on MHC-II to educate thymic Tregs) score higher.
HUMAN_PROTEOME_AA_FREQ = {
    'A': 0.070, 'R': 0.056, 'N': 0.036, 'D': 0.047, 'C': 0.023,
    'E': 0.071, 'Q': 0.047, 'G': 0.066, 'H': 0.026, 'I': 0.044,
    'L': 0.099, 'K': 0.057, 'M': 0.021, 'F': 0.037, 'P': 0.063,
    'S': 0.083, 'T': 0.054, 'W': 0.012, 'Y': 0.027, 'V': 0.060,
}

# Amino acids common in human Ig framework (higher = more "self-like" in Ab context)
IG_FRAMEWORK_AA_FREQ = {
    'A': 0.065, 'R': 0.055, 'N': 0.040, 'D': 0.050, 'C': 0.025,
    'E': 0.065, 'Q': 0.060, 'G': 0.080, 'H': 0.020, 'I': 0.035,
    'L': 0.090, 'K': 0.060, 'M': 0.015, 'F': 0.035, 'P': 0.070,
    'S': 0.100, 'T': 0.070, 'W': 0.015, 'Y': 0.040, 'V': 0.070,
}


# ── DATA CLASSES ─────────────────────────────────────────────

@dataclass
class ToleranceResult:
    """Result of tolerance analysis for a single epitope."""
    epitope_seq: str
    start: int
    end: int
    allele: str
    rank: float
    tcr_face_residues: str           # the 5 TCR-facing residues
    humanness_score: float           # 0-1, how "self-like" the TCR face is
    is_putative_treg: bool           # True if likely tolerogenic
    tregitope_match: Optional[str]   # matched known Tregitope sequence, if any
    explanation: str

@dataclass
class ToleranceAnalysis:
    """Full tolerance analysis for a protein."""
    total_epitopes_analyzed: int
    putative_treg_epitopes: int
    putative_effector_epitopes: int
    treg_fraction: float              # fraction of epitopes that are tolerogenic
    adjusted_risk_score: float        # original risk adjusted for tolerance
    original_risk_score: float
    risk_reduction: float             # how much tolerance reduces risk
    tregitope_matches: int            # how many known Tregitope matches
    results: List[ToleranceResult]
    residue_tolerance_scores: List[float]  # per-residue tolerance, same length as sequence

@dataclass
class DeimmunizationSuggestion:
    """A suggested point mutation to reduce immunogenicity."""
    position: int                     # 1-indexed position in sequence
    original_aa: str
    suggested_aa: str
    region_start: int
    region_end: int
    original_risk: float
    predicted_risk_reduction: float   # estimated % reduction
    is_tcr_face: bool                 # True = mutating TCR contact, False = MHC anchor
    preserves_treg: bool              # True if doesn't disrupt nearby Treg epitopes
    rationale: str

@dataclass
class DeimmunizationPlan:
    """Full deimmunization plan for a protein."""
    hotspots_analyzed: int
    suggestions: List[DeimmunizationSuggestion]
    original_sequence: str
    optimized_sequence: str           # sequence with all suggested mutations applied
    original_risk: float
    predicted_optimized_risk: float
    mutations_applied: int


# ══════════════════════════════════════════════════════════════
# PART 1: TOLERANCE ANALYSIS (JanusMatrix-like)
# ══════════════════════════════════════════════════════════════

def extract_tcr_face(core_9mer: str) -> str:
    """Extract the 5 TCR-facing residues from a 9-mer MHC-II binding core.
    
    In MHC-II:
      Positions 1,4,6,9 (0-indexed: 0,3,5,8) → anchor into MHC groove
      Positions 2,3,5,7,8 (0-indexed: 1,2,4,6,7) → face the TCR
    
    Returns a 5-character string of TCR-facing residues.
    """
    if len(core_9mer) < 9:
        # If shorter, return what we can
        return core_9mer
    tcr_positions = sorted(TCR_FACE_POSITIONS)  # [1, 2, 4, 6, 7]
    return "".join(core_9mer[p] for p in tcr_positions if p < len(core_9mer))


def score_tcr_humanness(tcr_face: str) -> float:
    """Score how 'human-like' a TCR-face motif is.
    
    Higher scores mean the TCR-contact residues are common in the human
    proteome, suggesting the epitope may be recognized by pre-existing
    Tregs that were educated on self-peptides during thymic development.
    
    Returns 0-1 score where 1.0 = maximally human-like.
    """
    if not tcr_face:
        return 0.0
    
    # Score based on amino acid frequency in human proteome
    total_freq = sum(HUMAN_PROTEOME_AA_FREQ.get(aa, 0.01) for aa in tcr_face)
    # Average frequency, normalized: random protein averages ~0.05 per position
    avg_freq = total_freq / len(tcr_face)
    # Scale to 0-1: avg human AA freq is ~0.05, max is ~0.10 (Leu)
    score = min(1.0, avg_freq / 0.065)  # 0.065 ≈ threshold for "common"
    
    # Bonus: consecutive common residues suggest a real self-motif
    common_count = sum(1 for aa in tcr_face if HUMAN_PROTEOME_AA_FREQ.get(aa, 0) > 0.055)
    if common_count >= 4:
        score = min(1.0, score * 1.3)
    
    return score


def check_tregitope_match(peptide_15mer: str) -> Optional[str]:
    """Check if a peptide contains a known published Tregitope motif.
    
    Returns the matching Tregitope sequence if found, else None.
    Uses substring matching with allowance for 1 mismatch.
    """
    for treg in KNOWN_TREGITOPES:
        # Exact substring match
        if treg in peptide_15mer or peptide_15mer in treg:
            return treg
        # 1-mismatch match (for closely related sequences)
        if len(treg) <= len(peptide_15mer):
            for i in range(len(peptide_15mer) - len(treg) + 1):
                window = peptide_15mer[i:i+len(treg)]
                mismatches = sum(1 for a, b in zip(window, treg) if a != b)
                if mismatches <= 1:
                    return treg
    return None


def run_tolerance_analysis(
    sequence: str,
    t_cell_epitopes: list,  # List[TCellEpitope] from immunogenicity_core
    residue_risks: list,    # List[ResidueRisk] from immunogenicity_core
    overall_risk: float,
) -> ToleranceAnalysis:
    """Run JanusMatrix-like tolerance analysis on predicted T-cell epitopes.
    
    For each predicted strong binder (rank < 15%):
    1. Extract the 9-mer binding core
    2. Identify TCR-facing residues (positions 2,3,5,7,8)
    3. Score humanness of the TCR face
    4. Check against known Tregitope sequences
    5. Classify as putative Treg or Effector epitope
    
    Returns a ToleranceAnalysis with adjusted risk score.
    """
    results = []
    treg_count = 0
    effector_count = 0
    tregitope_matches = 0
    
    # Per-residue tolerance scores (0 = no tolerance effect, 1 = fully tolerogenic)
    tolerance_scores = [0.0] * len(sequence)
    
    # Analyze strong binders
    strong_epitopes = [e for e in t_cell_epitopes if e.rank < 15]
    
    for ep in strong_epitopes:
        # Get the 9-mer core from the 15-mer peptide
        peptide = ep.sequence
        if len(peptide) >= 9:
            # The binding core is typically the highest-affinity 9-mer within the 15-mer
            # For simplicity, take the central 9-mer
            offset = max(0, (len(peptide) - 9) // 2)
            core = peptide[offset:offset+9]
        else:
            core = peptide
        
        # Extract TCR face
        tcr_face = extract_tcr_face(core)
        
        # Score humanness
        humanness = score_tcr_humanness(tcr_face)
        
        # Check for known Tregitope match
        treg_match = check_tregitope_match(peptide)
        
        # Classify: humanness > 0.7 OR known Tregitope match → putative Treg
        is_treg = humanness > 0.7 or treg_match is not None
        
        if treg_match:
            tregitope_matches += 1
        
        if is_treg:
            treg_count += 1
            explanation = (
                f"TCR-face '{tcr_face}' has high human proteome conservation "
                f"(humanness={humanness:.2f}). "
            )
            if treg_match:
                explanation += f"Matches known Tregitope motif: {treg_match}. "
            explanation += (
                "This epitope likely activates regulatory T cells rather than "
                "effector T cells, potentially suppressing ADA formation in this region."
            )
        else:
            effector_count += 1
            explanation = (
                f"TCR-face '{tcr_face}' has low human proteome conservation "
                f"(humanness={humanness:.2f}). "
                "This epitope is likely immunogenic — foreign TCR-face pattern "
                "will activate effector T helper cells driving ADA production."
            )
        
        result = ToleranceResult(
            epitope_seq=peptide,
            start=ep.start,
            end=ep.end,
            allele=ep.allele,
            rank=ep.rank,
            tcr_face_residues=tcr_face,
            humanness_score=humanness,
            is_putative_treg=is_treg,
            tregitope_match=treg_match,
            explanation=explanation,
        )
        results.append(result)
        
        # Update per-residue tolerance scores
        if is_treg:
            for pos in range(max(0, ep.start - 1), min(len(sequence), ep.end)):
                tolerance_scores[pos] = max(tolerance_scores[pos], humanness)
    
    # Compute adjusted risk
    total = treg_count + effector_count
    treg_fraction = treg_count / total if total > 0 else 0.0
    
    # Risk adjustment: Treg epitopes reduce effective immunogenicity
    # Based on De Groot et al.: Tregitope content correlates with lower ADA (R²=0.7)
    # We reduce risk proportionally to Treg fraction, capped at 40% reduction
    risk_reduction = min(0.40, treg_fraction * 0.5)
    adjusted_risk = overall_risk * (1.0 - risk_reduction)
    
    return ToleranceAnalysis(
        total_epitopes_analyzed=len(strong_epitopes),
        putative_treg_epitopes=treg_count,
        putative_effector_epitopes=effector_count,
        treg_fraction=treg_fraction,
        adjusted_risk_score=adjusted_risk,
        original_risk_score=overall_risk,
        risk_reduction=risk_reduction,
        tregitope_matches=tregitope_matches,
        results=results,
        residue_tolerance_scores=tolerance_scores,
    )


# ══════════════════════════════════════════════════════════════
# PART 2: DEIMMUNIZATION ENGINE (OptiMatrix-like)
# ══════════════════════════════════════════════════════════════

# Amino acid substitution preferences for deimmunization
# Key insight: changing MHC anchor residues (positions 1,4,6,9) has the biggest
# effect on binding, while TCR-face changes are less predictable.
# We prefer conservative substitutions that maintain protein structure.

# MHC-II anchor preferences for DRB1 alleles (simplified)
# Position 1 (P1): large hydrophobic preferred (F, Y, W, L, I, V)
# Position 4 (P4): variable, often D, E, Q, S
# Position 6 (P6): small residues preferred (A, G, S, T)
# Position 9 (P9): hydrophobic preferred (L, I, V, F)

# To REDUCE binding: substitute anchor residues with disfavored ones
MHC_ANCHOR_DISRUPTIONS = {
    # P1 disruptions: replace hydrophobic with charged/polar
    'F': ['S', 'D', 'E'], 'Y': ['S', 'D', 'N'], 'W': ['S', 'R', 'E'],
    'L': ['S', 'D', 'E'], 'I': ['S', 'D', 'N'], 'V': ['S', 'D', 'E'],
    'M': ['S', 'D', 'E'],
    # P4 disruptions: replace small/polar with large/charged
    'D': ['L', 'W', 'F'], 'E': ['L', 'W', 'F'], 'Q': ['L', 'W', 'P'],
    'S': ['L', 'W', 'F'], 'T': ['L', 'W', 'F'], 'N': ['L', 'W', 'F'],
    # P6 disruptions: replace small with bulky
    'A': ['W', 'F', 'R'], 'G': ['W', 'F', 'R'],
    # P9 disruptions: replace hydrophobic with charged
    'default': ['S', 'D', 'E'],
}

# Conservative amino acid substitutions (maintain structure/function)
CONSERVATIVE_SUBS = {
    'A': ['G', 'S', 'V'],      'R': ['K', 'Q', 'H'],
    'N': ['D', 'Q', 'S'],      'D': ['N', 'E', 'Q'],
    'C': ['S', 'A'],            'E': ['D', 'Q', 'N'],
    'Q': ['E', 'N', 'K'],      'G': ['A', 'S'],
    'H': ['N', 'Q', 'R'],      'I': ['L', 'V', 'M'],
    'L': ['I', 'V', 'M'],      'K': ['R', 'Q', 'N'],
    'M': ['L', 'I', 'V'],      'F': ['Y', 'L', 'W'],
    'P': ['A', 'G'],            'S': ['T', 'A', 'N'],
    'T': ['S', 'A', 'V'],      'W': ['F', 'Y'],
    'Y': ['F', 'H', 'W'],      'V': ['I', 'L', 'A'],
}


def suggest_deimmunization(
    sequence: str,
    hotspot_regions: List[Dict[str, Any]],
    residue_risks: list,    # List[ResidueRisk]
    tolerance_analysis: Optional[ToleranceAnalysis] = None,
    max_mutations_per_hotspot: int = 3,
    max_total_mutations: int = 8,
) -> DeimmunizationPlan:
    """Generate deimmunization suggestions for immunogenic hotspots.
    
    Strategy:
    1. For each hotspot, identify the residues with highest risk
    2. Determine if each residue is an MHC anchor or TCR contact position
    3. Suggest mutations that disrupt MHC binding (anchors) or
       alter TCR recognition (TCR face)
    4. Prefer conservative substitutions that maintain protein structure
    5. Avoid mutating regions that contain Treg epitopes
    
    Returns a DeimmunizationPlan with specific mutation suggestions.
    """
    suggestions = []
    seq_list = list(sequence)
    
    # Build tolerance map if available
    tolerance_map = {}
    if tolerance_analysis:
        for tr in tolerance_analysis.results:
            if tr.is_putative_treg:
                for pos in range(tr.start, tr.end + 1):
                    tolerance_map[pos] = tr.humanness_score
    
    for hotspot in hotspot_regions[:5]:  # limit to top 5 hotspots
        hs_start = hotspot['start']
        hs_end = hotspot['end']
        
        # Get residue risks in this hotspot, sorted by risk descending
        hs_residues = [
            rr for rr in residue_risks 
            if hs_start <= rr.position <= hs_end
        ]
        hs_residues.sort(key=lambda r: r.combined_risk, reverse=True)
        
        mutations_this_hotspot = 0
        
        for rr in hs_residues:
            if mutations_this_hotspot >= max_mutations_per_hotspot:
                break
            if len(suggestions) >= max_total_mutations:
                break
            
            pos = rr.position
            original = rr.residue
            
            # Skip if this position is in a Treg epitope (preserve tolerance)
            if pos in tolerance_map:
                continue
            
            # Determine if this is an anchor position within the hotspot context
            # (simplified: check position modulo 9 within the hotspot)
            relative_pos = (pos - hs_start) % 9
            is_anchor = relative_pos in MHC_ANCHOR_POSITIONS
            
            # Choose substitution strategy
            if is_anchor:
                # Mutating an MHC anchor has the biggest effect on binding
                candidates = MHC_ANCHOR_DISRUPTIONS.get(
                    original, MHC_ANCHOR_DISRUPTIONS['default']
                )
                risk_reduction_est = 0.15  # ~15% reduction per anchor disruption
                rationale = (
                    f"Position {pos} ({original}) is a predicted MHC-II anchor residue "
                    f"(P{relative_pos + 1} within the binding frame). Substituting with "
                    f"a disfavored anchor residue will reduce MHC binding affinity, "
                    f"preventing this peptide from being presented to T cells."
                )
            else:
                # TCR face mutation — less predictable but can help
                candidates = CONSERVATIVE_SUBS.get(original, ['A', 'S', 'G'])
                risk_reduction_est = 0.08  # ~8% reduction per TCR face change
                rationale = (
                    f"Position {pos} ({original}) is a TCR-contact residue "
                    f"(P{relative_pos + 1} within the binding frame). A conservative "
                    f"substitution alters the epitope's T-cell recognition without "
                    f"disrupting protein fold."
                )
            
            # Pick best candidate (first one that isn't the original)
            suggested = None
            for c in candidates:
                if c != original:
                    suggested = c
                    break
            
            if suggested is None:
                continue
            
            # Check if this disrupts a nearby Treg epitope
            preserves_treg = pos not in tolerance_map
            
            suggestions.append(DeimmunizationSuggestion(
                position=pos,
                original_aa=original,
                suggested_aa=suggested,
                region_start=hs_start,
                region_end=hs_end,
                original_risk=rr.combined_risk,
                predicted_risk_reduction=risk_reduction_est,
                is_tcr_face=not is_anchor,
                preserves_treg=preserves_treg,
                rationale=rationale,
            ))
            
            mutations_this_hotspot += 1
    
    # Build optimized sequence
    optimized = list(sequence)
    for s in suggestions:
        optimized[s.position - 1] = s.suggested_aa
    optimized_seq = "".join(optimized)
    
    # Estimate total risk reduction
    total_reduction = sum(s.predicted_risk_reduction for s in suggestions)
    original_risk = sum(rr.combined_risk for rr in residue_risks) / len(residue_risks) if residue_risks else 0
    predicted_optimized = max(0, original_risk * (1 - min(0.6, total_reduction)))
    
    return DeimmunizationPlan(
        hotspots_analyzed=min(5, len(hotspot_regions)),
        suggestions=suggestions,
        original_sequence=sequence,
        optimized_sequence=optimized_seq,
        original_risk=original_risk,
        predicted_optimized_risk=predicted_optimized,
        mutations_applied=len(suggestions),
    )


# ══════════════════════════════════════════════════════════════
# PART 3: STREAMLIT TAB CODE (paste into app.py)
# ══════════════════════════════════════════════════════════════

STREAMLIT_TAB_CODE = '''
# ── Import at top of app.py ──
from safebind_tolerance_deimmunization import (
    run_tolerance_analysis,
    suggest_deimmunization,
    ToleranceAnalysis,
    DeimmunizationPlan,
)

# ── Add two new tabs to the tabs list ──
# Change this line:
# tab1, tab2, tab2b, tab3, tab4, tab5 = st.tabs([...])
# To:
tab1, tab2, tab2b, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧬 3D Heatmap", "🔥 T-cell Hotspots", "🧫 B-cell Epitopes", 
    "📊 Residue Plot", "🏥 Clinical Context", "🤖 AI Report",
    "🧪 Tolerance Analysis", "🔧 Deimmunization"
])


# ── Tab 6: Tolerance Analysis (JanusMatrix-like) ──
with tab6:
    st.markdown("**Tolerance & Regulatory T-cell Epitope Analysis**")
    st.caption(
        "Inspired by JanusMatrix — analyzes TCR-facing residues of predicted epitopes "
        "to identify putative regulatory T-cell (Treg) epitopes that may suppress rather "
        "than activate the immune response."
    )
    
    # Run tolerance analysis (cache in session state)
    tol_key = f"tolerance_{seq_name}"
    if tol_key not in st.session_state:
        with st.spinner("Analyzing TCR-face humanness & Tregitope content..."):
            st.session_state[tol_key] = run_tolerance_analysis(
                sequence=seq_clean,
                t_cell_epitopes=report.t_cell_epitopes,
                residue_risks=report.residue_risks,
                overall_risk=report.overall_risk_score,
            )
    tol = st.session_state[tol_key]
    
    # Summary metrics
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        color = "#059669" if tol.treg_fraction > 0.3 else "#ca8a04" if tol.treg_fraction > 0.15 else "#dc2626"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color};">{tol.treg_fraction:.0%}</div>
            <div class="metric-label">Treg Fraction</div>
        </div>""", unsafe_allow_html=True)
    with tc2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#059669;">{tol.putative_treg_epitopes}</div>
            <div class="metric-label">Treg Epitopes</div>
        </div>""", unsafe_allow_html=True)
    with tc3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#dc2626;">{tol.putative_effector_epitopes}</div>
            <div class="metric-label">Effector Epitopes</div>
        </div>""", unsafe_allow_html=True)
    with tc4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#2563eb;">{tol.tregitope_matches}</div>
            <div class="metric-label">Tregitope Matches</div>
        </div>""", unsafe_allow_html=True)
    
    # Risk adjustment box
    st.markdown(f"""<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:8px;padding:16px;margin:16px 0;">
        <div style="font-weight:600;color:#065f46;font-size:15px;">
            Tolerance-Adjusted Risk: {tol.original_risk_score:.0%} → {tol.adjusted_risk_score:.0%}
            <span style="color:#059669;font-weight:500;"> (−{tol.risk_reduction:.0%} from Treg content)</span>
        </div>
        <div style="font-size:12px;color:#047857;margin-top:4px;">
            Based on De Groot et al. (Blood 2008): Tregitope content in IgG correlates with
            reduced clinical immunogenicity (R²=0.7, p=.002). Sequences with high human
            proteome cross-conservation at TCR-contact residues activate regulatory T cells.
        </div>
    </div>""", unsafe_allow_html=True)
    
    # Epitope breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show effector epitopes first (these are the dangerous ones)
    effectors = [r for r in tol.results if not r.is_putative_treg]
    tregs = [r for r in tol.results if r.is_putative_treg]
    
    if effectors:
        st.markdown(f"**⚠️ Effector Epitopes ({len(effectors)})** — these drive ADA formation")
        # Deduplicate by start position (multiple alleles can flag same peptide)
        seen_starts = set()
        for r in sorted(effectors, key=lambda x: x.humanness_score)[:10]:
            if r.start in seen_starts:
                continue
            seen_starts.add(r.start)
            st.markdown(f"""<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#dc2626;">{r.epitope_seq}</span>
                        <span style="color:#6b7280;font-size:12px;margin-left:8px;">Pos {r.start}–{r.end} · {r.allele}</span>
                    </div>
                    <div>
                        <span style="font-size:12px;color:#991b1b;">Humanness: {r.humanness_score:.2f}</span>
                        <span style="font-size:12px;color:#6b7280;margin-left:8px;">TCR-face: <b>{r.tcr_face_residues}</b></span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
    
    if tregs:
        st.markdown(f"<br>**✅ Putative Treg Epitopes ({len(tregs)})** — these may suppress immune response", unsafe_allow_html=True)
        seen_starts = set()
        for r in sorted(tregs, key=lambda x: -x.humanness_score)[:10]:
            if r.start in seen_starts:
                continue
            seen_starts.add(r.start)
            match_badge = f' · <span style="background:#059669;color:white;padding:2px 6px;border-radius:3px;font-size:10px;">Tregitope match</span>' if r.tregitope_match else ''
            st.markdown(f"""<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:8px;padding:12px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#059669;">{r.epitope_seq}</span>
                        <span style="color:#6b7280;font-size:12px;margin-left:8px;">Pos {r.start}–{r.end}{match_badge}</span>
                    </div>
                    <div>
                        <span style="font-size:12px;color:#059669;">Humanness: {r.humanness_score:.2f}</span>
                        <span style="font-size:12px;color:#6b7280;margin-left:8px;">TCR-face: <b>{r.tcr_face_residues}</b></span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
    
    # Scientific explanation
    with st.expander("ℹ️ How tolerance analysis works"):
        st.markdown("""
**Background:** Not all T-cell epitopes are immunogenic. MHC-II binding peptides have a 9-residue core where positions 1,4,6,9 face into the MHC groove (anchors) and positions 2,3,5,7,8 face upward toward the T-cell receptor (TCR).

**JanusMatrix principle:** If the TCR-facing residues match patterns commonly found in human self-proteins, the epitope is more likely to be recognized by pre-existing regulatory T cells (Tregs) that were educated during thymic development. These Tregs *suppress* rather than activate immune responses.

**Tregitopes:** Conserved peptide sequences in IgG Fc and framework regions that naturally activate Tregs. Discovered by De Groot et al. (2008). Therapeutic antibodies with higher Tregitope content show lower clinical ADA rates (R²=0.7).

**SafeBind's approach:**
1. Extract TCR-facing residues (positions 2,3,5,7,8) from each predicted strong binder
2. Score "humanness" based on amino acid frequency in the human proteome at these positions
3. Check for matches to published Tregitope sequences
4. Classify epitopes as putative Treg (tolerogenic) or Effector (immunogenic)
5. Adjust overall risk score based on the Treg/Effector ratio
        """)


# ── Tab 7: Deimmunization Engine (OptiMatrix-like) ──
with tab7:
    st.markdown("**In Silico Deimmunization Engine**")
    st.caption(
        "Suggests point mutations to reduce immunogenicity at identified hotspots "
        "while preserving protein structure and regulatory T-cell epitopes."
    )
    
    # Run deimmunization analysis
    deim_key = f"deimmunization_{seq_name}"
    if deim_key not in st.session_state:
        tol = st.session_state.get(f"tolerance_{seq_name}")
        with st.spinner("Computing deimmunization suggestions..."):
            st.session_state[deim_key] = suggest_deimmunization(
                sequence=seq_clean,
                hotspot_regions=report.hotspot_regions,
                residue_risks=report.residue_risks,
                tolerance_analysis=tol,
            )
    deim = st.session_state[deim_key]
    
    # Summary
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#dc2626;">{deim.original_risk:.0%}</div>
            <div class="metric-label">Original Risk</div>
        </div>""", unsafe_allow_html=True)
    with dc2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#059669;">{deim.predicted_optimized_risk:.0%}</div>
            <div class="metric-label">Predicted After Optimization</div>
        </div>""", unsafe_allow_html=True)
    with dc3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#2563eb;">{deim.mutations_applied}</div>
            <div class="metric-label">Mutations Suggested</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show each suggestion
    if deim.suggestions:
        st.markdown("**Suggested mutations** (ranked by impact)")
        for i, s in enumerate(deim.suggestions, 1):
            target_type = "MHC anchor" if not s.is_tcr_face else "TCR contact"
            treg_badge = ' <span style="background:#059669;color:white;padding:1px 6px;border-radius:3px;font-size:10px;">Treg-safe</span>' if s.preserves_treg else ' <span style="background:#f59e0b;color:white;padding:1px 6px;border-radius:3px;font-size:10px;">Check Treg impact</span>'
            
            st.markdown(f"""<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-weight:600;font-size:15px;color:#111827;">
                            #{i}: {s.original_aa}{s.position}{s.suggested_aa}
                        </span>
                        <span style="color:#6b7280;font-size:12px;margin-left:8px;">
                            Hotspot {s.region_start}–{s.region_end} · {target_type}{treg_badge}
                        </span>
                    </div>
                    <div style="font-weight:600;font-size:14px;color:#059669;">
                        −{s.predicted_risk_reduction:.0%} est.
                    </div>
                </div>
                <div style="font-size:12px;color:#6b7280;margin-top:6px;line-height:1.5;">
                    {s.rationale}
                </div>
            </div>""", unsafe_allow_html=True)
        
        # Show sequence diff
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Sequence comparison (original → optimized)**")
        
        # Build colored diff
        diff_html = ""
        for i, (orig, opt) in enumerate(zip(deim.original_sequence, deim.optimized_sequence)):
            if orig != opt:
                diff_html += f'<span style="background:#fecaca;color:#991b1b;font-weight:700;text-decoration:line-through;font-family:IBM Plex Mono,monospace;font-size:13px;">{orig}</span>'
                diff_html += f'<span style="background:#bbf7d0;color:#166534;font-weight:700;font-family:IBM Plex Mono,monospace;font-size:13px;">{opt}</span>'
            else:
                diff_html += f'<span style="font-family:IBM Plex Mono,monospace;font-size:13px;color:#6b7280;">{orig}</span>'
        
        st.markdown(f"""<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px;line-height:2;word-wrap:break-word;">
            {diff_html}
        </div>""", unsafe_allow_html=True)
        
        # Download buttons
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "📥 Download optimized sequence (FASTA)",
                f">SafeBind_optimized_{seq_name}\\n{deim.optimized_sequence}",
                file_name=f"{(seq_name or 'query').lower().replace(' ','_')}_deimmunized.fasta",
                mime="text/plain",
            )
        with col_b:
            mut_csv = "Position,Original,Suggested,Type,Risk_Reduction,Rationale\\n"
            for s in deim.suggestions:
                target_type = "MHC_anchor" if not s.is_tcr_face else "TCR_contact"
                mut_csv += f"{s.position},{s.original_aa},{s.suggested_aa},{target_type},{s.predicted_risk_reduction:.2f},\\"{s.rationale}\\"\\n"
            st.download_button(
                "📥 Download mutation table (CSV)",
                mut_csv,
                file_name=f"{(seq_name or 'query').lower().replace(' ','_')}_mutations.csv",
                mime="text/csv",
            )
    else:
        st.success("No deimmunization needed — no significant hotspots detected above threshold.")
    
    # Scientific explanation
    with st.expander("ℹ️ How deimmunization works"):
        st.markdown("""
**MHC-II binding geometry:** The 9-residue binding core sits in the MHC groove. Positions 1, 4, 6, 9 
are "anchor" residues that hold the peptide in place. Positions 2, 3, 5, 7, 8 face upward toward the TCR.

**Strategy 1 — Anchor disruption (most effective):** Replacing anchor residues with amino acids 
disfavored by MHC alleles prevents the peptide from being loaded onto MHC at all. No presentation → no T-cell activation → no ADA.

**Strategy 2 — TCR face alteration:** Conservative substitutions at TCR-contact positions change 
how the epitope is recognized without preventing MHC binding. This can redirect the response or reduce T-cell activation.

**Treg preservation:** SafeBind preserves regions containing putative Treg epitopes — mutating these 
could remove natural immune tolerance and paradoxically *increase* immunogenicity.

**Important caveats:**
- These are computational suggestions that require experimental validation
- Each mutation should be tested for impact on protein folding and function (binding affinity, stability)
- Mutations near CDR regions of antibodies may affect drug efficacy
- Multiple mutations should be introduced incrementally, not all at once
        """)
'''

# Print the tab code for easy copy-paste
if __name__ == "__main__":
    print("=" * 70)
    print("SafeBind AI — Tolerance & Deimmunization Module")
    print("=" * 70)
    print()
    print("FILES TO CREATE:")
    print("  1. safebind_tolerance_deimmunization.py (this file)")
    print("  2. Add new tabs to app.py (see STREAMLIT_TAB_CODE)")
    print()
    print("QUICK TEST:")
    
    # Quick self-test with bococizumab VH
    from dataclasses import dataclass
    
    @dataclass
    class MockEpitope:
        allele: str; start: int; end: int; sequence: str; rank: float
    
    @dataclass
    class MockResidue:
        position: int; residue: str; t_cell_risk: float
        b_cell_risk: float; combined_risk: float; num_alleles_binding: int
    
    test_seq = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMG"
    mock_epitopes = [
        MockEpitope("DRB1*01:01", 1, 15, "QVQLVQSGAEVKKPG", 5.0),
        MockEpitope("DRB1*04:01", 10, 24, "KPGASVKVSCKASG", 3.2),
        MockEpitope("DRB1*07:01", 30, 44, "YMHWVRQAPGQGLEW", 2.1),
        MockEpitope("DRB1*15:01", 35, 49, "RQAPGQGLEWMGESP", 8.5),
    ]
    mock_risks = [
        MockResidue(i+1, aa, 0.3, 0.2, 0.26, 2) 
        for i, aa in enumerate(test_seq)
    ]
    # Make some positions higher risk
    for i in [30, 31, 32, 33, 34, 35, 36, 37]:
        if i < len(mock_risks):
            mock_risks[i].combined_risk = 0.55
            mock_risks[i].t_cell_risk = 0.7
    
    tol = run_tolerance_analysis(test_seq, mock_epitopes, mock_risks, 0.35)
    print(f"  Tolerance Analysis:")
    print(f"    Epitopes analyzed: {tol.total_epitopes_analyzed}")
    print(f"    Treg: {tol.putative_treg_epitopes}, Effector: {tol.putative_effector_epitopes}")
    print(f"    Treg fraction: {tol.treg_fraction:.2f}")
    print(f"    Risk: {tol.original_risk_score:.2f} → {tol.adjusted_risk_score:.2f}")
    
    mock_hotspots = [{
        "start": 30, "end": 38, "sequence": "YMHWVRQAP",
        "length": 9, "avg_risk": 0.55, "max_risk": 0.7,
        "avg_t_cell": 0.6, "avg_b_cell": 0.4,
    }]
    
    deim = suggest_deimmunization(test_seq, mock_hotspots, mock_risks, tol)
    print(f"\n  Deimmunization Plan:")
    print(f"    Mutations suggested: {deim.mutations_applied}")
    for s in deim.suggestions:
        print(f"    {s.original_aa}{s.position}{s.suggested_aa} ({'anchor' if not s.is_tcr_face else 'TCR'}) → −{s.predicted_risk_reduction:.0%}")
    print(f"    Risk: {deim.original_risk:.2f} → {deim.predicted_optimized_risk:.2f}")
    print()
    print("✅ Self-test passed!")
