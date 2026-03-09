"""SafeBind Risk — Rule-based deimmunization engine with tolerance analysis.

Generates modified sequences by substituting MHC-II anchor residues
in predicted T-cell epitope hotspots with conservative alternatives
that disrupt binding while preserving structural integrity.

Includes JanusMatrix-inspired tolerance analysis that identifies putative
regulatory T-cell (Treg) epitopes via TCR-facing residue humanness scoring
and known Tregitope matching.  Treg-containing regions are preserved during
deimmunization to maintain natural immune tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# =====================================================================
# CONSTANTS
# =====================================================================

# MHC-II binding anchor positions within a 15-mer core
# P1 (pos 1), P4 (pos 4), P6 (pos 6), P9 (pos 9) are the key anchors
ANCHOR_OFFSETS = [0, 3, 5, 8]  # 0-indexed within 15-mer peptide

# Conservative substitutions that disrupt MHC-II binding at anchor positions
# These are chosen to:
#   1. Break hydrophobic/aromatic anchoring (MHC-II prefers large hydrophobic at P1)
#   2. Preserve backbone geometry (similar size/charge where possible)
#   3. Minimize structural disruption
ANCHOR_SUBSTITUTIONS = {
    # Large hydrophobic -> small/polar (disrupts P1 anchor)
    "F": ["S", "T", "A"],
    "Y": ["S", "T", "N"],
    "W": ["S", "Q", "A"],
    "L": ["A", "S", "T"],
    "I": ["A", "T", "S"],
    "V": ["A", "T", "S"],
    "M": ["A", "T", "S"],
    # Aromatic
    "H": ["N", "Q", "S"],
    # Charged -> opposite or neutral
    "R": ["Q", "N", "S"],
    "K": ["Q", "N", "T"],
    "D": ["N", "S", "A"],
    "E": ["Q", "S", "A"],
    # Polar -- less disruptive subs
    "N": ["S", "A", "G"],
    "Q": ["S", "A", "N"],
    "S": ["A", "G", "N"],
    "T": ["A", "S", "G"],
    # Small
    "A": ["G", "S"],
    "G": ["A", "S"],
    "P": ["A", "S", "G"],  # Proline breaks can disrupt binding
    "C": ["A", "S"],
}

# Positions to never mutate (structurally critical in most proteins)
# Cysteines in disulfide bonds, glycosylation sites, catalytic residues
AVOID_MUTATION = {"C"}  # Conservative default; could be expanded

# ── KNOWN TREGITOPE SEQUENCES ────────────────────────────────────────
# Published IgG-derived Tregitope sequences from EpiVax's foundational
# papers (De Groot et al., Blood 2008; Weber et al., ADDR 2009).
# They are conserved across human IgG1-4 and activate CD4+CD25+FoxP3+ Tregs.
KNOWN_TREGITOPES = [
    # IgG Fc-derived Tregitopes (De Groot et al. 2008)
    "EEQYNSTYR",       # Tregitope 167 (IgG1 CH2)
    "EEQFNSTFR",       # Tregitope 167 variant (IgG2)
    "VVSVLTVLH",       # Tregitope 289 (IgG1 CH2-CH3)
    "VVSVLTVVH",       # Tregitope 289 variant (IgG4)
    "YRVVSVLTV",       # Extended 289
    "SWNSGALTSG",      # Tregitope 134 (IgG1)
    "LSLSPGK",         # Framework tregitope
    # Framework-derived (VH/VL conserved)
    "WVRQAPGKGLE",     # VH FR2 conserved
    "RATISCKASQ",      # VL FR1 conserved (kappa)
    "DFTLTISSLQPED",   # VL FR3 conserved
    "SLRAEDTAVYYC",    # VH FR3 conserved
    "SGGGGSGGGG",      # Linker-associated (common in scFv)
]

# ── MHC-II ANCHOR vs TCR-FACE POSITIONS ──────────────────────────────
# In a 9-mer core binding to MHC-II:
#   Positions 1, 4, 6, 9 -> face into MHC groove (anchors)
#   Positions 2, 3, 5, 7, 8 -> face upward toward TCR
# (0-indexed: 0,3,5,8 are anchors; 1,2,4,6,7 are TCR-facing)
MHC_ANCHOR_POSITIONS = {0, 3, 5, 8}
TCR_FACE_POSITIONS = {1, 2, 4, 6, 7}

# ── HUMAN PROTEOME TCR-PENTAMER FREQUENCIES ──────────────────────────
# Amino acid frequencies in the human proteome (UniProt UP000005640).
# TCR-face pentamers composed of common residues are more likely to be
# recognised by pre-existing Tregs educated on self-peptides during
# thymic development.
HUMAN_PROTEOME_AA_FREQ = {
    "A": 0.070, "R": 0.056, "N": 0.036, "D": 0.047, "C": 0.023,
    "E": 0.071, "Q": 0.047, "G": 0.066, "H": 0.026, "I": 0.044,
    "L": 0.099, "K": 0.057, "M": 0.021, "F": 0.037, "P": 0.063,
    "S": 0.083, "T": 0.054, "W": 0.012, "Y": 0.027, "V": 0.060,
}

# =====================================================================
# DATA CLASSES
# =====================================================================


@dataclass
class ToleranceResult:
    """Result of JanusMatrix-like tolerance analysis for a protein sequence."""

    epitope_details: list  # list of dicts per epitope (see run_tolerance_analysis)
    treg_count: int
    effector_count: int
    risk_adjustment: float   # how much to *reduce* risk (0-0.4)
    adjusted_risk: float     # overall_risk * (1 - risk_adjustment)
    tolerance_score: float   # 0-1 summary (higher = more tolerogenic)


@dataclass
class DeimmunizationResult:
    """Result of deimmunization for a single epitope cluster."""

    region_start: int   # 1-indexed
    region_end: int
    original_peptide: str
    mutations: list     # [(position, original_aa, new_aa)]
    modified_peptide: str
    anchor_positions_targeted: list  # which P1/P4/P6/P9 were hit
    expected_binding_disruption: str  # "High", "Moderate", "Low"
    allele: str
    original_rank: float


@dataclass
class RedesignedSequence:
    """A full redesigned sequence with applied mutations."""

    name: str
    sequence: str
    n_mutations: int
    mutations: list  # [(position, original_aa, new_aa)]
    targeted_epitopes: int
    strategy: str


# =====================================================================
# TOLERANCE ANALYSIS (JanusMatrix-like)
# =====================================================================


def extract_tcr_face(core_9mer: str) -> str:
    """Extract the 5 TCR-facing residues from a 9-mer MHC-II binding core.

    In MHC-II:
      Positions 1,4,6,9 (0-indexed: 0,3,5,8) -> anchor into MHC groove
      Positions 2,3,5,7,8 (0-indexed: 1,2,4,6,7) -> face the TCR

    Returns a 5-character string of TCR-facing residues.
    """
    if len(core_9mer) < 9:
        return core_9mer
    tcr_positions = sorted(TCR_FACE_POSITIONS)  # [1, 2, 4, 6, 7]
    return "".join(core_9mer[p] for p in tcr_positions if p < len(core_9mer))


def score_tcr_humanness(tcr_face: str) -> float:
    """Score how 'human-like' a TCR-face motif is.

    Higher scores mean the TCR-contact residues are common in the human
    proteome, suggesting the epitope may be recognised by pre-existing
    Tregs educated on self-peptides during thymic development.

    Returns 0-1 score where 1.0 = maximally human-like.
    """
    if not tcr_face:
        return 0.0

    total_freq = sum(HUMAN_PROTEOME_AA_FREQ.get(aa, 0.01) for aa in tcr_face)
    avg_freq = total_freq / len(tcr_face)
    # Scale to 0-1: avg human AA freq is ~0.05, max is ~0.10 (Leu)
    score = min(1.0, avg_freq / 0.065)

    # Bonus: consecutive common residues suggest a real self-motif
    common_count = sum(
        1 for aa in tcr_face if HUMAN_PROTEOME_AA_FREQ.get(aa, 0) > 0.055
    )
    if common_count >= 4:
        score = min(1.0, score * 1.3)

    return score


def check_tregitope_match(peptide_15mer: str) -> Optional[str]:
    """Check if a peptide contains a known published Tregitope motif.

    Returns the matching Tregitope sequence if found (with <=1 mismatch),
    else ``None``.
    """
    for treg in KNOWN_TREGITOPES:
        # Exact substring match
        if treg in peptide_15mer or peptide_15mer in treg:
            return treg
        # 1-mismatch match (for closely related sequences)
        if len(treg) <= len(peptide_15mer):
            for i in range(len(peptide_15mer) - len(treg) + 1):
                window = peptide_15mer[i : i + len(treg)]
                mismatches = sum(1 for a, b in zip(window, treg) if a != b)
                if mismatches <= 1:
                    return treg
    return None


def run_tolerance_analysis(
    sequence: str,
    epitope_results: list,
    overall_risk: float,
) -> ToleranceResult:
    """Run JanusMatrix-like tolerance analysis on predicted T-cell epitopes.

    Parameters
    ----------
    sequence : str
        Full protein amino-acid sequence.
    epitope_results : list
        List of ``EpitopeResult`` objects (from ``sequence_engine.py``).
        Each must have attributes: ``start``, ``end``, ``peptide``,
        ``percentile_rank``, ``allele``.
    overall_risk : float
        Pre-existing overall immunogenicity risk score (0-1).

    Returns
    -------
    ToleranceResult
        Aggregated tolerance analysis including per-epitope details,
        Treg/effector counts, risk adjustment, and a 0-1 tolerance score.
    """
    epitope_details: list[dict] = []
    treg_count = 0
    effector_count = 0

    # Analyse only strong binders (percentile_rank < 15)
    strong_epitopes = [e for e in epitope_results if e.percentile_rank < 15]

    for ep in strong_epitopes:
        peptide = ep.peptide

        # Derive the 9-mer binding core (central 9-mer of the 15-mer)
        if len(peptide) >= 9:
            offset = max(0, (len(peptide) - 9) // 2)
            core = peptide[offset : offset + 9]
        else:
            core = peptide

        tcr_face = extract_tcr_face(core)
        humanness = score_tcr_humanness(tcr_face)
        treg_match = check_tregitope_match(peptide)

        # Classify: humanness > 0.7 OR known Tregitope match -> putative Treg
        is_treg = humanness > 0.7 or treg_match is not None

        if is_treg:
            treg_count += 1
        else:
            effector_count += 1

        epitope_details.append({
            "start": ep.start,
            "end": ep.end,
            "peptide": peptide,
            "allele": ep.allele,
            "percentile_rank": ep.percentile_rank,
            "tcr_face": tcr_face,
            "treg_score": humanness,
            "is_treg": is_treg,
            "tregitope_match": treg_match,
        })

    total = treg_count + effector_count
    treg_fraction = treg_count / total if total > 0 else 0.0

    # Risk adjustment: Treg epitopes reduce effective immunogenicity.
    # Based on De Groot et al.: Tregitope content correlates with lower ADA
    # (R^2=0.7).  We reduce risk proportionally to Treg fraction, capped at 40%.
    risk_adjustment = min(0.40, treg_fraction * 0.5)
    adjusted_risk = overall_risk * (1.0 - risk_adjustment)

    return ToleranceResult(
        epitope_details=epitope_details,
        treg_count=treg_count,
        effector_count=effector_count,
        risk_adjustment=risk_adjustment,
        adjusted_risk=adjusted_risk,
        tolerance_score=treg_fraction,
    )


# =====================================================================
# DEIMMUNIZATION ENGINE
# =====================================================================


def _pick_substitution(aa: str, neighbors: str = "") -> str | None:
    """Pick the best conservative substitution for an anchor residue."""
    if aa in AVOID_MUTATION:
        return None
    candidates = ANCHOR_SUBSTITUTIONS.get(aa, [])
    if not candidates:
        return None
    # Prefer substitution not already present in neighboring residues
    for sub in candidates:
        if sub not in neighbors:
            return sub
    return candidates[0]


def _build_treg_position_set(
    tolerance_result: Optional[ToleranceResult],
) -> set[int]:
    """Return the set of 1-indexed sequence positions covered by Treg epitopes."""
    protected: set[int] = set()
    if tolerance_result is None:
        return protected
    for detail in tolerance_result.epitope_details:
        if detail.get("is_treg"):
            for pos in range(detail["start"], detail["end"] + 1):
                protected.add(pos)
    return protected


def deimmunize_epitopes(
    sequence: str,
    epitope_results: list,
    max_epitopes: int = 10,
    max_mutations_per_epitope: int = 2,
    tolerance_result: Optional[ToleranceResult] = None,
) -> list[DeimmunizationResult]:
    """Generate deimmunization suggestions for top epitope hotspots.

    Strategy: For each epitope, mutate 1-2 anchor positions (P1, P4, P6, or P9)
    with conservative substitutions that disrupt MHC-II binding.

    When *tolerance_result* is provided, positions that fall inside a known Treg
    epitope are skipped so that tolerance-inducing sequences are preserved.
    """
    if not epitope_results:
        return []

    # Build set of positions protected by Treg epitopes
    treg_positions = _build_treg_position_set(tolerance_result)

    # Deduplicate overlapping epitopes -- keep the strongest binder per region
    seen_regions: set[int] = set()
    unique_epitopes: list = []
    for ep in epitope_results:
        region_key = ep.start // 5  # Cluster nearby epitopes
        if region_key not in seen_regions:
            seen_regions.add(region_key)
            unique_epitopes.append(ep)
        if len(unique_epitopes) >= max_epitopes:
            break

    results: list[DeimmunizationResult] = []
    for ep in unique_epitopes:
        peptide = ep.peptide
        if len(peptide) < 9:
            continue

        # If the entire epitope is within a Treg region, skip it entirely
        epitope_positions = set(range(ep.start, ep.end + 1))
        if treg_positions and epitope_positions.issubset(treg_positions):
            continue

        mutations: list[tuple] = []
        modified = list(peptide)
        anchors_hit: list[str] = []

        for offset in ANCHOR_OFFSETS:
            if offset >= len(peptide):
                continue
            if len(mutations) >= max_mutations_per_epitope:
                break

            seq_pos = ep.start + offset  # 1-indexed position in full sequence

            # Skip positions inside Treg epitopes
            if seq_pos in treg_positions:
                continue

            aa = peptide[offset]
            neighbors = peptide[max(0, offset - 1) : offset + 2]
            sub = _pick_substitution(aa, neighbors)

            if sub and sub != aa:
                mutations.append((seq_pos, aa, sub))
                modified[offset] = sub
                anchor_name = f"P{offset + 1}"
                anchors_hit.append(anchor_name)

        if not mutations:
            continue

        # Estimate binding disruption
        if len(mutations) >= 2 and any(
            o == 0
            for _, (o, _, _) in enumerate(
                [(m[0] - ep.start, m[1], m[2]) for m in mutations]
            )
        ):
            disruption = "High"
        elif len(mutations) >= 2:
            disruption = "High"
        elif mutations[0][0] - ep.start == 0:  # P1 hit
            disruption = "High"
        else:
            disruption = "Moderate"

        results.append(
            DeimmunizationResult(
                region_start=ep.start,
                region_end=ep.end,
                original_peptide=peptide,
                mutations=mutations,
                modified_peptide="".join(modified),
                anchor_positions_targeted=anchors_hit,
                expected_binding_disruption=disruption,
                allele=ep.allele,
                original_rank=ep.percentile_rank,
            )
        )

    return results


def generate_redesigned_sequences(
    original_sequence: str,
    deimmunization_results: list[DeimmunizationResult],
) -> list[RedesignedSequence]:
    """Generate full redesigned sequences from deimmunization results.

    Produces 3 variants:
        1. Conservative -- only top 3 highest-confidence mutations
        2. Moderate -- top 5-7 epitope regions, 1 mutation each
        3. Aggressive -- all suggested mutations applied
    """
    if not deimmunization_results:
        return []

    all_mutations = []
    for dr in deimmunization_results:
        for pos, orig, new in dr.mutations:
            if 1 <= pos <= len(original_sequence):
                all_mutations.append(
                    (pos, orig, new, dr.expected_binding_disruption, dr.original_rank)
                )

    # Deduplicate by position (keep lowest rank = strongest binder)
    pos_map: dict = {}
    for pos, orig, new, disr, rank in all_mutations:
        if pos not in pos_map or rank < pos_map[pos][4]:
            pos_map[pos] = (pos, orig, new, disr, rank)

    sorted_mutations = sorted(pos_map.values(), key=lambda x: x[4])  # by rank

    variants: list[RedesignedSequence] = []

    # Variant 1: Conservative (top 3 mutations at P1 anchors)
    conservative_muts = sorted_mutations[:3]
    if conservative_muts:
        seq = list(original_sequence)
        applied = []
        for pos, orig, new, _, _ in conservative_muts:
            idx = pos - 1
            if idx < len(seq) and seq[idx] == orig:
                seq[idx] = new
                applied.append((pos, orig, new))
        variants.append(
            RedesignedSequence(
                name="Conservative (3 mutations)",
                sequence="".join(seq),
                n_mutations=len(applied),
                mutations=applied,
                targeted_epitopes=len(applied),
                strategy="Targets only the 3 strongest-binding epitope P1/P4 anchors. "
                "Minimal structural risk, moderate deimmunization.",
            )
        )

    # Variant 2: Moderate (top 7 mutations)
    moderate_muts = sorted_mutations[:7]
    if len(moderate_muts) > 3:
        seq = list(original_sequence)
        applied = []
        for pos, orig, new, _, _ in moderate_muts:
            idx = pos - 1
            if idx < len(seq) and seq[idx] == orig:
                seq[idx] = new
                applied.append((pos, orig, new))
        variants.append(
            RedesignedSequence(
                name="Moderate (7 mutations)",
                sequence="".join(seq),
                n_mutations=len(applied),
                mutations=applied,
                targeted_epitopes=len(applied),
                strategy="Targets top 7 epitope anchor residues. "
                "Balanced deimmunization with acceptable structural risk.",
            )
        )

    # Variant 3: Aggressive (all mutations)
    if len(sorted_mutations) > 7:
        seq = list(original_sequence)
        applied = []
        for pos, orig, new, _, _ in sorted_mutations:
            idx = pos - 1
            if idx < len(seq) and seq[idx] == orig:
                seq[idx] = new
                applied.append((pos, orig, new))
        variants.append(
            RedesignedSequence(
                name=f"Aggressive ({len(applied)} mutations)",
                sequence="".join(seq),
                n_mutations=len(applied),
                mutations=applied,
                targeted_epitopes=len(applied),
                strategy="Targets all identified epitope anchors. "
                "Maximum deimmunization but higher structural risk -- "
                "validate with stability assays.",
            )
        )

    return variants
