"""SafeBind Risk — Rule-based deimmunization engine.

Generates modified sequences by substituting MHC-II anchor residues
in predicted T-cell epitope hotspots with conservative alternatives
that disrupt binding while preserving structural integrity.
"""

from dataclasses import dataclass, field

# MHC-II binding anchor positions within a 15-mer core
# P1 (pos 1), P4 (pos 4), P6 (pos 6), P9 (pos 9) are the key anchors
ANCHOR_OFFSETS = [0, 3, 5, 8]  # 0-indexed within 15-mer peptide

# Conservative substitutions that disrupt MHC-II binding at anchor positions
# These are chosen to:
#   1. Break hydrophobic/aromatic anchoring (MHC-II prefers large hydrophobic at P1)
#   2. Preserve backbone geometry (similar size/charge where possible)
#   3. Minimize structural disruption
ANCHOR_SUBSTITUTIONS = {
    # Large hydrophobic → small/polar (disrupts P1 anchor)
    "F": ["S", "T", "A"],
    "Y": ["S", "T", "N"],
    "W": ["S", "Q", "A"],
    "L": ["A", "S", "T"],
    "I": ["A", "T", "S"],
    "V": ["A", "T", "S"],
    "M": ["A", "T", "S"],
    # Aromatic
    "H": ["N", "Q", "S"],
    # Charged → opposite or neutral
    "R": ["Q", "N", "S"],
    "K": ["Q", "N", "T"],
    "D": ["N", "S", "A"],
    "E": ["Q", "S", "A"],
    # Polar — less disruptive subs
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


@dataclass
class DeimmunizationResult:
    """Result of deimmunization for a single epitope cluster."""
    region_start: int  # 1-indexed
    region_end: int
    original_peptide: str
    mutations: list  # [(position, original_aa, new_aa)]
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


def deimmunize_epitopes(
    sequence: str,
    epitope_results: list,
    max_epitopes: int = 10,
    max_mutations_per_epitope: int = 2,
) -> list[DeimmunizationResult]:
    """Generate deimmunization suggestions for top epitope hotspots.

    Strategy: For each epitope, mutate 1-2 anchor positions (P1, P4, P6, or P9)
    with conservative substitutions that disrupt MHC-II binding.
    """
    if not epitope_results:
        return []

    # Deduplicate overlapping epitopes — keep the strongest binder per region
    seen_regions = set()
    unique_epitopes = []
    for ep in epitope_results:
        region_key = ep.start // 5  # Cluster nearby epitopes
        if region_key not in seen_regions:
            seen_regions.add(region_key)
            unique_epitopes.append(ep)
        if len(unique_epitopes) >= max_epitopes:
            break

    results = []
    for ep in unique_epitopes:
        peptide = ep.peptide
        if len(peptide) < 9:
            continue

        mutations = []
        modified = list(peptide)
        anchors_hit = []

        for offset in ANCHOR_OFFSETS:
            if offset >= len(peptide):
                continue
            if len(mutations) >= max_mutations_per_epitope:
                break

            aa = peptide[offset]
            # Get neighbors for context
            neighbors = peptide[max(0, offset - 1):offset + 2]
            sub = _pick_substitution(aa, neighbors)

            if sub and sub != aa:
                seq_pos = ep.start + offset  # 1-indexed position in full sequence
                mutations.append((seq_pos, aa, sub))
                modified[offset] = sub
                anchor_name = f"P{offset + 1}"
                anchors_hit.append(anchor_name)

        if not mutations:
            continue

        # Estimate binding disruption
        if len(mutations) >= 2 and any(o == 0 for _, (o, _, _) in enumerate([(m[0] - ep.start, m[1], m[2]) for m in mutations])):
            disruption = "High"
        elif len(mutations) >= 2:
            disruption = "High"
        elif mutations[0][0] - ep.start == 0:  # P1 hit
            disruption = "High"
        else:
            disruption = "Moderate"

        results.append(DeimmunizationResult(
            region_start=ep.start,
            region_end=ep.end,
            original_peptide=peptide,
            mutations=mutations,
            modified_peptide="".join(modified),
            anchor_positions_targeted=anchors_hit,
            expected_binding_disruption=disruption,
            allele=ep.allele,
            original_rank=ep.percentile_rank,
        ))

    return results


def generate_redesigned_sequences(
    original_sequence: str,
    deimmunization_results: list[DeimmunizationResult],
) -> list[RedesignedSequence]:
    """Generate full redesigned sequences from deimmunization results.

    Produces 3 variants:
        1. Conservative — only top 3 highest-confidence mutations
        2. Moderate — top 5-7 epitope regions, 1 mutation each
        3. Aggressive — all suggested mutations applied
    """
    if not deimmunization_results:
        return []

    all_mutations = []
    for dr in deimmunization_results:
        for pos, orig, new in dr.mutations:
            if 1 <= pos <= len(original_sequence):
                all_mutations.append((pos, orig, new, dr.expected_binding_disruption, dr.original_rank))

    # Deduplicate by position (keep lowest rank = strongest binder)
    pos_map = {}
    for pos, orig, new, disr, rank in all_mutations:
        if pos not in pos_map or rank < pos_map[pos][4]:
            pos_map[pos] = (pos, orig, new, disr, rank)

    sorted_mutations = sorted(pos_map.values(), key=lambda x: x[4])  # by rank

    variants = []

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
        variants.append(RedesignedSequence(
            name="Conservative (3 mutations)",
            sequence="".join(seq),
            n_mutations=len(applied),
            mutations=applied,
            targeted_epitopes=len(applied),
            strategy="Targets only the 3 strongest-binding epitope P1/P4 anchors. "
                     "Minimal structural risk, moderate deimmunization.",
        ))

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
        variants.append(RedesignedSequence(
            name="Moderate (7 mutations)",
            sequence="".join(seq),
            n_mutations=len(applied),
            mutations=applied,
            targeted_epitopes=len(applied),
            strategy="Targets top 7 epitope anchor residues. "
                     "Balanced deimmunization with acceptable structural risk.",
        ))

    # Variant 3: Aggressive (all mutations)
    if len(sorted_mutations) > 7:
        seq = list(original_sequence)
        applied = []
        for pos, orig, new, _, _ in sorted_mutations:
            idx = pos - 1
            if idx < len(seq) and seq[idx] == orig:
                seq[idx] = new
                applied.append((pos, orig, new))
        variants.append(RedesignedSequence(
            name=f"Aggressive ({len(applied)} mutations)",
            sequence="".join(seq),
            n_mutations=len(applied),
            mutations=applied,
            targeted_epitopes=len(applied),
            strategy="Targets all identified epitope anchors. "
                     "Maximum deimmunization but higher structural risk — "
                     "validate with stability assays.",
        ))

    return variants
