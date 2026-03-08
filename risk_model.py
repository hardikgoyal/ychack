"""SafeBind Risk — Composite ADA risk scoring engine."""

from dataclasses import dataclass, field
from config import (
    W_LOOKUP, W_SEQUENCE, W_FEATURE,
    GLOBAL_BASELINE_ADA, SPECIES_ADA, CONJUGATE_ADA,
    ROUTE_MULTIPLIER, MODALITY_ADA, RISK_TIERS,
    MIN_IDENTITY_THRESHOLD,
)


@dataclass
class RiskResult:
    composite_score: float
    risk_tier: str
    tier_color: str
    lookup_score: float
    sequence_score: float | None
    feature_score: float
    nearest_drugs: list
    risk_factors: list = field(default_factory=list)
    lookup_level: str = ""  # Which lookup granularity was used


def get_risk_tier(score: float):
    """Return (tier_name, color) for a given ADA score."""
    for threshold, name, color in RISK_TIERS:
        if score <= threshold:
            return name, color
    return "Very High", "#8e44ad"


def _lookup_score(route, disease, modality, lookup_tables):
    """Hierarchical lookup: route+disease+modality → route+modality → modality → global."""
    l1 = lookup_tables["route_disease_modality"]
    l2 = lookup_tables["route_modality"]
    l3 = lookup_tables["modality"]

    route_col = "Therapeutic Route of Administration"
    disease_col = "Disease Indication Category"
    modality_col = "Protein Modality"

    # Level 1: full match
    match = l1[
        (l1[route_col] == route)
        & (l1[disease_col] == disease)
        & (l1[modality_col] == modality)
    ]
    if len(match) > 0:
        row = match.iloc[0]
        return row["weighted_ada"], f"Route + Disease + Modality ({int(row['n_cohorts'])} cohorts, {int(row['total_patients'])} patients)"

    # Level 2: route + modality
    match = l2[(l2[route_col] == route) & (l2[modality_col] == modality)]
    if len(match) > 0:
        row = match.iloc[0]
        return row["weighted_ada"], f"Route + Modality ({int(row['n_cohorts'])} cohorts, {int(row['total_patients'])} patients)"

    # Level 3: modality only
    match = l3[l3[modality_col] == modality]
    if len(match) > 0:
        row = match.iloc[0]
        return row["weighted_ada"], f"Modality only ({int(row['n_cohorts'])} cohorts, {int(row['total_patients'])} patients)"

    # Fallback to hardcoded modality or global baseline
    if modality in MODALITY_ADA:
        return MODALITY_ADA[modality], "Modality baseline (hardcoded)"
    return GLOBAL_BASELINE_ADA, "Global baseline"


def _sequence_score(alignment_results, drug_ada_map):
    """Weighted average of top-k matches' known ADA%, weighted by alignment score."""
    if not alignment_results:
        return None, []

    valid = []
    for r in alignment_results:
        if r.pct_identity >= MIN_IDENTITY_THRESHOLD and r.inn_name in drug_ada_map:
            valid.append(r)

    if not valid:
        return None, alignment_results

    total_weight = sum(r.score for r in valid)
    if total_weight == 0:
        return None, alignment_results

    weighted_ada = sum(r.score * drug_ada_map[r.inn_name] for r in valid) / total_weight
    return weighted_ada, alignment_results


def _feature_adjustment(species, conjugate, route, modality):
    """Additive deltas from species/conjugate/route/modality baselines relative to global mean."""
    delta = 0.0
    factors = []

    # Modality delta — often the largest signal
    modality_ada = MODALITY_ADA.get(modality, GLOBAL_BASELINE_ADA)
    modality_delta = modality_ada - GLOBAL_BASELINE_ADA
    delta += modality_delta
    if abs(modality_delta) > 3:
        direction = "increases" if modality_delta > 0 else "decreases"
        factors.append(f"{modality} modality {direction} risk by {abs(modality_delta):.0f}pp vs baseline")

    # Species delta
    species_ada = SPECIES_ADA.get(species, GLOBAL_BASELINE_ADA)
    species_delta = species_ada - GLOBAL_BASELINE_ADA
    delta += species_delta
    if abs(species_delta) > 2:
        direction = "increases" if species_delta > 0 else "decreases"
        factors.append(f"{species} origin {direction} risk by {abs(species_delta):.0f}pp vs baseline")

    # Conjugate delta
    conj_ada = CONJUGATE_ADA.get(conjugate, GLOBAL_BASELINE_ADA)
    conj_delta = conj_ada - GLOBAL_BASELINE_ADA
    delta += conj_delta
    if abs(conj_delta) > 2:
        direction = "increases" if conj_delta > 0 else "decreases"
        factors.append(f"{conjugate} {direction} risk by {abs(conj_delta):.0f}pp")

    # Route multiplier effect
    route_mult = ROUTE_MULTIPLIER.get(route, 1.0)
    if route_mult != 1.0:
        route_effect = (route_mult - 1.0) * GLOBAL_BASELINE_ADA
        delta += route_effect
        direction = "increases" if route_effect > 0 else "decreases"
        factors.append(f"{route} administration {direction} risk (×{route_mult:.1f})")

    return GLOBAL_BASELINE_ADA + delta, factors


def predict_ada(
    modality: str,
    species: str,
    route: str,
    disease: str,
    conjugate: str,
    lookup_tables: dict,
    drug_ada_map: dict,
    alignment_results: list | None = None,
) -> RiskResult:
    """Compute composite ADA risk score.

    Components:
        - Lookup (40%): Historical ADA rates from clinical data
        - Sequence (35%): Nearest-neighbor ADA from sequence alignment
        - Feature (25%): Adjustments from species/conjugate/route
    """
    # Lookup component
    lookup_val, lookup_level = _lookup_score(route, disease, modality, lookup_tables)

    # Sequence component
    seq_val, nearest_drugs = _sequence_score(
        alignment_results or [], drug_ada_map
    )

    # Feature adjustment component
    feature_val, risk_factors = _feature_adjustment(species, conjugate, route, modality)

    # Compute composite
    if seq_val is not None:
        # All three components
        composite = (
            W_LOOKUP * lookup_val
            + W_SEQUENCE * seq_val
            + W_FEATURE * feature_val
        )
    else:
        # No sequence: split weight evenly between lookup and features
        # (features now carry modality signal, so they deserve equal weight)
        composite = (
            0.55 * lookup_val
            + 0.45 * feature_val
        )

    # Clamp to [0, 100]
    composite = max(0.0, min(100.0, composite))

    tier_name, tier_color = get_risk_tier(composite)

    return RiskResult(
        composite_score=round(composite, 1),
        risk_tier=tier_name,
        tier_color=tier_color,
        lookup_score=round(lookup_val, 1),
        sequence_score=round(seq_val, 1) if seq_val is not None else None,
        feature_score=round(feature_val, 1),
        nearest_drugs=nearest_drugs,
        risk_factors=risk_factors,
        lookup_level=lookup_level,
    )
