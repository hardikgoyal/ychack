"""SafeBind Risk — Composite ADA risk scoring engine."""

from dataclasses import dataclass, field
from config import (
    W_LOOKUP, W_SEQUENCE, W_FEATURE,
    GLOBAL_BASELINE_ADA, SPECIES_ADA, CONJUGATE_ADA,
    ROUTE_MULTIPLIER, MODALITY_ADA, RISK_TIERS,
    MIN_IDENTITY_THRESHOLD, EXPRESSION_SYSTEM_ADA,
    NADA_SEVERITY, TIME_TO_ADA_MULTIPLIER,
)


@dataclass
class NadaResult:
    """Neutralizing ADA risk assessment."""
    nada_pct: float
    nada_ratio: float  # nADA/ADA ratio
    severity: str
    severity_color: str
    description: str
    source: str  # Where the estimate came from


@dataclass
class TimeAdaResult:
    """Time-to-ADA profile."""
    expected_onset: str  # Which time bin has highest risk
    peak_ada_pct: float
    time_multiplier: float
    profile: list  # [(time_bin, ada_pct, n_cohorts)]


@dataclass
class ConfidenceInfo:
    """Prediction confidence assessment."""
    level: str  # "High", "Moderate", "Low"
    color: str
    reasons: list
    prediction_range: tuple  # (low, high) in %
    n_similar_drugs: int


@dataclass
class RiskResult:
    composite_score: float
    risk_tier: str
    tier_color: str
    lookup_score: float
    sequence_score: float | None
    feature_score: float
    ml_score: float | None
    nearest_drugs: list
    risk_factors: list = field(default_factory=list)
    lookup_level: str = ""
    nada: NadaResult | None = None
    time_ada: TimeAdaResult | None = None
    confidence: ConfidenceInfo | None = None


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


def _feature_adjustment(species, conjugate, route, modality, expression_system=None):
    """Additive deltas from species/conjugate/route/modality/expression baselines."""
    delta = 0.0
    factors = []

    # Modality delta
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

    # Expression system delta
    if expression_system and expression_system != "Undetermined":
        expr_ada = EXPRESSION_SYSTEM_ADA.get(expression_system, GLOBAL_BASELINE_ADA)
        expr_delta = expr_ada - GLOBAL_BASELINE_ADA
        delta += expr_delta
        if abs(expr_delta) > 2:
            direction = "increases" if expr_delta > 0 else "decreases"
            factors.append(f"{expression_system} expression {direction} risk by {abs(expr_delta):.0f}pp")

    return GLOBAL_BASELINE_ADA + delta, factors


def estimate_nada(composite_ada, modality, route, nada_lookup):
    """Estimate neutralizing ADA risk from composite ADA score + nADA lookup.

    Returns NadaResult.
    """
    modality_col = "Protein Modality"
    route_col = "Therapeutic Route of Administration"

    # Try modality + route specific nADA ratio
    by_mr = nada_lookup["by_mod_route"]
    match = by_mr[(by_mr[modality_col] == modality) & (by_mr[route_col] == route)]
    if len(match) > 0 and match.iloc[0]["n_cohorts"] >= 3:
        ratio = match.iloc[0]["nada_ratio"]
        source = f"Modality + Route ({int(match.iloc[0]['n_cohorts'])} cohorts)"
    else:
        # Fallback to modality only
        by_m = nada_lookup["by_mod"]
        match = by_m[by_m[modality_col] == modality]
        if len(match) > 0 and match.iloc[0]["n_cohorts"] >= 3:
            ratio = match.iloc[0]["nada_ratio"]
            source = f"Modality ({int(match.iloc[0]['n_cohorts'])} cohorts)"
        else:
            ratio = nada_lookup["global_ratio"]
            source = "Global baseline"

    # Cap ratio at 1.0 (nADA can't exceed ADA)
    ratio = min(ratio, 1.0)
    nada_pct = composite_ada * ratio

    # Determine severity tier
    severity = "Low nADA Risk"
    severity_color = "#2ecc71"
    description = "Most ADA are non-neutralizing"
    for threshold, name, color, desc in NADA_SEVERITY:
        if nada_pct <= threshold:
            severity = name
            severity_color = color
            description = desc
            break

    return NadaResult(
        nada_pct=round(nada_pct, 1),
        nada_ratio=round(ratio, 2),
        severity=severity,
        severity_color=severity_color,
        description=description,
        source=source,
    )


def estimate_time_ada(composite_ada, modality, time_ada_lookup):
    """Estimate time-to-ADA profile.

    Returns TimeAdaResult with expected onset timing and ADA trajectory.
    """
    modality_col = "Protein Modality"

    # Try modality-specific profile
    by_mod = time_ada_lookup["by_modality"]
    mod_data = by_mod[by_mod[modality_col] == modality]

    if len(mod_data) >= 2:
        profile = [(row["time_bin"], row["weighted_ada"], int(row["n_cohorts"]))
                    for _, row in mod_data.iterrows()]
        source = "modality-specific"
    else:
        global_data = time_ada_lookup["global"]
        profile = [(row["time_bin"], row["weighted_ada"], int(row["n_cohorts"]))
                    for _, row in global_data.iterrows()]
        source = "global"

    if not profile:
        return None

    # Find peak
    peak_bin, peak_ada, _ = max(profile, key=lambda x: x[1])

    # Time multiplier: ratio of peak to overall
    avg_ada = sum(p[1] for p in profile) / len(profile) if profile else 1
    time_mult = peak_ada / avg_ada if avg_ada > 0 else 1.0

    return TimeAdaResult(
        expected_onset=peak_bin,
        peak_ada_pct=round(peak_ada, 1),
        time_multiplier=round(time_mult, 2),
        profile=profile,
    )


def _estimate_confidence(composite, lookup_val, seq_val, feature_val, ml_val,
                         alignment_results, lookup_level):
    """Estimate prediction confidence based on agreement between components
    and availability of data."""
    reasons = []
    scores = [s for s in [lookup_val, seq_val, feature_val, ml_val] if s is not None]

    # 1. Component agreement — if all components point the same direction, higher confidence
    if len(scores) >= 3:
        spread = max(scores) - min(scores)
        if spread < 15:
            reasons.append("Components agree well (spread < 15pp)")
            agreement_score = 2
        elif spread < 30:
            reasons.append(f"Moderate component disagreement (spread {spread:.0f}pp)")
            agreement_score = 1
        else:
            reasons.append(f"High component disagreement (spread {spread:.0f}pp)")
            agreement_score = 0
    else:
        agreement_score = 0
        reasons.append("Limited scoring components available")

    # 2. Sequence match quality
    seq_quality = 0
    if alignment_results:
        best_identity = max(r.pct_identity for r in alignment_results)
        if best_identity > 0.90:
            seq_quality = 2
            reasons.append(f"Strong sequence match ({best_identity:.0%} identity)")
        elif best_identity > 0.60:
            seq_quality = 1
            reasons.append(f"Moderate sequence match ({best_identity:.0%} identity)")
        else:
            reasons.append(f"Weak sequence match ({best_identity:.0%} identity)")
    else:
        reasons.append("No sequence provided — prediction based on class averages only")

    # 3. Lookup data density
    lookup_quality = 0
    if "cohorts" in lookup_level:
        import re
        cohort_match = re.search(r"(\d+) cohorts", lookup_level)
        patient_match = re.search(r"(\d+) patients", lookup_level)
        if cohort_match:
            n_cohorts = int(cohort_match.group(1))
            if n_cohorts >= 20:
                lookup_quality = 2
            elif n_cohorts >= 5:
                lookup_quality = 1
        if patient_match:
            n_patients = int(patient_match.group(1))
            if n_patients > 1000:
                reasons.append(f"Backed by {n_patients:,} patient-observations")

    # Total confidence
    total = agreement_score + seq_quality + lookup_quality
    if total >= 4:
        level, color = "High", "#2ecc71"
    elif total >= 2:
        level, color = "Moderate", "#f39c12"
    else:
        level, color = "Low", "#e74c3c"

    # Prediction range
    # Based on benchmark: median AE is ~7pp, worst is ~60pp
    if level == "High":
        margin = 10
    elif level == "Moderate":
        margin = 20
    else:
        margin = 35

    pred_range = (round(max(0, composite - margin), 1), round(min(100, composite + margin), 1))

    n_similar = len(alignment_results) if alignment_results else 0

    return ConfidenceInfo(
        level=level,
        color=color,
        reasons=reasons,
        prediction_range=pred_range,
        n_similar_drugs=n_similar,
    )


def predict_ada(
    modality: str,
    species: str,
    route: str,
    disease: str,
    conjugate: str,
    lookup_tables: dict,
    drug_ada_map: dict,
    alignment_results: list | None = None,
    expression_system: str | None = None,
    nada_lookup: dict | None = None,
    time_ada_lookup: dict | None = None,
) -> RiskResult:
    """Compute composite ADA risk score.

    Components:
        - Lookup (40%): Historical ADA rates from clinical data
        - Sequence (35%): Nearest-neighbor ADA from sequence alignment
        - Feature (25%): Adjustments from species/conjugate/route/expression
        - ML model: Ensemble member (when available)
    """
    # Lookup component
    lookup_val, lookup_level = _lookup_score(route, disease, modality, lookup_tables)

    # Sequence component
    seq_val, nearest_drugs = _sequence_score(
        alignment_results or [], drug_ada_map
    )

    # Feature adjustment component
    feature_val, risk_factors = _feature_adjustment(
        species, conjugate, route, modality, expression_system
    )

    # Compute composite — rule-based ensemble
    ml_val = None
    if seq_val is not None:
        composite = (
            W_LOOKUP * lookup_val
            + W_SEQUENCE * seq_val
            + W_FEATURE * feature_val
        )
    else:
        composite = (
            0.55 * lookup_val
            + 0.45 * feature_val
        )

    # Clamp to [0, 100]
    composite = max(0.0, min(100.0, composite))

    tier_name, tier_color = get_risk_tier(composite)

    # Confidence estimation
    confidence = _estimate_confidence(
        composite, lookup_val, seq_val, feature_val, ml_val,
        alignment_results, lookup_level,
    )

    # nADA estimate
    nada = None
    if nada_lookup:
        nada = estimate_nada(composite, modality, route, nada_lookup)

    # Time-to-ADA profile
    time_ada = None
    if time_ada_lookup:
        time_ada = estimate_time_ada(composite, modality, time_ada_lookup)

    return RiskResult(
        composite_score=round(composite, 1),
        risk_tier=tier_name,
        tier_color=tier_color,
        lookup_score=round(lookup_val, 1),
        sequence_score=round(seq_val, 1) if seq_val is not None else None,
        feature_score=round(feature_val, 1),
        ml_score=round(ml_val, 1) if ml_val is not None else None,
        nearest_drugs=nearest_drugs,
        risk_factors=risk_factors,
        lookup_level=lookup_level,
        nada=nada,
        time_ada=time_ada,
        confidence=confidence,
    )
