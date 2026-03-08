"""SafeBind Risk — Claude AI risk memo generation."""

import os
import streamlit as st


def generate_risk_memo(risk_result, user_inputs, sequence_diffs=None, epitope_results=None):
    """Generate an executive risk memo using Claude Sonnet 4.6.

    Falls back to structured data display if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    # Also check Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        st.info("Set `ANTHROPIC_API_KEY` environment variable for AI-powered analysis.")
        return _fallback_memo(risk_result, user_inputs, sequence_diffs, epitope_results)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Could not initialize Anthropic client: {e}")
        return _fallback_memo(risk_result, user_inputs, sequence_diffs, epitope_results)

    # Build the prompt
    nearest_drugs_text = ""
    if risk_result.nearest_drugs:
        for r in risk_result.nearest_drugs[:5]:
            ada_str = ""
            if hasattr(r, "inn_name"):
                ada_str = f"  - {r.inn_name} ({r.chain_descriptor}): {r.pct_identity:.0%} identity, score={r.score:.0f}"
            nearest_drugs_text += ada_str + "\n"

    diffs_text = ""
    if sequence_diffs:
        diffs_text = f"Sequence differences from closest match ({len(sequence_diffs)} mutations):\n"
        for pos, q_aa, r_aa in sequence_diffs[:20]:
            diffs_text += f"  Position {pos}: {r_aa} → {q_aa}\n"
        if len(sequence_diffs) > 20:
            diffs_text += f"  ... and {len(sequence_diffs) - 20} more\n"

    epitope_text = ""
    if epitope_results:
        epitope_text = f"T-cell epitope hotspots ({len(epitope_results)} high-affinity binders):\n"
        for ep in epitope_results[:15]:
            epitope_text += f"  Positions {ep.start}-{ep.end}: {ep.peptide} (rank={ep.percentile_rank:.1f}, {ep.allele})\n"

    prompt = f"""You are an immunogenicity risk expert for biotherapeutics. Generate an executive risk memo based on this analysis.

## Candidate Properties
- Protein Modality: {user_inputs.get('modality', 'N/A')}
- Species Origin: {user_inputs.get('species', 'N/A')}
- Route: {user_inputs.get('route', 'N/A')}
- Disease Indication: {user_inputs.get('disease', 'N/A')}
- Conjugate: {user_inputs.get('conjugate', 'N/A')}
- Backbone: {user_inputs.get('backbone', 'N/A')}
- Dose: {user_inputs.get('dose', 'N/A')}
- Schedule: {user_inputs.get('schedule', 'N/A')}

## Risk Assessment Results
- **Composite ADA Risk Score: {risk_result.composite_score}%** ({risk_result.risk_tier})
- Lookup benchmark score: {risk_result.lookup_score}% ({risk_result.lookup_level})
- Sequence similarity score: {risk_result.sequence_score if risk_result.sequence_score is not None else 'N/A (no sequence provided)'}%
- Feature adjustment score: {risk_result.feature_score}%

## Risk Factors
{chr(10).join('- ' + f for f in risk_result.risk_factors) if risk_result.risk_factors else 'None identified'}

## Nearest Clinical Precedents
{nearest_drugs_text or 'No sequence matches available'}

{diffs_text}

{epitope_text}

---

Generate a structured memo with these sections:
1. **Executive Risk Summary** (2-3 sentences)
2. **Risk Factor Analysis** (table format with factor, impact, severity)
3. **Clinical Precedent Comparison** (how this candidate compares to approved drugs)
4. **Recommended Sequence Modifications** (target surface-exposed epitopes if data available; otherwise provide general guidance)
5. **Dosing & Formulation Strategy** (based on route, dose, and immunogenicity risk)

Be specific, quantitative where possible, and actionable. Reference the clinical precedent data directly."""

    try:
        with st.spinner("Generating AI risk memo..."):
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system="You are an expert immunologist and biotherapeutics risk assessor. Provide clear, actionable risk memos for drug development teams. Use markdown formatting.",
                messages=[{"role": "user", "content": prompt}],
            )
        return response.content[0].text
    except Exception as e:
        st.warning(f"Claude API error: {e}")
        return _fallback_memo(risk_result, user_inputs, sequence_diffs, epitope_results)


def _fallback_memo(risk_result, user_inputs, sequence_diffs=None, epitope_results=None):
    """Structured data display when API is not available."""
    memo = f"""## Executive Risk Summary

**Composite ADA Risk: {risk_result.composite_score}% ({risk_result.risk_tier})**

This assessment is based on analysis of 3,334 clinical trial cohorts covering 218 approved biotherapeutics.

### Score Breakdown
| Component | Score | Weight |
|-----------|-------|--------|
| Clinical Lookup | {risk_result.lookup_score}% | 40% |
| Sequence Similarity | {risk_result.sequence_score if risk_result.sequence_score is not None else 'N/A'}% | {'35%' if risk_result.sequence_score is not None else '0% (redistributed)'} |
| Feature Adjustment | {risk_result.feature_score}% | 25% |

### Risk Factors
"""
    for f in risk_result.risk_factors:
        memo += f"- {f}\n"

    if not risk_result.risk_factors:
        memo += "- No elevated risk factors identified\n"

    if sequence_diffs:
        memo += f"\n### Sequence Differences ({len(sequence_diffs)} mutations from closest match)\n"
        for pos, q_aa, r_aa in sequence_diffs[:10]:
            memo += f"- Position {pos}: {r_aa} → {q_aa}\n"

    if epitope_results:
        memo += f"\n### T-cell Epitope Hotspots ({len(epitope_results)} high-affinity binders)\n"
        memo += "| Position | Peptide | Rank | Allele |\n|----------|---------|------|--------|\n"
        for ep in epitope_results[:10]:
            memo += f"| {ep.start}-{ep.end} | {ep.peptide} | {ep.percentile_rank:.1f} | {ep.allele} |\n"

    memo += "\n\n*Set `ANTHROPIC_API_KEY` environment variable for AI-powered analysis with specific modification recommendations.*"
    return memo
