"""SafeBind Risk — Immunogenicity Risk Assessment Dashboard."""

import streamlit as st
import pandas as pd
import altair as alt
import os

from config import (
    MODALITY_OPTIONS, SPECIES_OPTIONS, ROUTE_OPTIONS,
    DISEASE_OPTIONS, CONJUGATE_OPTIONS, BACKBONE_OPTIONS,
    W_LOOKUP, W_SEQUENCE, W_FEATURE,
)
from data_loader import (
    load_therapeutic, load_sequences, load_clinical,
    build_lookup_table, build_drug_ada_map, get_historical_precedents,
)
from risk_model import predict_ada
from sequence_engine import (
    parse_fasta, align_to_references, get_sequence_diffs,
    predict_epitopes, compute_epitope_density,
)
from claude_report import generate_risk_memo
from deimmunize import deimmunize_epitopes, generate_redesigned_sequences

st.set_page_config(
    page_title="SafeBind Risk",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Inputs ---
st.sidebar.title("SafeBind Risk")
st.sidebar.markdown("*Immunogenicity Risk Assessment*")
st.sidebar.divider()

sequence_input = st.sidebar.text_area(
    "Protein Sequence (FASTA or raw)",
    height=150,
    placeholder="Paste amino acid sequence here...\nRequired for structural analysis (Parts 2 & 3)",
)

st.sidebar.divider()
modality = st.sidebar.selectbox("Protein Modality", MODALITY_OPTIONS)
species = st.sidebar.selectbox("Species Origin", SPECIES_OPTIONS)
route = st.sidebar.selectbox("Route of Administration", ROUTE_OPTIONS)
disease = st.sidebar.selectbox("Disease Indication", DISEASE_OPTIONS)
conjugate = st.sidebar.selectbox("Conjugate Modification", CONJUGATE_OPTIONS)
backbone = st.sidebar.selectbox("Antibody Backbone", BACKBONE_OPTIONS)

st.sidebar.divider()
dose = st.sidebar.text_input("Dose Level", placeholder="e.g., 10 mg/kg")
schedule = st.sidebar.text_input("Dosing Schedule", placeholder="e.g., Q2W IV")

analyze = st.sidebar.button("Analyze Risk", type="primary", use_container_width=True)

# --- Main Area ---
if not analyze:
    st.title("SafeBind Risk")
    st.markdown("""
    ### Biotherapeutics Immunogenicity Risk Assessment

    Predict anti-drug antibody (ADA) probability for your protein therapeutic candidate
    using clinical data from **218 approved drugs** and **3,334 clinical trial cohorts**.

    **How it works:**
    - **Empirical Benchmarking** — Historical ADA rates matched by route, disease, and modality
    - **Sequence Similarity** — Alignment to 222 reference sequences from approved drugs
    - **T-cell Epitope Prediction** — IEDB MHC-II binding analysis
    - **AI Synthesis** — Claude-powered risk memo with modification recommendations

    Configure your candidate in the sidebar and click **Analyze Risk**.
    """)

    # Show data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Approved Drugs", "218")
    with col2:
        st.metric("Clinical Cohorts", "3,334")
    with col3:
        st.metric("Reference Sequences", "222")
    st.stop()


# --- Run Analysis ---
# Load data
lookup_tables = build_lookup_table()
drug_ada_map = build_drug_ada_map()
sequences_df = load_sequences()

# Parse sequence if provided
parsed_seq = None
alignment_results = None
epitope_results = None
sequence_diffs = None

if sequence_input.strip():
    try:
        parsed_seq = parse_fasta(sequence_input)
        st.sidebar.success(f"Sequence parsed: {len(parsed_seq)} residues")
    except ValueError as e:
        st.sidebar.error(str(e))
        parsed_seq = None

# Run sequence alignment
if parsed_seq:
    alignment_results = align_to_references(parsed_seq, sequences_df)

# Run risk prediction
risk_result = predict_ada(
    modality=modality,
    species=species,
    route=route,
    disease=disease,
    conjugate=conjugate,
    lookup_tables=lookup_tables,
    drug_ada_map=drug_ada_map,
    alignment_results=alignment_results,
)

# Get sequence diffs if we have alignment
if alignment_results and len(alignment_results) > 0:
    best_match = alignment_results[0]
    sequence_diffs = get_sequence_diffs(parsed_seq, best_match.ref_sequence)

# ==============================
# PART 1 — Prediction & Benchmarking
# ==============================
st.title("SafeBind Risk Assessment")

tab1, tab2, tab3 = st.tabs([
    "Prediction & Benchmarking",
    "Structural Risk Viewer",
    "Redesign Copilot",
])

with tab1:
    # Main risk metric
    col_score, col_tier = st.columns([1, 2])
    with col_score:
        st.markdown(
            f"""<div style="text-align:center; padding:20px; background:{risk_result.tier_color}22;
            border-radius:12px; border:2px solid {risk_result.tier_color}">
            <h1 style="color:{risk_result.tier_color}; margin:0; font-size:3.5rem">{risk_result.composite_score}%</h1>
            <h3 style="color:{risk_result.tier_color}; margin:0">ADA Risk — {risk_result.risk_tier}</h3>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_tier:
        st.markdown("#### Score Breakdown")
        breakdown = {
            "Component": ["Clinical Lookup", "Sequence Similarity", "Feature Adjustment"],
            "Score (%)": [
                risk_result.lookup_score,
                risk_result.sequence_score if risk_result.sequence_score is not None else "N/A",
                risk_result.feature_score,
            ],
            "Weight": [
                f"{W_LOOKUP:.0%}",
                f"{W_SEQUENCE:.0%}" if risk_result.sequence_score is not None else "0% (redistributed)",
                f"{W_FEATURE:.0%}",
            ],
            "Source": [
                risk_result.lookup_level,
                f"Top {len(alignment_results)} matches" if alignment_results else "No sequence",
                "Species + Conjugate + Route",
            ],
        }
        st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True)

    st.divider()

    # Benchmarking comparison
    col_chart, col_factors = st.columns([3, 2])

    with col_chart:
        st.markdown(f"#### Benchmarking: Your Candidate vs. {disease} + {route}")
        precedents = get_historical_precedents(route, disease, modality)

        if len(precedents) > 0:
            chart_data = precedents[["Therapeutic Assessed for ADA INN Name", "ada_pct", "total_patients"]].copy()
            chart_data.columns = ["Drug", "ADA Rate (%)", "Patients"]
            # Add the candidate
            candidate_row = pd.DataFrame([{
                "Drug": "YOUR CANDIDATE",
                "ADA Rate (%)": risk_result.composite_score,
                "Patients": 0,
            }])
            chart_data = pd.concat([candidate_row, chart_data], ignore_index=True)

            bars = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("ADA Rate (%):Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Drug:N", sort="-x"),
                color=alt.condition(
                    alt.datum.Drug == "YOUR CANDIDATE",
                    alt.value(risk_result.tier_color),
                    alt.value("#4a90d9"),
                ),
                tooltip=["Drug", "ADA Rate (%)", "Patients"],
            ).properties(height=max(200, len(chart_data) * 30))

            st.altair_chart(bars, use_container_width=True)
        else:
            st.info("No exact precedents found for this combination. Showing modality-level benchmarks.")

    with col_factors:
        st.markdown("#### Risk Factors")
        if risk_result.risk_factors:
            for f in risk_result.risk_factors:
                if "increases" in f:
                    st.markdown(f"🔴 {f}")
                elif "decreases" in f:
                    st.markdown(f"🟢 {f}")
                else:
                    st.markdown(f"🟡 {f}")
        else:
            st.success("No elevated risk factors identified.")

        # Nearest drugs by sequence
        if alignment_results:
            st.markdown("#### Nearest Drugs (by sequence)")
            for r in alignment_results:
                ada_str = f" — ADA: {drug_ada_map[r.inn_name]:.1f}%" if r.inn_name in drug_ada_map else ""
                st.markdown(f"- **{r.inn_name}** ({r.chain_descriptor}): {r.pct_identity:.0%} identity{ada_str}")

    # Historical precedent table
    st.divider()
    st.markdown("#### Historical Precedent Table")
    precedents = get_historical_precedents(route, disease, modality, top_n=15)
    if len(precedents) > 0:
        display_cols = {
            "Therapeutic Assessed for ADA INN Name": "Drug",
            "ada_pct": "ADA Rate (%)",
            "total_patients": "Total Patients",
            "n_cohorts": "Cohorts",
            "route": "Route",
            "disease": "Disease",
            "modality": "Modality",
        }
        display_df = precedents.rename(columns=display_cols)[list(display_cols.values())]
        display_df["ADA Rate (%)"] = display_df["ADA Rate (%)"].round(1)
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.info("No historical precedents found for this combination.")

# ==============================
# PART 2 — Structural Risk Viewer
# ==============================
with tab2:
    if not parsed_seq:
        st.warning("Provide a protein sequence in the sidebar to enable structural analysis.")
    else:
        st.markdown(f"#### Sequence Analysis ({len(parsed_seq)} residues)")

        col_3d, col_epitopes = st.columns([3, 2])

        with col_3d:
            st.markdown("#### 3D Protein Structure")

            # Try Tamarind Bio ESMFold, fallback to sequence-only view
            tamarind_key = os.environ.get("TAMARIND_API_KEY")
            pdb_data = None

            if tamarind_key:
                try:
                    from tamarind_integration import fold_protein
                    pdb_data = fold_protein(parsed_seq, tamarind_key)
                except Exception as e:
                    st.info(f"Tamarind API unavailable: {e}")

            if pdb_data:
                try:
                    import stmol
                    import py3Dmol

                    view = py3Dmol.view(width=600, height=400)
                    view.addModel(pdb_data, "pdb")
                    view.setStyle({"cartoon": {"color": "spectrum"}})

                    # Highlight epitope hotspots in red
                    if epitope_results:
                        epitope_positions = set()
                        for ep in epitope_results:
                            epitope_positions.update(range(ep.start, ep.end + 1))
                        for pos in epitope_positions:
                            view.addStyle(
                                {"resi": pos},
                                {"cartoon": {"color": "red"}, "stick": {"color": "red"}},
                            )

                    view.zoomTo()
                    stmol.showmol(view, height=400, width=600)
                except Exception as e:
                    st.info(f"3D viewer unavailable: {e}. Showing sequence view.")
                    pdb_data = None

            if not pdb_data:
                st.info("Set `TAMARIND_API_KEY` for 3D protein folding. Showing sequence view.")
                # Show sequence with epitope highlighting
                if epitope_results:
                    epitope_positions = set()
                    for ep in epitope_results:
                        epitope_positions.update(range(ep.start, ep.end + 1))

                    # Render sequence with color blocks
                    seq_html = '<div style="font-family:monospace; word-wrap:break-word; line-height:1.8">'
                    for i, aa in enumerate(parsed_seq):
                        pos = i + 1  # 1-indexed
                        if pos in epitope_positions:
                            seq_html += f'<span style="background:#ff4444; color:white; padding:1px 2px; border-radius:2px" title="Epitope at {pos}">{aa}</span>'
                        else:
                            seq_html += f'<span style="color:#666" title="Position {pos}">{aa}</span>'
                        if pos % 60 == 0:
                            seq_html += f' <span style="color:#999; font-size:0.8em">{pos}</span><br>'
                    seq_html += "</div>"
                    st.markdown(seq_html, unsafe_allow_html=True)
                    st.caption("Red = predicted T-cell epitope hotspots")
                else:
                    # Plain sequence display
                    formatted = "\n".join(
                        parsed_seq[i:i + 60] + f"  {min(i + 60, len(parsed_seq))}"
                        for i in range(0, len(parsed_seq), 60)
                    )
                    st.code(formatted)

        with col_epitopes:
            st.markdown("#### T-cell Epitope Prediction")

            # Run IEDB prediction
            with st.spinner("Querying IEDB API..."):
                epitope_results = predict_epitopes(parsed_seq)

            if epitope_results:
                density = compute_epitope_density(epitope_results, len(parsed_seq))
                st.metric("Epitope Density", f"{density:.1f} per 100 residues")
                st.metric("High-Affinity Binders", len(epitope_results))

                ep_df = pd.DataFrame([
                    {
                        "Position": f"{ep.start}-{ep.end}",
                        "Peptide": ep.peptide,
                        "Rank": round(ep.percentile_rank, 2),
                        "Allele": ep.allele,
                    }
                    for ep in epitope_results[:30]
                ])
                st.dataframe(ep_df, hide_index=True, use_container_width=True)
            else:
                st.info("No epitope predictions available (IEDB API may be unreachable).")

            # Sequence diffs from closest match
            if sequence_diffs:
                st.markdown(f"#### Mutations vs. Closest Match ({len(sequence_diffs)} differences)")
                diff_df = pd.DataFrame([
                    {"Position": pos, "Query": q_aa, "Reference": r_aa}
                    for pos, q_aa, r_aa in sequence_diffs[:30]
                ])
                st.dataframe(diff_df, hide_index=True, use_container_width=True)

# ==============================
# PART 3 — Redesign Copilot
# ==============================
with tab3:
    user_inputs = {
        "modality": modality,
        "species": species,
        "route": route,
        "disease": disease,
        "conjugate": conjugate,
        "backbone": backbone,
        "dose": dose,
        "schedule": schedule,
    }

    # --- Redesigned Sequences Section ---
    if parsed_seq and epitope_results:
        st.markdown("### Deimmunized Sequence Variants")
        st.caption("Anchor residues in predicted T-cell epitopes are substituted with conservative alternatives to disrupt MHC-II binding.")

        deimm_results = deimmunize_epitopes(parsed_seq, epitope_results)
        variants = generate_redesigned_sequences(parsed_seq, deimm_results)

        if variants:
            # Mutation map overview
            st.markdown("#### Mutation Map")
            mut_df = pd.DataFrame([
                {
                    "Region": f"{dr.region_start}-{dr.region_end}",
                    "Original": dr.original_peptide,
                    "Modified": dr.modified_peptide,
                    "Mutations": ", ".join(f"{o}{p}{n}" for p, o, n in dr.mutations),
                    "Anchors": ", ".join(dr.anchor_positions_targeted),
                    "Disruption": dr.expected_binding_disruption,
                    "Allele": dr.allele,
                    "Rank": dr.original_rank,
                }
                for dr in deimm_results
            ])
            st.dataframe(mut_df, hide_index=True, use_container_width=True)

            st.divider()

            # Sequence variants
            for i, var in enumerate(variants):
                with st.expander(f"Variant {i+1}: {var.name} — {var.n_mutations} mutations", expanded=(i == 0)):
                    st.markdown(f"**Strategy:** {var.strategy}")

                    # Show mutations as a table
                    var_mut_df = pd.DataFrame([
                        {"Position": pos, "Original": orig, "New": new}
                        for pos, orig, new in var.mutations
                    ])
                    st.dataframe(var_mut_df, hide_index=True)

                    # Show the sequence with mutations highlighted
                    seq_html = '<div style="font-family:monospace; word-wrap:break-word; line-height:1.8; font-size:0.85em">'
                    mut_positions = {pos for pos, _, _ in var.mutations}
                    for j, aa in enumerate(var.sequence):
                        pos = j + 1
                        if pos in mut_positions:
                            orig_aa = parsed_seq[j]
                            seq_html += f'<span style="background:#2ecc71; color:white; padding:1px 3px; border-radius:2px; font-weight:bold" title="Pos {pos}: {orig_aa}→{aa}">{aa}</span>'
                        else:
                            seq_html += aa
                        if pos % 60 == 0:
                            seq_html += f' <span style="color:#999; font-size:0.8em">{pos}</span><br>'
                    seq_html += "</div>"
                    st.markdown(seq_html, unsafe_allow_html=True)
                    st.caption("Green = mutated positions")

                    # Copy-ready FASTA
                    fasta = f">SafeBind_{var.name.replace(' ', '_')}|{var.n_mutations}_mutations\n"
                    fasta += "\n".join(var.sequence[k:k+60] for k in range(0, len(var.sequence), 60))

                    col_copy, col_dl = st.columns(2)
                    with col_copy:
                        st.code(fasta, language=None)
                    with col_dl:
                        st.download_button(
                            f"Download FASTA",
                            fasta,
                            f"safebind_variant_{i+1}.fasta",
                            mime="text/plain",
                            key=f"dl_var_{i}",
                        )
        else:
            st.info("Could not generate deimmunized variants — no actionable epitope mutations found.")

    elif parsed_seq and not epitope_results:
        st.info("Waiting for epitope predictions (IEDB API) to generate redesigned sequences.")
    else:
        st.info("Provide a protein sequence in the sidebar to generate deimmunized variants.")

    st.divider()

    # --- AI Risk Memo Section ---
    st.markdown("### AI Risk Memo")

    memo = generate_risk_memo(
        risk_result=risk_result,
        user_inputs=user_inputs,
        sequence_diffs=sequence_diffs,
        epitope_results=epitope_results,
    )

    st.markdown(memo)

    st.download_button(
        label="Download Risk Memo (Markdown)",
        data=memo,
        file_name="safebind_risk_memo.md",
        mime="text/markdown",
    )
