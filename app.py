"""SafeBind Risk — Immunogenicity Risk Assessment Dashboard."""

import streamlit as st
import pandas as pd
import altair as alt
import os

from config import (
    MODALITY_OPTIONS, SPECIES_OPTIONS, ROUTE_OPTIONS,
    DISEASE_OPTIONS, CONJUGATE_OPTIONS, BACKBONE_OPTIONS,
    EXPRESSION_SYSTEM_OPTIONS,
    W_LOOKUP, W_SEQUENCE, W_FEATURE,
)
from data_loader import (
    load_therapeutic, load_sequences, load_clinical,
    build_lookup_table, build_drug_ada_map, get_historical_precedents,
    build_nada_lookup, build_time_ada_lookup,
)
from risk_model import predict_ada
from sequence_engine import (
    parse_multi_fasta, align_to_references, get_sequence_diffs,
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
    "Protein Sequence (multi-chain FASTA or raw)",
    height=180,
    placeholder=">Heavy Chain 1\nEVQLVESGGG...\n>Heavy Chain 2\nEVQLVESGGG...\n>Light Chain\nDIQMTQSPS...",
)

st.sidebar.divider()
modality = st.sidebar.selectbox("Protein Modality", MODALITY_OPTIONS)
species = st.sidebar.selectbox("Species Origin", SPECIES_OPTIONS)
route = st.sidebar.selectbox("Route of Administration", ROUTE_OPTIONS)
disease = st.sidebar.selectbox("Disease Indication", DISEASE_OPTIONS)
conjugate = st.sidebar.selectbox("Conjugate Modification", CONJUGATE_OPTIONS)
backbone = st.sidebar.selectbox("Antibody Backbone", BACKBONE_OPTIONS)
expression_system = st.sidebar.selectbox("Expression System", EXPRESSION_SYSTEM_OPTIONS)

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
    - **T-cell Epitope Prediction** — IEDB MHC-II binding analysis per chain
    - **AI Synthesis** — Claude-powered risk memo with per-chain modification recommendations

    Paste multi-chain FASTA (with `>headers`) or a single raw sequence, configure in sidebar, and click **Analyze Risk**.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Approved Drugs", "218")
    with col2:
        st.metric("Clinical Cohorts", "3,334")
    with col3:
        st.metric("Reference Sequences", "222")
    st.stop()


# --- Run Analysis ---
lookup_tables = build_lookup_table()
drug_ada_map = build_drug_ada_map()
nada_lookup = build_nada_lookup()
time_ada_lookup = build_time_ada_lookup()
sequences_df = load_sequences()

# Parse multi-chain sequences
chains = {}  # {chain_name: sequence}
if sequence_input.strip():
    try:
        chains = parse_multi_fasta(sequence_input)
        chain_summary = ", ".join(f"{name} ({len(seq)}aa)" for name, seq in chains.items())
        st.sidebar.success(f"{len(chains)} chain(s): {chain_summary}")
    except ValueError as e:
        st.sidebar.error(str(e))

# Per-chain analysis results
chain_alignments = {}   # {chain_name: [AlignmentResult]}
chain_epitopes = {}     # {chain_name: [EpitopeResult]}
chain_diffs = {}        # {chain_name: [(pos, q, r)]}

# Run alignment per chain
for chain_name, seq in chains.items():
    chain_alignments[chain_name] = align_to_references(seq, sequences_df)

# Aggregate: use the best alignment across all chains for risk model
all_alignments = []
for name, results in chain_alignments.items():
    for r in results:
        r.chain_descriptor = f"{name} → {r.chain_descriptor}"
        all_alignments.append(r)
# Sort by score descending, deduplicate by INN
seen_inns = set()
best_alignments = []
for r in sorted(all_alignments, key=lambda x: x.score, reverse=True):
    if r.inn_name not in seen_inns:
        seen_inns.add(r.inn_name)
        best_alignments.append(r)
    if len(best_alignments) >= 5:
        break

# Run risk prediction
risk_result = predict_ada(
    modality=modality,
    species=species,
    route=route,
    disease=disease,
    conjugate=conjugate,
    lookup_tables=lookup_tables,
    drug_ada_map=drug_ada_map,
    alignment_results=best_alignments if best_alignments else None,
    expression_system=expression_system,
    nada_lookup=nada_lookup,
    time_ada_lookup=time_ada_lookup,
)

# Per-chain sequence diffs
for chain_name, results in chain_alignments.items():
    if results:
        best = results[0]
        chain_diffs[chain_name] = get_sequence_diffs(chains[chain_name], best.ref_sequence)

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
                f"Top {len(best_alignments)} matches across {len(chains)} chain(s)" if best_alignments else "No sequence",
                "Modality + Species + Conjugate + Route",
            ],
        }
        st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True)

    st.divider()

    col_chart, col_factors = st.columns([3, 2])

    with col_chart:
        st.markdown(f"#### Benchmarking: Your Candidate vs. {disease} + {route}")
        precedents = get_historical_precedents(route, disease, modality)

        if len(precedents) > 0:
            chart_data = precedents[["Therapeutic Assessed for ADA INN Name", "ada_pct", "total_patients"]].copy()
            chart_data.columns = ["Drug", "ADA Rate (%)", "Patients"]
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

        # Nearest drugs by sequence (aggregated across chains)
        if best_alignments:
            st.markdown("#### Nearest Drugs (by sequence)")
            for r in best_alignments:
                ada_str = f" — ADA: {drug_ada_map[r.inn_name]:.1f}%" if r.inn_name in drug_ada_map else ""
                st.markdown(f"- **{r.inn_name}** ({r.chain_descriptor}): {r.pct_identity:.0%} identity{ada_str}")

    # nADA + Time-to-ADA row
    st.divider()
    col_nada, col_time = st.columns(2)

    with col_nada:
        st.markdown("#### Neutralizing ADA Risk")
        if risk_result.nada:
            nada = risk_result.nada
            st.markdown(
                f"""<div style="padding:12px; background:{nada.severity_color}22;
                border-radius:8px; border-left:4px solid {nada.severity_color}">
                <span style="font-size:1.8rem; font-weight:bold; color:{nada.severity_color}">{nada.nada_pct}%</span>
                <span style="color:{nada.severity_color}; margin-left:8px">{nada.severity}</span>
                <br><span style="font-size:0.85rem; color:#666">{nada.description}</span>
                <br><span style="font-size:0.8rem; color:#999">nADA/ADA ratio: {nada.nada_ratio} | Source: {nada.source}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("nADA data not available.")

    with col_time:
        st.markdown("#### ADA Onset Timeline")
        if risk_result.time_ada:
            tad = risk_result.time_ada
            st.markdown(f"**Expected peak:** {tad.expected_onset} (ADA rate: {tad.peak_ada_pct}%)")

            if tad.profile:
                time_df = pd.DataFrame([
                    {"Time Window": tb, "ADA Rate (%)": round(ada, 1), "Cohorts": n}
                    for tb, ada, n in tad.profile
                ])
                time_chart = alt.Chart(time_df).mark_bar().encode(
                    x=alt.X("Time Window:N", sort=None),
                    y=alt.Y("ADA Rate (%):Q"),
                    color=alt.value("#4a90d9"),
                    tooltip=["Time Window", "ADA Rate (%)", "Cohorts"],
                ).properties(height=200)
                st.altair_chart(time_chart, use_container_width=True)
        else:
            st.info("Time-to-ADA data not available.")

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
# PART 2 — Structural Risk Viewer (per-chain)
# ==============================
with tab2:
    if not chains:
        st.warning("Provide a protein sequence in the sidebar to enable structural analysis.")
    else:
        st.markdown(f"### Structural Analysis — {len(chains)} Chain(s)")

        # Create a tab per chain
        chain_tabs = st.tabs(list(chains.keys()))

        for chain_tab, (chain_name, chain_seq) in zip(chain_tabs, chains.items()):
            with chain_tab:
                st.markdown(f"**{chain_name}** — {len(chain_seq)} residues")

                col_seq, col_ep = st.columns([3, 2])

                with col_seq:
                    # Run IEDB for this chain
                    ep_key = chain_name
                    if ep_key not in chain_epitopes:
                        chain_epitopes[ep_key] = predict_epitopes(chain_seq)

                    ep_results = chain_epitopes[ep_key]

                    # Try 3D view
                    tamarind_key = os.environ.get("TAMARIND_API_KEY")
                    pdb_data = None
                    if tamarind_key:
                        try:
                            from tamarind_integration import fold_protein
                            pdb_data = fold_protein(chain_seq, tamarind_key)
                        except Exception:
                            pass

                    if pdb_data:
                        try:
                            import stmol
                            import py3Dmol
                            view = py3Dmol.view(width=600, height=400)
                            view.addModel(pdb_data, "pdb")
                            view.setStyle({"cartoon": {"color": "spectrum"}})
                            if ep_results:
                                for ep in ep_results:
                                    for pos in range(ep.start, ep.end + 1):
                                        view.addStyle({"resi": pos}, {"cartoon": {"color": "red"}, "stick": {"color": "red"}})
                            view.zoomTo()
                            stmol.showmol(view, height=400, width=600)
                        except Exception:
                            pdb_data = None

                    if not pdb_data:
                        # Sequence view with epitope highlighting
                        if ep_results:
                            epitope_positions = set()
                            for ep in ep_results:
                                epitope_positions.update(range(ep.start, ep.end + 1))

                            seq_html = '<div style="font-family:monospace; word-wrap:break-word; line-height:1.8">'
                            for i, aa in enumerate(chain_seq):
                                pos = i + 1
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
                            formatted = "\n".join(
                                chain_seq[i:i + 60] + f"  {min(i + 60, len(chain_seq))}"
                                for i in range(0, len(chain_seq), 60)
                            )
                            st.code(formatted)

                with col_ep:
                    st.markdown("#### T-cell Epitopes")

                    if ep_results:
                        density = compute_epitope_density(ep_results, len(chain_seq))
                        c1, c2 = st.columns(2)
                        c1.metric("Density", f"{density:.1f}/100aa")
                        c2.metric("Binders", len(ep_results))

                        ep_df = pd.DataFrame([
                            {
                                "Position": f"{ep.start}-{ep.end}",
                                "Peptide": ep.peptide,
                                "Rank": round(ep.percentile_rank, 2),
                                "Allele": ep.allele,
                            }
                            for ep in ep_results[:25]
                        ])
                        st.dataframe(ep_df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No epitope predictions (IEDB may be unreachable).")

                    # Alignment results for this chain
                    if chain_name in chain_alignments and chain_alignments[chain_name]:
                        st.markdown("#### Nearest Matches")
                        for r in chain_alignments[chain_name][:3]:
                            ada_str = f" (ADA: {drug_ada_map[r.inn_name]:.1f}%)" if r.inn_name in drug_ada_map else ""
                            st.markdown(f"- **{r.inn_name}** {r.chain_descriptor}: {r.pct_identity:.0%}{ada_str}")

                    # Sequence diffs
                    diffs = chain_diffs.get(chain_name, [])
                    if diffs:
                        st.markdown(f"#### Diffs vs. Closest ({len(diffs)})")
                        st.dataframe(
                            pd.DataFrame([{"Pos": p, "Query": q, "Ref": r} for p, q, r in diffs[:20]]),
                            hide_index=True, use_container_width=True,
                        )

        # Cross-chain summary
        if len(chains) > 1:
            st.divider()
            st.markdown("#### Cross-Chain Epitope Summary")
            summary_rows = []
            for name in chains:
                eps = chain_epitopes.get(name, [])
                summary_rows.append({
                    "Chain": name,
                    "Length (aa)": len(chains[name]),
                    "Epitopes": len(eps),
                    "Density (/100aa)": round(compute_epitope_density(eps, len(chains[name])), 1) if eps else 0,
                    "Top Binder Rank": round(eps[0].percentile_rank, 2) if eps else "—",
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# ==============================
# PART 3 — Redesign Copilot (per-chain)
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

    if chains and any(chain_epitopes.get(name) for name in chains):
        st.markdown("### Deimmunized Sequence Variants")
        st.caption("Per-chain deimmunization: anchor residues in predicted T-cell epitopes are substituted to disrupt MHC-II binding.")

        # Per-chain redesign tabs
        redesign_chain_tabs = st.tabs(list(chains.keys()))

        # Collect all variants for combined FASTA download
        all_chain_variants = {}  # {chain_name: [RedesignedSequence]}

        for rtab, (chain_name, chain_seq) in zip(redesign_chain_tabs, chains.items()):
            with rtab:
                ep_results = chain_epitopes.get(chain_name, [])
                if not ep_results:
                    st.info(f"No epitope data for {chain_name} — skipping deimmunization.")
                    continue

                deimm_results = deimmunize_epitopes(chain_seq, ep_results)
                variants = generate_redesigned_sequences(chain_seq, deimm_results)
                all_chain_variants[chain_name] = variants

                if not variants:
                    st.info(f"No actionable mutations found for {chain_name}.")
                    continue

                # Mutation map
                st.markdown(f"#### Mutation Map — {chain_name}")
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

                # Variant expanders
                for i, var in enumerate(variants):
                    with st.expander(f"Variant {i+1}: {var.name} — {var.n_mutations} mutations", expanded=(i == 0)):
                        st.markdown(f"**Strategy:** {var.strategy}")

                        var_mut_df = pd.DataFrame([
                            {"Position": pos, "Original": orig, "New": new}
                            for pos, orig, new in var.mutations
                        ])
                        st.dataframe(var_mut_df, hide_index=True)

                        # Highlighted sequence
                        seq_html = '<div style="font-family:monospace; word-wrap:break-word; line-height:1.8; font-size:0.85em">'
                        mut_positions = {pos for pos, _, _ in var.mutations}
                        for j, aa in enumerate(var.sequence):
                            pos = j + 1
                            if pos in mut_positions:
                                orig_aa = chain_seq[j]
                                seq_html += f'<span style="background:#2ecc71; color:white; padding:1px 3px; border-radius:2px; font-weight:bold" title="Pos {pos}: {orig_aa}→{aa}">{aa}</span>'
                            else:
                                seq_html += aa
                            if pos % 60 == 0:
                                seq_html += f' <span style="color:#999; font-size:0.8em">{pos}</span><br>'
                        seq_html += "</div>"
                        st.markdown(seq_html, unsafe_allow_html=True)
                        st.caption("Green = mutated positions")

                        # Per-chain FASTA
                        safe_name = chain_name.replace(" ", "_")
                        fasta = f">{safe_name}|{var.name.replace(' ', '_')}|{var.n_mutations}_mutations\n"
                        fasta += "\n".join(var.sequence[k:k+60] for k in range(0, len(var.sequence), 60))

                        st.download_button(
                            f"Download {chain_name} FASTA",
                            fasta,
                            f"safebind_{safe_name}_variant_{i+1}.fasta",
                            mime="text/plain",
                            key=f"dl_{safe_name}_{i}",
                        )

        # Combined multi-chain FASTA download (one variant level across all chains)
        if len(chains) > 1 and all_chain_variants:
            st.divider()
            st.markdown("### Download Combined Multi-Chain FASTA")
            st.caption("Downloads all chains at the selected deimmunization level in a single FASTA file.")

            # Find which variant levels are available across all chains
            variant_levels = ["Conservative", "Moderate", "Aggressive"]
            for level in variant_levels:
                combined_fasta = ""
                chain_count = 0
                for cname, variants in all_chain_variants.items():
                    for var in variants:
                        if level.lower() in var.name.lower():
                            safe_name = cname.replace(" ", "_")
                            combined_fasta += f">{safe_name}|{var.name.replace(' ', '_')}|{var.n_mutations}_mutations\n"
                            combined_fasta += "\n".join(var.sequence[k:k+60] for k in range(0, len(var.sequence), 60))
                            combined_fasta += "\n"
                            chain_count += 1
                            break
                    else:
                        # No variant at this level — use original
                        safe_name = cname.replace(" ", "_")
                        combined_fasta += f">{safe_name}|original\n"
                        combined_fasta += "\n".join(chains[cname][k:k+60] for k in range(0, len(chains[cname]), 60))
                        combined_fasta += "\n"
                        chain_count += 1

                if combined_fasta:
                    st.download_button(
                        f"Download {level} — All {chain_count} Chains",
                        combined_fasta,
                        f"safebind_all_chains_{level.lower()}.fasta",
                        mime="text/plain",
                        key=f"dl_combined_{level}",
                    )

    elif chains:
        st.info("Waiting for epitope predictions (IEDB API) to generate redesigned sequences.")
    else:
        st.info("Provide a protein sequence in the sidebar to generate deimmunized variants.")

    st.divider()

    # --- AI Risk Memo ---
    st.markdown("### AI Risk Memo")

    # Flatten epitope + diff data for the memo
    all_epitopes = []
    all_diffs = []
    for name in chains:
        for ep in chain_epitopes.get(name, []):
            ep_copy = type(ep)(start=ep.start, end=ep.end, peptide=ep.peptide,
                              percentile_rank=ep.percentile_rank, allele=f"[{name}] {ep.allele}")
            all_epitopes.append(ep_copy)
        for pos, q, r in chain_diffs.get(name, []):
            all_diffs.append((pos, q, r))

    memo = generate_risk_memo(
        risk_result=risk_result,
        user_inputs=user_inputs,
        sequence_diffs=all_diffs if all_diffs else None,
        epitope_results=all_epitopes if all_epitopes else None,
    )

    st.markdown(memo)

    st.download_button(
        label="Download Risk Memo (Markdown)",
        data=memo,
        file_name="safebind_risk_memo.md",
        mime="text/markdown",
    )
