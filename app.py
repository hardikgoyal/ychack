"""SafeBind Risk — Immunogenicity Risk Assessment Dashboard."""

import streamlit as st
import pandas as pd
import altair as alt
import os
import time

from config import (
    MODALITY_OPTIONS, SPECIES_OPTIONS, ROUTE_OPTIONS,
    DISEASE_OPTIONS, CONJUGATE_OPTIONS, BACKBONE_OPTIONS,
    EXPRESSION_SYSTEM_OPTIONS,
    W_LOOKUP, W_SEQUENCE, W_FEATURE,
)

# ── Drug Presets ──────────────────────────────────────────────
DRUG_PRESETS = {
    "Custom": {
        "sequence": "",
        "modality": "Monoclonal Antibody",
        "species": "Human",
        "route": "Intravenous",
        "disease": "Cancer and neoplasms",
        "backbone": "human IgG1",
        "conjugate": "Unconjugated",
        "expression_system": "Chinese hamster ovary (CHO) cells",
        "dose": "",
        "schedule": "",
    },
    "Odronextamab (CD20xCD3 bispecific)": {
        "sequence": """>Heavy Chain 1 (anti-CD20)
EVQLVESGGGLVQPGRSLRLSCVASGFTFNDYAMHWVRQAPGKGLEWVSVISWNSDSIGY
ADSVKGRFTISRDNAKNSLYLQMHSLRAEDTALYYCAKDNHYGSGSYYYYQYGMDVWGQG
TTVTVSSASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTF
PAVLQSSGLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRVESKYGPPCPPCPAPP
VAGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSQEDPEVQFNWYVDGVEVHNAKTKPREE
QFNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKGLPSSIEKTISKAKGQPREPQVYTLPPS
QEEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSRLTVDK
SRWQEGNVFSCSVMHEALHNHYTQKSLSLSLG
>Heavy Chain 2 (anti-CD3)
EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYTMHWVRQAPGKGLEWVSGISWNSGSIGY
ADSVKGRFTISRDNAKKSLYLQMNSLRAEDTALYYCAKDNSGYGHYYYGMDVWGQGTTVT
VASASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVL
QSSGLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRVESKYGPPCPPCPAPPVAGP
SVFLFPPKPKDTLMISRTPEVTCVVVDVSQEDPEVQFNWYVDGVEVHNAKTKPREEQFNS
TYRVVSVLTVLHQDWLNGKEYKCKVSNKGLPSSIEKTISKAKGQPREPQVYTLPPSQEEM
TKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSRLTVDKSRWQ
EGNVFSCSVMHEALHNRFTQKSLSLSLG
>Light Chain
EIVMTQSPATLSVSPGERATLSCRASQSVSSNLAWYQQKPGQAPRLLIYGASTRATGIPA
RFSGSGSGTEFTLTISSLQSEDFAVYYCQHYINWPLTFGGGTKVEIKRTVAAPSVFIFPP
SDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLT
LSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC""",
        "modality": "IgG-like Bispecific",
        "species": "Human",
        "route": "Intravenous",
        "disease": "Cancer and neoplasms",
        "backbone": "Human IgG4",
        "conjugate": "Unconjugated",
        "expression_system": "Chinese hamster ovary (CHO) cells",
        "dose": "80 mg weekly / 160 mg Q2W",
        "schedule": "Step-up: 0.2 > 2 > 20 > 80 mg weekly, then 160 mg Q2W IV",
    },
    "Adalimumab (anti-TNF mAb)": {
        "sequence": "",
        "modality": "Monoclonal Antibody",
        "species": "Human",
        "route": "Subcutaneous",
        "disease": "Inflammation and autoimmunity",
        "backbone": "human IgG1",
        "conjugate": "Unconjugated",
        "expression_system": "Chinese hamster ovary (CHO) cells",
        "dose": "40 mg",
        "schedule": "Q2W SC",
    },
    "Infliximab (anti-TNF chimeric)": {
        "sequence": "",
        "modality": "Monoclonal Antibody",
        "species": "Chimeric",
        "route": "Intravenous",
        "disease": "Inflammation and autoimmunity",
        "backbone": "human IgG1",
        "conjugate": "Unconjugated",
        "expression_system": "Murine myeloma cells",
        "dose": "5 mg/kg",
        "schedule": "Q8W IV after induction",
    },
    "Brentuximab vedotin (ADC)": {
        "sequence": "",
        "modality": "Antibody-drug Conjugate",
        "species": "Chimeric",
        "route": "Intravenous",
        "disease": "Cancer and neoplasms",
        "backbone": "human IgG1",
        "conjugate": "Drug Conjugate",
        "expression_system": "Chinese hamster ovary (CHO) cells",
        "dose": "1.8 mg/kg",
        "schedule": "Q3W IV",
    },
    "Moxetumomab pasudotox (scFv immunotoxin)": {
        "sequence": """>VL
DIQMTQTTSSLSASLGDRVTISCRASQDISKYLNWYQQKPDGTVKLLIYHTSRLHSGVPS
RFSGSGSGTDYSLTISNLEQEDIATYFCQQGNTLPYTFGGGTKLEIT
>Linker (Whitlow)
GSTSGSGKPGSGEGSTKG
>VH
EVKLQESGPGLVAPSQSLSVTCTVSGVSLPDYGVSWIRQPPRKGLEWLGVIWGSETTYYN
SALKSRLTIIKDNSKSQVFLKMNSLQTDDTAIYYCAKHYYYGGSYAMDYWGQGTSVTVSS""",
        "modality": "Immunotoxin",
        "species": "Mouse",
        "route": "Intravenous",
        "disease": "Cancer and neoplasms",
        "backbone": "ScFv",
        "conjugate": "Unconjugated",
        "expression_system": "E. coli bacteria",
        "dose": "40 mcg/kg",
        "schedule": "QOD x3 per 28-day cycle IV",
    },
}
from data_loader import (
    load_therapeutic, load_sequences, load_clinical,
    build_lookup_table, build_drug_ada_map, get_historical_precedents,
    build_nada_lookup, build_time_ada_lookup,
)
from risk_model import predict_ada
from sequence_engine import (
    parse_multi_fasta, align_to_references, get_sequence_diffs,
    predict_epitopes, compute_epitope_density,
    predict_bcell_epitopes, calculate_sasa_from_pdb, is_surface_exposed,
    filter_surface_epitopes, detect_cdr_regions, check_cdr_epitope_overlap,
)
from claude_report import generate_risk_memo
from deimmunize import (
    deimmunize_epitopes, generate_redesigned_sequences,
    run_tolerance_analysis,
)
from tamarind_integration import (
    fold_protein, suggest_redesigns, _get_api_key,
    submit_fold_job, check_fold_status, fetch_fold_result,
)

st.set_page_config(
    page_title="SafeBind Risk",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global CSS ---
st.markdown("""
<style>
    /* Tighter header spacing */
    h1, h2, h3, h4 { margin-top: 0.3rem; margin-bottom: 0.2rem; }
    /* Cards */
    .sb-card {
        padding: 1.2rem; border-radius: 10px; margin-bottom: 1rem;
        border: 1px solid rgba(128,128,128,0.15);
    }
    /* Metric overrides — less padding */
    [data-testid="stMetric"] { padding: 0.4rem 0; }
    /* Tab content spacing */
    [data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    /* Sidebar breathing room */
    section[data-testid="stSidebar"] .stSelectbox { margin-bottom: -0.5rem; }
    /* Better expander styling */
    .streamlit-expanderHeader { font-size: 0.95rem; font-weight: 600; }
    /* Hide fullscreen buttons on small charts */
    button[title="View fullscreen"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("### SafeBind Risk")
st.sidebar.caption("Immunogenicity Risk Assessment")

# Preset callback — runs BEFORE widgets re-render, so state is ready
def _apply_preset():
    name = st.session_state["_sb_preset"]
    p = DRUG_PRESETS[name]
    st.session_state["_sb_seq"] = p["sequence"]
    st.session_state["_sb_mod"] = p["modality"]
    st.session_state["_sb_route"] = p["route"]
    st.session_state["_sb_disease"] = p["disease"]
    st.session_state["_sb_bone"] = p["backbone"]
    st.session_state["_sb_species"] = p["species"]
    st.session_state["_sb_conj"] = p["conjugate"]
    st.session_state["_sb_expr"] = p["expression_system"]
    st.session_state["_sb_dose"] = p.get("dose", "")
    st.session_state["_sb_sched"] = p.get("schedule", "")

preset_name = st.sidebar.selectbox(
    "Drug Preset", list(DRUG_PRESETS.keys()), key="_sb_preset", on_change=_apply_preset,
)
preset = DRUG_PRESETS[preset_name]

# Initialize defaults on first load
if "_sb_mod" not in st.session_state:
    p = DRUG_PRESETS[preset_name]
    st.session_state["_sb_seq"] = p["sequence"]
    st.session_state["_sb_mod"] = p["modality"]
    st.session_state["_sb_route"] = p["route"]
    st.session_state["_sb_disease"] = p["disease"]
    st.session_state["_sb_bone"] = p["backbone"]
    st.session_state["_sb_species"] = p["species"]
    st.session_state["_sb_conj"] = p["conjugate"]
    st.session_state["_sb_expr"] = p["expression_system"]
    st.session_state["_sb_dose"] = p.get("dose", "")
    st.session_state["_sb_sched"] = p.get("schedule", "")

sequence_input = st.sidebar.text_area(
    "Protein Sequence",
    height=120 if not preset["sequence"] else 80,
    placeholder=">Heavy Chain\nEVQLVESGGG...\n>Light Chain\nDIQMTQSPS...",
    key="_sb_seq",
)

modality = st.sidebar.selectbox("Modality", MODALITY_OPTIONS, key="_sb_mod")
route = st.sidebar.selectbox("Route", ROUTE_OPTIONS, key="_sb_route")
disease = st.sidebar.selectbox("Disease", DISEASE_OPTIONS, key="_sb_disease")
backbone = st.sidebar.selectbox("Backbone", BACKBONE_OPTIONS, key="_sb_bone")

with st.sidebar.expander("More"):
    species = st.selectbox("Species Origin", SPECIES_OPTIONS, key="_sb_species")
    conjugate = st.selectbox("Conjugate", CONJUGATE_OPTIONS, key="_sb_conj")
    expression_system = st.selectbox("Expression System", EXPRESSION_SYSTEM_OPTIONS, key="_sb_expr")
    dose = st.text_input("Dose", placeholder="e.g., 10 mg/kg", key="_sb_dose")
    schedule = st.text_input("Schedule", placeholder="e.g., Q2W IV", key="_sb_sched")

tamarind_key = _get_api_key()

st.sidebar.markdown("")
if st.sidebar.button("Analyze Risk", type="primary", use_container_width=True):
    st.session_state["analyze_clicked"] = True
analyze = st.session_state.get("analyze_clicked", False)

# --- Main Area ---
if not analyze:
    st.markdown("")
    col_hero_l, col_hero_r = st.columns([2, 1])
    with col_hero_l:
        st.markdown("# SafeBind Risk")
        st.markdown(
            "Predict anti-drug antibody (ADA) risk for biotherapeutic candidates. "
            "Built on clinical data from 218 approved drugs and 3,334 trial cohorts."
        )
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Drugs in Database", "218")
        c2.metric("Clinical Cohorts", "3,334")
        c3.metric("Reference Sequences", "222")

    with col_hero_r:
        st.markdown("""
        <div class="sb-card" style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); margin-top: 1rem;">
        <p style="font-weight: 600; margin-bottom: 0.6rem;">Get started</p>
        <ol style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; line-height: 1.7;">
            <li>Pick a <strong>drug preset</strong> or paste your own sequence</li>
            <li>Set modality, route, and disease</li>
            <li>Click <strong>Analyze Risk</strong></li>
        </ol>
        <p style="font-size: 0.82rem; color: #888; margin-top: 0.8rem; margin-bottom: 0;">
        Try the <strong>Odronextamab</strong> preset for a full demo with 3 chains.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown(
        '<p style="text-align:center; color:#aaa; font-size:0.85rem;">'
        'Empirical benchmarking &middot; Sequence similarity &middot; T/B-cell epitopes &middot; '
        '3D structure &middot; AI-powered redesign</p>',
        unsafe_allow_html=True,
    )
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
chain_bcell = {}        # {chain_name: [BCellEpitope]}
chain_cdr = {}          # {chain_name: [cdr_dict]}
chain_sasa = {}         # {chain_name: {residue: sasa}}
chain_diffs = {}        # {chain_name: [(pos, q, r)]}
chain_pdb = {}          # {chain_name: pdb_string} — populated in Tab 2 if Tamarind key present

# Run alignment + B-cell prediction per chain (before tabs so all tabs can use results)
for i, (chain_name, seq) in enumerate(chains.items()):
    chain_alignments[chain_name] = align_to_references(seq, sequences_df)
    # Stagger IEDB B-cell calls to avoid rate-limiting
    if i > 0:
        time.sleep(2)
    chain_bcell[chain_name] = predict_bcell_epitopes(seq)

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
st.markdown("## SafeBind Risk Assessment")

tab1, tab2, tab_bcell, tab3 = st.tabs([
    "Prediction",
    "Structure",
    "B-cell Epitopes",
    "Redesign",
])

with tab1:
    # --- Hero score ---
    col_score, col_breakdown, col_conf = st.columns([1, 2, 1])
    with col_score:
        st.markdown(
            f"""<div style="text-align:center; padding:1.5rem 1rem; background:{risk_result.tier_color}15;
            border-radius:12px; border:2px solid {risk_result.tier_color}">
            <div style="color:{risk_result.tier_color}; font-size:3rem; font-weight:700; line-height:1">{risk_result.composite_score}%</div>
            <div style="color:{risk_result.tier_color}; font-size:1rem; font-weight:600; margin-top:0.3rem">{risk_result.risk_tier} Risk</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if risk_result.nada:
            nada = risk_result.nada
            st.markdown(
                f"""<div style="text-align:center; padding:0.8rem; background:{nada.severity_color}12;
                border-radius:8px; margin-top:0.5rem; border:1px solid {nada.severity_color}30">
                <div style="color:{nada.severity_color}; font-size:1.4rem; font-weight:600">{nada.nada_pct}%</div>
                <div style="font-size:0.8rem; color:#888">Neutralizing ADA</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with col_breakdown:
        components = ["Clinical Lookup", "Sequence Similarity", "Feature Adjustment"]
        scores = [
            risk_result.lookup_score,
            risk_result.sequence_score if risk_result.sequence_score is not None else "N/A",
            risk_result.feature_score,
        ]
        weights = [
            f"{W_LOOKUP:.0%}",
            f"{W_SEQUENCE:.0%}" if risk_result.sequence_score is not None else "—",
            f"{W_FEATURE:.0%}",
        ]
        sources = [
            risk_result.lookup_level,
            f"Top {len(best_alignments)} matches" if best_alignments else "No sequence",
            "Modality + Species + Route",
        ]
        st.dataframe(
            pd.DataFrame({"Component": components, "Score (%)": scores, "Weight": weights, "Source": sources}),
            hide_index=True, use_container_width=True,
        )

        if risk_result.risk_factors:
            for f in risk_result.risk_factors:
                icon = "+" if "increases" in f else "-" if "decreases" in f else "~"
                color = "#c0392b" if "increases" in f else "#27ae60" if "decreases" in f else "#7f8c8d"
                st.markdown(f'<span style="color:{color}; font-weight:600">[{icon}]</span> {f}', unsafe_allow_html=True)

    with col_conf:
        if risk_result.confidence:
            conf = risk_result.confidence
            low, high = conf.prediction_range
            st.markdown(
                f"""<div class="sb-card" style="background:{conf.color}08;">
                <div style="font-weight:600; color:{conf.color}; margin-bottom:0.3rem">Confidence: {conf.level}</div>
                <div style="font-size:0.9rem; color:#555">Range: {low:.0f}–{high:.0f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
            for reason in conf.reasons:
                st.caption(f"  {reason}")

        if best_alignments:
            st.markdown("**Nearest drugs**")
            for r in best_alignments[:3]:
                ada_str = f" ({drug_ada_map[r.inn_name]:.0f}%)" if r.inn_name in drug_ada_map else ""
                st.caption(f"{r.inn_name} — {r.pct_identity:.0%}{ada_str}")

    st.markdown("")

    # --- Benchmarking chart + Timeline ---
    col_chart, col_time = st.columns([3, 2])

    with col_chart:
        st.markdown(f"**Benchmarking** — {disease}, {route}")
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

            bars = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("ADA Rate (%):Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Drug:N", sort="-x"),
                color=alt.condition(
                    alt.datum.Drug == "YOUR CANDIDATE",
                    alt.value(risk_result.tier_color),
                    alt.value("#94b8d9"),
                ),
                tooltip=["Drug", "ADA Rate (%)", "Patients"],
            ).properties(height=max(180, len(chart_data) * 28))

            st.altair_chart(bars, use_container_width=True)
        else:
            st.info("No exact precedents for this combination.")

    with col_time:
        st.markdown("**ADA Onset Timeline**")
        if risk_result.time_ada:
            tad = risk_result.time_ada
            st.caption(f"Peak: {tad.expected_onset} ({tad.peak_ada_pct}%)")

            if tad.profile:
                time_df = pd.DataFrame([
                    {"Time Window": tb, "ADA Rate (%)": round(ada, 1), "Cohorts": n}
                    for tb, ada, n in tad.profile
                ])
                time_chart = alt.Chart(time_df).mark_bar(cornerRadiusEnd=3).encode(
                    x=alt.X("Time Window:N", sort=None, title=None),
                    y=alt.Y("ADA Rate (%):Q"),
                    color=alt.value("#94b8d9"),
                    tooltip=["Time Window", "ADA Rate (%)", "Cohorts"],
                ).properties(height=180)
                st.altair_chart(time_chart, use_container_width=True)
        else:
            st.info("Timeline data not available.")

        if risk_result.nada:
            nada = risk_result.nada
            st.caption(f"nADA/ADA ratio: {nada.nada_ratio} ({nada.source})")

    # Historical precedent table — collapsed by default
    with st.expander("Historical Precedents"):
        precedents = get_historical_precedents(route, disease, modality, top_n=15)
        if len(precedents) > 0:
            display_cols = {
                "Therapeutic Assessed for ADA INN Name": "Drug",
                "ada_pct": "ADA (%)",
                "total_patients": "Patients",
                "n_cohorts": "Cohorts",
                "route": "Route",
                "modality": "Modality",
            }
            display_df = precedents.rename(columns=display_cols)
            available = [c for c in display_cols.values() if c in display_df.columns]
            display_df = display_df[available]
            if "ADA (%)" in display_df.columns:
                display_df["ADA (%)"] = display_df["ADA (%)"].round(1)
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info("No historical precedents found.")

# ==============================
# PART 2 — Structural Risk Viewer (per-chain)
# ==============================
with tab2:
    if not chains:
        st.info("Paste a protein sequence in the sidebar to enable structural analysis.")
    else:
        if len(chains) > 1:
            chain_tabs = st.tabs(list(chains.keys()))
        else:
            chain_tabs = [st.container()]

        for chain_tab, (chain_name, chain_seq) in zip(chain_tabs, chains.items()):
            with chain_tab:
                st.markdown(f"**{chain_name}** — {len(chain_seq)} aa")
                col_seq, col_ep = st.columns([3, 2])

                with col_seq:
                    # Run IEDB for this chain
                    ep_key = chain_name
                    if ep_key not in chain_epitopes:
                        chain_epitopes[ep_key] = predict_epitopes(chain_seq)

                    ep_results = chain_epitopes[ep_key]

                    # --- Non-blocking Tamarind ESMFold via session_state polling ---
                    job_key = f"fold_job_{chain_name}"
                    pdb_ss_key = f"fold_pdb_{chain_name}"
                    start_key = f"fold_start_{chain_name}"
                    existing_job = st.session_state.get(job_key)
                    pdb_data = st.session_state.get(pdb_ss_key)

                    # Also populate chain_pdb from session_state if available
                    if pdb_data:
                        chain_pdb[chain_name] = pdb_data

                    if existing_job:
                        # ── Polling loop: check status and show progress ──
                        status = check_fold_status(existing_job, tamarind_key)

                        if status == "complete":
                            with st.spinner("Downloading structure..."):
                                pdb_result = fetch_fold_result(existing_job, tamarind_key)
                            if pdb_result:
                                st.session_state[pdb_ss_key] = pdb_result
                                del st.session_state[job_key]
                                if start_key in st.session_state:
                                    del st.session_state[start_key]
                                chain_pdb[chain_name] = pdb_result
                                pdb_data = pdb_result
                                st.rerun()
                            else:
                                st.warning("Job complete but PDB not found. Try again.")
                                del st.session_state[job_key]
                                if start_key in st.session_state:
                                    del st.session_state[start_key]
                        elif status == "failed":
                            st.error("ESMFold job failed. Try again.")
                            del st.session_state[job_key]
                            if start_key in st.session_state:
                                del st.session_state[start_key]
                        else:
                            if start_key not in st.session_state:
                                st.session_state[start_key] = time.time()

                            elapsed = time.time() - st.session_state[start_key]
                            estimated_total = 90
                            progress = min(elapsed / estimated_total, 0.95)

                            st.info(f"Folding with AlphaFold2 via Tamarind Bio... ({int(elapsed)}s)")
                            st.progress(progress)

                            time.sleep(5)
                            st.rerun()

                    if pdb_data:
                        try:
                            import stmol
                            import py3Dmol
                            view = py3Dmol.view(width="100%", height=380)
                            view.addModel(pdb_data, "pdb")
                            view.setStyle({"cartoon": {"color": "spectrum"}})

                            if ep_results:
                                for ep in ep_results:
                                    for pos in range(ep.start, ep.end + 1):
                                        view.addStyle(
                                            {"resi": pos},
                                            {"cartoon": {"color": "red"}, "stick": {"color": "red", "radius": 0.15}},
                                        )

                            view.zoomTo()
                            stmol.showmol(view, height=380, width=None)
                            st.caption("Red = T-cell epitope hotspots")

                            st.download_button(
                                "Download PDB",
                                pdb_data,
                                f"safebind_{chain_name.replace(' ', '_')}.pdb",
                                mime="chemical/x-pdb",
                                key=f"pdb_dl_{chain_name}",
                            )
                        except Exception as e:
                            st.warning(f"3D viewer error: {e}")
                            pdb_data = None

                    if not pdb_data and not existing_job:
                        if tamarind_key:
                            if st.button(
                                "Predict 3D Structure",
                                key=f"fold_btn_{chain_name}",
                            ):
                                with st.spinner("Submitting..."):
                                    submitted = submit_fold_job(chain_seq, tamarind_key)
                                if submitted:
                                    st.session_state[job_key] = submitted
                                    st.session_state[start_key] = time.time()
                                    st.rerun()
                                else:
                                    st.error("Submission failed.")
                        else:
                            st.caption("Set TAMARIND_API_KEY for 3D structure.")

                        # Show sequence with epitope highlights
                        if ep_results:
                            epitope_positions = set()
                            for ep in ep_results:
                                epitope_positions.update(range(ep.start, ep.end + 1))

                            seq_html = '<div style="font-family:monospace; word-wrap:break-word; line-height:1.8; font-size:0.82rem">'
                            for i, aa in enumerate(chain_seq):
                                pos = i + 1
                                if pos in epitope_positions:
                                    seq_html += f'<span style="background:#c0392b; color:white; padding:1px 2px; border-radius:2px" title="Epitope at {pos}">{aa}</span>'
                                else:
                                    seq_html += f'<span style="color:#aaa" title="{pos}">{aa}</span>'
                                if pos % 60 == 0:
                                    seq_html += f' <span style="color:#bbb; font-size:0.75em">{pos}</span><br>'
                            seq_html += "</div>"
                            st.markdown(seq_html, unsafe_allow_html=True)
                            st.caption("Red = T-cell epitope hotspots")
                        else:
                            st.code(chain_seq[:300] + ("..." if len(chain_seq) > 300 else ""))

                with col_ep:
                    if ep_results:
                        density = compute_epitope_density(ep_results, len(chain_seq))
                        c1, c2 = st.columns(2)
                        c1.metric("T-cell Epitopes", len(ep_results))
                        c2.metric("Density", f"{density:.1f}/100aa")

                        ep_df = pd.DataFrame([
                            {
                                "Pos": f"{ep.start}-{ep.end}",
                                "Peptide": ep.peptide,
                                "Rank": round(ep.percentile_rank, 2),
                                "Allele": ep.allele,
                            }
                            for ep in ep_results[:20]
                        ])
                        st.dataframe(ep_df, hide_index=True, use_container_width=True, height=250)
                    else:
                        st.info("No T-cell epitope predictions available.")

                    # CDR Detection
                    chain_type = "heavy" if any(kw in chain_name.lower() for kw in ["heavy", "vh", "hc"]) else "light"
                    cdrs = detect_cdr_regions(chain_seq, chain_type)
                    if cdrs:
                        chain_cdr[chain_name] = cdrs
                        st.markdown("**CDR Regions**")
                        for cdr in cdrs:
                            st.caption(f'{cdr["label"]} ({cdr["start"]}-{cdr["end"]}): {cdr["sequence"][:25]}')

                        # nADA risk flags
                        if ep_results:
                            nada_warnings = []
                            for ep in ep_results[:10]:
                                overlap = check_cdr_epitope_overlap(cdrs, ep.start, ep.end)
                                if overlap:
                                    nada_warnings.append(overlap)
                            if nada_warnings:
                                for w in nada_warnings:
                                    risk_color = "#c0392b" if w["nada_risk"] == "HIGH" else "#e67e22"
                                    st.markdown(
                                        f'<div style="padding:4px 8px; background:{risk_color}10; border-left:3px solid {risk_color}; border-radius:3px; margin-bottom:3px; font-size:0.85rem">'
                                        f'<strong style="color:{risk_color}">{w["nada_risk"]}</strong> nADA — '
                                        f'epitope {w["overlap_start"]}-{w["overlap_end"]} in {w["cdr"]}</div>',
                                        unsafe_allow_html=True,
                                    )

                    # Nearest matches & diffs collapsed
                    if chain_name in chain_alignments and chain_alignments[chain_name]:
                        with st.expander("Nearest matches & diffs"):
                            for r in chain_alignments[chain_name][:3]:
                                ada_str = f" ({drug_ada_map[r.inn_name]:.0f}%)" if r.inn_name in drug_ada_map else ""
                                st.caption(f"{r.inn_name} {r.chain_descriptor}: {r.pct_identity:.0%}{ada_str}")
                            diffs = chain_diffs.get(chain_name, [])
                            if diffs:
                                st.dataframe(
                                    pd.DataFrame([{"Pos": p, "Query": q, "Ref": r} for p, q, r in diffs[:15]]),
                                    hide_index=True, use_container_width=True, height=200,
                                )

        if len(chains) > 1:
            st.markdown("")
            st.markdown("**Cross-chain summary**")
            summary_rows = []
            for name in chains:
                eps = chain_epitopes.get(name, [])
                bcells = chain_bcell.get(name, [])
                summary_rows.append({
                    "Chain": name,
                    "Length": len(chains[name]),
                    "T-cell": len(eps),
                    "B-cell": len(bcells),
                    "CDRs": len(chain_cdr.get(name, [])),
                    "Density": round(compute_epitope_density(eps, len(chains[name])), 1) if eps else 0,
                    "3D": "Yes" if name in chain_pdb else "—",
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# ==============================
# PART 2B — B-cell & Surface Epitopes
# ==============================
with tab_bcell:
    if not chains:
        st.info("Paste a protein sequence in the sidebar to enable B-cell epitope analysis.")
    else:
        if len(chains) > 1:
            bcell_tabs = st.tabs(list(chains.keys()))
        else:
            bcell_tabs = [st.container()]

        for btab, (chain_name, chain_seq) in zip(bcell_tabs, chains.items()):
            with btab:
                if chain_name not in chain_bcell:
                    chain_bcell[chain_name] = predict_bcell_epitopes(chain_seq)

                bcell_results = chain_bcell[chain_name]

                pdb_data = chain_pdb.get(chain_name)
                sasa_scores = {}
                if pdb_data and chain_name not in chain_sasa:
                    sasa_scores = calculate_sasa_from_pdb(pdb_data, chain_id="A")
                    chain_sasa[chain_name] = sasa_scores
                elif chain_name in chain_sasa:
                    sasa_scores = chain_sasa[chain_name]

                cdrs = chain_cdr.get(chain_name, [])
                tcell_eps = chain_epitopes.get(chain_name, [])

                # Metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("B-cell", len(bcell_results))
                if sasa_scores:
                    surface_bcells = filter_surface_epitopes(bcell_results, sasa_scores)
                    mc2.metric("Surface", len(surface_bcells))
                else:
                    surface_bcells = []
                    mc2.metric("Surface", "—")
                mc3.metric("CDRs", len(cdrs))
                mc4.metric("T-cell", len(tcell_eps))

                col_bcell_left, col_bcell_right = st.columns([3, 2])

                with col_bcell_left:
                    # Combined heatmap
                    if bcell_results or tcell_eps:
                        st.markdown(f"**Epitope map** — {chain_name}")
                        tcell_positions = set()
                        for ep in tcell_eps:
                            tcell_positions.update(range(ep.start, ep.end + 1))
                        bcell_positions = set()
                        for ep in bcell_results:
                            bcell_positions.update(range(ep.start, ep.end + 1))

                        seq_html = '<div style="font-family:monospace; word-wrap:break-word; line-height:1.8; font-size:0.82rem">'
                        for i, aa in enumerate(chain_seq):
                            pos = i + 1
                            in_t = pos in tcell_positions
                            in_b = pos in bcell_positions
                            if in_t and in_b:
                                seq_html += f'<span style="background:#8e44ad; color:white; padding:1px 2px; border-radius:2px" title="T+B at {pos}">{aa}</span>'
                            elif in_t:
                                seq_html += f'<span style="background:#c0392b; color:white; padding:1px 2px; border-radius:2px" title="T-cell at {pos}">{aa}</span>'
                            elif in_b:
                                seq_html += f'<span style="background:#2980b9; color:white; padding:1px 2px; border-radius:2px" title="B-cell at {pos}">{aa}</span>'
                            else:
                                seq_html += f'<span style="color:#aaa" title="{pos}">{aa}</span>'
                            if pos % 60 == 0:
                                seq_html += f' <span style="color:#bbb; font-size:0.75em">{pos}</span><br>'
                        seq_html += "</div>"
                        st.markdown(seq_html, unsafe_allow_html=True)
                        st.caption("Red = T-cell | Blue = B-cell | Purple = both")

                with col_bcell_right:
                    # CDR + nADA warnings
                    if cdrs:
                        st.markdown("**CDR Regions**")
                        for cdr in cdrs:
                            st.caption(f'{cdr["label"]} ({cdr["start"]}-{cdr["end"]}): {cdr["sequence"][:25]}')

                        nada_bcell_warnings = []
                        for ep in bcell_results:
                            overlap = check_cdr_epitope_overlap(cdrs, ep.start, ep.end)
                            if overlap:
                                nada_bcell_warnings.append((ep, overlap))
                        if nada_bcell_warnings:
                            for ep, w in nada_bcell_warnings[:5]:
                                risk_color = "#c0392b" if w["nada_risk"] == "HIGH" else "#e67e22"
                                st.markdown(
                                    f'<div style="padding:4px 8px; background:{risk_color}10; border-left:3px solid {risk_color}; border-radius:3px; margin-bottom:3px; font-size:0.85rem">'
                                    f'<strong style="color:{risk_color}">{w["nada_risk"]}</strong> — '
                                    f'B-cell {ep.start}-{ep.end} in {w["cdr"]}</div>',
                                    unsafe_allow_html=True,
                                )

                    # Surface-exposed table
                    if sasa_scores and surface_bcells:
                        st.markdown("**Surface-exposed**")
                        surf_df = pd.DataFrame([
                            {
                                "Pos": f"{ep.start}-{ep.end}",
                                "Seq": ep.sequence[:20],
                                "Score": round(ep.avg_score, 3),
                                "SASA": round(ep.avg_sasa, 1) if ep.avg_sasa else "—",
                            }
                            for ep in surface_bcells[:10]
                        ])
                        st.dataframe(surf_df, hide_index=True, use_container_width=True, height=200)

                # Full epitope table — collapsed
                if bcell_results:
                    with st.expander(f"All B-cell epitopes ({len(bcell_results)})"):
                        all_bcell_df = pd.DataFrame([
                            {
                                "Position": f"{ep.start}-{ep.end}",
                                "Sequence": ep.sequence[:30],
                                "Score": round(ep.avg_score, 3),
                                "Surface": ("SURFACE" if sasa_scores and any(
                                    is_surface_exposed(sasa_scores.get(p, 0))
                                    for p in range(ep.start, ep.end + 1)
                                ) else ("BURIED" if sasa_scores else "?")),
                            }
                            for ep in bcell_results
                        ])
                        st.dataframe(all_bcell_df, hide_index=True, use_container_width=True)

                        csv_data = all_bcell_df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv_data,
                            f"safebind_{chain_name.replace(' ', '_')}_bcell.csv",
                            mime="text/csv",
                            key=f"dl_bcell_{chain_name}",
                        )
                else:
                    st.info("No B-cell epitopes predicted.")

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
        # --- Tolerance Analysis ---
        chain_tolerance = {}
        for chain_name, chain_seq in chains.items():
            ep_results = chain_epitopes.get(chain_name, [])
            if not ep_results:
                continue
            tol = run_tolerance_analysis(chain_seq, ep_results, risk_result.composite_score / 100.0)
            chain_tolerance[chain_name] = tol

        if chain_tolerance:
            st.markdown("**Immune Tolerance**")
            tol_cols = st.columns(min(len(chain_tolerance), 4))
            for i, (cname, tol) in enumerate(chain_tolerance.items()):
                with tol_cols[i % len(tol_cols)]:
                    tol_color = "#27ae60" if tol.tolerance_score > 0.5 else "#e67e22" if tol.tolerance_score > 0.3 else "#c0392b"
                    treg_pct = tol.treg_count / max(tol.treg_count + tol.effector_count, 1) * 100
                    st.markdown(
                        f'<div style="padding:0.6rem 0.8rem; background:{tol_color}08; border-left:3px solid {tol_color}; '
                        f'border-radius:4px;">'
                        f'<div style="display:flex; justify-content:space-between; align-items:baseline">'
                        f'<span style="font-size:0.85rem">{cname}</span>'
                        f'<span style="font-size:1.1rem; font-weight:700; color:{tol_color}">{tol.tolerance_score:.2f}</span>'
                        f'</div>'
                        f'<div style="font-size:0.78rem; color:#888; margin-top:2px">'
                        f'Treg {tol.treg_count} ({treg_pct:.0f}%) &middot; Eff {tol.effector_count} &middot; '
                        f'adj {tol.risk_adjustment:+.1%}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with st.expander("Tolerance details"):
                for cname, tol in chain_tolerance.items():
                    if tol.epitope_details:
                        tol_df = pd.DataFrame([
                            {
                                "Peptide": d.get("peptide", "")[:20],
                                "Humanness": round(d.get("treg_score", 0), 2),
                                "Tregitope": d.get("tregitope_match") or "—",
                                "Type": "Treg" if d.get("is_treg") else "Eff",
                            }
                            for d in tol.epitope_details
                        ])
                        st.dataframe(tol_df, hide_index=True, use_container_width=True, height=200)

        st.markdown("")

        # --- Rule-Based Deimmunization ---
        st.markdown("**Deimmunized Variants**")

        all_chain_variants = {}

        for chain_name, chain_seq in chains.items():
            ep_results = chain_epitopes.get(chain_name, [])
            if not ep_results:
                continue

            tol_for_chain = chain_tolerance.get(chain_name)
            deimm_results = deimmunize_epitopes(chain_seq, ep_results, tolerance_result=tol_for_chain)
            variants = generate_redesigned_sequences(chain_seq, deimm_results)
            all_chain_variants[chain_name] = variants

            if not variants:
                continue

            with st.expander(f"{chain_name} — {len(deimm_results)} mutations across {len(variants)} variants", expanded=(len(chains) == 1)):
                # Mutation map
                mut_df = pd.DataFrame([
                    {
                        "Region": f"{dr.region_start}-{dr.region_end}",
                        "Mutations": ", ".join(f"{o}{p}{n}" for p, o, n in dr.mutations),
                        "Disruption": dr.expected_binding_disruption,
                        "Allele": dr.allele,
                    }
                    for dr in deimm_results
                ])
                st.dataframe(mut_df, hide_index=True, use_container_width=True)

                # Variants
                for i, var in enumerate(variants):
                    st.markdown(f"**{var.name}** — {var.n_mutations} mutations ({var.strategy})")

                    # Highlighted sequence
                    seq_html = '<div style="font-family:monospace; word-wrap:break-word; line-height:1.8; font-size:0.82rem">'
                    mut_positions = {pos for pos, _, _ in var.mutations}
                    for j, aa in enumerate(var.sequence):
                        pos = j + 1
                        if pos in mut_positions:
                            seq_html += f'<span style="background:#27ae60; color:white; padding:1px 3px; border-radius:2px; font-weight:bold" title="{chain_seq[j]}->{aa}">{aa}</span>'
                        else:
                            seq_html += aa
                        if pos % 80 == 0:
                            seq_html += f' <span style="color:#bbb; font-size:0.75em">{pos}</span><br>'
                    seq_html += "</div>"
                    st.markdown(seq_html, unsafe_allow_html=True)

                    safe_name = chain_name.replace(" ", "_")
                    fasta = f">{safe_name}|{var.name.replace(' ', '_')}|{var.n_mutations}_mutations\n"
                    fasta += "\n".join(var.sequence[k:k+60] for k in range(0, len(var.sequence), 60))
                    st.download_button(
                        f"Download FASTA",
                        fasta,
                        f"safebind_{safe_name}_variant_{i+1}.fasta",
                        mime="text/plain",
                        key=f"dl_{safe_name}_{i}",
                    )

        # Combined download
        if len(chains) > 1 and all_chain_variants:
            st.markdown("")
            dl_cols = st.columns(3)
            for idx, level in enumerate(["Conservative", "Moderate", "Aggressive"]):
                combined_fasta = ""
                chain_count = 0
                for cname, variants in all_chain_variants.items():
                    for var in variants:
                        if level.lower() in var.name.lower():
                            safe_name = cname.replace(" ", "_")
                            combined_fasta += f">{safe_name}|{var.name.replace(' ', '_')}\n"
                            combined_fasta += "\n".join(var.sequence[k:k+60] for k in range(0, len(var.sequence), 60))
                            combined_fasta += "\n"
                            chain_count += 1
                            break
                    else:
                        safe_name = cname.replace(" ", "_")
                        combined_fasta += f">{safe_name}|original\n"
                        combined_fasta += "\n".join(chains[cname][k:k+60] for k in range(0, len(chains[cname]), 60))
                        combined_fasta += "\n"
                        chain_count += 1
                if combined_fasta:
                    with dl_cols[idx]:
                        st.download_button(
                            f"{level} ({chain_count} chains)",
                            combined_fasta,
                            f"safebind_all_{level.lower()}.fasta",
                            mime="text/plain",
                            key=f"dl_combined_{level}",
                        )

        # --- ProteinMPNN ---
        if tamarind_key:
            st.markdown("")
            st.markdown("**ProteinMPNN Redesign**")
            st.caption("Structure-aware sequence design at epitope anchor positions.")

            for chain_name, chain_seq in chains.items():
                ep_results = chain_epitopes.get(chain_name, [])
                if not ep_results:
                    continue

                hotspot_positions = set()
                for ep in ep_results:
                    for offset in [0, 3, 5, 8]:
                        pos = ep.start + offset
                        if pos <= len(chain_seq):
                            hotspot_positions.add(pos)

                if not hotspot_positions:
                    continue

                pdb_for_chain = chain_pdb.get(chain_name)

                with st.expander(f"{chain_name} — {len(hotspot_positions)} designable positions"):
                    if st.button(f"Run ProteinMPNN", key=f"mpnn_run_{chain_name}"):
                        mpnn_results = suggest_redesigns(
                            chain_seq, sorted(hotspot_positions), tamarind_key,
                            n_designs=3, pdb_data=pdb_for_chain,
                        )

                        if mpnn_results:
                            for idx, result in enumerate(mpnn_results):
                                mpnn_seq = result["sequence"]
                                mpnn_score = result.get("score", 0)

                                mutations = []
                                for j, (orig, new) in enumerate(zip(chain_seq, mpnn_seq)):
                                    if orig != new:
                                        mutations.append((j + 1, orig, new))

                                st.markdown(f"**Design {idx + 1}** — {len(mutations)} mutations (score: {mpnn_score:.3f})")

                                if mutations:
                                    st.dataframe(
                                        pd.DataFrame([{"Pos": p, "From": o, "To": n} for p, o, n in mutations]),
                                        hide_index=True,
                                    )

                                safe_name = chain_name.replace(" ", "_")
                                mpnn_fasta = f">{safe_name}|MPNN_design_{idx+1}|score={mpnn_score:.3f}\n"
                                mpnn_fasta += "\n".join(mpnn_seq[k:k+60] for k in range(0, len(mpnn_seq), 60))
                                st.download_button(
                                    f"Download Design {idx + 1}",
                                    mpnn_fasta,
                                    f"safebind_{safe_name}_mpnn_{idx+1}.fasta",
                                    mime="text/plain",
                                    key=f"dl_mpnn_{safe_name}_{idx}",
                                )
                        else:
                            st.warning("ProteinMPNN returned no results.")

    elif chains:
        st.info("Waiting for epitope predictions to generate redesigned sequences.")
    else:
        st.info("Paste a protein sequence in the sidebar to generate deimmunized variants.")

    st.markdown("")

    # --- AI Risk Memo ---
    st.markdown("**AI Risk Memo**")

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

    with st.expander("View memo", expanded=True):
        st.markdown(memo)

    st.download_button(
        "Download Risk Memo",
        memo,
        "safebind_risk_memo.md",
        mime="text/markdown",
    )
