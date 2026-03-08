"""
SafeBind AI — Immunogenicity Risk Assessment Platform
Bio x AI Hackathon @ YC HQ — March 8, 2026

Streamlit web app that:
1. Accepts antibody/protein sequences
2. Predicts T-cell & B-cell epitopes via IEDB
3. Generates interactive 3D immunogenicity heatmaps
4. Cross-references IDC DB V1 clinical ADA data
5. Uses Claude to synthesize a natural language risk report
"""

import streamlit as st
import streamlit.components.v1 as components
import time
import json
import os
import sys

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from immunogenicity_core_2 import (
    run_immunogenicity_assessment,
    generate_3d_heatmap_html,
    generate_risk_summary,
    export_residue_csv,
    HLA_ALLELES,
    identify_hotspots,
    submit_tamarind_structure,
    get_tamarind_job_status,
    fetch_tamarind_pdb,
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SafeBind AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* Global - clean white background */
    .stApp { background-color: #ffffff; }
    html, body, [class*="css"] { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
        color: #1a1a1a; 
    }

    /* Header */
    .hero-title {
        font-size: 36px; font-weight: 700; line-height: 1.2;
        color: #111827;
        margin-bottom: 4px;
    }
    .hero-sub { color: #6b7280; font-size: 15px; margin-bottom: 24px; font-weight: 400; }

    /* Cards */
    .metric-card {
        background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 20px; text-align: center;
    }
    .metric-value { font-size: 32px; font-weight: 600; }
    .metric-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; font-weight: 500; }

    .risk-low { color: #0891b2; }
    .risk-moderate { color: #ca8a04; }
    .risk-high { color: #ea580c; }
    .risk-very-high { color: #dc2626; }

    /* Hotspot table */
    .hotspot-row {
        background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 14px 18px; margin-bottom: 10px;
    }
    .hotspot-seq { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #0891b2; }

    /* Disclaimer */
    .disclaimer {
        background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 14px; font-size: 11px; color: #6b7280; text-align: center; margin-top: 24px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f9fafb; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextArea label { color: #374151; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #e5e7eb; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; border-radius: 0;
        color: #6b7280; padding: 10px 16px; font-weight: 500;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] { 
        background-color: transparent; color: #2563eb; 
        border-bottom: 2px solid #2563eb;
    }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Claude report */
    .claude-report {
        background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
        padding: 20px; line-height: 1.7; font-size: 14px; color: #1e3a5f;
    }
    .claude-badge {
        display: inline-block; background: #2563eb; color: #ffffff;
        font-size: 11px; font-weight: 500; padding: 4px 10px;
        border-radius: 4px; margin-bottom: 12px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563eb; color: white; border: none;
        font-weight: 500; border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 1px solid #d1d5db; border-radius: 6px;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Preloaded sequences ──────────────────────────────────────
PRELOADED = {
    "Bococizumab (Pfizer — TERMINATED, 44% ADA)": {
        "seq": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGEISPFGGRTNYNEKFKSRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARERPLYASDLWGQGTTVTVSS",
        "species": "Humanized",
        "pdb": None,
    },
    "Adalimumab / Humira (30-93% ADA)": {
        "seq": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
        "species": "Human",
        "pdb": None,
    },
    "Trastuzumab / Herceptin (0-14% ADA)": {
        "seq": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        "species": "Humanized",
        "pdb": "1N8Z",
    },
    "Nivolumab / Opdivo (11-26% ADA)": {
        "seq": "QVQLVESGGGVVQPGRSLRLDCKASGITFSNSGMHWVRQAPGKGLEWVAVIWYDGSKRYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCATNDDYWGQGTLVTVSS",
        "species": "Human",
        "pdb": None,
    },
    "── Gene Therapy ──": None,
    "AAV9 Capsid VP1 (SGT-001, NGN-401 failures)": {
        "seq": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDQQLKAGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRLLEPLGLVEEAAKTAPGKKRPVEQSPQEPDSSA",
        "species": "Viral",
        "pdb": None,
    },
    "── Enzyme Replacement ──": None,
    "Agalsidase beta / Fabrazyme (~40% nADA)": {
        "seq": "LDNGLARTPTMGWLHWERFMCNLDCQEEPDSCISEKLFMEMAELMVSEGWKDAGYEYLCIDDCWMAPQRDSEGRLQADPQRFPHGIRQLANYVHSKGLKLGIYADVGNKTCAGFPGSFGYYDIDAQTFAD",
        "species": "Human",
        "pdb": None,
    },
    "Alglucosidase alfa / Myozyme (~100% ADA CRIM-neg)": {
        "seq": "AHPGRPRAVPTQCDVPPNSRFDCAPDKAITQEQCEARGCCYIPAKQGLQGAQMGQPWCFFPPSYPSYKLENLSSSEMGYTATLTRTTPTFFPKDILTLRLDVMMETENRLHFTIKDPANRRYEVPLETPH",
        "species": "Human",
        "pdb": None,
    },
    "── Bispecific ──": None,
    "Blinatumomab / Blincyto (BiTE, 1-2% ADA)": {
        "seq": "DIQLTQSPASLAVSLGQRATISCKASQSVDYDGDSYLNWYQQIPGQPPKLLIYDASNLVSGIPPRFSGSGSGTDFTLNIHPVEKVDAATYHCQQSTEDPWTFGGGTKLEIKGGGGSGGGGSGGGGSQVQL",
        "species": "Mouse",
        "pdb": None,
    },
    "Custom sequence": {
        "seq": "",
        "species": "Humanized",
        "pdb": None,
    },
}


# ── Claude API integration ───────────────────────────────────
def generate_claude_report(report_data: dict) -> str:
    """Use Claude API to generate a natural language risk narrative."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are an expert immunologist and drug development scientist. Analyze this immunogenicity risk assessment and write a concise, actionable risk report (3-4 paragraphs). Be specific about regions, residues, and clinical implications. Reference the comparable therapeutics data.

ASSESSMENT DATA:
- Therapeutic: {report_data['name']}
- Sequence length: {report_data['seq_len']} amino acids
- Overall risk score: {report_data['overall_risk']:.1%}
- Risk category: {report_data['risk_category']}
- T-cell epitopes (strong binders, rank <10%): {report_data['n_strong_binders']}
- B-cell epitope regions: {report_data['n_bcell']}
- HLA alleles tested: 9 (covering ~85% global population)

HOTSPOT REGIONS:
{json.dumps(report_data['hotspots'], indent=2)}

COMPARABLE THERAPEUTICS (from IDC DB V1 clinical data):
{json.dumps(report_data['comparables'], indent=2)}

Write the report. Be direct and scientific. Include:
1. Overall risk assessment with specific numbers
2. Which hotspot regions are most concerning and why
3. How this compares to known therapeutics with real clinical ADA data
4. 1-2 specific de-immunization recommendations (e.g., which regions to prioritize for engineering)

Keep it under 250 words. Do not use headers or bullet points — write in paragraphs."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Claude API error: {str(e)}"


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:24px;">🛡️ SafeBind AI</div>', unsafe_allow_html=True)
    st.markdown("**Immunogenicity Risk Assessment**")
    st.markdown("---")

    # Sequence selection
    selected = st.selectbox("Select therapeutic", list(PRELOADED.keys()))

    if selected.startswith("──") or PRELOADED.get(selected) is None:
        st.warning("Select a therapeutic from the dropdown.")
        seq_input = ""
        species = "Humanized"
        pdb_id = ""
        seq_name = ""
    elif selected == "Custom sequence":
        seq_input = st.text_area(
            "Paste amino acid sequence",
            height=120,
            placeholder="EVQLVESGGGLVQPGG..."
        )
        species = st.selectbox("Species origin", ["Human", "Humanized", "Chimeric", "Mouse", "Viral", "Bacterial"])
        pdb_id = st.text_input("PDB ID (optional)", placeholder="e.g., 1N8Z")
        seq_name = st.text_input("Name", placeholder="My antibody")
    else:
        info = PRELOADED[selected]
        seq_input = info["seq"]
        species = info["species"]
        pdb_id = info.get("pdb") or ""
        seq_name = selected.split("(")[0].strip().split("/")[0].strip()
        st.code(seq_input[:60] + "...", language=None)
        st.caption(f"Species: {species} | Length: {len(seq_input)} aa")

    st.markdown("---")

    # Tamarind API key
    tamarind_key_input = st.text_input(
        "Tamarind API key (for 3D folding)",
        type="password",
        placeholder="tmrd-...",
        help="Get your key at app.tamarind.bio — used for ESMFold / AlphaFold2 structure prediction"
    )
    if tamarind_key_input:
        st.session_state["tamarind_api_key"] = tamarind_key_input

    st.markdown("---")

    # Run button
    run_clicked = st.button("🔬 Analyze Immunogenicity", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#6b7280; line-height:1.6;">
    <b style="color:#374151;">Data sources</b><br>
    IEDB (tools.iedb.org)<br>
    IDC DB V1 (CC BY 4.0)<br>
    RCSB PDB<br>
    Tamarind Bio (ESMFold/AF2)<br><br>
    <b style="color:#374151;">HLA coverage</b><br>
    9 DRB1 alleles (~85% global)
    </div>
    """, unsafe_allow_html=True)


# ── Main content ─────────────────────────────────────────────
st.markdown('<div class="hero-title">SafeBind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Immunogenicity risk assessment for therapeutic proteins · Powered by IEDB, IDC DB V1, and Claude</div>', unsafe_allow_html=True)

if run_clicked and seq_input:
    # Clean sequence
    seq_clean = "".join(c for c in seq_input.upper() if c.isalpha())

    if len(seq_clean) < 20:
        st.error("Sequence too short. Please enter at least 20 amino acids.")
    else:
        # Progress tracking
        progress = st.progress(0, text="Initializing analysis...")

        # Step 1: T-cell epitopes
        progress.progress(5, text="Predicting T-cell epitopes across 9 HLA alleles...")

        report = run_immunogenicity_assessment(
            sequence=seq_clean,
            name=seq_name or "Query",
            pdb_id=pdb_id if pdb_id else None,
            pdb_chain="A",
            idc_data_path="idc_db_v1_table_s4.xlsx",
            species=species,
            modality="Monoclonal antibody",
            verbose=False,
        )

        progress.progress(80, text="Generating visualizations...")

        # ── Risk category color ──
        risk_class = {
            "LOW": "risk-low", "MODERATE": "risk-moderate",
            "HIGH": "risk-high", "VERY HIGH": "risk-very-high"
        }.get(report.risk_category, "risk-moderate")

        # ── Metrics row ──
        progress.progress(85, text="Rendering results...")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value {risk_class}">{report.overall_risk_score:.0%}</div>
                <div class="metric-label">Overall Risk</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#111827;">{report.risk_category}</div>
                <div class="metric-label">Risk Category</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            strong = len([e for e in report.t_cell_epitopes if e.rank < 10])
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#ea580c;">{strong}</div>
                <div class="metric-label">Strong T-cell Binders</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#0891b2;">{len(report.hotspot_regions)}</div>
                <div class="metric-label">Hotspot Regions</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ──
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧬 3D Heatmap", "🔥 Hotspots", "📊 Residue Plot",
            "🏥 Clinical Context", "🤖 AI Report"
        ])

        # ── Tab 1: 3D Heatmap ──
        with tab1:
            # Prefer: 1) Tamarind-folded PDB in session state, 2) PDB from RCSB (pdb_id), 3) text fallback
            tamarind_pdb = st.session_state.get(f"tamarind_pdb_{seq_name}", None)
            active_pdb = tamarind_pdb or report.pdb_data

            if active_pdb:
                source_label = "Tamarind ESMFold/AlphaFold2" if tamarind_pdb else f"RCSB PDB ({pdb_id})"
                st.caption(f"Structure source: {source_label}")
                html = generate_3d_heatmap_html(
                    active_pdb, report.residue_risks,
                    chain="A", title=f"{seq_name} Immunogenicity Heatmap"
                )
                components.html(html, height=620, scrolling=False)
            else:
                # ── Sequence text fallback ──
                st.markdown("**Per-residue risk (text view):**")
                risk_html = ""
                for rr in report.residue_risks:
                    if rr.combined_risk > 0.5:
                        color = "#dc2626"
                    elif rr.combined_risk > 0.35:
                        color = "#ea580c"
                    elif rr.combined_risk > 0.2:
                        color = "#ca8a04"
                    else:
                        color = "#2563eb"
                    risk_html += (
                        f'<span style="color:{color};font-family:IBM Plex Mono,monospace;'
                        f'font-size:14px;font-weight:500;" '
                        f'title="Pos {rr.position}: {rr.combined_risk:.0%}">{rr.residue}</span>'
                    )
                st.markdown(
                    f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
                    f'padding:16px;line-height:2;word-wrap:break-word;">'
                    f'<div style="font-size:11px;color:#6b7280;margin-bottom:8px;">Hover over residues · '
                    f'<span style="color:#2563eb;">■</span> Low '
                    f'<span style="color:#ca8a04;">■</span> Moderate '
                    f'<span style="color:#ea580c;">■</span> High '
                    f'<span style="color:#dc2626;">■</span> Very High</div>'
                    f'{risk_html}</div>',
                    unsafe_allow_html=True,
                )

                # ── Tamarind fold button ──
                st.markdown("<br>", unsafe_allow_html=True)
                tamarind_key = st.session_state.get("tamarind_api_key", "")
                job_key = f"tamarind_job_{seq_name}"
                existing_job = st.session_state.get(job_key)

                if not tamarind_key:
                    st.info("💡 Enter your Tamarind API key in the sidebar to predict the 3D structure and unlock the heatmap visualization.")
                elif existing_job:
                    # Poll for existing job
                    with st.spinner(f"Folding structure with Tamarind ({existing_job})…"):
                        for _ in range(18):  # max ~3 min
                            status = get_tamarind_job_status(tamarind_key, existing_job)
                            if status == "complete":
                                pdb_result = fetch_tamarind_pdb(tamarind_key, existing_job)
                                if pdb_result:
                                    st.session_state[f"tamarind_pdb_{seq_name}"] = pdb_result
                                    del st.session_state[job_key]
                                    st.success("Structure predicted! Reloading heatmap…")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.warning("Job complete but PDB download failed. Try the PDB ID field in the sidebar instead.")
                                    del st.session_state[job_key]
                                break
                            elif status == "failed":
                                st.error("Tamarind job failed. Check your API key or try a different sequence.")
                                del st.session_state[job_key]
                                break
                            time.sleep(10)
                        else:
                            st.info("Still running… click **Analyze Immunogenicity** again to check status.")
                else:
                    fold_clicked = st.button(
                        "🔮 Predict 3D Structure with Tamarind",
                        help="Submits sequence to ESMFold / AlphaFold2 on Tamarind Bio (uses 1 compute job)",
                        use_container_width=True,
                    )
                    if fold_clicked:
                        safe_name = "".join(c if c.isalnum() else "_" for c in (seq_name or "query"))[:30]
                        job_name = f"safebind_{safe_name}_{int(time.time())}"
                        with st.spinner("Submitting to Tamarind Bio…"):
                            submitted = submit_tamarind_structure(tamarind_key, seq_clean, job_name)
                        if submitted:
                            st.session_state[job_key] = submitted
                            st.success(f"Job submitted: `{submitted}`. Click **Analyze Immunogenicity** to check progress.")
                        else:
                            st.error("Submission failed. Check your Tamarind API key.")

        # ── Tab 2: Hotspot details ──
        with tab2:
            if report.hotspot_regions:
                for i, hs in enumerate(report.hotspot_regions, 1):
                    risk_pct = hs['avg_risk']
                    bar_color = "#dc2626" if risk_pct > 0.5 else "#ea580c" if risk_pct > 0.35 else "#ca8a04"

                    st.markdown(f"""<div class="hotspot-row">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <span style="font-weight:600;font-size:15px;color:#111827;">Hotspot {i}</span>
                                <span style="color:#6b7280;font-size:13px;margin-left:8px;">Positions {hs['start']}–{hs['end']} ({hs['length']} residues)</span>
                            </div>
                            <div style="font-weight:600;font-size:18px;color:{bar_color};">{hs['avg_risk']:.0%}</div>
                        </div>
                        <div class="hotspot-seq" style="margin-top:8px;">{hs['sequence']}</div>
                        <div style="display:flex;gap:24px;margin-top:8px;font-size:12px;color:#6b7280;">
                            <span>T-cell risk: <b style="color:#ea580c;">{hs['avg_t_cell']:.0%}</b></span>
                            <span>B-cell risk: <b style="color:#0891b2;">{hs['avg_b_cell']:.0%}</b></span>
                            <span>Max risk: <b style="color:#dc2626;">{hs['max_risk']:.0%}</b></span>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No significant hotspot regions detected above threshold.")

        # ── Tab 3: Residue-level plot ──
        with tab3:
            import pandas as pd
            df = pd.DataFrame([{
                "Position": rr.position,
                "Residue": rr.residue,
                "T-cell Risk": rr.t_cell_risk,
                "B-cell Risk": rr.b_cell_risk,
                "Combined Risk": rr.combined_risk,
                "Alleles Binding": rr.num_alleles_binding
            } for rr in report.residue_risks])

            st.markdown("**Per-residue immunogenicity risk profile**")

            # Use streamlit's built-in chart
            chart_df = df.set_index("Position")[["T-cell Risk", "B-cell Risk", "Combined Risk"]]
            st.area_chart(chart_df, color=["#ea580c", "#0891b2", "#dc2626"], height=350)

            # Data table
            with st.expander("View full residue data table"):
                st.dataframe(
                    df.style.background_gradient(subset=["Combined Risk"], cmap="YlOrRd"),
                    use_container_width=True,
                    height=400,
                )

            # CSV download
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                file_name=f"{seq_name.lower().replace(' ','_')}_risk_scores.csv",
                mime="text/csv"
            )

        # ── Tab 4: Clinical context (IDC DB V1) ──
        with tab4:
            st.markdown("**Comparable therapeutics from IDC DB V1**")
            st.caption("4,146 ADA datapoints across 218 therapeutics and 727 clinical trials")

            if report.comparable_therapeutics:
                for comp in report.comparable_therapeutics:
                    ada_color = "#dc2626" if comp['median_ada_freq'] > 30 else "#ea580c" if comp['median_ada_freq'] > 10 else "#0891b2"
                    st.markdown(f"""<div class="hotspot-row">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <span style="font-weight:600;font-size:15px;color:#111827;">{comp['name']}</span>
                                <span style="color:#6b7280;font-size:12px;margin-left:8px;">{comp['species']} · {comp['modality']}</span>
                            </div>
                            <div style="font-weight:600;font-size:18px;color:{ada_color};">{comp['median_ada_freq']:.0f}% median ADA</div>
                        </div>
                        <div style="display:flex;gap:24px;margin-top:6px;font-size:12px;color:#6b7280;">
                            <span>Target: <b style="color:#374151;">{comp['target']}</b></span>
                            <span>Range: {comp['min_ada_freq']:.0f}–{comp['max_ada_freq']:.0f}%</span>
                            <span>n={comp['n_datapoints']} datapoints</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No comparable therapeutics found in IDC DB V1.")

            st.markdown("""<div style="margin-top:16px;font-size:12px;color:#6b7280;line-height:1.6;">
                <b>Key finding from IDC DB V1 multivariate analysis:</b> The top 3 drivers of clinical ADA frequency are
                (1) therapeutic immune MOA type, (2) disease indication, and (3) predicted T-cell epitope content.
                Sequence-level predictions should always be interpreted in clinical context.
            </div>""", unsafe_allow_html=True)

        # ── Tab 5: Claude AI report ──
        with tab5:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

            if not api_key:
                st.warning("Set the ANTHROPIC_API_KEY environment variable to generate an AI-powered risk narrative.")
            else:
                progress.progress(90, text="Generating AI risk report with Claude...")

                report_data = {
                    "name": seq_name,
                    "seq_len": len(seq_clean),
                    "overall_risk": report.overall_risk_score,
                    "risk_category": report.risk_category,
                    "n_strong_binders": len([e for e in report.t_cell_epitopes if e.rank < 10]),
                    "n_bcell": len(report.b_cell_epitopes),
                    "hotspots": report.hotspot_regions,
                    "comparables": report.comparable_therapeutics[:3],
                }

                claude_text = generate_claude_report(report_data)

                if claude_text and not claude_text.startswith("Claude API error"):
                    st.markdown(f"""<div class="claude-report">
                        <div class="claude-badge">Generated by Claude · Anthropic</div>
                        <div>{claude_text}</div>
                    </div>""", unsafe_allow_html=True)
                elif claude_text.startswith("Claude API error"):
                    st.error(claude_text)

        progress.progress(100, text="Analysis complete!")
        time.sleep(0.5)
        progress.empty()

        # ── Disclaimer ──
        st.markdown("""<div class="disclaimer">
            FOR RESEARCH AND DEMONSTRATION PURPOSES ONLY. NOT FOR CLINICAL USE.<br>
            This tool is not FDA-cleared. Treatment decisions should be made by qualified healthcare professionals.<br>
            Data: IEDB (tools.iedb.org) · IDC DB V1 (CC BY 4.0, Agnihotri et al. 2025) · RCSB PDB · Claude (Anthropic)
        </div>""", unsafe_allow_html=True)

elif not run_clicked:
    # Landing state
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
        <div style="font-size:48px;margin-bottom:16px;">🛡️</div>
        <div style="font-size:20px;font-weight:600;color:#111827;margin-bottom:8px;">
            Predict immunogenicity risk before it satisfies your drug
        </div>
        <div style="font-size:14px;color:#6b7280;max-width:600px;margin:0 auto;line-height:1.7;">
            Paste any protein sequence to get T-cell and B-cell epitope predictions across 9 HLA alleles
            covering 85% of the global population. Visualize immune hotspots and
            benchmark against 218 real therapeutics with clinical ADA outcomes from the IDC DB V1.
        </div>
        <div style="margin-top:32px;display:flex;justify-content:center;gap:40px;">
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:600;color:#2563eb;">218</div>
                <div style="font-size:11px;color:#6b7280;">Therapeutics in IDC DB</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:600;color:#0891b2;">4,146</div>
                <div style="font-size:11px;color:#6b7280;">Clinical ADA Datapoints</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:600;color:#ea580c;">9</div>
                <div style="font-size:11px;color:#6b7280;">HLA-DR Alleles</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:600;color:#dc2626;">~30s</div>
                <div style="font-size:11px;color:#6b7280;">Per Analysis</div>
            </div>
        </div>
        <div style="margin-top:40px;font-size:13px;color:#9ca3af;">
            Select a therapeutic or paste a custom sequence in the sidebar to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show the pitch below
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **The Problem**

        Anti-drug antibodies (ADAs) are the silent killer of biologics.
        Bococizumab cost Pfizer over $1B when 44% of patients developed
        neutralizing antibodies in Phase 3. Existing prediction tools
        haven't been meaningfully updated since 1998.
        """)
    with col2:
        st.markdown("""
        **Our Approach**

        SafeBind AI combines real-time epitope prediction (IEDB, 9 HLA alleles)
        with clinical ground truth from the IDC DB V1 — the first standardized
        database linking therapeutic sequences to actual patient ADA outcomes
        across 727 clinical trials.
        """)
