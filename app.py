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

try:
    from api_keys import TAMARIND_API_KEY, ANTHROPIC_API_KEY  # local-only, gitignored
except ImportError:
    TAMARIND_API_KEY = os.environ.get("TAMARIND_API_KEY", "")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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
    api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
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
    
    # Clinical context (affects risk interpretation)
    with st.expander("⚙️ Clinical Context (optional)", expanded=False):
        route = st.selectbox("Route of administration", ["IV (intravenous)", "SC (subcutaneous)", "Unknown"])
        indication = st.selectbox("Disease indication", [
            "Oncology", "Autoimmune/Inflammation", "Healthy volunteers", 
            "Metabolic/Enzyme replacement", "Other/Unknown"
        ])
        backbone = st.selectbox("Antibody backbone", ["IgG4", "IgG1", "IgG2", "Unknown"])
        immunosuppressants = st.checkbox("Patient on immunosuppressants (MTX, etc.)")
    
    # Store clinical context in session state
    st.session_state["clinical_context"] = {
        "route": route, "indication": indication, 
        "backbone": backbone, "immunosuppressants": immunosuppressants
    }

    st.markdown("---")

    # Run button
    run_clicked = st.button("🔬 Analyze Immunogenicity", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#6b7280; line-height:1.6;">
    <b style="color:#374151;">Data sources</b><br>
    IEDB (tools.iedb.org)<br>
    IDC DB V1 (CC BY 4.0)<br>
    RCSB PDB · Tamarind Bio<br><br>
    <b style="color:#374151;">HLA coverage</b><br>
    9 DRB1 alleles (~85% global)
    </div>
    """, unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────
def compute_clinical_adjustment(raw_risk: float, context: dict) -> tuple:
    """Apply clinical context adjustments based on IDC DB V1 findings.
    Returns (adjusted_risk, adjustment_factors).
    """
    factors = []
    multiplier = 1.0
    
    # Route: SC has ~2x higher ADA than IV (Bococizumab: 46.7% SC vs 5.9% IV)
    route = context.get("route", "Unknown")
    if "IV" in route:
        multiplier *= 0.5
        factors.append("IV route: -50% (lower ADA vs SC)")
    elif "SC" in route:
        multiplier *= 1.2
        factors.append("SC route: +20% (higher ADA than IV)")
    
    # Indication: Oncology patients often immunosuppressed
    indication = context.get("indication", "Unknown")
    if indication == "Oncology":
        multiplier *= 0.4
        factors.append("Oncology: -60% (immunosuppressed patients)")
    elif "Autoimmune" in indication:
        multiplier *= 1.3
        factors.append("Autoimmune: +30% (hyperactive immune system)")
    elif "Metabolic" in indication or "Enzyme" in indication:
        multiplier *= 1.5
        factors.append("Enzyme replacement: +50% (high ADA historically)")
    
    # Backbone: IgG4 is less immunostimulatory
    backbone = context.get("backbone", "Unknown")
    if backbone == "IgG4":
        multiplier *= 0.7
        factors.append("IgG4 backbone: -30% (less immunostimulatory)")
    elif backbone == "IgG1":
        multiplier *= 1.0
        factors.append("IgG1 backbone: baseline")
    
    # Immunosuppressants
    if context.get("immunosuppressants"):
        multiplier *= 0.5
        factors.append("Immunosuppressants: -50%")
    
    adjusted = min(raw_risk * multiplier, 1.0)
    return adjusted, factors


def render_results(report, seq_clean, seq_name, pdb_id):
    """Render all metrics + tabs from a completed AssessmentReport."""
    # Get clinical context
    context = st.session_state.get("clinical_context", {})
    adjusted_risk, adj_factors = compute_clinical_adjustment(report.overall_risk_score, context)
    
    # Determine which risk to emphasize
    has_context = any(context.get(k) not in [None, "Unknown", False, "Other/Unknown"] 
                      for k in ["route", "indication", "backbone", "immunosuppressants"])
    
    display_risk = adjusted_risk if has_context else report.overall_risk_score
    
    # Categorize adjusted risk
    if display_risk >= 0.6:
        adj_category = "VERY HIGH"
    elif display_risk >= 0.4:
        adj_category = "HIGH"
    elif display_risk >= 0.2:
        adj_category = "MODERATE"
    else:
        adj_category = "LOW"
    
    risk_class = {
        "LOW": "risk-low", "MODERATE": "risk-moderate",
        "HIGH": "risk-high", "VERY HIGH": "risk-very-high"
    }.get(adj_category if has_context else report.risk_category, "risk-moderate")

    col1, col2, col3, col4, col5 = st.columns(5)
    strong = len([e for e in report.t_cell_epitopes if e.rank < 10])
    
    with col1:
        if has_context:
            st.markdown(f'<div class="metric-card"><div class="metric-value {risk_class}">{adjusted_risk:.0%}</div><div class="metric-label">Adjusted Risk</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card"><div class="metric-value {risk_class}">{report.overall_risk_score:.0%}</div><div class="metric-label">Sequence Risk</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#111827;">{adj_category if has_context else report.risk_category}</div><div class="metric-label">Risk Category</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ea580c;">{strong}</div><div class="metric-label">T-cell Epitopes</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#0891b2;">{len(report.b_cell_epitopes)}</div><div class="metric-label">B-cell Epitopes</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#7c3aed;">{len(report.hotspot_regions)}</div><div class="metric-label">Hotspot Regions</div></div>', unsafe_allow_html=True)
    
    # Show adjustment explanation if context was provided
    if has_context and adj_factors:
        with st.expander("📊 Risk adjustment factors applied", expanded=True):
            st.markdown(f"**Raw sequence risk:** {report.overall_risk_score:.0%} → **Context-adjusted:** {adjusted_risk:.0%}")
            for f in adj_factors:
                st.markdown(f"- {f}")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab2b, tab3, tab4, tab5 = st.tabs([
        "🧬 3D Heatmap", "🔥 T-cell Hotspots", "🧫 B-cell Epitopes", "📊 Residue Plot", "🏥 Clinical Context", "🤖 AI Report"
    ])

    # ── Tab 1: 3D Heatmap ──
    with tab1:
        tamarind_pdb = st.session_state.get(f"tamarind_pdb_{seq_name}")
        active_pdb = tamarind_pdb or report.pdb_data
        job_key = f"tamarind_job_{seq_name}"
        existing_job = st.session_state.get(job_key)

        if active_pdb:
            source = "Tamarind ESMFold/AlphaFold2" if tamarind_pdb else f"RCSB PDB ({pdb_id})"
            st.caption(f"Structure source: {source}")
            html = generate_3d_heatmap_html(active_pdb, report.residue_risks, chain="A",
                                            title=f"{seq_name} Immunogenicity Heatmap")
            components.html(html, height=620, scrolling=False)
        elif existing_job:
            # ── Polling loop via st.rerun() ──
            with st.spinner(f"Folding with Tamarind… job: `{existing_job}`"):
                status = get_tamarind_job_status(TAMARIND_API_KEY, existing_job)
            if status == "complete":
                pdb_result = fetch_tamarind_pdb(TAMARIND_API_KEY, existing_job)
                if pdb_result:
                    st.session_state[f"tamarind_pdb_{seq_name}"] = pdb_result
                    del st.session_state[job_key]
                    st.rerun()
                else:
                    st.warning("Job complete but PDB not found — try entering the PDB ID manually in the sidebar.")
                    del st.session_state[job_key]
            elif status == "failed":
                st.error("Tamarind job failed. Try again or enter a PDB ID in the sidebar.")
                del st.session_state[job_key]
            else:
                st.info(f"Still folding… (status: {status}). This page will auto-refresh.")
                time.sleep(8)
                st.rerun()
        else:
            # ── Text fallback + fold button ──
            st.markdown("**Per-residue risk (text view):**")
            risk_html = ""
            for rr in report.residue_risks:
                color = "#dc2626" if rr.combined_risk > 0.5 else "#ea580c" if rr.combined_risk > 0.35 else "#ca8a04" if rr.combined_risk > 0.2 else "#2563eb"
                risk_html += f'<span style="color:{color};font-family:IBM Plex Mono,monospace;font-size:14px;font-weight:500;" title="Pos {rr.position}: {rr.combined_risk:.0%}">{rr.residue}</span>'
            st.markdown(
                f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px;line-height:2;word-wrap:break-word;">'
                f'<div style="font-size:11px;color:#6b7280;margin-bottom:8px;">'
                f'<span style="color:#2563eb;">■</span> Low <span style="color:#ca8a04;">■</span> Moderate '
                f'<span style="color:#ea580c;">■</span> High <span style="color:#dc2626;">■</span> Very High</div>'
                f'{risk_html}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Predict 3D Structure with Tamarind",
                         help="Submits to ESMFold / AlphaFold2 on Tamarind Bio (uses 1 job)",
                         use_container_width=True):
                safe_name = "".join(c if c.isalnum() else "_" for c in (seq_name or "query"))[:30]
                job_name = f"safebind_{safe_name}_{int(time.time())}"
                with st.spinner("Submitting to Tamarind Bio…"):
                    submitted = submit_tamarind_structure(TAMARIND_API_KEY, seq_clean, job_name)
                if submitted:
                    st.session_state[job_key] = submitted
                    st.rerun()  # immediately enter polling loop
                else:
                    st.error("Submission failed — check the Tamarind API key or network.")

    # ── Tab 2: Hotspots ──
    with tab2:
        if report.hotspot_regions:
            for i, hs in enumerate(report.hotspot_regions, 1):
                bar_color = "#dc2626" if hs['avg_risk'] > 0.5 else "#ea580c" if hs['avg_risk'] > 0.35 else "#ca8a04"
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
                        <span>T-cell: <b style="color:#ea580c;">{hs['avg_t_cell']:.0%}</b></span>
                        <span>B-cell: <b style="color:#0891b2;">{hs['avg_b_cell']:.0%}</b></span>
                        <span>Max: <b style="color:#dc2626;">{hs['max_risk']:.0%}</b></span>
                    </div></div>""", unsafe_allow_html=True)
        else:
            st.success("No significant T-cell hotspot regions detected above threshold.")

    # ── Tab 2b: B-cell epitopes ──
    with tab2b:
        st.markdown("**Linear B-cell epitope regions** (predicted by IEDB Bepipred)")
        st.caption("These regions are likely accessible on the protein surface and may trigger antibody responses.")
        if report.b_cell_epitopes:
            for i, be in enumerate(report.b_cell_epitopes, 1):
                score_pct = be.avg_score
                bar_color = "#0891b2" if score_pct > 0.7 else "#06b6d4" if score_pct > 0.5 else "#22d3ee"
                st.markdown(f"""<div class="hotspot-row">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-weight:600;font-size:15px;color:#111827;">B-cell Epitope {i}</span>
                            <span style="color:#6b7280;font-size:13px;margin-left:8px;">Positions {be.start}–{be.end} ({be.end - be.start + 1} residues)</span>
                        </div>
                        <div style="font-weight:600;font-size:18px;color:{bar_color};">{be.avg_score:.0%}</div>
                    </div>
                    <div class="hotspot-seq" style="margin-top:8px;">{be.sequence}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No significant B-cell epitope regions detected above threshold.")

    # ── Tab 3: Residue plot ──
    with tab3:
        import pandas as pd
        df = pd.DataFrame([{
            "Position": rr.position, "Residue": rr.residue,
            "T-cell Risk": rr.t_cell_risk, "B-cell Risk": rr.b_cell_risk,
            "Combined Risk": rr.combined_risk, "Alleles Binding": rr.num_alleles_binding
        } for rr in report.residue_risks])
        st.markdown("**Per-residue immunogenicity risk profile**")
        st.area_chart(df.set_index("Position")[["T-cell Risk", "B-cell Risk", "Combined Risk"]],
                      color=["#ea580c", "#0891b2", "#dc2626"], height=350)
        with st.expander("View full residue data table"):
            st.dataframe(df.style.background_gradient(subset=["Combined Risk"], cmap="YlOrRd"),
                         use_container_width=True, height=400)
        st.download_button("📥 Download CSV", df.to_csv(index=False),
                           file_name=f"{(seq_name or 'query').lower().replace(' ','_')}_risk_scores.csv",
                           mime="text/csv")

    # ── Tab 4: Clinical context ──
    with tab4:
        st.markdown("**Comparable therapeutics from IDC DB V1**")
        st.caption("3,334 ADA datapoints across 218 therapeutics and 726 clinical trials")
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
                    </div></div>""", unsafe_allow_html=True)
        else:
            st.info("No comparable therapeutics found in IDC DB V1.")
        st.markdown("""<div style="margin-top:16px;font-size:12px;color:#6b7280;line-height:1.6;">
            <b>Key finding from IDC DB V1:</b> Top 3 drivers of clinical ADA frequency are
            (1) therapeutic immune MOA type, (2) disease indication, and (3) predicted T-cell epitope content.
        </div>""", unsafe_allow_html=True)

    # ── Tab 5: Claude AI report ──
    with tab5:
        api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            st.warning("No Anthropic API key configured.")
        else:
            if f"claude_report_{seq_name}" not in st.session_state:
                with st.spinner("Generating AI risk report with Claude…"):
                    report_data = {
                        "name": seq_name, "seq_len": len(seq_clean),
                        "overall_risk": report.overall_risk_score,
                        "risk_category": report.risk_category,
                        "n_strong_binders": strong,
                        "n_bcell": len(report.b_cell_epitopes),
                        "hotspots": report.hotspot_regions,
                        "comparables": report.comparable_therapeutics[:3],
                    }
                    st.session_state[f"claude_report_{seq_name}"] = generate_claude_report(report_data)
            claude_text = st.session_state.get(f"claude_report_{seq_name}", "")
            if claude_text and not claude_text.startswith("Claude API error"):
                st.markdown(f'<div class="claude-report"><div class="claude-badge">Generated by Claude · Anthropic</div><div>{claude_text}</div></div>', unsafe_allow_html=True)
            elif claude_text.startswith("Claude API error"):
                st.error(claude_text)

    st.markdown("""<div class="disclaimer">
        FOR RESEARCH AND DEMONSTRATION PURPOSES ONLY. NOT FOR CLINICAL USE.<br>
        Data: IEDB (tools.iedb.org) · IDC DB V1 (CC BY 4.0, Agnihotri et al. 2025) · RCSB PDB · Claude (Anthropic) · Tamarind Bio
    </div>""", unsafe_allow_html=True)


# ── Main content ─────────────────────────────────────────────
st.markdown('<div class="hero-title">SafeBind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Immunogenicity risk assessment for therapeutic proteins · Powered by IEDB, IDC DB V1, and Claude</div>', unsafe_allow_html=True)

if run_clicked and seq_input:
    seq_clean = "".join(c for c in seq_input.upper() if c.isalpha())
    if len(seq_clean) < 20:
        st.error("Sequence too short. Please enter at least 20 amino acids.")
    else:
        progress = st.progress(0, text="Initializing analysis...")
        progress.progress(5, text="Predicting T-cell epitopes across 9 HLA alleles...")
        report = run_immunogenicity_assessment(
            sequence=seq_clean, name=seq_name or "Query",
            pdb_id=pdb_id if pdb_id else None, pdb_chain="A",
            idc_data_path="idc_db_v1_table_s4.xlsx",
            species=species, modality="Monoclonal antibody", verbose=False,
        )
        progress.progress(95, text="Rendering results...")
        # Persist to session state so Tamarind button reruns don't lose the report
        st.session_state["report"] = report
        st.session_state["seq_clean"] = seq_clean
        st.session_state["seq_name"] = seq_name
        st.session_state["pdb_id"] = pdb_id
        # Clear any cached Claude report when re-analyzing
        st.session_state.pop(f"claude_report_{seq_name}", None)
        progress.progress(100, text="Analysis complete!")
        time.sleep(0.3)
        progress.empty()

if "report" in st.session_state:
    render_results(
        st.session_state["report"],
        st.session_state["seq_clean"],
        st.session_state["seq_name"],
        st.session_state.get("pdb_id", ""),
    )

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
