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
    "Bococizumab (anti-PCSK9)": {
        "sequence": """>Heavy Chain
QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGEISPFGGRTNYNEKFKSRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARERPLYASDLWGQGTTVTVSS""",
        "modality": "Monoclonal Antibody",
        "species": "Humanized",
        "route": "Subcutaneous",
        "disease": "Cardiovascular",
        "backbone": "human IgG2",
        "conjugate": "Unconjugated",
        "expression_system": "Chinese hamster ovary (CHO) cells",
        "dose": "150 mg",
        "schedule": "Q2W SC",
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
from safebind_mhc1_cytotoxic import (
    run_cytotoxic_assessment, CytotoxicReport,
    MHCIEpitope, CytotoxicResidueRisk,
)
from safebind_composite_scorer import compute_composite_score, CompositeScore
import hashlib
import json

_MHC1_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".mhc1_cache")
_PDB_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".pdb_cache")


def _load_mhc1_cache(chain_name: str, chain_seq: str):
    """Load pre-computed MHC-I results from disk cache.

    Tries exact name+hash match first, then falls back to hash-only match
    (handles FASTA header variations like 'VL' vs 'VL|original').
    """
    seq_hash = hashlib.sha256(chain_seq.encode()).hexdigest()[:12]
    safe_name = chain_name.replace(" ", "_").replace("/", "_")
    cache_path = os.path.join(_MHC1_CACHE_DIR, f"{safe_name}_{seq_hash}.json")
    if not os.path.exists(cache_path):
        # Fallback: search by hash suffix in any cached file
        if os.path.isdir(_MHC1_CACHE_DIR):
            for fname in os.listdir(_MHC1_CACHE_DIR):
                if fname.endswith(f"_{seq_hash}.json"):
                    cache_path = os.path.join(_MHC1_CACHE_DIR, fname)
                    break
            else:
                return None
        else:
            return None
    with open(cache_path) as f:
        d = json.load(f)
    # Reconstruct dataclass
    epitopes = [MHCIEpitope(**ep) for ep in d.get("epitopes", [])]
    residue_risks = [CytotoxicResidueRisk(**rr) for rr in d.get("residue_risks", [])]
    return CytotoxicReport(
        total_epitopes_predicted=d["total_epitopes_predicted"],
        strong_binders=d["strong_binders"],
        moderate_binders=d["moderate_binders"],
        epitopes=epitopes,
        residue_risks=residue_risks,
        hotspot_regions=d.get("hotspot_regions", []),
        validated_hits=d.get("validated_hits", 0),
        validated_details=d.get("validated_details", []),
        overall_cytotoxic_risk=d["overall_cytotoxic_risk"],
        risk_category=d["risk_category"],
        prediction_sources=d.get("prediction_sources", []),
        aav_epitope_recovery=d.get("aav_epitope_recovery"),
        data_references=d.get("data_references"),
    )
from deimmunize import (
    deimmunize_epitopes, generate_redesigned_sequences,
    run_tolerance_analysis, compute_variant_risk_comparison,
)
from tamarind_integration import (
    fold_protein, _get_api_key,
    submit_fold_job, check_fold_status, fetch_fold_result,
)
from safebind_downselect import render_downselect_tab
import streamlit.components.v1 as components


# ── Adapter for downselect module ──
# Wraps existing functions to match the expected interface
class MockReport:
    """Adapter to wrap this branch's results into the expected report format."""
    def __init__(self, overall_risk_score, t_cell_epitopes, b_cell_epitopes, hotspot_regions, risk_category, residue_risks=None):
        self.overall_risk_score = overall_risk_score
        self.t_cell_epitopes = t_cell_epitopes
        self.b_cell_epitopes = b_cell_epitopes
        self.hotspot_regions = hotspot_regions
        self.risk_category = risk_category
        self.residue_risks = residue_risks or []

class MockEpitope:
    """Mock epitope with rank attribute for compatibility."""
    def __init__(self, rank, start=0, end=0, sequence=""):
        self.rank = rank
        self.start = start
        self.end = end
        self.sequence = sequence

def run_immunogenicity_assessment_adapter(
    sequence, name="Query", pdb_id=None, pdb_chain="A",
    idc_data_path=None, species="Humanized", modality="Monoclonal antibody", verbose=False
):
    """Adapter function that wraps this branch's prediction functions."""
    # Predict T-cell epitopes
    from sequence_engine import predict_epitopes, predict_bcell_epitopes, compute_epitope_density
    
    t_cell_raw = predict_epitopes(sequence)
    t_cell_epitopes = [
        MockEpitope(rank=ep.rank_percentile if hasattr(ep, 'rank_percentile') else ep.rank if hasattr(ep, 'rank') else 5, 
                    start=ep.position if hasattr(ep, 'position') else 0,
                    sequence=ep.peptide if hasattr(ep, 'peptide') else "")
        for ep in t_cell_raw
    ]
    
    # Predict B-cell epitopes
    b_cell_epitopes = predict_bcell_epitopes(sequence)
    
    # Calculate epitope density as risk score
    epitope_density = compute_epitope_density(t_cell_raw, len(sequence))
    overall_risk = min(1.0, epitope_density / 50)  # Normalize: 50 epitopes/100aa = 100% risk
    
    # Identify hotspot regions (clusters of epitopes)
    hotspot_regions = []
    if t_cell_raw:
        # Simple clustering: find regions with multiple epitopes
        positions = sorted(set(ep.position if hasattr(ep, 'position') else 0 for ep in t_cell_raw))
        if positions:
            current_start = positions[0]
            current_end = positions[0]
            for pos in positions[1:]:
                if pos <= current_end + 20:  # within 20aa
                    current_end = pos + 15
                else:
                    if current_end - current_start >= 15:
                        hotspot_regions.append({
                            "start": current_start,
                            "end": current_end,
                            "length": current_end - current_start,
                            "sequence": sequence[current_start:current_end],
                            "avg_risk": overall_risk,
                            "avg_t_cell": overall_risk,
                            "avg_b_cell": 0.3,
                            "max_risk": overall_risk,
                        })
                    current_start = pos
                    current_end = pos + 15
            if current_end - current_start >= 15:
                hotspot_regions.append({
                    "start": current_start,
                    "end": min(current_end, len(sequence)),
                    "length": min(current_end, len(sequence)) - current_start,
                    "sequence": sequence[current_start:min(current_end, len(sequence))],
                    "avg_risk": overall_risk,
                    "avg_t_cell": overall_risk,
                    "avg_b_cell": 0.3,
                    "max_risk": overall_risk,
                })
    
    # Risk category
    if overall_risk >= 0.6:
        risk_category = "VERY HIGH"
    elif overall_risk >= 0.4:
        risk_category = "HIGH"
    elif overall_risk >= 0.2:
        risk_category = "MODERATE"
    else:
        risk_category = "LOW"
    
    return MockReport(
        overall_risk_score=overall_risk,
        t_cell_epitopes=t_cell_epitopes,
        b_cell_epitopes=b_cell_epitopes,
        hotspot_regions=hotspot_regions,
        risk_category=risk_category,
    )


def _load_pdb_cache(chain_name: str, chain_seq: str) -> str | None:
    """Load cached PDB structure from disk."""
    seq_hash = hashlib.sha256(chain_seq.encode()).hexdigest()[:12]
    safe_name = chain_name.replace(" ", "_").replace("/", "_")
    cache_path = os.path.join(_PDB_CACHE_DIR, f"{safe_name}_{seq_hash}.pdb")
    if not os.path.exists(cache_path):
        # Fallback: search by hash suffix
        if os.path.isdir(_PDB_CACHE_DIR):
            for fname in os.listdir(_PDB_CACHE_DIR):
                if fname.endswith(f"_{seq_hash}.pdb"):
                    cache_path = os.path.join(_PDB_CACHE_DIR, fname)
                    break
            else:
                return None
        else:
            return None
    with open(cache_path) as f:
        return f.read()


def _save_pdb_cache(chain_name: str, chain_seq: str, pdb_data: str):
    """Save PDB structure to disk cache."""
    os.makedirs(_PDB_CACHE_DIR, exist_ok=True)
    seq_hash = hashlib.sha256(chain_seq.encode()).hexdigest()[:12]
    safe_name = chain_name.replace(" ", "_").replace("/", "_")
    cache_path = os.path.join(_PDB_CACHE_DIR, f"{safe_name}_{seq_hash}.pdb")
    with open(cache_path, "w") as f:
        f.write(pdb_data)


# ── 3-D Risk Heatmap helpers ────────────────────────────────────────
def _compute_residue_risk_map(
    seq_len: int,
    epitopes=None,
    bcell_epitopes=None,
    mhc1_report=None,
    mode="combined",
):
    """Return a list of floats (0-1) per residue for the chosen risk layer.

    Modes: 'humoral' (MHC-II + B-cell), 'cytotoxic' (MHC-I), 'combined'.
    """
    humoral = [0.0] * seq_len
    cytotoxic = [0.0] * seq_len

    # MHC-II / T-cell epitope density  → humoral layer
    if epitopes:
        for ep in epitopes:
            rank_score = max(0.0, 1.0 - ep.percentile_rank / 10.0)  # rank<2 → ~0.8+
            for pos in range(ep.start - 1, min(ep.end, seq_len)):
                humoral[pos] = max(humoral[pos], rank_score)

    # B-cell epitope density → humoral layer (additive, capped at 1)
    if bcell_epitopes:
        for bep in bcell_epitopes:
            for pos in range(bep.start - 1, min(bep.end, seq_len)):
                humoral[pos] = min(1.0, humoral[pos] + bep.avg_score * 0.5)

    # MHC-I per-residue risk → cytotoxic layer
    if mhc1_report and mhc1_report.residue_risks:
        for rr in mhc1_report.residue_risks:
            idx = rr.position - 1
            if 0 <= idx < seq_len:
                cytotoxic[idx] = rr.mhc1_risk

    if mode == "humoral":
        return humoral
    elif mode == "cytotoxic":
        return cytotoxic
    else:  # combined — max of both layers
        return [max(h, c) for h, c in zip(humoral, cytotoxic)]


def _inject_bfactor(pdb_text: str, risk_scores: list[float]) -> str:
    """Replace B-factor column in ATOM/HETATM records with risk scores (0-99)."""
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                resseq = int(line[22:26].strip())
                idx = resseq - 1
                bval = risk_scores[idx] * 99.0 if 0 <= idx < len(risk_scores) else 0.0
            except (ValueError, IndexError):
                bval = 0.0
            # B-factor occupies columns 60-65 (6 chars, right-justified, 2 decimal)
            line = line[:60] + f"{bval:6.2f}" + line[66:]
        lines.append(line)
    return "\n".join(lines)


def _render_3d_heatmap(pdb_data, risk_scores, view_key, caption=""):
    """Render a py3Dmol viewer colored by risk scores injected as B-factors."""
    import stmol
    import py3Dmol

    modified_pdb = _inject_bfactor(pdb_data, risk_scores)
    view = py3Dmol.view(width="100%", height=400)
    view.addModel(modified_pdb, "pdb")
    view.setStyle({
        "cartoon": {
            "colorscheme": {
                "prop": "b",
                "gradient": "rwb",
                "min": 0,
                "max": 99,
            }
        }
    })
    # Highlight high-risk residues (>50%) with sticks
    view.addStyle(
        {"b": [50, 99]},
        {"stick": {"colorscheme": {"prop": "b", "gradient": "rwb", "min": 0, "max": 99}, "radius": 0.12}},
    )
    view.zoomTo()
    stmol.showmol(view, height=400, width=None)
    if caption:
        st.caption(caption)


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
chain_mhc1 = {}         # {chain_name: CytotoxicReport}

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

tab1, tab2, tab_bcell, tab_mhc1, tab_composite, tab3, tab_ds = st.tabs([
    "Prediction",
    "Structure",
    "B-cell Epitopes",
    "Cytotoxic (MHC-I)",
    "Composite Score",
    "Redesign",
    "Downselect",
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

                    # Try disk cache if not in session state
                    if not pdb_data:
                        pdb_data = _load_pdb_cache(chain_name, chain_seq)
                        if pdb_data:
                            st.session_state[pdb_ss_key] = pdb_data

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
                                _save_pdb_cache(chain_name, chain_seq, pdb_result)
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
                            # Gather available risk layers for this chain
                            _mhc1_for_view = chain_mhc1.get(chain_name)
                            if _mhc1_for_view is None:
                                _mhc1_ss = st.session_state.get(f"mhc1_{chain_name}")
                                if isinstance(_mhc1_ss, CytotoxicReport):
                                    _mhc1_for_view = _mhc1_ss

                            _has_humoral = bool(ep_results or chain_bcell.get(chain_name))
                            _has_cytotoxic = bool(_mhc1_for_view and _mhc1_for_view.residue_risks)

                            heatmap_mode = st.radio(
                                "Color by risk",
                                ["Combined", "Humoral (MHC-II + B-cell)", "Cytotoxic (MHC-I)", "Rainbow (no risk)"],
                                horizontal=True,
                                key=f"heatmap_mode_{chain_name}",
                            )

                            if heatmap_mode == "Rainbow (no risk)":
                                import stmol, py3Dmol
                                view = py3Dmol.view(width="100%", height=400)
                                view.addModel(pdb_data, "pdb")
                                view.setStyle({"cartoon": {"color": "spectrum"}})
                                view.zoomTo()
                                stmol.showmol(view, height=400, width=None)
                                st.caption("Rainbow coloring (N→C terminus)")
                            else:
                                mode_map = {
                                    "Combined": "combined",
                                    "Humoral (MHC-II + B-cell)": "humoral",
                                    "Cytotoxic (MHC-I)": "cytotoxic",
                                }
                                risk_scores = _compute_residue_risk_map(
                                    seq_len=len(chain_seq),
                                    epitopes=ep_results,
                                    bcell_epitopes=chain_bcell.get(chain_name),
                                    mhc1_report=_mhc1_for_view,
                                    mode=mode_map[heatmap_mode],
                                )
                                hot_count = sum(1 for s in risk_scores if s > 0.5)
                                _render_3d_heatmap(
                                    pdb_data, risk_scores,
                                    view_key=f"heatmap_{chain_name}_{heatmap_mode}",
                                    caption=f"Blue = low risk → Red = high risk | {hot_count} high-risk residues (>50%)",
                                )

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
# PART 2C — Cytotoxic (MHC-I) Analysis
# ==============================
with tab_mhc1:
    if not chains:
        st.info("Paste a protein sequence in the sidebar to enable MHC-I cytotoxic analysis.")
    else:
        st.markdown("**MHC Class I / CD8+ Cytotoxic T-cell Assessment**")
        st.caption(
            "Dual-pathway analysis: MHC-I presents intracellular peptides (8-11mers) to CD8+ cytotoxic T cells. "
            "Unlike MHC-II/ADA (humoral), this pathway causes direct cell killing — critical for gene therapy and cell therapy."
        )

        # Load cached MHC-I results: session state first, then disk cache
        for chain_name, chain_seq in chains.items():
            mhc1_key = f"mhc1_{chain_name}"
            if mhc1_key in st.session_state:
                chain_mhc1[chain_name] = st.session_state[mhc1_key]
            else:
                cached = _load_mhc1_cache(chain_name, chain_seq)
                if cached:
                    st.session_state[mhc1_key] = cached
                    chain_mhc1[chain_name] = cached

        _mhc1_cached = len(chain_mhc1) == len(chains)

        if not _mhc1_cached:
            missing = [cn for cn in chains if cn not in chain_mhc1]
            st.info(
                f"MHC-I prediction queries IEDB NetMHCpan 4.1 across 12 HLA supertypes "
                f"for {len(missing)} chain(s). This takes ~10-15 seconds per chain."
            )
            if st.button("Run MHC-I Cytotoxic Analysis", type="primary"):
                for chain_name, chain_seq in chains.items():
                    mhc1_key = f"mhc1_{chain_name}"
                    if mhc1_key not in st.session_state:
                        with st.spinner(f"Predicting MHC-I epitopes for {chain_name} ({len(chain_seq)} aa)..."):
                            mhc1_report = run_cytotoxic_assessment(
                                chain_seq, name=chain_name, use_mhcflurry=False, use_iedb=True,
                            )
                            st.session_state[mhc1_key] = mhc1_report
                    chain_mhc1[chain_name] = st.session_state[mhc1_key]
                st.rerun()

        if chain_mhc1:
            # Summary metrics across chains
            total_strong = sum(r.strong_binders for r in chain_mhc1.values())
            total_moderate = sum(r.moderate_binders for r in chain_mhc1.values())
            total_validated = sum(r.validated_hits for r in chain_mhc1.values())
            avg_risk = sum(r.overall_cytotoxic_risk for r in chain_mhc1.values()) / max(len(chain_mhc1), 1)

            mc1, mc2, mc3, mc4 = st.columns(4)
            risk_color = "#c0392b" if avg_risk >= 0.35 else "#e67e22" if avg_risk >= 0.18 else "#27ae60"
            mc1.markdown(
                f'<div style="text-align:center;padding:0.8rem;background:{risk_color}10;border-radius:8px;border:1px solid {risk_color}30">'
                f'<div style="font-size:1.8rem;font-weight:700;color:{risk_color}">{avg_risk:.0%}</div>'
                f'<div style="font-size:0.8rem;color:#666">MHC-I Risk</div></div>',
                unsafe_allow_html=True,
            )
            mc2.metric("Strong Binders", total_strong, help="Rank < 2%")
            mc3.metric("Moderate Binders", total_moderate, help="Rank 2-10%")
            mc4.metric("Validated Hits", total_validated, help="Matches to known epitopes")

            st.markdown("")

        # Dual pathway comparison
        if chain_mhc1 and any(chain_epitopes.get(name) for name in chains):
            st.markdown("**Dual Pathway Comparison**")
            pathway_rows = []
            for cname in chains:
                mhc2_eps = chain_epitopes.get(cname, [])
                mhc1_rpt = chain_mhc1.get(cname)
                if mhc1_rpt:
                    pathway_rows.append({
                        "Chain": cname,
                        "MHC-II (ADA)": len(mhc2_eps),
                        "MHC-I (Cytotoxic)": mhc1_rpt.total_epitopes_predicted,
                        "MHC-II Density": round(len(mhc2_eps) / max(len(chains[cname]), 1) * 100, 1),
                        "MHC-I Density": round(mhc1_rpt.total_epitopes_predicted / max(len(chains[cname]), 1) * 100, 1),
                        "Cytotoxic Risk": mhc1_rpt.risk_category,
                    })
            if pathway_rows:
                st.dataframe(pd.DataFrame(pathway_rows), hide_index=True, use_container_width=True)

        # Per-chain details
        for chain_name in chains:
            mhc1_rpt = chain_mhc1.get(chain_name)
            if not mhc1_rpt:
                continue

            with st.expander(f"{chain_name} — {mhc1_rpt.risk_category} risk ({mhc1_rpt.overall_cytotoxic_risk:.0%})", expanded=(len(chains) == 1)):
                # Hotspot regions
                if mhc1_rpt.hotspot_regions:
                    st.markdown("**CTL Hotspot Regions**")
                    hs_df = pd.DataFrame([
                        {
                            "Region": f"{hs['start']}-{hs['end']}",
                            "Sequence": hs["sequence"][:20] + ("..." if len(hs["sequence"]) > 20 else ""),
                            "Avg Risk": round(hs["avg_risk"], 2),
                            "Max Risk": round(hs["max_risk"], 2),
                            "Alleles": hs["max_alleles"],
                            "Validated": "Yes" if hs.get("has_validated") else "—",
                        }
                        for hs in mhc1_rpt.hotspot_regions[:10]
                    ])
                    st.dataframe(hs_df, hide_index=True, use_container_width=True)

                # Top epitopes
                if mhc1_rpt.epitopes:
                    st.markdown("**Top MHC-I Epitopes**")
                    top_eps = sorted(mhc1_rpt.epitopes, key=lambda e: e.rank)[:20]
                    ep_df = pd.DataFrame([
                        {
                            "Pos": f"{ep.start}-{ep.end}",
                            "Peptide": ep.sequence,
                            "Allele": ep.allele,
                            "Rank (%)": round(ep.rank, 2),
                            "Length": ep.length,
                            "Source": ep.source,
                        }
                        for ep in top_eps
                    ])
                    st.dataframe(ep_df, hide_index=True, use_container_width=True, height=300)

                # Validated epitope cross-reference
                if mhc1_rpt.validated_details:
                    st.markdown("**Validated Epitope Cross-Reference**")
                    if mhc1_rpt.aav_epitope_recovery is not None:
                        st.caption(f"Recovery rate: {mhc1_rpt.aav_epitope_recovery:.0%} of known epitopes detected")
                    val_df = pd.DataFrame([
                        {
                            "Peptide": v["validated_peptide"],
                            "Position": v["position"],
                            "HLA": v["hla"],
                            "Source": v["source"],
                            "Recovered": "Yes" if v["is_recovered"] else "No",
                        }
                        for v in mhc1_rpt.validated_details
                    ])
                    st.dataframe(val_df, hide_index=True, use_container_width=True)

                # Per-residue risk heatmap (Altair)
                if mhc1_rpt.residue_risks:
                    st.markdown("**Per-Residue Cytotoxic Risk**")
                    risk_data = pd.DataFrame([
                        {"Position": rr.position, "Risk": rr.mhc1_risk, "Alleles": rr.num_alleles_binding}
                        for rr in mhc1_rpt.residue_risks
                    ])
                    risk_chart = alt.Chart(risk_data).mark_bar(width=1).encode(
                        x=alt.X("Position:Q", title="Residue Position"),
                        y=alt.Y("Risk:Q", title="MHC-I Risk", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Risk:Q", scale=alt.Scale(scheme="reds"), legend=None),
                        tooltip=["Position", "Risk", "Alleles"],
                    ).properties(height=200)
                    st.altair_chart(risk_chart, use_container_width=True)

        # Data sources
        with st.expander("Data Sources"):
            first_report = next(iter(chain_mhc1.values()), None)
            if first_report and first_report.data_references:
                for key, val in first_report.data_references.items():
                    st.caption(f"**{key}**: {val}")

# ==============================
# PART 2D — Composite Score
# ==============================
with tab_composite:
    if not chains:
        st.info("Paste a protein sequence in the sidebar to compute the composite score.")
    else:
        st.markdown("**4-Signal Composite Immunogenicity Score**")
        st.caption(
            "Fuses clinical benchmarks (IDC DB V1), sequence similarity, "
            "dual-pathway epitope load (MHC-I + MHC-II), and AI synthesis."
        )

        # Load any cached MHC-I results for composite scoring
        for cname in chains:
            mhc1_key = f"mhc1_{cname}"
            if mhc1_key in st.session_state and cname not in chain_mhc1:
                chain_mhc1[cname] = st.session_state[mhc1_key]

        # Aggregate epitope data across chains
        total_mhc2 = sum(len(chain_epitopes.get(cname, [])) for cname in chains)
        total_mhc2_hotspots = 0
        total_mhc1 = sum(r.total_epitopes_predicted for r in chain_mhc1.values())
        total_mhc1_hotspots = sum(len(r.hotspot_regions) for r in chain_mhc1.values())
        mhc1_overall = sum(r.overall_cytotoxic_risk for r in chain_mhc1.values()) / max(len(chain_mhc1), 1) if chain_mhc1 else 0.0

        # Estimate MHC-II hotspots from epitope clusters
        for cname in chains:
            eps = chain_epitopes.get(cname, [])
            if eps:
                positions = set()
                for ep in eps:
                    positions.update(range(ep.start, ep.end + 1))
                if positions:
                    sorted_pos = sorted(positions)
                    clusters = 1
                    for i in range(1, len(sorted_pos)):
                        if sorted_pos[i] - sorted_pos[i-1] > 5:
                            clusters += 1
                    total_mhc2_hotspots += clusters

        combined_seq = "".join(chains.values())

        composite_key = f"composite_{hash(combined_seq)}"
        if composite_key not in st.session_state:
            comp = compute_composite_score(
                sequence=combined_seq,
                name="Candidate",
                modality=modality,
                route=route,
                species=species,
                indication=disease,
                mhc2_epitope_count=total_mhc2,
                mhc2_hotspot_count=total_mhc2_hotspots,
                mhc2_overall_risk=risk_result.composite_score / 100.0,
                mhc1_epitope_count=total_mhc1,
                mhc1_hotspot_count=total_mhc1_hotspots,
                mhc1_overall_risk=mhc1_overall,
            )
            st.session_state[composite_key] = comp
        comp = st.session_state[composite_key]

        # Hero composite score
        comp_color = "#c0392b" if comp.composite_score >= 60 else "#e67e22" if comp.composite_score >= 40 else "#f39c12" if comp.composite_score >= 20 else "#27ae60"
        col_comp, col_signals = st.columns([1, 3])

        with col_comp:
            st.markdown(
                f'<div style="text-align:center;padding:1.5rem;background:{comp_color}10;border-radius:12px;border:2px solid {comp_color}">'
                f'<div style="color:{comp_color};font-size:2.8rem;font-weight:700;line-height:1">{comp.composite_score:.0f}</div>'
                f'<div style="color:{comp_color};font-size:0.9rem;font-weight:600;margin-top:0.3rem">{comp.composite_category}</div>'
                f'<div style="font-size:0.75rem;color:#888;margin-top:0.3rem">'
                f'95% CI: [{comp.confidence_interval[0]:.0f}, {comp.confidence_interval[1]:.0f}]</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(
                f'<div style="display:flex;justify-content:space-around;text-align:center">'
                f'<div><div style="font-size:1.2rem;font-weight:600;color:#2563eb">{comp.humoral_risk:.0f}</div>'
                f'<div style="font-size:0.7rem;color:#666">Humoral</div></div>'
                f'<div><div style="font-size:1.2rem;font-weight:600;color:#dc2626">{comp.cytotoxic_risk:.0f}</div>'
                f'<div style="font-size:0.7rem;color:#666">Cytotoxic</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_signals:
            signals = [
                ("Clinical Benchmark", comp.benchmark_signal, "#3b82f6"),
                ("Sequence Similarity", comp.similarity_signal, "#8b5cf6"),
                ("Epitope Load", comp.epitope_signal, "#ef4444"),
            ]
            for sig_name, sig, color in signals:
                bar_width = min(sig.score, 100)
                st.markdown(
                    f'<div style="margin-bottom:0.8rem">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
                    f'<span style="font-size:0.85rem;font-weight:600">{sig_name} ({sig.weight:.0%})</span>'
                    f'<span style="font-size:0.85rem;font-weight:700;color:{color}">{sig.score:.0f}/100</span>'
                    f'</div>'
                    f'<div style="background:#e5e7eb;border-radius:4px;height:8px;overflow:hidden">'
                    f'<div style="background:{color};height:100%;width:{bar_width}%;border-radius:4px"></div>'
                    f'</div>'
                    f'<div style="font-size:0.72rem;color:#888;margin-top:2px">{sig.explanation[:120]}{"..." if len(sig.explanation) > 120 else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Risk flags
        if comp.flags:
            st.markdown("")
            st.markdown("**Risk Flags**")
            for flag in comp.flags:
                st.markdown(
                    f'<div style="padding:6px 10px;background:#fef2f2;border-left:3px solid #dc2626;'
                    f'border-radius:4px;margin-bottom:4px;font-size:0.85rem">{flag}</div>',
                    unsafe_allow_html=True,
                )

        # Signal details
        with st.expander("Signal Details"):
            for sig_name, sig, _ in signals:
                st.markdown(f"**{sig_name}** (weight: {sig.weight:.0%}, confidence: {sig.confidence:.0%})")
                st.markdown(sig.explanation)
                if sig.data_sources:
                    for ds in sig.data_sources:
                        st.caption(f"  {ds}")
                st.markdown("")

        # Data references
        with st.expander("Data Sources & References"):
            st.caption(f"Total data sources: {comp.total_data_sources}")
            for key, val in comp.data_references.items():
                st.caption(f"**{key}**: {val}")

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

        # --- Structural constraints summary ---
        _has_cdrs = any(chain_cdr.get(n) for n in chains)
        _has_sasa = any(chain_sasa.get(n) for n in chains)
        _has_mhc1 = any(chain_mhc1.get(n) or st.session_state.get(f"mhc1_{n}") for n in chains)
        constraint_parts = []
        if _has_cdrs:
            constraint_parts.append("CDR-protected")
        if _has_sasa:
            constraint_parts.append("SASA-filtered")
        if _has_mhc1:
            constraint_parts.append("MHC-I + MHC-II")
        else:
            constraint_parts.append("MHC-II only")
        st.caption("Constraints: " + " | ".join(constraint_parts))

        # --- Rule-Based Deimmunization ---
        st.markdown("**Deimmunized Variants**")

        all_chain_variants = {}

        for chain_name, chain_seq in chains.items():
            ep_results = chain_epitopes.get(chain_name, [])
            if not ep_results:
                continue

            tol_for_chain = chain_tolerance.get(chain_name)
            cdrs_for_chain = chain_cdr.get(chain_name)
            sasa_for_chain = chain_sasa.get(chain_name)

            # Gather MHC-I epitopes if available
            mhc1_for_chain = chain_mhc1.get(chain_name)
            if mhc1_for_chain is None:
                _ss = st.session_state.get(f"mhc1_{chain_name}")
                if hasattr(_ss, 'epitopes'):
                    mhc1_for_chain = _ss
            mhc1_eps = mhc1_for_chain.epitopes if mhc1_for_chain else None

            deimm_results = deimmunize_epitopes(
                chain_seq, ep_results,
                tolerance_result=tol_for_chain,
                cdr_regions=cdrs_for_chain,
                sasa_scores=sasa_for_chain,
                mhc1_epitopes=mhc1_eps,
            )
            variants = generate_redesigned_sequences(
                chain_seq, deimm_results,
                original_epitopes=ep_results,
            )
            all_chain_variants[chain_name] = variants

            if not variants:
                st.caption(f"{chain_name}: no high-confidence mutations identified.")
                continue

            # Count MHC class breakdown
            n_c1 = sum(1 for dr in deimm_results if dr.mhc_class == "I")
            n_c2 = sum(1 for dr in deimm_results if dr.mhc_class == "II")
            class_label = f"Class II: {n_c2}"
            if n_c1:
                class_label += f", Class I: {n_c1}"

            with st.expander(f"{chain_name} — {len(deimm_results)} targets ({class_label}), {len(variants)} variants", expanded=(len(chains) == 1)):
                # Mutation map
                mut_df = pd.DataFrame([
                    {
                        "Region": f"{dr.region_start}-{dr.region_end}",
                        "MHC": dr.mhc_class,
                        "Mutations": ", ".join(f"{o}{p}{n}" for p, o, n in dr.mutations),
                        "Disruption": dr.expected_binding_disruption,
                        "Allele": dr.allele,
                    }
                    for dr in deimm_results
                ])
                st.dataframe(mut_df, hide_index=True, use_container_width=True)

                # Variants with re-scoring
                for i, var in enumerate(variants):
                    # Re-score badge
                    rescore_badge = ""
                    if var.epitopes_before > 0:
                        reduction = var.epitopes_before - var.epitopes_after
                        pct = reduction / var.epitopes_before * 100
                        rescore_badge = f" — **{reduction}/{var.epitopes_before} epitopes disrupted ({pct:.0f}%)**"

                    st.markdown(f"**{var.name}** — {var.n_mutations} mutations{rescore_badge}")
                    st.caption(var.strategy)

                    # Highlighted sequence
                    seq_html = '<div style="font-family:monospace; word-wrap:break-word; line-height:1.8; font-size:0.82rem">'
                    mut_positions = {pos for pos, _, _ in var.mutations}
                    cdr_pos_set = set()
                    if cdrs_for_chain:
                        for cdr in cdrs_for_chain:
                            cdr_pos_set.update(range(cdr["start"], cdr["end"] + 1))
                    for j, aa in enumerate(var.sequence):
                        pos = j + 1
                        if pos in mut_positions:
                            seq_html += f'<span style="background:#27ae60; color:white; padding:1px 3px; border-radius:2px; font-weight:bold" title="{chain_seq[j]}->{aa}">{aa}</span>'
                        elif pos in cdr_pos_set:
                            seq_html += f'<span style="background:#3498db22; border-bottom:2px solid #3498db" title="CDR">{aa}</span>'
                        else:
                            seq_html += aa
                        if pos % 80 == 0:
                            seq_html += f' <span style="color:#bbb; font-size:0.75em">{pos}</span><br>'
                    seq_html += "</div>"
                    st.markdown(seq_html, unsafe_allow_html=True)
                    st.caption("Green = mutation, Blue underline = CDR (protected)")

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

                    # --- Risk Factor Comparison Card ---
                    mhc1_eps_list = None
                    if mhc1_for_chain and hasattr(mhc1_for_chain, 'epitopes'):
                        mhc1_eps_list = mhc1_for_chain.epitopes
                    rc = compute_variant_risk_comparison(
                        original_sequence=chain_seq,
                        variant=var,
                        mhc2_epitopes=ep_results,
                        mhc1_epitopes=mhc1_eps_list,
                        tolerance_result=tol_for_chain,
                        cdr_regions=cdrs_for_chain,
                        sasa_scores=sasa_for_chain,
                        composite_risk=risk_result.composite_score,
                    )

                    def _delta_str(before, after, fmt=".0f", lower_is_better=True, suffix=""):
                        """Format a delta value with color arrow."""
                        delta = after - before
                        if delta == 0:
                            return f'<span style="color:#7f8c8d">— {suffix}</span>'
                        improving = (delta < 0) if lower_is_better else (delta > 0)
                        color = "#27ae60" if improving else "#c0392b"
                        arrow = "↓" if delta < 0 else "↑"
                        return f'<span style="color:{color}; font-weight:600">{arrow}{abs(delta):{fmt}}{suffix}</span>'

                    risk_delta = rc.estimated_risk_before - rc.estimated_risk_after
                    risk_color = "#27ae60" if risk_delta > 0 else "#c0392b" if risk_delta < 0 else "#7f8c8d"
                    verdict = "Improved" if risk_delta > 2 else "Marginal" if risk_delta > 0 else "No change"
                    verdict_color = "#27ae60" if risk_delta > 2 else "#f39c12" if risk_delta > 0 else "#7f8c8d"

                    st.markdown(
                        f'<div style="margin:0.5rem 0 1rem; padding:0.8rem 1rem; border-radius:8px; '
                        f'border:1px solid {verdict_color}30; background:{verdict_color}06;">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem">'
                        f'<span style="font-weight:600; font-size:0.9rem">Risk Comparison</span>'
                        f'<span style="font-weight:700; color:{verdict_color}; font-size:0.9rem">{verdict}'
                        f' ({rc.estimated_risk_before:.0f}% → {rc.estimated_risk_after:.0f}%)</span>'
                        f'</div>'
                        f'<table style="width:100%; font-size:0.82rem; border-collapse:collapse">'
                        f'<tr style="border-bottom:1px solid #eee">'
                        f'<td style="padding:3px 6px; color:#888">Metric</td>'
                        f'<td style="padding:3px 6px; text-align:right; color:#888">Original</td>'
                        f'<td style="padding:3px 6px; text-align:right; color:#888">Variant</td>'
                        f'<td style="padding:3px 6px; text-align:right; color:#888">Delta</td></tr>'
                        f'<tr><td style="padding:3px 6px">MHC-II epitopes</td>'
                        f'<td style="padding:3px 6px; text-align:right">{rc.mhc2_before}</td>'
                        f'<td style="padding:3px 6px; text-align:right">{rc.mhc2_after}</td>'
                        f'<td style="padding:3px 6px; text-align:right">{_delta_str(rc.mhc2_before, rc.mhc2_after)}</td></tr>'
                        + (
                            f'<tr><td style="padding:3px 6px">MHC-I epitopes</td>'
                            f'<td style="padding:3px 6px; text-align:right">{rc.mhc1_before}</td>'
                            f'<td style="padding:3px 6px; text-align:right">{rc.mhc1_after}</td>'
                            f'<td style="padding:3px 6px; text-align:right">{_delta_str(rc.mhc1_before, rc.mhc1_after)}</td></tr>'
                            if rc.mhc1_before > 0 else ""
                        )
                        + f'<tr><td style="padding:3px 6px">Epitope density</td>'
                        f'<td style="padding:3px 6px; text-align:right">{rc.density_before}/100aa</td>'
                        f'<td style="padding:3px 6px; text-align:right">{rc.density_after}/100aa</td>'
                        f'<td style="padding:3px 6px; text-align:right">{_delta_str(rc.density_before, rc.density_after, ".1f")}</td></tr>'
                        f'<tr><td style="padding:3px 6px">Hotspot clusters</td>'
                        f'<td style="padding:3px 6px; text-align:right">{rc.hotspots_before}</td>'
                        f'<td style="padding:3px 6px; text-align:right">{rc.hotspots_after}</td>'
                        f'<td style="padding:3px 6px; text-align:right">{_delta_str(rc.hotspots_before, rc.hotspots_after)}</td></tr>'
                        f'<tr><td style="padding:3px 6px">Tolerance score</td>'
                        f'<td style="padding:3px 6px; text-align:right">{rc.tolerance_before:.2f}</td>'
                        f'<td style="padding:3px 6px; text-align:right">{rc.tolerance_after:.2f}</td>'
                        f'<td style="padding:3px 6px; text-align:right">{_delta_str(rc.tolerance_before, rc.tolerance_after, ".2f", lower_is_better=False)}</td></tr>'
                        f'<tr style="border-top:1px solid #ddd; font-weight:600">'
                        f'<td style="padding:4px 6px">Est. ADA risk</td>'
                        f'<td style="padding:4px 6px; text-align:right">{rc.estimated_risk_before:.0f}%</td>'
                        f'<td style="padding:4px 6px; text-align:right; color:{risk_color}">{rc.estimated_risk_after:.0f}%</td>'
                        f'<td style="padding:4px 6px; text-align:right">{_delta_str(rc.estimated_risk_before, rc.estimated_risk_after, ".0f", suffix="%")}</td></tr>'
                        f'</table>'
                        + (
                            f'<div style="margin-top:0.4rem; font-size:0.78rem">'
                            + "".join(
                                f'<span style="color:#27ae60; margin-right:0.8rem">✓ Treg epitopes preserved</span>'
                                if rc.treg_preserved else
                                f'<span style="color:#c0392b; margin-right:0.8rem">⚠ Treg epitopes affected</span>'
                                for _ in [1]
                            )
                            + "".join(
                                f'<span style="color:#c0392b; margin-right:0.8rem">⚠ {flag}</span>'
                                for flag in rc.structural_flags
                            )
                            + (f'<span style="color:#27ae60">✓ No structural concerns</span>'
                               if not rc.structural_flags and rc.treg_preserved else "")
                            + f'</div>'
                        )
                        + f'</div>',
                        unsafe_allow_html=True,
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

# ══════════════════════════════════════════════════════════════
# TAB: DOWNSELECT — Batch Immunogenicity Comparison
# ══════════════════════════════════════════════════════════════
with tab_ds:
    # Build preloaded sequences dict from DRUG_PRESETS
    preloaded_for_ds = {}
    for name, preset in DRUG_PRESETS.items():
        if preset.get("sequence") and name != "Custom":
            preloaded_for_ds[name] = {
                "seq": preset["sequence"],
                "species": preset.get("species", "Humanized"),
            }
    
    render_downselect_tab(
        st_module=st,
        components_module=components,
        run_immunogenicity_fn=run_immunogenicity_assessment_adapter,
        run_tolerance_fn=run_tolerance_analysis,
        run_cytotoxic_fn=run_cytotoxic_assessment,
        preloaded_sequences=preloaded_for_ds,
        idc_data_path="media-1__Clinical_Trial.csv",
    )
