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
    calculate_sasa_from_pdb,
    is_surface_exposed,
    filter_surface_bcell_epitopes,
    detect_cdr_regions,
    check_cdr_epitope_overlap,
)

from safebind_tolerance_deimmunization import (
    run_tolerance_analysis,
    suggest_deimmunization,
)

from safebind_mhc1_cytotoxic import (
    run_cytotoxic_assessment,
    CytotoxicReport,
    HLA_CLASS_I_ALLELES,
    AAV_VALIDATED_EPITOPES_HUI_2015,
    AAV_IMMUNOPEPTIDOMICS_2023,
)

from safebind_composite_scorer import (
    compute_composite_score,
    CompositeScore,
)

from safebind_advanced_analysis import (
    run_advanced_analysis,
    detect_checkpoint_inhibitor,
    analyze_germline_humanness,
    search_iedb_epitopes,
)

try:
    from api_keys import TAMARIND_API_KEY, ANTHROPIC_API_KEY  # local-only, gitignored
except ImportError:
    TAMARIND_API_KEY = os.environ.get("TAMARIND_API_KEY", "")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── PDB Cache ────────────────────────────────────────────────
PDB_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".pdb_cache")
os.makedirs(PDB_CACHE_DIR, exist_ok=True)

SAVED_SEQUENCES_FILE = os.path.join(os.path.dirname(__file__), "saved_sequences.json")


def get_cached_pdb(seq_hash: str) -> str:
    """Get PDB from local cache if it exists."""
    cache_file = os.path.join(PDB_CACHE_DIR, f"{seq_hash}.pdb")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return f.read()
    return None


def save_pdb_to_cache(seq_hash: str, pdb_data: str):
    """Save PDB to local cache."""
    cache_file = os.path.join(PDB_CACHE_DIR, f"{seq_hash}.pdb")
    with open(cache_file, "w") as f:
        f.write(pdb_data)


def get_sequence_hash(sequence: str) -> str:
    """Get a hash of the sequence for caching."""
    import hashlib
    return hashlib.md5(sequence.encode()).hexdigest()[:16]


def load_saved_sequences() -> dict:
    """Load saved sequences from JSON file."""
    if os.path.exists(SAVED_SEQUENCES_FILE):
        try:
            with open(SAVED_SEQUENCES_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_sequence(name: str, sequence: str, species: str, notes: str = ""):
    """Save a sequence to the JSON file."""
    saved = load_saved_sequences()
    saved[name] = {
        "sequence": sequence,
        "species": species,
        "notes": notes,
        "saved_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(SAVED_SEQUENCES_FILE, "w") as f:
        json.dump(saved, f, indent=2)


def delete_saved_sequence(name: str):
    """Delete a saved sequence."""
    saved = load_saved_sequences()
    if name in saved:
        del saved[name]
        with open(SAVED_SEQUENCES_FILE, "w") as f:
            json.dump(saved, f, indent=2)

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

    /* Sidebar - ALWAYS visible, cannot collapse */
    [data-testid="stSidebar"] { 
        background-color: #f9fafb; 
        min-width: 320px !important;
        width: 320px !important;
        transform: translateX(0) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 320px !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        transform: translateX(0) !important;
        min-width: 320px !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextArea label { color: #374151; }
    
    /* Hide the collapse button since sidebar is always visible */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Adjust main content to account for permanent sidebar */
    .main .block-container {
        max-width: calc(100% - 340px) !important;
        margin-left: 320px !important;
    }

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


# ── Preloaded sequences with default clinical contexts ───────
PRELOADED = {
    "Bococizumab (Pfizer — TERMINATED, 44% ADA)": {
        "seq": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGEISPFGGRTNYNEKFKSRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARERPLYASDLWGQGTTVTVSS",
        "species": "Humanized",
        "pdb": None,
        "context": {"modality": "Monoclonal antibody (mAb)", "route": "SC (subcutaneous)", "indication": "Metabolic", "backbone": "IgG2"},
    },
    "Adalimumab / Humira (FDA: 5% ADA, 1% with MTX)": {
        "seq": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
        "species": "Human",
        "pdb": None,
        "context": {"modality": "Monoclonal antibody (mAb)", "route": "SC (subcutaneous)", "indication": "Autoimmune/Inflammation", "backbone": "IgG1"},
    },
    "Trastuzumab / Herceptin (0-14% ADA)": {
        "seq": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        "species": "Humanized",
        "pdb": "1N8Z",
        "context": {"modality": "Monoclonal antibody (mAb)", "route": "IV (intravenous)", "indication": "Oncology", "backbone": "IgG1"},
    },
    "Nivolumab / Opdivo (11-26% ADA)": {
        "seq": "QVQLVESGGGVVQPGRSLRLDCKASGITFSNSGMHWVRQAPGKGLEWVAVIWYDGSKRYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCATNDDYWGQGTLVTVSS",
        "species": "Human",
        "pdb": None,
        "context": {"modality": "Monoclonal antibody (mAb)", "route": "IV (intravenous)", "indication": "Oncology", "backbone": "IgG4"},
    },
    "── Gene Therapy (AAV) ──": None,
    "AAV9 VP1 (Zolgensma, SGT-001)": {
        "seq": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDQQLKAGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRLLEPLGLVEEAAKTAPGKKRPVEQSPQEPDSSAGIGKSGAQPAKKRLNFGQTGDTESVPDPQPIGEPPAAPSGVGSLTMASGGGAPVADNNEGADGVGSSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL",
        "species": "Viral",
        "pdb": None,
        "context": {"modality": "AAV gene therapy", "route": "IV (intravenous)", "indication": "Neuromuscular (DMD, SMA)", "serotype": "AAV9"},
    },
    "AAV8 VP1 (Hemgenix)": {
        "seq": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNAADAAALEHDKAYDQQLQAGDNPYLRYNHADAEFQERLQEDTSFGGNLGRAVFQAKKRVLEPLGLVEEGAKTAPGKKRPVEPSPQRSPDSSTGIGKKGQQPARKRLNFGQTGDSESVPDPQPLGEPPATPAAVGPTTMASGGGAPMADNNEGADGVGNASGNWHCDSTWLGDRVITTSTRTWALPTYNNHLYKQISSASTGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLSFKLFNIQVKEVTTNDGVTTIANNLTSTVQVFSDSEYQLPYVLGSAHQGCLPPFPADVFMIPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEEVPFHSSYAHSQSLDRLMNPLIDQYLYYLNRTQNQSGSAQNKDLLFSRGSPAGMSVQPKNWLPGPCYRQQRVSKTKTDNNNSNFTWTGASKYNLNGRESIINPGTAMASHKDDKDKFFPMSGVMIFGKESAGASNTALDNVMITDEEEIKATNPVATERFGTVAVNFQSSSTDPATGDVHAMGALPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKNPPPQILIKNTPVPANPPAEFSATKFASFITQYSTGQVSVEIEWELQKENSKRWNPEVQYTSNYAKSANVDFTVDNNGLYTEPRPIGTRYLTRPL",
        "species": "Viral",
        "pdb": None,
        "context": {"modality": "AAV gene therapy", "route": "IV (intravenous)", "indication": "Hemophilia", "serotype": "AAV8"},
    },
    "AT132 / AAV8 (XLMTM, 4 deaths)": {
        "seq": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNAADAAALEHDKAYDQQLQAGDNPYLRYNHADAEFQERLQEDTSFGGNLGRAVFQAKKRVLEPLGLVEEGAKTAPGKKRPVEPSPQRSPDSSTGIGKKGQQPARKRLNFGQTGDSESVPDPQPLGEPPATPAAVGPTTMASGGGAPMADNNEGADGVGNASGNWHCDSTWLGDRVITTSTRTWALPTYNNHLYKQISSASTGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLSFKLFNIQVKEVTTNDGVTTIANNLTSTVQVFSDSEYQLPYVLGSAHQGCLPPFPADVFMIPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEEVPFHSSYAHSQSLDRLMNPLIDQYLYYLNRTQNQSGSAQNKDLLFSRGSPAGMSVQPKNWLPGPCYRQQRVSKTKTDNNNSNFTWTGASKYNLNGRESIINPGTAMASHKDDKDKFFPMSGVMIFGKESAGASNTALDNVMITDEEEIKATNPVATERFGTVAVNFQSSSTDPATGDVHAMGALPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKNPPPQILIKNTPVPANPPAEFSATKFASFITQYSTGQVSVEIEWELQKENSKRWNPEVQYTSNYAKSANVDFTVDNNGLYTEPRPIGTRYLTRPL",
        "species": "Viral",
        "pdb": None,
        "context": {"modality": "AAV gene therapy", "route": "IV (intravenous)", "indication": "Neuromuscular (DMD, SMA)", "serotype": "AAV8"},
    },
    "AAV5 VP1 (Luxturna, Roctavian)": {
        "seq": "MSFVDHPPDWLEEVGEGLREFLGLEAGPPKPKPNQQHQDQARGLVLPGYNYLGPGNGLDRGEPVNRADEVAREHDISYNEQLEAGDNPYLKYNHADAEFQEKLADDTSFGGNLGKAVFQAKKRVLEPFGLVEEGAKTAPTGKRIDDHFPKRKKARTEEDSKPSTSSDAEAGPSGSQQLQIPAQPASSLGADTMSAGGGGPLGDNNQGADGVGNASGDWHCDSTWSEGHVTTTSTRTWVLPTYNNHLYKRLGESLQSNTYNKFSTPWGYFDFNRFHCHFSPRDWQRLINNNWGMRPKAMRVKIFNIQVKEVTTSNGETTVANNLTSTVQIFADSSYELPYVMDAGQEGSLPPFPNDVFMVPQYGYCGLVTGNTSQQQTDRNAFYCLEYFPSQMLRTGNNFEITYSFEKVPFHSMYAHSQSLDRLMNPLIDQYLWGLQSTTTGTTLNAGTATTNFTKLRPTNFSNFKKNWLPGPSIKQQGFSKTANQNYKIPATGSDSLIKYETHSTLDGRWSALTPGPPMATAGPADSKFSNSQLIFAGPKQNGNTATVPGTLIFTSEEELAATNATDTDMWGNLPGGDQSNSNLPTVDRLTALGAVPGMVWQNRDIYYQGPIWAKIPHTDGHFHPSPLIGGFGLKHPPPQIFIKNTPVPANPATTFSSTPVNSFITQYSTGQVSVQIDWEIQKERSKRWNPEVQFTSNYGQQNSLLWAPDAAGKYTEPRAIGTRYLTHHL",
        "species": "Viral",
        "pdb": None,
        "context": {"modality": "AAV gene therapy", "route": "Subretinal", "indication": "Retinal/Ocular", "serotype": "AAV5"},
    },
    "AAVrh74 VP1 (delandistrogene, SRP-9001)": {
        "seq": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNAADAAALEHDKAYDQQLKAGDNPYLRYNHADAEFQERLQEDTSFGGNLGRAVFQAKKRVLEPLGLVEEGAKTAPAKKRPVEPSPQRSPDSSTGIGKKGQQPAKKRLNFGQTGDSESVPDPQPIGEPPAGPSGLGSGTMAAGGGAPMADNNEGADGVGNASGNWHCDSTWLGDRVITTSTRTWALPTYNNHLYKQISNGTSGGSTNDNTYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNEGTKTIANNLTSTIQVFTDSEYQLPYVLGSAHEGCLPPFPADVFMIPQYGYLTLNNGSQALGRSSFYCLEYFPSQMLRTGNNFQFTYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTQSTGGTAGTQQLLFSQAGPANMSVQPKNWLPGPCYRQQRVSTTLSQNNNSNFAWTGATKYHLNGRDSLVNPGVAMATHKDDEERFFPSSGVLIFGKTGATNKTTLENVLMTNEEEIRPTNPVATEQYGVVADNLQQQNAAPIVGAVNSQGALPGMVWQNRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPADPPTTFNQAKLASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYYKSTNVDFAVNTEGTYSEPRPIGTRYLTRN",
        "species": "Viral",
        "pdb": None,
        "context": {"modality": "AAV gene therapy", "route": "IV (intravenous)", "indication": "Neuromuscular (DMD, SMA)", "serotype": "AAVrh74"},
    },
    "AAV2 VP1 (high seroprevalence ~70%)": {
        "seq": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDQQLKAGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL",
        "species": "Viral",
        "pdb": None,
        "context": {"modality": "AAV gene therapy", "route": "IV (intravenous)", "indication": "Other", "serotype": "AAV2"},
    },
    "── Bispecific ──": None,
    "Blinatumomab / Blincyto (BiTE, 1-2% ADA)": {
        "seq": "DIQLTQSPASLAVSLGQRATISCKASQSVDYDGDSYLNWYQQIPGQPPKLLIYDASNLVSGIPPRFSGSGSGTDFTLNIHPVEKVDAATYHCQQSTEDPWTFGGGTKLEIKGGGGSGGGGSGGGGSQVQL",
        "species": "Mouse",
        "pdb": None,
        "context": {"modality": "Bispecific antibody", "route": "IV (intravenous)", "indication": "Hematologic malignancy (ALL, DLBCL)", "backbone": "scFv/Fab"},
    },
    "Custom sequence": {
        "seq": "",
        "species": "Humanized",
        "pdb": None,
        "context": {},
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


# ── Update report with structure-aware metrics ────────────────
def _update_report_with_structure(report, pdb_data: str, chain: str = "A"):
    """Recalculate structure-aware metrics when PDB is obtained (e.g., from Tamarind).
    
    This updates the report in-place with:
    - SASA values for each residue
    - Surface-exposed B-cell epitopes
    - CDR/epitope overlap warnings
    """
    if not pdb_data:
        return
    
    # Calculate SASA
    sasa_scores = calculate_sasa_from_pdb(pdb_data, chain=chain)
    
    if not sasa_scores:
        return
    
    # Update residue risks with SASA
    for rr in report.residue_risks:
        if rr.position in sasa_scores:
            rr.sasa = sasa_scores[rr.position]
            rr.is_surface = is_surface_exposed(rr.sasa)
    
    # Filter surface-exposed B-cell epitopes
    surface_epitopes = filter_surface_bcell_epitopes(
        report.b_cell_epitopes, sasa_scores, threshold=25.0
    )
    report.surface_b_cell_epitopes = surface_epitopes if surface_epitopes else None
    
    # Detect CDRs (if not already done)
    if not report.cdr_regions:
        cdrs = detect_cdr_regions(report.sequence, chain_type="heavy")
        report.cdr_regions = cdrs if cdrs else None
    
    # Recalculate CDR/epitope overlaps
    if report.cdr_regions:
        overlaps = []
        for hs in report.hotspot_regions:
            overlap = check_cdr_epitope_overlap(report.cdr_regions, hs["start"], hs["end"])
            if overlap:
                overlaps.append({
                    "hotspot_start": hs["start"],
                    "hotspot_end": hs["end"],
                    "hotspot_sequence": hs["sequence"],
                    **overlap
                })
        for be in report.b_cell_epitopes:
            overlap = check_cdr_epitope_overlap(report.cdr_regions, be.start, be.end)
            if overlap:
                overlaps.append({
                    "epitope_type": "B-cell",
                    "epitope_start": be.start,
                    "epitope_end": be.end,
                    "epitope_sequence": be.sequence,
                    **overlap
                })
        report.cdr_epitope_overlaps = overlaps if overlaps else None
        
        # Update residue risks with CDR info
        for rr in report.residue_risks:
            for cdr in report.cdr_regions:
                if cdr["start"] <= rr.position <= cdr["end"]:
                    rr.in_cdr = True
                    rr.cdr_label = cdr["label"]
                    break
    
    # Store PDB data
    report.pdb_data = pdb_data


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:24px;">🛡️ SafeBind AI</div>', unsafe_allow_html=True)
    st.markdown("**Immunogenicity Risk Assessment**")
    st.markdown("---")

    # Build options list with saved sequences
    saved_seqs = load_saved_sequences()
    options = list(PRELOADED.keys())
    if saved_seqs:
        options = ["── Saved Sequences ──"] + [f"💾 {name}" for name in saved_seqs.keys()] + options
    
    # Sequence selection
    selected = st.selectbox("Select therapeutic", options)

    if selected.startswith("──") or PRELOADED.get(selected) is None:
        if selected.startswith("💾 "):
            # Load saved sequence
            saved_name = selected[3:]  # Remove "💾 " prefix
            saved_data = saved_seqs[saved_name]
            seq_input = saved_data["sequence"]
            species = saved_data.get("species", "Humanized")
            pdb_id = ""
            seq_name = saved_name
            st.code(seq_input[:60] + "..." if len(seq_input) > 60 else seq_input, language=None)
            st.caption(f"Species: {species} | Length: {len(seq_input)} aa | Saved: {saved_data.get('saved_at', 'N/A')}")
            if saved_data.get("notes"):
                st.caption(f"Notes: {saved_data['notes']}")
            # Delete button
            if st.button("🗑️ Delete saved sequence", key="delete_saved"):
                delete_saved_sequence(saved_name)
                st.rerun()
        else:
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
        
        # Save sequence option
        if seq_input and seq_name:
            with st.expander("💾 Save this sequence"):
                save_notes = st.text_input("Notes (optional)", placeholder="My project notes...")
                if st.button("Save sequence"):
                    save_sequence(seq_name, seq_input, species, save_notes)
                    st.success(f"Saved '{seq_name}'!")
                    st.rerun()
    else:
        info = PRELOADED[selected]
        seq_input = info["seq"]
        species = info["species"]
        pdb_id = info.get("pdb") or ""
        seq_name = selected.split("(")[0].strip().split("/")[0].strip()
        default_context = info.get("context", {})
        st.code(seq_input[:60] + "...", language=None)
        st.caption(f"Species: {species} | Length: {len(seq_input)} aa")

    # Get default context from preloaded therapeutic (if any)
    if 'default_context' not in dir():
        default_context = {}

    st.markdown("---")
    
    # Clinical context (affects risk interpretation)
    with st.expander("⚙️ Clinical Context (optional)", expanded=bool(default_context)):
        # Modality selection first - this affects other options
        modality_options = [
            "Monoclonal antibody (mAb)",
            "Bispecific antibody",
            "AAV gene therapy",
            "Enzyme replacement",
            "Fc-fusion protein",
            "ADC (antibody-drug conjugate)",
            "CAR-T / cell therapy",
            "Other/Unknown"
        ]
        default_modality = default_context.get("modality", "Monoclonal antibody (mAb)")
        modality_idx = modality_options.index(default_modality) if default_modality in modality_options else 0
        modality = st.selectbox("Modality", modality_options, index=modality_idx)
        
        # Route options vary by modality
        if modality == "AAV gene therapy":
            route_options = ["IV (intravenous)", "Intrathecal", "Intramuscular", "Subretinal", "Intravitreal", "Other"]
        elif modality == "Enzyme replacement":
            route_options = ["IV (intravenous)", "SC (subcutaneous)", "Intrathecal", "Other"]
        else:
            route_options = ["IV (intravenous)", "SC (subcutaneous)", "Intramuscular", "Other"]
        default_route = default_context.get("route", "IV (intravenous)")
        route_idx = route_options.index(default_route) if default_route in route_options else 0
        route = st.selectbox("Route of administration", route_options, index=route_idx)
        
        # Indication options vary by modality
        if modality == "AAV gene therapy":
            indication_options = ["Neuromuscular (DMD, SMA)", "Hemophilia", "Retinal/Ocular", "CNS/Neurological", "Metabolic/Lysosomal", "Oncology", "Other"]
        elif modality == "Enzyme replacement":
            indication_options = ["Lysosomal storage (Fabry, Gaucher, Pompe)", "Metabolic (PKU, Homocystinuria)", "Other"]
        elif modality == "CAR-T / cell therapy":
            indication_options = ["Hematologic malignancy (ALL, DLBCL)", "Solid tumor", "Autoimmune", "Other"]
        else:
            indication_options = ["Oncology", "Autoimmune/Inflammation", "Healthy volunteers", "Metabolic", "Infectious disease", "Other"]
        default_indication = default_context.get("indication", indication_options[0])
        indication_idx = indication_options.index(default_indication) if default_indication in indication_options else 0
        indication = st.selectbox("Disease indication", indication_options, index=indication_idx)
        
        # Backbone only for antibodies
        if modality in ["Monoclonal antibody (mAb)", "Bispecific antibody", "ADC (antibody-drug conjugate)", "Fc-fusion protein"]:
            backbone_options = ["IgG4", "IgG1", "IgG2", "scFv/Fab", "Unknown"]
            default_backbone = default_context.get("backbone", "IgG1")
            backbone_idx = backbone_options.index(default_backbone) if default_backbone in backbone_options else 0
            backbone = st.selectbox("Antibody backbone", backbone_options, index=backbone_idx)
        else:
            backbone = "N/A"
        
        # AAV-specific options
        if modality == "AAV gene therapy":
            serotype_options = ["AAV9", "AAV8", "AAVrh74", "AAV5", "AAV2", "AAVrh10", "Other/Novel"]
            default_serotype = default_context.get("serotype", "AAV9")
            serotype_idx = serotype_options.index(default_serotype) if default_serotype in serotype_options else 0
            serotype = st.selectbox("AAV serotype", serotype_options, index=serotype_idx)
            prior_aav_exposure = st.checkbox("Pre-existing AAV antibodies expected (high seroprevalence)", 
                                             value=default_context.get("prior_aav_exposure", False))
        else:
            serotype = None
            prior_aav_exposure = False
        
        # CRIM status for enzyme replacement
        if modality == "Enzyme replacement":
            crim_options = ["CRIM-positive (residual enzyme)", "CRIM-negative (no enzyme)", "Unknown"]
            default_crim = default_context.get("crim_status", "Unknown")
            crim_idx = crim_options.index(default_crim) if default_crim in crim_options else 0
            crim_status = st.selectbox("CRIM status", crim_options, index=crim_idx)
        else:
            crim_status = None
        
        immunosuppressants = st.checkbox("Patient on immunosuppressants/immunomodulation",
                                         value=default_context.get("immunosuppressants", False))
    
    # Store clinical context in session state
    st.session_state["clinical_context"] = {
        "modality": modality,
        "route": route, 
        "indication": indication, 
        "backbone": backbone, 
        "immunosuppressants": immunosuppressants,
        "serotype": serotype,
        "prior_aav_exposure": prior_aav_exposure,
        "crim_status": crim_status,
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
    """Apply clinical context adjustments based on IDC DB V1 findings and literature.
    Returns (adjusted_risk, adjustment_factors).
    """
    factors = []
    multiplier = 1.0
    
    modality = context.get("modality", "")
    
    # ── Modality-specific baseline adjustments ──
    if "AAV" in modality:
        multiplier *= 1.8
        factors.append("AAV gene therapy: +80% (high pre-existing immunity)")
    elif "Enzyme replacement" in modality:
        multiplier *= 1.6
        factors.append("Enzyme replacement: +60% (high ADA historically)")
    elif "Bispecific" in modality:
        multiplier *= 1.2
        factors.append("Bispecific: +20% (novel architecture)")
    elif "CAR-T" in modality:
        multiplier *= 0.7
        factors.append("CAR-T: -30% (lymphodepletion pre-conditioning)")
    
    # ── Route adjustments ──
    route = context.get("route", "")
    if "IV" in route:
        multiplier *= 0.6
        factors.append("IV route: -40% (lower ADA vs SC)")
    elif "SC" in route:
        multiplier *= 1.3
        factors.append("SC route: +30% (higher ADA than IV)")
    elif "Intrathecal" in route:
        multiplier *= 0.5
        factors.append("Intrathecal: -50% (immune-privileged CNS)")
    elif "Subretinal" in route or "Intravitreal" in route:
        multiplier *= 0.4
        factors.append("Ocular: -60% (immune-privileged eye)")
    
    # ── Indication adjustments ──
    indication = context.get("indication", "")
    if indication == "Oncology" or "malignancy" in indication.lower():
        multiplier *= 0.5
        factors.append("Oncology: -50% (immunosuppressed patients)")
    elif "Autoimmune" in indication:
        multiplier *= 1.3
        factors.append("Autoimmune: +30% (hyperactive immune system)")
    elif "Lysosomal" in indication or "Fabry" in indication or "Pompe" in indication:
        multiplier *= 1.8
        factors.append("Lysosomal storage: +80% (very high ADA, e.g. Fabrazyme 67%)")
    elif "Neuromuscular" in indication or "DMD" in indication or "SMA" in indication:
        multiplier *= 1.4
        factors.append("Neuromuscular: +40% (nAb concern for retreatment)")
    elif "Hemophilia" in indication:
        multiplier *= 1.5
        factors.append("Hemophilia: +50% (factor inhibitor history)")
    elif "Healthy" in indication:
        multiplier *= 1.2
        factors.append("Healthy volunteers: +20% (intact immune system)")
    
    # ── Backbone adjustments (antibodies only) ──
    backbone = context.get("backbone", "")
    if backbone == "IgG4":
        multiplier *= 0.7
        factors.append("IgG4 backbone: -30% (less immunostimulatory)")
    elif backbone == "IgG1":
        pass  # baseline, no adjustment
    elif backbone == "scFv/Fab":
        multiplier *= 1.4
        factors.append("scFv/Fab: +40% (foreign scaffolds, e.g. brolucizumab)")
    
    # ── AAV-specific adjustments ──
    serotype = context.get("serotype", "")
    if serotype:
        if serotype in ["AAV9", "AAV8", "AAVrh74"]:
            multiplier *= 1.3
            factors.append(f"{serotype}: +30% (high seroprevalence)")
        elif serotype == "AAV5":
            multiplier *= 0.8
            factors.append("AAV5: -20% (lower seroprevalence)")
        elif serotype == "AAV2":
            multiplier *= 1.5
            factors.append("AAV2: +50% (highest seroprevalence, ~70%)")
    
    if context.get("prior_aav_exposure"):
        multiplier *= 2.0
        factors.append("Pre-existing AAV Ab: +100% (may neutralize transduction)")
    
    # ── CRIM status for enzyme replacement ──
    crim_status = context.get("crim_status") or ""
    if crim_status and "CRIM-negative" in crim_status:
        multiplier *= 2.5
        factors.append("CRIM-negative: +150% (near-universal ADA, e.g. Pompe)")
    elif crim_status and "CRIM-positive" in crim_status:
        multiplier *= 0.7
        factors.append("CRIM-positive: -30% (immune tolerance to endogenous protein)")
    
    # ── Immunosuppressants/immunomodulation ──
    if context.get("immunosuppressants"):
        multiplier *= 0.4
        factors.append("Immunomodulation: -60% (ITI, rituximab, MTX, etc.)")
    
    # Apply multiplier with soft cap to avoid alarming 100% display
    # Use sigmoid-like compression for values approaching/exceeding 1.0
    adjusted = raw_risk * multiplier
    if adjusted > 0.85:
        # Soft cap: compress values above 85% so they approach but rarely hit 100%
        # Maps 0.85->0.85, 1.0->0.90, 1.5->0.95, 2.0->0.97
        excess = adjusted - 0.85
        adjusted = 0.85 + 0.15 * (1 - 1/(1 + excess * 2))
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
            st.markdown(f'<div class="metric-card"><div class="metric-value {risk_class}">{adjusted_risk:.0%}</div><div class="metric-label">Epitope Density</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card"><div class="metric-value {risk_class}">{report.overall_risk_score:.0%}</div><div class="metric-label">Epitope Density</div></div>', unsafe_allow_html=True)
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
        with st.expander("Risk adjustment factors applied", expanded=True):
            st.markdown(f"**Raw sequence risk:** {report.overall_risk_score:.0%} → **Context-adjusted:** {adjusted_risk:.0%}")
            for f in adj_factors:
                st.markdown(f"- {f}")
    
    # Important disclaimer about sequence risk vs clinical ADA
    st.caption("⚠️ **Note:** Sequence risk score reflects predicted MHC binding density (% of residues in immunogenic regions), "
               "not clinical ADA incidence. Clinical ADA rates depend heavily on formulation, dosing, co-medications, and patient factors. "
               "For example, Humira shows 76% sequence risk but only 5% clinical ADA (1% with MTX).")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab2b, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "3D Heatmap", "T-cell Hotspots", "B-cell Epitopes", "Residue Plot", 
        "Clinical Context", "AI Report", "Tolerance", "Deimmunize",
        "Cytotoxic (MHC-I)", "Composite Score", "Advanced"
    ])

    # ── Tab 1: 3D Heatmap ──
    with tab1:
        # Check for cached PDB first
        seq_hash = get_sequence_hash(seq_clean)
        cached_pdb = get_cached_pdb(seq_hash)
        tamarind_pdb = st.session_state.get(f"tamarind_pdb_{seq_name}")
        active_pdb = cached_pdb or tamarind_pdb or report.pdb_data
        job_key = f"tamarind_job_{seq_name}"
        existing_job = st.session_state.get(job_key)

        if active_pdb:
            if cached_pdb:
                source = "Local cache (instant load)"
            elif tamarind_pdb:
                source = "Tamarind ESMFold/AlphaFold2"
            else:
                source = f"RCSB PDB ({pdb_id})"
            st.caption(f"Structure source: {source}")
            html = generate_3d_heatmap_html(active_pdb, report.residue_risks, chain="A",
                                            title=f"{seq_name} Immunogenicity Heatmap")
            components.html(html, height=620, scrolling=False)
            
            # Update structure-aware metrics if not already done
            if not report.surface_b_cell_epitopes and active_pdb:
                _update_report_with_structure(report, active_pdb)
                st.session_state["report"] = report
        elif existing_job:
            # ── Polling loop with progress bar ──
            status = get_tamarind_job_status(TAMARIND_API_KEY, existing_job)
            
            if status == "complete":
                with st.spinner("Downloading structure..."):
                    pdb_result = fetch_tamarind_pdb(TAMARIND_API_KEY, existing_job)
                if pdb_result:
                    # Save to cache for instant loading next time
                    save_pdb_to_cache(seq_hash, pdb_result)
                    st.session_state[f"tamarind_pdb_{seq_name}"] = pdb_result
                    del st.session_state[job_key]
                    if f"tamarind_start_{seq_name}" in st.session_state:
                        del st.session_state[f"tamarind_start_{seq_name}"]
                    
                    # Recalculate structure-aware metrics now that we have PDB
                    _update_report_with_structure(report, pdb_result)
                    st.session_state["report"] = report
                    
                    st.rerun()
                else:
                    st.warning("Job complete but PDB not found — try entering the PDB ID manually in the sidebar.")
                    del st.session_state[job_key]
            elif status == "failed":
                st.error("Tamarind job failed. Try again or enter a PDB ID in the sidebar.")
                del st.session_state[job_key]
                if f"tamarind_start_{seq_name}" in st.session_state:
                    del st.session_state[f"tamarind_start_{seq_name}"]
            else:
                # Show progress UI
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 24px; color: white; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                        <div style="font-size: 32px; color: #7c3aed;">◆</div>
                        <div>
                            <div style="font-size: 18px; font-weight: 600;">Folding protein structure with ESMFold</div>
                            <div style="font-size: 13px; opacity: 0.9;">Powered by Tamarind Bio</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Track elapsed time
                start_key = f"tamarind_start_{seq_name}"
                if start_key not in st.session_state:
                    st.session_state[start_key] = time.time()
                
                elapsed = time.time() - st.session_state[start_key]
                estimated_total = 90  # ESMFold typically takes 60-120 seconds
                progress = min(elapsed / estimated_total, 0.95)  # Cap at 95% until complete
                
                # Progress bar
                st.progress(progress, text=f"Processing... ({int(elapsed)}s elapsed)")
                
                # Status details
                status_display = status.replace("_", " ").title() if status else "Queued"
                remaining = max(0, estimated_total - elapsed)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", status_display)
                with col2:
                    st.metric("Elapsed", f"{int(elapsed)}s")
                with col3:
                    if remaining > 0:
                        st.metric("Est. Remaining", f"~{int(remaining)}s")
                    else:
                        st.metric("Est. Remaining", "Almost done...")
                
                st.caption("The page will automatically update when folding completes.")
                
                time.sleep(5)
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
            if st.button("Predict 3D Structure with Tamarind",
                         help="Submits to ESMFold / AlphaFold2 on Tamarind Bio (uses 1 job)",
                         use_container_width=True):
                safe_name = "".join(c if c.isalnum() else "_" for c in (seq_name or "query"))[:30]
                job_name = f"safebind_{safe_name}_{int(time.time())}"
                with st.spinner("Submitting to Tamarind Bio…"):
                    submitted = submit_tamarind_structure(TAMARIND_API_KEY, seq_clean, job_name)
                if submitted:
                    st.session_state[job_key] = submitted
                    st.session_state[f"tamarind_start_{seq_name}"] = time.time()  # Track start time
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
                    <div style="display:flex;gap:24px;margin-top:8px;font-size:12px;">
                        <span style="color:#6b7280;">T-cell:</span> <span style="color:#ea580c;font-weight:600;">{hs['avg_t_cell']:.0%}</span>
                        <span style="color:#6b7280;margin-left:8px;">B-cell:</span> <span style="color:#0891b2;font-weight:600;">{hs['avg_b_cell']:.0%}</span>
                        <span style="color:#6b7280;margin-left:8px;">Max:</span> <span style="color:#dc2626;font-weight:600;">{hs['max_risk']:.0%}</span>
                    </div></div>""", unsafe_allow_html=True)
        else:
            st.success("No significant T-cell hotspot regions detected above threshold.")

    # ── Tab 2b: B-cell epitopes ──
    with tab2b:
        # Show CDR regions if detected (for antibodies)
        if report.cdr_regions:
            st.markdown("**Detected CDR Regions** (binding site)")
            cdr_cols = st.columns(len(report.cdr_regions))
            for idx, cdr in enumerate(report.cdr_regions):
                with cdr_cols[idx]:
                    st.markdown(f"""<div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;padding:10px;text-align:center;">
                        <div style="font-weight:600;color:#92400e;">{cdr['label']}</div>
                        <div style="font-size:11px;color:#78350f;">Pos {cdr['start']}–{cdr['end']}</div>
                        <div style="font-family:monospace;font-size:10px;color:#451a03;margin-top:4px;">{cdr['sequence'][:15]}{'...' if len(cdr['sequence']) > 15 else ''}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Show nADA risk warnings for CDR/epitope overlaps
        if report.cdr_epitope_overlaps:
            st.markdown("**⚠️ Neutralizing ADA (nADA) Risk Flags**")
            st.caption("Epitopes overlapping CDR regions may generate neutralizing antibodies that block drug activity.")
            for overlap in report.cdr_epitope_overlaps:
                risk_color = "#dc2626" if overlap['nada_risk'] == "HIGH" else "#ea580c"
                st.markdown(f"""<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-weight:600;color:#991b1b;">⚠️ {overlap['cdr']} Overlap</span>
                            <span style="color:#7f1d1d;font-size:12px;margin-left:8px;">
                                Positions {overlap['overlap_start']}–{overlap['overlap_end']} ({overlap['overlap_length']} residues)
                            </span>
                        </div>
                        <div style="font-weight:600;color:{risk_color};">{overlap['nada_risk']} nADA Risk</div>
                    </div>
                    <div style="font-size:12px;color:#991b1b;margin-top:6px;">
                        Epitope in binding site → antibodies may neutralize drug efficacy
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Surface-exposed B-cell epitopes (if structure available)
        if report.surface_b_cell_epitopes:
            st.markdown("**🔬 Surface-Exposed B-cell Epitopes** (SASA-filtered)")
            st.caption(f"Filtered from {len(report.b_cell_epitopes)} total → {len(report.surface_b_cell_epitopes)} surface-accessible (antibody-reachable)")
            for i, be in enumerate(report.surface_b_cell_epitopes, 1):
                bar_color = "#0891b2"
                sasa_info = f" · SASA: {be.avg_sasa:.0f}Å²" if be.avg_sasa else ""
                html = f'<div class="hotspot-row" style="border-left:3px solid #0891b2;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><span style="font-weight:600;font-size:15px;color:#111827;">Surface Epitope {i}</span><span style="color:#6b7280;font-size:13px;margin-left:8px;">Positions {be.start}–{be.end}{sasa_info}</span></div><div style="font-weight:600;font-size:18px;color:{bar_color};">{be.avg_score:.0%}</div></div><div class="hotspot-seq" style="margin-top:8px;">{be.sequence}</div></div>'
                st.markdown(html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # All linear B-cell epitopes
        st.markdown("**📋 All Linear B-cell Epitopes** (IEDB Bepipred)")
        if not report.surface_b_cell_epitopes:
            st.caption("Note: Surface accessibility filtering requires a 3D structure. Use the Tamarind fold button or enter a PDB ID.")
        
        if report.b_cell_epitopes:
            for i, be in enumerate(report.b_cell_epitopes, 1):
                score_pct = be.avg_score
                bar_color = "#0891b2" if score_pct > 0.7 else "#06b6d4" if score_pct > 0.5 else "#22d3ee"
                surface_badge = ""
                if report.surface_b_cell_epitopes:
                    is_surface = any(sbe.start == be.start and sbe.end == be.end for sbe in report.surface_b_cell_epitopes)
                    surface_badge = '<span style="background:#dcfce7;color:#166534;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">SURFACE</span>' if is_surface else '<span style="background:#f3f4f6;color:#6b7280;font-size:10px;padding:2px 6px;border-radius:4px;margin-left:8px;">BURIED</span>'
                html = f'<div class="hotspot-row"><div style="display:flex;justify-content:space-between;align-items:center;"><div><span style="font-weight:600;font-size:15px;color:#111827;">B-cell Epitope {i}</span><span style="color:#6b7280;font-size:13px;margin-left:8px;">Positions {be.start}–{be.end}</span>{surface_badge}</div><div style="font-weight:600;font-size:18px;color:{bar_color};">{be.avg_score:.0%}</div></div><div class="hotspot-seq" style="margin-top:8px;">{be.sequence}</div></div>'
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.success("No significant B-cell epitope regions detected above threshold.")
        
        # Add Tamarind fold button at the bottom of B-cell tab if no structure
        if not report.pdb_data and not report.surface_b_cell_epitopes:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Get Surface Accessibility Analysis**")
            st.caption("Fold the protein with Tamarind to filter for surface-exposed (antibody-accessible) epitopes.")
            if st.button("Predict 3D Structure with Tamarind", key="tamarind_bcell_tab", use_container_width=True):
                safe_name = "".join(c if c.isalnum() else "_" for c in (seq_name or "query"))[:30]
                job_name = f"safebind_{safe_name}_{int(time.time())}"
                job_key = f"tamarind_job_{seq_name}"
                with st.spinner("Submitting to Tamarind Bio…"):
                    submitted = submit_tamarind_structure(TAMARIND_API_KEY, seq_clean, job_name)
                if submitted:
                    st.session_state[job_key] = submitted
                    st.rerun()
                else:
                    st.error("Submission failed — check the Tamarind API key or network.")

    # ── Tab 3: Residue plot ──
    with tab3:
        import pandas as pd
        # Build enhanced dataframe with SASA and CDR info
        rows = []
        for rr in report.residue_risks:
            row = {
                "Position": rr.position, "Residue": rr.residue,
                "T-cell Risk": rr.t_cell_risk, "B-cell Risk": rr.b_cell_risk,
                "Combined Risk": rr.combined_risk, "Alleles Binding": rr.num_alleles_binding,
            }
            if rr.sasa is not None:
                row["SASA"] = rr.sasa
                row["Surface"] = "Yes" if rr.is_surface else "No"
            if rr.in_cdr:
                row["CDR"] = rr.cdr_label
            else:
                row["CDR"] = ""
            rows.append(row)
        df = pd.DataFrame(rows)
        
        st.markdown("**Per-residue immunogenicity risk profile**")
        
        # Main risk chart
        chart_cols = ["T-cell Risk", "B-cell Risk", "Combined Risk"]
        st.area_chart(df.set_index("Position")[chart_cols],
                      color=["#ea580c", "#0891b2", "#dc2626"], height=350)
        
        # SASA chart if available
        if "SASA" in df.columns:
            st.markdown("**Solvent Accessibility (SASA)** — higher = more surface-exposed")
            st.caption("Surface-exposed residues (SASA > 25) are accessible to antibodies")
            st.bar_chart(df.set_index("Position")["SASA"], color="#10b981", height=200)
        
        # CDR overlay legend
        if report.cdr_regions:
            cdr_text = " | ".join([f"{c['label']}: {c['start']}–{c['end']}" for c in report.cdr_regions])
            st.info(f"**CDR regions (binding site):** {cdr_text}")
        
        with st.expander("View full residue data table"):
            styled = df.style.background_gradient(subset=["Combined Risk"], cmap="YlOrRd")
            if "SASA" in df.columns:
                styled = styled.background_gradient(subset=["SASA"], cmap="Greens")
            st.dataframe(styled, use_container_width=True, height=400)
        
        st.download_button("Download CSV", df.to_csv(index=False),
                           file_name=f"{(seq_name or 'query').lower().replace(' ','_')}_risk_scores.csv",
                           mime="text/csv")

    # ── Tab 4: Clinical context ──
    with tab4:
        ctx = st.session_state.get("clinical_context", {})
        ctx_modality = ctx.get("modality", "All")
        st.markdown(f"**Comparable therapeutics from IDC DB V1** ({ctx_modality})")
        st.caption("Showing comparables filtered by modality. 3,334 ADA datapoints across 218 therapeutics and 726 clinical trials.")
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

    # ── Tab 6: Tolerance Analysis ──
    with tab6:
        st.markdown("**Tolerance & Regulatory T-cell Epitope Analysis**")
        st.caption("Inspired by JanusMatrix — analyzes TCR-facing residues to identify putative Treg epitopes")
        
        tol_key = f"tolerance_{seq_name}"
        if tol_key not in st.session_state:
            with st.spinner("Analyzing TCR-face humanness & Tregitope content..."):
                st.session_state[tol_key] = run_tolerance_analysis(
                    sequence=seq_clean,
                    t_cell_epitopes=report.t_cell_epitopes,
                    residue_risks=report.residue_risks,
                    overall_risk=report.overall_risk_score,
                )
        tol = st.session_state[tol_key]
        
        # Summary metrics
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            color = "#059669" if tol.treg_fraction > 0.3 else "#ca8a04" if tol.treg_fraction > 0.15 else "#dc2626"
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color};">{tol.treg_fraction:.0%}</div><div class="metric-label">Treg Fraction</div></div>', unsafe_allow_html=True)
        with tc2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#059669;">{tol.putative_treg_epitopes}</div><div class="metric-label">Treg Epitopes</div></div>', unsafe_allow_html=True)
        with tc3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#dc2626;">{tol.putative_effector_epitopes}</div><div class="metric-label">Effector Epitopes</div></div>', unsafe_allow_html=True)
        with tc4:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2563eb;">{tol.tregitope_matches}</div><div class="metric-label">Tregitope Matches</div></div>', unsafe_allow_html=True)
        
        # Risk adjustment box
        st.markdown(f'<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:8px;padding:16px;margin:16px 0;"><div style="font-weight:600;color:#065f46;font-size:15px;">Tolerance-Adjusted Risk: {tol.original_risk_score:.0%} → {tol.adjusted_risk_score:.0%}<span style="color:#059669;font-weight:500;"> (−{tol.risk_reduction:.0%} from Treg content)</span></div><div style="font-size:12px;color:#047857;margin-top:4px;">Based on De Groot et al. (Blood 2008): Tregitope content correlates with reduced clinical immunogenicity.</div></div>', unsafe_allow_html=True)
        
        # Epitope breakdown
        effectors = [r for r in tol.results if not r.is_putative_treg]
        tregs = [r for r in tol.results if r.is_putative_treg]
        
        if effectors:
            st.markdown(f"**⚠️ Effector Epitopes ({len(effectors)})** — these drive ADA formation")
            seen_starts = set()
            for r in sorted(effectors, key=lambda x: x.humanness_score)[:8]:
                if r.start in seen_starts:
                    continue
                seen_starts.add(r.start)
                st.markdown(f'<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px;margin-bottom:8px;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><span style="font-family:monospace;font-size:13px;color:#dc2626;">{r.epitope_seq}</span><span style="color:#6b7280;font-size:12px;margin-left:8px;">Pos {r.start}–{r.end}</span></div><div><span style="font-size:12px;color:#991b1b;">Humanness: {r.humanness_score:.2f}</span></div></div></div>', unsafe_allow_html=True)
        
        if tregs:
            st.markdown(f"<br>**Putative Treg Epitopes ({len(tregs)})** — may suppress immune response", unsafe_allow_html=True)
            seen_starts = set()
            for r in sorted(tregs, key=lambda x: -x.humanness_score)[:8]:
                if r.start in seen_starts:
                    continue
                seen_starts.add(r.start)
                match_badge = ' <span style="background:#059669;color:white;padding:2px 6px;border-radius:3px;font-size:10px;">Tregitope</span>' if r.tregitope_match else ''
                st.markdown(f'<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:8px;padding:12px;margin-bottom:8px;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><span style="font-family:monospace;font-size:13px;color:#059669;">{r.epitope_seq}</span><span style="color:#6b7280;font-size:12px;margin-left:8px;">Pos {r.start}–{r.end}{match_badge}</span></div><div><span style="font-size:12px;color:#059669;">Humanness: {r.humanness_score:.2f}</span></div></div></div>', unsafe_allow_html=True)
        
        with st.expander("How tolerance analysis works"):
            st.markdown("""
**JanusMatrix principle:** MHC-II binding peptides have positions 2,3,5,7,8 facing the TCR. If these residues match patterns in human self-proteins, the epitope may activate regulatory T cells (Tregs) that suppress immune responses.

**Tregitopes:** Conserved IgG sequences that activate Tregs. Therapeutics with higher Tregitope content show lower clinical ADA (De Groot et al., Blood 2008).
            """)

    # ── Tab 7: Deimmunization Engine ──
    with tab7:
        st.markdown("**In Silico Deimmunization Engine**")
        st.caption("Suggests point mutations to reduce immunogenicity while preserving Treg epitopes")
        
        deim_key = f"deimmunization_{seq_name}"
        if deim_key not in st.session_state:
            tol = st.session_state.get(f"tolerance_{seq_name}")
            with st.spinner("Computing deimmunization suggestions..."):
                st.session_state[deim_key] = suggest_deimmunization(
                    sequence=seq_clean,
                    hotspot_regions=report.hotspot_regions,
                    residue_risks=report.residue_risks,
                    tolerance_analysis=tol,
                )
        deim = st.session_state[deim_key]
        
        # Summary
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#dc2626;">{deim.original_risk:.0%}</div><div class="metric-label">Original Risk</div></div>', unsafe_allow_html=True)
        with dc2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#059669;">{deim.predicted_optimized_risk:.0%}</div><div class="metric-label">Predicted Optimized</div></div>', unsafe_allow_html=True)
        with dc3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2563eb;">{deim.mutations_applied}</div><div class="metric-label">Mutations Suggested</div></div>', unsafe_allow_html=True)
        
        if deim.suggestions:
            st.markdown("<br>**Suggested mutations** (ranked by impact)", unsafe_allow_html=True)
            for i, s in enumerate(deim.suggestions, 1):
                target_type = "MHC anchor" if not s.is_tcr_face else "TCR contact"
                treg_badge = ' <span style="background:#059669;color:white;padding:1px 6px;border-radius:3px;font-size:10px;">Treg-safe</span>' if s.preserves_treg else ''
                st.markdown(f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin-bottom:10px;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><span style="font-weight:600;font-size:15px;color:#111827;">#{i}: {s.original_aa}{s.position}{s.suggested_aa}</span><span style="color:#6b7280;font-size:12px;margin-left:8px;">{target_type}{treg_badge}</span></div><div style="font-weight:600;font-size:14px;color:#059669;">−{s.predicted_risk_reduction:.0%}</div></div><div style="font-size:12px;color:#6b7280;margin-top:6px;">{s.rationale[:150]}...</div></div>', unsafe_allow_html=True)
            
            # Sequence diff
            st.markdown("<br>**Sequence comparison**", unsafe_allow_html=True)
            diff_html = ""
            for i, (orig, opt) in enumerate(zip(deim.original_sequence, deim.optimized_sequence)):
                if orig != opt:
                    diff_html += f'<span style="background:#fecaca;color:#991b1b;font-weight:700;text-decoration:line-through;font-family:monospace;font-size:12px;">{orig}</span>'
                    diff_html += f'<span style="background:#bbf7d0;color:#166534;font-weight:700;font-family:monospace;font-size:12px;">{opt}</span>'
                else:
                    diff_html += f'<span style="font-family:monospace;font-size:12px;color:#6b7280;">{orig}</span>'
            st.markdown(f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px;line-height:2;word-wrap:break-word;">{diff_html}</div>', unsafe_allow_html=True)
            
            # Download buttons
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "Download optimized FASTA",
                    f">SafeBind_optimized_{seq_name}\n{deim.optimized_sequence}",
                    file_name=f"{(seq_name or 'query').lower().replace(' ','_')}_deimmunized.fasta",
                    mime="text/plain",
                )
            with col_b:
                mut_csv = "Position,Original,Suggested,Type,Risk_Reduction\n"
                for s in deim.suggestions:
                    target_type = "MHC_anchor" if not s.is_tcr_face else "TCR_contact"
                    mut_csv += f"{s.position},{s.original_aa},{s.suggested_aa},{target_type},{s.predicted_risk_reduction:.2f}\n"
                st.download_button("Download mutations CSV", mut_csv, file_name=f"{(seq_name or 'query').lower().replace(' ','_')}_mutations.csv", mime="text/csv")
        else:
            st.success("No deimmunization needed — no significant hotspots detected.")
        
        with st.expander("How deimmunization works"):
            st.markdown("""
**Strategy 1 — Anchor disruption:** Replace MHC anchor residues (positions 1,4,6,9) with disfavored amino acids to prevent peptide loading.

**Strategy 2 — TCR face alteration:** Conservative substitutions at TCR-contact positions change recognition without disrupting MHC binding.

**Treg preservation:** Regions with putative Treg epitopes are protected from mutation.
            """)

    # ── Tab 8: MHC-I / Cytotoxic T-cell Analysis ──
    with tab8:
        st.markdown("**MHC Class I / CD8+ Cytotoxic T-cell Analysis**")
        st.caption(
            "Predicts where CD8+ cytotoxic T cells will attack — the pathway that causes "
            "liver toxicity in gene therapy and cell rejection in CAR-T. Uses IEDB NetMHCpan 4.1 "
            "(trained on 185,985 peptide-MHC-I pairs from TDC) across 12 HLA-A/B supertypes."
        )

        cyto_key = f"cytotoxic_{seq_name}"
        if cyto_key not in st.session_state:
            ctx = st.session_state.get("clinical_context", {})
            serotype = ctx.get("serotype")
            with st.spinner("Running MHC-I predictions across 12 HLA Class I alleles..."):
                st.session_state[cyto_key] = run_cytotoxic_assessment(
                    sequence=seq_clean,
                    name=seq_name or "Query",
                    serotype=serotype,
                    use_mhcflurry=True,
                    use_iedb=True,
                    verbose=False,
                )
        cyto = st.session_state[cyto_key]

        # ── Summary metrics ──
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        risk_color = "#dc2626" if cyto.overall_cytotoxic_risk > 0.4 else "#ea580c" if cyto.overall_cytotoxic_risk > 0.25 else "#ca8a04" if cyto.overall_cytotoxic_risk > 0.12 else "#0891b2"
        with cc1:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{risk_color};">{cyto.overall_cytotoxic_risk:.0%}</div><div class="metric-label">Cytotoxic Risk</div></div>', unsafe_allow_html=True)
        with cc2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#111827;">{cyto.risk_category}</div><div class="metric-label">Risk Category</div></div>', unsafe_allow_html=True)
        with cc3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#dc2626;">{cyto.strong_binders}</div><div class="metric-label">Strong Binders (&lt;2%)</div></div>', unsafe_allow_html=True)
        with cc4:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#7c3aed;">{len(cyto.hotspot_regions)}</div><div class="metric-label">CTL Hotspots</div></div>', unsafe_allow_html=True)
        with cc5:
            val_color = "#059669" if cyto.validated_hits > 0 else "#6b7280"
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{val_color};">{cyto.validated_hits}</div><div class="metric-label">Validated Hits</div></div>', unsafe_allow_html=True)

        # ── Prediction sources ──
        if cyto.prediction_sources:
            src_badges = " ".join(
                f'<span style="background:#eff6ff;color:#2563eb;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:500;">{s}</span>'
                for s in cyto.prediction_sources
            )
        else:
            src_badges = '<span style="color:#6b7280;font-size:12px;">Predictions pending</span>'
        st.markdown(f'<div style="margin:12px 0;">{src_badges}</div>', unsafe_allow_html=True)

        # ── Dual pathway comparison ──
        mhc2_risk = report.overall_risk_score
        st.markdown(f"""<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:16px;margin:12px 0;">
            <div style="font-weight:600;color:#0c4a6e;font-size:14px;margin-bottom:8px;">Dual Pathway Risk Comparison</div>
            <div style="display:flex;gap:32px;">
                <div style="flex:1;">
                    <div style="font-size:12px;color:#6b7280;">MHC-II → CD4+ → ADA (humoral)</div>
                    <div style="font-size:24px;font-weight:600;color:#ea580c;">{mhc2_risk:.0%}</div>
                    <div style="background:#e5e7eb;border-radius:4px;height:8px;margin-top:4px;">
                        <div style="background:#ea580c;width:{min(100, mhc2_risk*100):.0f}%;height:100%;border-radius:4px;"></div>
                    </div>
                </div>
                <div style="flex:1;">
                    <div style="font-size:12px;color:#6b7280;">MHC-I → CD8+ → Cytotoxic (cellular)</div>
                    <div style="font-size:24px;font-weight:600;color:#7c3aed;">{cyto.overall_cytotoxic_risk:.0%}</div>
                    <div style="background:#e5e7eb;border-radius:4px;height:8px;margin-top:4px;">
                        <div style="background:#7c3aed;width:{min(100, cyto.overall_cytotoxic_risk*100):.0f}%;height:100%;border-radius:4px;"></div>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── AAV Validated Epitope Recovery ──
        if cyto.aav_epitope_recovery is not None and cyto.validated_details:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**AAV Validated Epitope Cross-Reference**")
            st.caption(
                f"Cross-checked against {len(AAV_VALIDATED_EPITOPES_HUI_2015)} validated CD8+ epitopes "
                f"(Hui et al. 2015) + {len(AAV_IMMUNOPEPTIDOMICS_2023)} immunopeptidomics peptides (2023)."
            )
            rec_color = "#059669" if cyto.aav_epitope_recovery > 0.6 else "#ca8a04" if cyto.aav_epitope_recovery > 0.3 else "#dc2626"
            st.markdown(f"""<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:8px;padding:12px;margin-bottom:12px;">
                <span style="font-weight:600;color:{rec_color};font-size:16px;">{cyto.aav_epitope_recovery:.0%}</span>
                <span style="color:#065f46;font-size:13px;margin-left:8px;">
                    of known epitopes recovered ({cyto.validated_hits}/{len(cyto.validated_details)} in sequence)
                </span>
            </div>""", unsafe_allow_html=True)
            for vd in cyto.validated_details:
                status = "✅" if vd["is_recovered"] else "❌"
                style_bg = "#ecfdf5" if vd["is_recovered"] else "#fef2f2"
                style_border = "#6ee7b7" if vd["is_recovered"] else "#fecaca"
                kd_text = f" · Kd={vd['kd_um']:.1f}µM" if vd.get("kd_um") else ""
                conserved_badge = ' <span style="background:#dbeafe;color:#1e40af;padding:1px 5px;border-radius:3px;font-size:10px;">cross-serotype</span>' if vd.get("conserved") else ""
                st.markdown(f"""<div style="background:{style_bg};border:1px solid {style_border};border-radius:6px;padding:10px;margin-bottom:6px;">
                    <span style="font-size:14px;">{status}</span>
                    <span style="font-family:monospace;font-size:13px;margin-left:6px;">{vd['validated_peptide']}</span>
                    <span style="color:#6b7280;font-size:12px;margin-left:8px;">
                        {vd['position']} · {vd['hla']}{kd_text} · {vd['source']}{conserved_badge}
                    </span>
                </div>""", unsafe_allow_html=True)

        # ── Cytotoxic Hotspots ──
        st.markdown("<br>", unsafe_allow_html=True)
        if cyto.hotspot_regions:
            st.markdown(f"**Cytotoxic Hotspot Regions ({len(cyto.hotspot_regions)})**")
            st.caption("Regions with high MHC-I binding density — CD8+ T cells will target these areas")
            for i, hs in enumerate(cyto.hotspot_regions[:8], 1):
                bar_color = "#7c3aed" if hs['avg_risk'] > 0.4 else "#a855f7" if hs['avg_risk'] > 0.25 else "#c084fc"
                val_badge = ' <span style="background:#fef3c7;color:#92400e;padding:1px 5px;border-radius:3px;font-size:10px;">VALIDATED</span>' if hs.get("has_validated") else ""
                st.markdown(f"""<div class="hotspot-row" style="border-left:3px solid {bar_color};padding:12px 16px;margin-bottom:8px;background:#faf5ff;border-radius:6px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-weight:600;font-size:15px;color:#111827;">CTL Hotspot {i}</span>
                            <span style="color:#6b7280;font-size:13px;margin-left:8px;">Pos {hs['start']}–{hs['end']} ({hs['length']} aa){val_badge}</span>
                        </div>
                        <div style="font-weight:600;font-size:18px;color:{bar_color};">{hs['avg_risk']:.0%}</div>
                    </div>
                    <div style="font-family:monospace;font-size:13px;color:#7c3aed;margin-top:8px;">{hs['sequence']}</div>
                    <div style="display:flex;gap:24px;margin-top:8px;font-size:12px;color:#6b7280;">
                        <span>Max alleles: <b>{hs['max_alleles']}</b></span>
                        <span>Max risk: <b style="color:#7c3aed;">{hs['max_risk']:.0%}</b></span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No significant MHC-I hotspot regions detected.")

        # ── Per-residue cytotoxic risk chart ──
        if cyto.residue_risks:
            import pandas as pd
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Per-residue MHC-I risk profile**")
            cyto_df = pd.DataFrame([
                {"Position": rr.position, "MHC-I Risk": rr.mhc1_risk}
                for rr in cyto.residue_risks
            ])
            st.area_chart(cyto_df.set_index("Position")["MHC-I Risk"], color="#7c3aed", height=250)

        with st.expander("Data sources & methodology"):
            st.markdown("""
**MHC-I Prediction:** IEDB NetMHCpan 4.1 across 12 HLA-A/B supertypes covering ~95% of the global population.

**AAV Validation:** Hui et al. 2015 (21 validated CD8+ epitopes, PMC4588448) + 2023 immunopeptidomics (65 HLA-I peptides from AAV capsids). Clinical correlation: GENEr8-1 Phase 3 trial, 134 patients.

**Biology:** MHC-I presents intracellular peptides (8–11mers) to CD8+ cytotoxic T cells, triggering direct cell killing. This is the primary pathway driving hepatotoxicity in AAV gene therapy (AT132 patient deaths, Elevidys liver failures, Roctavian withdrawal).
            """)

    # ── Tab 9: Composite Scoring Engine ──
    with tab9:
        st.markdown("**Composite Immunogenicity Scoring Engine**")
        st.caption(
            "Fuses 4 orthogonal signals: clinical benchmarking (3,334 datapoints), "
            "sequence similarity (222 references), dual-pathway epitope load (MHC-I + MHC-II), "
            "and AI synthesis — into a single clinically-meaningful risk prediction."
        )

        comp_key = f"composite_{seq_name}"
        if comp_key not in st.session_state:
            ctx = st.session_state.get("clinical_context", {})
            cyto = st.session_state.get(f"cytotoxic_{seq_name}")
            mhc1_eps = cyto.strong_binders if cyto else 0
            mhc1_hs = len(cyto.hotspot_regions) if cyto else 0
            mhc1_risk = cyto.overall_cytotoxic_risk if cyto else 0.0
            mhc2_eps = len([e for e in report.t_cell_epitopes if e.rank < 10])
            mhc2_hs = len(report.hotspot_regions)
            tol = st.session_state.get(f"tolerance_{seq_name}")
            tol_data = None
            if tol:
                tol_data = {
                    "treg_fraction": tol.treg_fraction,
                    "treg_count": tol.putative_treg_epitopes,
                    "effector_count": tol.putative_effector_epitopes,
                    "adjusted_risk": tol.adjusted_risk_score,
                }
            with st.spinner("Computing composite score from 4 signals..."):
                st.session_state[comp_key] = compute_composite_score(
                    sequence=seq_clean,
                    name=seq_name or "Query",
                    modality=ctx.get("modality", "Monoclonal antibody (mAb)"),
                    route=ctx.get("route", "IV (intravenous)"),
                    species=species,
                    indication=ctx.get("indication", ""),
                    crim_status=ctx.get("crim_status"),
                    immunosuppressants=ctx.get("immunosuppressants", False),
                    mhc2_epitope_count=mhc2_eps,
                    mhc2_hotspot_count=mhc2_hs,
                    mhc2_overall_risk=report.overall_risk_score,
                    mhc1_epitope_count=mhc1_eps,
                    mhc1_hotspot_count=mhc1_hs,
                    mhc1_overall_risk=mhc1_risk,
                    tolerance_data=tol_data,
                    clinical_context=ctx,
                )
        comp = st.session_state[comp_key]

        # ── Big composite score display ──
        # Calibrated thresholds based on clinical outcomes
        score_color = "#dc2626" if comp.composite_score > 40 else "#ea580c" if comp.composite_score > 25 else "#ca8a04" if comp.composite_score > 15 else "#059669"
        
        # Pass/fail prediction
        if comp.composite_score <= 15:
            prediction = "LIKELY TO PASS"
            pred_color = "#059669"
            pred_bg = "#ecfdf5"
            comparables = "Similar to: Trastuzumab, Pembrolizumab, Adalimumab+MTX"
        elif comp.composite_score <= 25:
            prediction = "MAY PASS WITH MITIGATION"
            pred_color = "#ca8a04"
            pred_bg = "#fefce8"
            comparables = "Similar to: Infliximab (needs TDM), Adalimumab monotherapy"
        elif comp.composite_score <= 40:
            prediction = "HIGH FAILURE RISK"
            pred_color = "#ea580c"
            pred_bg = "#fff7ed"
            comparables = "Similar to: Bococizumab (failed Phase 3)"
        else:
            prediction = "LIKELY TO FAIL"
            pred_color = "#dc2626"
            pred_bg = "#fef2f2"
            comparables = "Similar to: Catumaxomab (withdrawn), AT132 (4 deaths)"
        
        st.markdown(f"""<div style="background:linear-gradient(135deg,#f8fafc 0%,#f1f5f9 100%);border:2px solid {score_color};border-radius:12px;padding:24px;text-align:center;margin-bottom:12px;">
            <div style="font-size:56px;font-weight:700;color:{score_color};">{comp.composite_score:.0f}<span style="font-size:24px;color:#6b7280;">/100</span></div>
            <div style="font-size:18px;font-weight:600;color:#111827;margin-top:4px;">{comp.composite_category}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:4px;">
                95% CI: [{comp.confidence_interval[0]:.0f}, {comp.confidence_interval[1]:.0f}] · {comp.total_data_sources} data sources
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Pass/fail prediction banner
        st.markdown(f"""<div style="background:{pred_bg};border:2px solid {pred_color};border-radius:8px;padding:12px 16px;margin-bottom:20px;text-align:center;">
            <div style="font-size:14px;font-weight:700;color:{pred_color};">{prediction}</div>
            <div style="font-size:11px;color:#6b7280;margin-top:4px;">{comparables}</div>
        </div>""", unsafe_allow_html=True)

        # ── Dual pathway bars ──
        st.markdown(f"""<div style="display:flex;gap:16px;margin-bottom:20px;">
            <div style="flex:1;background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:14px;">
                <div style="font-size:12px;color:#9a3412;font-weight:500;">HUMORAL (MHC-II → ADA)</div>
                <div style="font-size:28px;font-weight:600;color:#ea580c;margin:4px 0;">{comp.humoral_risk:.0f}</div>
                <div style="background:#fed7aa;border-radius:4px;height:6px;">
                    <div style="background:#ea580c;width:{min(100, comp.humoral_risk):.0f}%;height:100%;border-radius:4px;"></div>
                </div>
            </div>
            <div style="flex:1;background:#f5f3ff;border:1px solid #ddd6fe;border-radius:8px;padding:14px;">
                <div style="font-size:12px;color:#5b21b6;font-weight:500;">CYTOTOXIC (MHC-I → CD8+)</div>
                <div style="font-size:28px;font-weight:600;color:#7c3aed;margin:4px 0;">{comp.cytotoxic_risk:.0f}</div>
                <div style="background:#ddd6fe;border-radius:4px;height:6px;">
                    <div style="background:#7c3aed;width:{min(100, comp.cytotoxic_risk):.0f}%;height:100%;border-radius:4px;"></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Risk flags ──
        if comp.flags:
            for flag in comp.flags:
                flag_name = flag.split(":")[0] if ":" in flag else flag
                flag_desc = flag.split(":", 1)[1].strip() if ":" in flag else flag
                is_critical = "VERY_HIGH" in flag or "CONVERGENT" in flag
                bg = "#fef2f2" if is_critical else "#fffbeb"
                border = "#fecaca" if is_critical else "#fde68a"
                icon = "⚠️" if is_critical else "⚠️"
                st.markdown(f"""<div style="background:{bg};border:1px solid {border};border-radius:6px;padding:10px 14px;margin-bottom:6px;">
                    {icon} <span style="font-weight:600;font-size:12px;color:#374151;">{flag_name}</span>
                    <span style="font-size:12px;color:#6b7280;margin-left:6px;">{flag_desc}</span>
                </div>""", unsafe_allow_html=True)

        # ── Signal breakdown ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Signal Breakdown**")
        signals = [
            (comp.benchmark_signal, "#2563eb", "1."),
            (comp.similarity_signal, "#059669", "2."),
            (comp.epitope_signal, "#dc2626", "3."),
        ]
        for sig, color, icon in signals:
            bar_width = min(100, sig.score)
            conf_pct = sig.confidence * 100
            st.markdown(f"""<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div>
                        <span style="font-size:16px;">{icon}</span>
                        <span style="font-weight:600;font-size:14px;color:#111827;margin-left:6px;">{sig.name}</span>
                        <span style="font-size:11px;color:#6b7280;margin-left:8px;">weight {sig.weight:.0%} · confidence {conf_pct:.0f}%</span>
                    </div>
                    <div style="font-weight:600;font-size:20px;color:{color};">{sig.score:.0f}</div>
                </div>
                <div style="background:#e5e7eb;border-radius:4px;height:8px;margin-bottom:8px;">
                    <div style="background:{color};width:{bar_width}%;height:100%;border-radius:4px;"></div>
                </div>
                <div style="font-size:12px;color:#6b7280;line-height:1.5;">
                    {sig.explanation[:300]}{'...' if len(sig.explanation) > 300 else ''}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Claude AI Synthesis (Signal 4) ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Signal 4: AI Synthesis**")
        api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        synthesis_key = f"composite_synthesis_{seq_name}"
        if api_key:
            if synthesis_key not in st.session_state:
                with st.spinner("Claude is synthesizing the risk assessment..."):
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key)
                        message = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=600,
                            messages=[{"role": "user", "content": comp.synthesis_prompt}]
                        )
                        st.session_state[synthesis_key] = message.content[0].text
                    except Exception as e:
                        st.session_state[synthesis_key] = f"Claude API error: {e}"
            synthesis_text = st.session_state.get(synthesis_key, "")
            if synthesis_text and not synthesis_text.startswith("Claude API error"):
                st.markdown(f"""<div class="claude-report">
                    <div class="claude-badge">Composite Synthesis · Claude</div>
                    <div>{synthesis_text}</div>
                </div>""", unsafe_allow_html=True)
            elif synthesis_text.startswith("Claude API error"):
                st.error(synthesis_text)
        else:
            st.info("Add an Anthropic API key to enable AI synthesis of the composite score.")
            with st.expander("View synthesis prompt"):
                st.code(comp.synthesis_prompt, language=None)

        with st.expander("All data sources & references"):
            st.markdown(f"**{comp.total_data_sources} data sources referenced:**")
            for key, desc in comp.data_references.items():
                st.markdown(f"- **{key}**: {desc}")
            st.markdown("""
**Composite scoring methodology:** Fuses three quantitative signals weighted by confidence:
1. **Clinical Benchmark (30%):** Empirical ADA rates from IDC DB V1 for the drug's class/route/disease
2. **Sequence Similarity (20%):** 5-mer Jaccard similarity against 222 reference therapeutics with known outcomes  
3. **Epitope Load (35%):** Combined MHC-II (70%) + MHC-I (30%) epitope density vs benchmarks
4. **AI Synthesis (15%):** Claude integrates all signals and recommends next steps
            """)

    # ── Tab 10: Advanced Analysis ──
    with tab10:
        st.markdown("**Advanced Analysis**")
        st.caption(
            "Checkpoint inhibitor detection, germline humanness scoring, and IEDB epitope homology search. "
            "These features address gaps identified in comparison with EpiVax ISPRI."
        )
        
        # Run advanced analysis (cache in session state)
        adv_key = f"advanced_{seq_name}"
        if adv_key not in st.session_state:
            ctx = st.session_state.get("clinical_context", {})
            hotspot_peptides = [hs["sequence"] for hs in report.hotspot_regions[:5]] if report.hotspot_regions else []
            
            with st.spinner("Running advanced analysis..."):
                st.session_state[adv_key] = run_advanced_analysis(
                    sequence=seq_clean,
                    name=seq_name or "Query",
                    indication=ctx.get("indication", ""),
                    target="",
                    hotspot_peptides=hotspot_peptides,
                    chain_type="auto",
                )
        adv = st.session_state[adv_key]
        
        # ── Checkpoint Inhibitor Analysis ──
        st.markdown("### 1. Checkpoint Inhibitor Detection")
        
        if adv.checkpoint_analysis.is_checkpoint_inhibitor:
            st.markdown(f"""<div style="background:#fef2f2;border:2px solid #dc2626;border-radius:8px;padding:16px;margin-bottom:16px;">
                <div style="font-weight:700;color:#dc2626;font-size:16px;">⚠️ CHECKPOINT INHIBITOR DETECTED</div>
                <div style="color:#991b1b;margin-top:8px;">
                    <b>Target:</b> {adv.checkpoint_analysis.target}<br>
                    <b>Recommendation:</b> {adv.checkpoint_analysis.recommendation}<br>
                </div>
                <div style="color:#6b7280;font-size:13px;margin-top:12px;line-height:1.5;">
                    {adv.checkpoint_analysis.explanation}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:8px;padding:16px;margin-bottom:16px;">
                <div style="font-weight:600;color:#059669;">Not a checkpoint inhibitor</div>
                <div style="color:#6b7280;font-size:13px;margin-top:8px;">
                    Standard Treg-adjusted scoring applies. Regulatory T cell epitopes will appropriately 
                    reduce the predicted immunogenicity score.
                </div>
            </div>""", unsafe_allow_html=True)
        
        # ── Germline Humanness Analysis ──
        st.markdown("### 2. Germline Humanness Score")
        st.caption("Compares sequence against human V-gene germline to assess 'foreignness'")
        
        if adv.germline_analysis:
            germ = adv.germline_analysis
            
            # Color based on humanness
            if germ.humanness_score >= 90:
                hum_color = "#059669"
                hum_bg = "#ecfdf5"
            elif germ.humanness_score >= 75:
                hum_color = "#ca8a04"
                hum_bg = "#fefce8"
            elif germ.humanness_score >= 60:
                hum_color = "#ea580c"
                hum_bg = "#fff7ed"
            else:
                hum_color = "#dc2626"
                hum_bg = "#fef2f2"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div style="background:{hum_bg};border:1px solid {hum_color};border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:32px;font-weight:700;color:{hum_color};">{germ.humanness_score:.0f}%</div>
                    <div style="font-size:12px;color:#6b7280;">Humanness Score</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:16px;font-weight:600;color:#111827;">{germ.chain_type}</div>
                    <div style="font-size:12px;color:#6b7280;">Chain Type</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                best_name = germ.best_match.germline_name if germ.best_match else "None"
                st.markdown(f"""<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:14px;font-weight:600;color:#111827;">{best_name}</div>
                    <div style="font-size:12px;color:#6b7280;">Best Germline Match</div>
                </div>""", unsafe_allow_html=True)
            
            st.markdown(f"<div style='margin-top:12px;color:#6b7280;font-size:13px;'>{germ.recommendation}</div>", unsafe_allow_html=True)
            
            # Foreign regions
            if germ.foreign_regions:
                st.markdown(f"<br>**Foreign Regions** ({len(germ.foreign_regions)} detected)", unsafe_allow_html=True)
                for region in germ.foreign_regions[:5]:
                    st.markdown(f"""<div style="background:#fef2f2;border-left:3px solid #dc2626;padding:8px 12px;margin-bottom:4px;font-size:13px;">
                        <span style="color:#6b7280;">Pos {region['start']}-{region['end']}:</span>
                        <span style="font-family:monospace;color:#dc2626;margin-left:8px;">{region['query']}</span>
                        <span style="color:#6b7280;margin-left:8px;">vs germline:</span>
                        <span style="font-family:monospace;color:#059669;margin-left:4px;">{region['germline']}</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("Germline analysis not available for this sequence (requires antibody-length sequence)")
        
        # ── IEDB Homology Search ──
        st.markdown("### 3. IEDB Epitope Homology")
        st.caption("Cross-references predicted hotspots against published T-cell epitopes in IEDB")
        
        if adv.iedb_matches:
            validated_count = sum(1 for r in adv.iedb_matches.values() if r.has_published_epitope)
            positive_count = sum(
                1 for r in adv.iedb_matches.values() 
                if r.published_response_type and "POSITIVE" in r.published_response_type
            )
            
            st.markdown(f"""<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:16px;margin-bottom:16px;">
                <div style="display:flex;gap:32px;">
                    <div>
                        <span style="font-size:24px;font-weight:600;color:#0284c7;">{len(adv.iedb_matches)}</span>
                        <span style="color:#6b7280;font-size:13px;margin-left:8px;">Hotspots searched</span>
                    </div>
                    <div>
                        <span style="font-size:24px;font-weight:600;color:#059669;">{validated_count}</span>
                        <span style="color:#6b7280;font-size:13px;margin-left:8px;">Found in IEDB</span>
                    </div>
                    <div>
                        <span style="font-size:24px;font-weight:600;color:#dc2626;">{positive_count}</span>
                        <span style="color:#6b7280;font-size:13px;margin-left:8px;">Published as immunogenic</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            for peptide, result in list(adv.iedb_matches.items())[:5]:
                if result.has_published_epitope:
                    status_color = "#dc2626" if result.published_response_type and "POSITIVE" in result.published_response_type else "#059669"
                    status_text = result.published_response_type or "Found in IEDB"
                    st.markdown(f"""<div style="background:#f9fafb;border-left:3px solid {status_color};padding:10px 14px;margin-bottom:6px;">
                        <span style="font-family:monospace;font-size:13px;">{peptide}</span>
                        <span style="color:{status_color};font-size:12px;margin-left:12px;font-weight:500;">{status_text}</span>
                        <span style="color:#6b7280;font-size:11px;margin-left:8px;">({result.total_matches} matches)</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="background:#f9fafb;border-left:3px solid #e5e7eb;padding:10px 14px;margin-bottom:6px;">
                        <span style="font-family:monospace;font-size:13px;color:#6b7280;">{peptide}</span>
                        <span style="color:#6b7280;font-size:12px;margin-left:12px;">No IEDB matches</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("No hotspot peptides to search (run T-cell epitope prediction first)")
        
        # ── Summary Flags ──
        if adv.additional_flags:
            st.markdown("### Summary Flags")
            for flag in adv.additional_flags:
                flag_name = flag.split(":")[0]
                flag_desc = flag.split(":", 1)[1].strip() if ":" in flag else flag
                st.markdown(f"""<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:10px 14px;margin-bottom:6px;">
                    ⚠️ <span style="font-weight:600;font-size:12px;color:#92400e;">{flag_name}</span>
                    <span style="font-size:12px;color:#6b7280;margin-left:6px;">{flag_desc}</span>
                </div>""", unsafe_allow_html=True)
        
        with st.expander("About these analyses"):
            st.markdown("""
**Checkpoint Inhibitor Detection:** Per EpiVax guidance (Mattei et al. 2024), checkpoint inhibitors 
targeting PD-1, PD-L1, or CTLA-4 impair regulatory T cell function. For these drugs, Treg-adjusted 
scores underestimate immunogenicity risk. We recommend using raw epitope scores instead.

**Germline Humanness:** Compares the query sequence against human V-gene germlines (IMGT reference). 
Higher similarity to germline indicates the sequence is more "self-like" and less likely to trigger 
T-cell responses, as T cells recognizing germline sequences would have been deleted during thymic development.

**IEDB Homology:** Searches the Immune Epitope Database for published epitopes matching predicted hotspots. 
If a predicted epitope has been experimentally validated as immunogenic in T-cell assays, this increases 
confidence in the prediction.
            """)

    st.markdown("""<div class="disclaimer">
        FOR RESEARCH AND DEMONSTRATION PURPOSES ONLY. NOT FOR CLINICAL USE.<br>
        Data: IEDB (tools.iedb.org) · IDC DB V1 (CC BY 4.0, Agnihotri et al. 2025) · RCSB PDB · Claude (Anthropic) · Tamarind Bio
    </div>""", unsafe_allow_html=True)


# ── Main content ─────────────────────────────────────────────
st.markdown('<div class="hero-title">SafeBind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Dual-pathway immunogenicity prediction for biologics · MHC-I (cytotoxic) + MHC-II (humoral) · 3,334 clinical datapoints</div>', unsafe_allow_html=True)

if run_clicked and seq_input:
    seq_clean = "".join(c for c in seq_input.upper() if c.isalpha())
    if len(seq_clean) < 20:
        st.error("Sequence too short. Please enter at least 20 amino acids.")
    else:
        # Create a styled progress container
        progress_container = st.container()
        with progress_container:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); border-radius: 12px; padding: 20px; color: white; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 28px;">🔬</div>
                    <div>
                        <div style="font-size: 16px; font-weight: 600;">Analyzing Immunogenicity</div>
                        <div style="font-size: 12px; opacity: 0.9;">Running IEDB predictions across 9 HLA-DR alleles</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            progress = st.progress(0)
            status_text = st.empty()
            details_text = st.empty()
            
            # Step 1: Initialize
            status_text.markdown("**Step 1/5:** Initializing analysis...")
            details_text.caption(f"Sequence: {len(seq_clean)} amino acids")
            progress.progress(5)
            
            # Step 2: T-cell epitopes (this is the slow part)
            status_text.markdown("**Step 2/5:** Predicting T-cell epitopes...")
            details_text.caption("Querying IEDB MHC Class II binding predictions for 9 HLA-DRB1 alleles (~85% global coverage)")
            progress.progress(10)
            
        # Run the actual analysis
        # Use modality from clinical context
        ctx = st.session_state.get("clinical_context", {})
        analysis_modality = ctx.get("modality", "Monoclonal antibody")
        
        report = run_immunogenicity_assessment(
            sequence=seq_clean, name=seq_name or "Query",
            pdb_id=pdb_id if pdb_id else None, pdb_chain="A",
            idc_data_path="idc_db_v1_table_s4.xlsx",
            species=species, modality=analysis_modality, verbose=False,
        )
        
        with progress_container:
            # Step 3: B-cell epitopes
            status_text.markdown("**Step 3/5:** Analyzing B-cell epitopes...")
            details_text.caption("Running BepiPred linear epitope prediction")
            progress.progress(60)
            time.sleep(0.2)
            
            # Step 4: Risk scoring
            status_text.markdown("**Step 4/5:** Computing risk scores...")
            details_text.caption("Calculating per-residue immunogenicity and identifying hotspots")
            progress.progress(80)
            time.sleep(0.2)
            
            # Step 5: Comparables
            status_text.markdown("**Step 5/5:** Loading clinical comparables...")
            details_text.caption("Cross-referencing IDC DB V1 (218 therapeutics, 4,146 ADA datapoints)")
            progress.progress(95)
            time.sleep(0.2)
            
            # Complete
            status_text.markdown("**Analysis complete!**")
            details_text.caption(f"Found {len([e for e in report.t_cell_epitopes if e.rank < 10])} T-cell epitopes, {len(report.b_cell_epitopes)} B-cell epitopes, {len(report.hotspot_regions)} hotspots")
            progress.progress(100)
            time.sleep(0.5)
        
        # Clear the progress container
        progress_container.empty()
        
        # Persist to session state so Tamarind button reruns don't lose the report
        st.session_state["report"] = report
        st.session_state["seq_clean"] = seq_clean
        st.session_state["seq_name"] = seq_name
        st.session_state["pdb_id"] = pdb_id
        # Clear any cached Claude report when re-analyzing
        st.session_state.pop(f"claude_report_{seq_name}", None)

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
            Predict both ADA risk and cytotoxic T-cell liability
        </div>
        <div style="font-size:14px;color:#6b7280;max-width:650px;margin:0 auto;line-height:1.7;">
            First platform to predict <b>both immunogenicity pathways</b>: MHC-II (humoral/ADA)
            and MHC-I (cytotoxic/CD8+). Fuses epitope predictions across 21 HLA alleles with
            3,334 clinical outcomes from IDC DB V1 into a single composite risk score.
        </div>
        <div style="margin-top:32px;display:flex;justify-content:center;gap:40px;">
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:600;color:#2563eb;">21</div>
                <div style="font-size:11px;color:#6b7280;">HLA Alleles (I + II)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:600;color:#0891b2;">3,334</div>
                <div style="font-size:11px;color:#6b7280;">Clinical Outcomes</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:600;color:#ea580c;">222</div>
                <div style="font-size:11px;color:#6b7280;">Reference Drugs</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:28px;font-weight:600;color:#7c3aed;">2</div>
                <div style="font-size:11px;color:#6b7280;">Immune Pathways</div>
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

        Immunogenicity kills drugs through two pathways: **ADAs** (anti-drug antibodies)
        that neutralize efficacy, and **CD8+ cytotoxic T cells** that cause liver toxicity
        in gene therapy. Bococizumab cost Pfizer $1B+ (44% nADA). AAV gene therapies
        have killed patients from hepatotoxicity. Existing tools predict only one pathway.
        """)
    with col2:
        st.markdown("""
        **Our Approach**

        SafeBind is the first platform predicting **both pathways**: MHC-II → CD4+ → ADA
        (humoral) and MHC-I → CD8+ → cytotoxicity (cellular). We fuse 21 HLA alleles,
        3,334 clinical outcomes from IDC DB V1, sequence similarity against 222 approved
        drugs, and validated AAV epitopes into a single composite score (0-100).
        """)
