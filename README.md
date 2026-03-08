# 🛡️ SafeBind AI — Immunogenicity Risk Assessment

**Bio x AI Hackathon @ YC HQ — March 8, 2026**

SafeBind AI predicts where the human immune system will attack therapeutic proteins, helping drug developers fix immunogenicity hotspots before a billion-dollar clinical trial fails.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key (from hackathon credits)
export ANTHROPIC_API_KEY="sk-ant-..."

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## What It Does

1. **Paste any protein sequence** (antibody, enzyme, Fc-fusion, etc.)
2. **T-cell epitope prediction** via IEDB MHC Class II API across 9 HLA-DR alleles (~85% global population)
3. **B-cell epitope prediction** via IEDB Bepipred
4. **Per-residue risk scoring** combining T-cell and B-cell signals
5. **Interactive 3D heatmap** (if PDB structure available) via py3Dmol
6. **Clinical benchmarking** against 218 real therapeutics from IDC DB V1 (4,146 ADA datapoints)
7. **AI risk narrative** via Claude (Anthropic) synthesizing predictions with clinical context

## Files

- `app.py` — Streamlit web application
- `immunogenicity_core.py` — Backend prediction pipeline
- `idc_db_v1_table_s4.xlsx` — IDC DB V1 clinical immunogenicity database (CC BY 4.0)
- `idc_db_v1_table_s5.xlsx` — IDC DB V1 aggregated ADA frequencies
- `requirements.txt` — Python dependencies

## Demo Sequences (pre-loaded in app)

| Therapeutic | Clinical ADA Rate | Outcome |
|---|---|---|
| Bococizumab | 44% (27% neutralizing) | **TERMINATED** — $1B+ loss for Pfizer |
| Adalimumab (Humira) | 30-93% | Best-selling drug ever despite high ADA |
| Trastuzumab (Herceptin) | 0-14% | Low ADA, well-tolerated blockbuster |
| Nivolumab (Opdivo) | 11-26% | Moderate ADA, checkpoint inhibitor context |

## Data Sources

- **IEDB** (tools.iedb.org) — Immune Epitope Database, zero-auth API
- **IDC DB V1** — Immunogenicity Database Collaborative (Agnihotri et al., bioRxiv 2025), CC BY 4.0
- **RCSB PDB** — Protein Data Bank for 3D structures
- **Claude** (Anthropic) — AI-powered risk narrative generation

## Disclaimer

FOR RESEARCH AND DEMONSTRATION PURPOSES ONLY. NOT FOR CLINICAL USE.
