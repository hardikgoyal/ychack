# SafeBind Risk 🧬

**Biotherapeutics Immunogenicity Risk Assessment Dashboard**

SafeBind Risk is a Streamlit-based web application designed to predict anti-drug antibody (ADA) probability for protein therapeutic candidates. It leverages empirical clinical data, advanced sequence analysis, structural modeling, and AI to provide a comprehensive immunogenicity risk profile. 

Recent updates have expanded SafeBind beyond simple MHC-II screening to a full-scale candidate downselection tool encompassing cytotoxic risk (MHC-I) and global composite scoring.

## Key Features

- **Empirical Benchmarking**: Compare your candidate against historical ADA rates matched by route of administration, disease indication, and modality (powered by clinical data from over 200 approved drugs and 3,300+ cohorts).
- **Sequence Similarity**: Aligns query sequences to reference sequences of approved drugs using k-mer prefiltering and global sequence alignment.
- **T-cell (MHC-II & MHC-I) & B-cell Epitope Prediction**: 
  - *MHC-II (Helper T-cell):* Integration with the IEDB API for CD4+ binding analysis.
  - *MHC-I (Cytotoxic T-cell):* High-throughput NetMHCpan-4.1 integration to evaluate CD8+ T-cell responses and cross-presentation risks.
  - *B-cell:* Linear epitope prediction via Bepipred.
- **Structural Analysis & Solvent Accessibility (SASA)**: Real-time 3D protein structure folding via **ESMFold** (powered by Tamarind Bio API), enabling calculation of surface-exposed epitopes and a 3D Risk Heatmap viewer. Includes a local `.pdb_cache` to speed up repeated runs.
- **Composite Scoring & Candidate Downselection**: A unified risk model that aggregates clinical baselines, MHC-II, MHC-I, and B-cell data into a single `CompositeScore`. The new **Downselect** module allows teams to load entirely un-optimized sequences and automatically filter them based on viability.
- **CDR Detection & nADA Risk Check**: Identifies complementarity-determining regions (CDRs) and assesses overlapping epitopes to estimate neutralizing ADA risk.
- **Tolerance Analysis & AI Redesign**: Uses a JanusMatrix-like analysis to find Tregitopes and leverages **Claude 3.5 Sonnet** to automatically generate an executive risk assessment memo and redesign strategies.

## Setup and Installation

1. **Clone the repository and enter the directory:**
   ```bash
   cd SafeBind
   ```

2. **Set up a Python virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your API Keys (optional but highly recommended):**
   To fully utilize 3D structure folding and the AI Risk Memo generator, provide API keys either via environment variables or in `.streamlit/secrets.toml`:

   ```bash
   export TAMARIND_API_KEY="your-tamarind-bio-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

## Usage

Run the dashboard locally using Streamlit:

```bash
streamlit run app.py
```

1. Enter your multi-chain FASTA (e.g., Heavy and Light chains) or a raw sequence in the sidebar. **Note:** You can also use the integrated Drug Presets (like Odronextamab) for a quick demo.
2. Select the therapeutic features such as Modality, Route, and Disease.
3. Click **Analyze Risk**.
4. Browse the tabs to view the clinical benchmark score, explore the 3D structural risk viewer, review MHC-I (Cytotoxic) risks, and utilize the Candidate Downselection module to rank multiple variants.

## File Structure

- `app.py`: Main Streamlit web application.
- `safebind_composite_scorer.py`: Unifies disparate risk signals into a single global score.
- `safebind_mhc1_cytotoxic.py`: NetMHCpan-based evaluation mapping CD8+ cytotoxic T-cell risks.
- `safebind_downselect.py`: UI and logic for evaluating and ranking multiple sequences at once.
- `risk_model.py`: Historical ADA risk scoring engine and neutralizing ADA (nADA) estimators.
- `sequence_engine.py`: Sequence alignment, CDR detection, and IEDB epitope prediction calling.
- `deimmunize.py`: Tolerance analysis and deimmunization sequence redesigns incorporating structural constraints.
- `tamarind_integration.py`: Connects to Tamarind Bio for ESMFold protein structure generation.
- `claude_report.py`: Prompts Anthropic's Claude 3.5 Sonnet for detailed risk and executive memos.
- `precompute_mhc1.py` / `precompute_structures.py`: Utilities for pre-calculating and caching data prior to analysis.
