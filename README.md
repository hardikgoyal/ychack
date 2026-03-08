# SafeBind Risk 🧬

Hey there. If you're building protein therapeutics, you already know that immunogenicity is the silent killer of drug development programs. You spend millions getting a candidate through discovery and preclinical, only to have it fail in Phase I/II because patients develop anti-drug antibodies (ADAs). It tanks efficacy, causes adverse events, and potentially kills the entire asset.

That's exactly why we built **SafeBind Risk**.

SafeBind is a Streamlit dashboard that predicts the ADA risk of your biologics *before* you commit to costly trials. We look at empirical clinical data from over 200 approved drugs and 3,300+ patient cohorts to benchmark your candidate against historical realities, not just theoretical models.

## The Business Case (Why this matters)

Drug development has a horrific failure rate, and immunogenicity is one of the top reasons biologics fail in clinical stages. Fast-tracking a high-risk candidate is throwing good money after bad. The math is brutal:

- **Cost of failure:** A clinical-stage failure costs anywhere from $50M to $200M+ in sunk R&D costs.
- **Time lost:** 2-4 years of clinical development down the drain, giving competitors a massive edge.
- **The solution:** By screening sequences for T-cell/B-cell epitopes, analyzing surface exposure (SASA) via 3D folding, and benchmarking against real-world clinical benchmarks, SafeBind acts as a cheap, high-ROI insurance policy. We identify high-risk assets early so your team can either de-immunize them or kill them fast before they burn your runway and resources.

## What it actually does

- **Real-World Benchmarking:** Compare your candidate's risk profile against our internal database of 218 approved drugs, matched precisely by route, disease, and modality.
- **Sequence Analysis:** We align your sequence against known drugs to see if you're building on safe foundations or treading into risky territory.
- **Epitope Prediction (T-cell & B-cell):** Full, out-of-the-box integration with the IEDB API to predict MHC-II binders and linear B-cell epitopes.
- **Structural Risk (ESMFold):** Natively calls the Tamarind Bio API to fold your protein in real-time. This lets us see which epitopes are actually exposed on the surface (SASA) versus safely buried inside.
- **AI Redesign Copilot:** Analyzes neutralizing ADA (nADA) risks around CDRs, runs a JanusMatrix-style tolerance analysis to find Tregitopes, and leverages Claude 3.5 Sonnet to spit out an executive risk memo and actionable de-immunization strategies.

## Getting Started

1. **Clone & setup:**
   ```bash
   git clone <your-repo>
   cd SafeBind
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **API Keys (Highly Recommended):**
   To get the good stuff (3D folding and AI memos), drop your keys in `.streamlit/secrets.toml` or export them as environment variables:
   ```bash
   export TAMARIND_API_KEY="your-tamarind-bio-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

3. **Run it:**
   ```bash
   streamlit run app.py
   ```
   Drop in your multi-chain FASTA, pick your parameters, and hit the Analyze Risk button.

---

## What's inside the repo?
- `app.py`: The main Streamlit dashboard. Start here.
- `risk_model.py`: The brain of the operation. Handles the composite ADA scoring and nADA estimates based on the clinical benchmarking data.
- `sequence_engine.py`: Sequence alignment, IEDB calls, and CDR detection routines.
- `tamarind_integration.py`: Handles ESMFold calls via Tamarind Bio so we can get 3D structures on the fly.
- `deimmunize.py` & `claude_report.py`: The redesign tools and Anthropic integration for the executive memos.
- `data_loader.py`: Whips the clinical cohort CSVs into shape.
