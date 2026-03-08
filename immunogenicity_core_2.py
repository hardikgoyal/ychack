"""
immunogenicity_core.py — Backend prediction pipeline for SafeBind AI
Calls IEDB APIs for T-cell and B-cell epitope prediction,
computes per-residue risk scores, fetches PDB structures,
and cross-references IDC DB V1 clinical data.
"""

import requests
import time
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ── HLA alleles (9 DRB1 alleles, ~85% global population coverage) ──
HLA_ALLELES = [
    "HLA-DRB1*01:01",
    "HLA-DRB1*03:01",
    "HLA-DRB1*04:01",
    "HLA-DRB1*07:01",
    "HLA-DRB1*08:01",
    "HLA-DRB1*11:01",
    "HLA-DRB1*13:01",
    "HLA-DRB1*15:01",
    "HLA-DRB1*04:04",
]


# ── Data classes ─────────────────────────────────────────────

@dataclass
class TCellEpitope:
    """A predicted T-cell epitope from IEDB MHC Class II."""
    allele: str
    start: int
    end: int
    sequence: str
    rank: float  # percentile rank; lower = stronger binder
    score: float = 0.0

@dataclass
class BCellEpitope:
    """A predicted B-cell epitope region."""
    start: int
    end: int
    sequence: str
    avg_score: float

@dataclass
class ResidueRisk:
    """Per-residue immunogenicity risk score."""
    position: int
    residue: str
    t_cell_risk: float
    b_cell_risk: float
    combined_risk: float
    num_alleles_binding: int

@dataclass
class AssessmentReport:
    """Full immunogenicity assessment report."""
    name: str
    sequence: str
    overall_risk_score: float
    risk_category: str  # LOW, MODERATE, HIGH, VERY HIGH
    t_cell_epitopes: List[TCellEpitope]
    b_cell_epitopes: List[BCellEpitope]
    residue_risks: List[ResidueRisk]
    hotspot_regions: List[Dict[str, Any]]
    comparable_therapeutics: List[Dict[str, Any]]
    pdb_data: Optional[str] = None  # PDB file content


# ── IEDB API calls ───────────────────────────────────────────

def predict_tcell_epitopes(sequence: str, alleles: List[str] = None, method: str = "recommended") -> List[TCellEpitope]:
    """Call IEDB MHC Class II binding prediction API."""
    if alleles is None:
        alleles = HLA_ALLELES

    epitopes = []
    url = "http://tools-cluster-interface.iedb.org/tools_api/mhcii/"

    for allele in alleles:
        try:
            data = {
                "method": method,
                "sequence_text": sequence,
                "allele": allele,
            }
            resp = requests.post(url, data=data, timeout=60)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                if len(lines) > 1:
                    header = lines[0].split("\t")
                    # Find column indices
                    try:
                        allele_idx = header.index("allele") if "allele" in header else 0
                        start_idx = header.index("start") if "start" in header else 3
                        end_idx = header.index("end") if "end" in header else 4
                        seq_idx = header.index("peptide") if "peptide" in header else 5
                        rank_idx = header.index("rank") if "rank" in header else len(header) - 1
                    except (ValueError, IndexError):
                        allele_idx, start_idx, end_idx, seq_idx, rank_idx = 0, 3, 4, 5, -1

                    for line in lines[1:]:
                        cols = line.split("\t")
                        if len(cols) > max(start_idx, end_idx, seq_idx, abs(rank_idx)):
                            try:
                                ep = TCellEpitope(
                                    allele=allele,
                                    start=int(cols[start_idx]),
                                    end=int(cols[end_idx]),
                                    sequence=cols[seq_idx],
                                    rank=float(cols[rank_idx]) if rank_idx >= 0 else float(cols[-1]),
                                )
                                epitopes.append(ep)
                            except (ValueError, IndexError):
                                continue
            # Small delay to be nice to IEDB servers
            time.sleep(0.3)
        except requests.exceptions.RequestException:
            continue

    return epitopes


def predict_bcell_epitopes(sequence: str) -> List[float]:
    """Call IEDB Bepipred Linear Epitope Prediction API.
    Returns per-residue scores.
    """
    url = "http://tools-cluster-interface.iedb.org/tools_api/bcell/"
    scores = [0.0] * len(sequence)

    try:
        data = {
            "method": "Bepipred",
            "sequence_text": sequence,
        }
        resp = requests.post(url, data=data, timeout=60)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            if len(lines) > 1:
                for line in lines[1:]:
                    cols = line.split("\t")
                    if len(cols) >= 3:
                        try:
                            pos = int(cols[0]) - 1  # 0-indexed
                            score = float(cols[2])
                            if 0 <= pos < len(sequence):
                                scores[pos] = score
                        except (ValueError, IndexError):
                            continue
    except requests.exceptions.RequestException:
        pass

    return scores


# ── Risk scoring ─────────────────────────────────────────────

def compute_residue_risks(sequence: str, t_cell_epitopes: List[TCellEpitope],
                          b_cell_scores: List[float]) -> List[ResidueRisk]:
    """Compute per-residue combined risk from T-cell and B-cell predictions."""
    n = len(sequence)

    # T-cell: for each residue, count how many alleles have strong binders overlapping it
    t_cell_counts = [0] * n  # number of alleles with rank < 10% at this position
    t_cell_best_rank = [100.0] * n  # best (lowest) rank at each position

    for ep in t_cell_epitopes:
        for pos in range(max(0, ep.start - 1), min(n, ep.end)):
            if ep.rank < t_cell_best_rank[pos]:
                t_cell_best_rank[pos] = ep.rank
            if ep.rank < 10:
                t_cell_counts[pos] += 1

    # Normalize T-cell risk: map rank to 0-1 (rank 0 = risk 1.0, rank 50+ = risk 0)
    t_cell_risk = []
    for i in range(n):
        rank = t_cell_best_rank[i]
        if rank >= 50:
            risk = 0.0
        else:
            risk = max(0, 1.0 - (rank / 50.0))
        t_cell_risk.append(risk)

    # Normalize B-cell scores to 0-1
    b_min = min(b_cell_scores) if b_cell_scores else 0
    b_max = max(b_cell_scores) if b_cell_scores else 1
    b_range = b_max - b_min if b_max > b_min else 1.0
    b_cell_risk = [(s - b_min) / b_range for s in b_cell_scores]

    # Combined risk: weighted average (T-cell 60%, B-cell 40%)
    residue_risks = []
    for i in range(n):
        combined = 0.6 * t_cell_risk[i] + 0.4 * b_cell_risk[i]
        residue_risks.append(ResidueRisk(
            position=i + 1,
            residue=sequence[i],
            t_cell_risk=t_cell_risk[i],
            b_cell_risk=b_cell_risk[i],
            combined_risk=combined,
            num_alleles_binding=t_cell_counts[i],
        ))

    return residue_risks


def identify_hotspots(residue_risks: List[ResidueRisk], threshold: float = 0.35,
                      min_length: int = 5) -> List[Dict[str, Any]]:
    """Identify contiguous regions of high immunogenicity risk."""
    hotspots = []
    current_start = None
    current_residues = []

    for rr in residue_risks:
        if rr.combined_risk >= threshold:
            if current_start is None:
                current_start = rr.position
            current_residues.append(rr)
        else:
            if current_residues and len(current_residues) >= min_length:
                hotspots.append(_make_hotspot(current_start, current_residues))
            current_start = None
            current_residues = []

    # Handle trailing hotspot
    if current_residues and len(current_residues) >= min_length:
        hotspots.append(_make_hotspot(current_start, current_residues))

    # Sort by average risk descending
    hotspots.sort(key=lambda h: h["avg_risk"], reverse=True)
    return hotspots


def _make_hotspot(start: int, residues: List[ResidueRisk]) -> Dict[str, Any]:
    return {
        "start": start,
        "end": residues[-1].position,
        "sequence": "".join(r.residue for r in residues),
        "length": len(residues),
        "avg_risk": sum(r.combined_risk for r in residues) / len(residues),
        "max_risk": max(r.combined_risk for r in residues),
        "avg_t_cell": sum(r.t_cell_risk for r in residues) / len(residues),
        "avg_b_cell": sum(r.b_cell_risk for r in residues) / len(residues),
    }


def categorize_risk(score: float) -> str:
    """Categorize overall risk score."""
    if score >= 0.6:
        return "VERY HIGH"
    elif score >= 0.4:
        return "HIGH"
    elif score >= 0.2:
        return "MODERATE"
    else:
        return "LOW"


# ── PDB fetching ─────────────────────────────────────────────

def fetch_pdb(pdb_id: str) -> Optional[str]:
    """Fetch PDB file content from RCSB."""
    if not pdb_id:
        return None
    try:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.text
    except requests.exceptions.RequestException:
        pass
    return None


# ── 3D Heatmap HTML generation ───────────────────────────────

def generate_3d_heatmap_html(pdb_data: str, residue_risks: List[ResidueRisk],
                              chain: str = "A", title: str = "Immunogenicity Heatmap") -> str:
    """Generate a standalone HTML page with py3Dmol visualization."""
    # Build a JS array of {resi, risk} for coloring
    risk_data = []
    for rr in residue_risks:
        risk_data.append({"resi": rr.position, "risk": round(rr.combined_risk, 3)})

    import json
    risk_json = json.dumps(risk_data)
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
    <style>
        body {{ margin:0; background:#0a0a0f; font-family:sans-serif; color:#e0e0e0; }}
        #viewer {{ width:100%; height:500px; position:relative; }}
        .controls {{ padding:8px; text-align:center; }}
        .controls button {{
            background:#1e1e2e; color:#888; border:1px solid #333;
            padding:6px 14px; border-radius:6px; cursor:pointer; margin:0 4px;
            font-size:12px;
        }}
        .controls button:hover {{ color:#e0e0e0; border-color:#555; }}
        .legend {{
            display:flex; justify-content:center; gap:4px; margin-top:8px;
            font-size:11px; color:#888;
        }}
        .legend span {{ padding:2px 8px; border-radius:3px; }}
    </style>
</head>
<body>
    <div id="viewer"></div>
    <div class="controls">
        <button onclick="showSurface()">Surface</button>
        <button onclick="showCartoon()">Cartoon</button>
        <button onclick="showStick()">Stick</button>
        <button onclick="viewer.zoomTo();viewer.render()">Reset View</button>
    </div>
    <div class="legend">
        <span style="background:#3b82f6;">Low</span>
        <span style="background:#22d3ee;">Moderate</span>
        <span style="background:#facc15;color:#000;">High</span>
        <span style="background:#f97316;">V.High</span>
        <span style="background:#ef4444;">Critical</span>
    </div>
    <script>
        var viewer = $3Dmol.createViewer("viewer", {{backgroundColor:"#0a0a0f"}});
        var pdbData = `{pdb_escaped}`;
        viewer.addModel(pdbData, "pdb");

        var riskData = {risk_json};
        var riskMap = {{}};
        riskData.forEach(function(d) {{ riskMap[d.resi] = d.risk; }});

        function riskToColor(risk) {{
            if (risk > 0.6) return "#ef4444";
            if (risk > 0.45) return "#f97316";
            if (risk > 0.3) return "#facc15";
            if (risk > 0.15) return "#22d3ee";
            return "#3b82f6";
        }}

        function colorByRisk(style) {{
            viewer.removeAllSurfaces();
            viewer.setStyle({{}}, {{}});
            if (style === "surface") {{
                viewer.setStyle({{chain:"{chain}"}}, {{cartoon:{{color:"#333"}}}});
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity:0.9,
                    colorfunc: function(atom) {{
                        var risk = riskMap[atom.resi] || 0;
                        var c = riskToColor(risk);
                        return c;
                    }}
                }}, {{chain:"{chain}"}});
            }} else if (style === "cartoon") {{
                viewer.setStyle({{chain:"{chain}"}}, {{
                    cartoon:{{
                        colorfunc: function(atom) {{
                            var risk = riskMap[atom.resi] || 0;
                            return riskToColor(risk);
                        }}
                    }}
                }});
            }} else {{
                viewer.setStyle({{chain:"{chain}"}}, {{
                    stick:{{
                        colorfunc: function(atom) {{
                            var risk = riskMap[atom.resi] || 0;
                            return riskToColor(risk);
                        }}
                    }}
                }});
            }}
            viewer.zoomTo({{chain:"{chain}"}});
            viewer.render();
        }}

        function showSurface() {{ colorByRisk("surface"); }}
        function showCartoon() {{ colorByRisk("cartoon"); }}
        function showStick() {{ colorByRisk("stick"); }}

        colorByRisk("surface");
    </script>
</body>
</html>"""
    return html


# ── IDC DB V1 integration ────────────────────────────────────

def load_idc_comparables(idc_path: str, species: str = None,
                         modality: str = None, top_n: int = 5) -> List[Dict[str, Any]]:
    """Load comparable therapeutics from IDC DB V1 Excel files.
    
    The IDC DB V1 is split across two files:
    - media-1.xlsx (Table S4): Multi-sheet workbook with Therapeutic, Sequence, Clinical Trial tables
    - media-2.xlsx (Table S5): Aggregated ADA frequencies with 'Prevalence of ADA+ patients'
    
    We need to join Therapeutic metadata (species, modality, target) from media-1 
    with ADA frequency data from media-2.
    """
    try:
        import pandas as pd
        
        # Find the data files - try multiple naming conventions
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        
        therapeutic_file = None
        clinical_file = None
        
        # Look for the files under various names
        therapeutic_names = ["media-1.xlsx", "idc_db_v1_table_s4.xlsx"]
        clinical_names = ["media-2.xlsx", "idc_db_v1_table_s5.xlsx"]
        search_dirs = [base_dir, cwd, os.path.join(base_dir, "data"), os.path.join(cwd, "data")]
        
        for d in search_dirs:
            for name in therapeutic_names:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    therapeutic_file = path
                    break
            if therapeutic_file:
                break
        
        for d in search_dirs:
            for name in clinical_names:
                path = os.path.join(d, name)
                if os.path.exists(path):
                    clinical_file = path
                    break
            if clinical_file:
                break
        
        # Also check the provided idc_path directly
        if not therapeutic_file and os.path.exists(idc_path):
            # Check if it's a multi-sheet workbook
            try:
                xl = pd.ExcelFile(idc_path)
                if "Therapeutic" in xl.sheet_names:
                    therapeutic_file = idc_path
                elif "in" in xl.sheet_names:
                    clinical_file = idc_path
            except Exception:
                pass
        
        if not therapeutic_file and not clinical_file:
            return _get_fallback_comparables(species, modality)
        
        # Load therapeutic metadata from media-1 (Table S4)
        therapeutics_df = None
        if therapeutic_file:
            try:
                therapeutics_df = pd.read_excel(therapeutic_file, sheet_name="Therapeutic", engine="openpyxl")
            except Exception:
                pass
        
        # Load clinical ADA data from media-2 (Table S5) or Clinical Trial sheet
        clinical_df = None
        if clinical_file:
            try:
                xl = pd.ExcelFile(clinical_file)
                # media-2 has sheets: 'Licensing', 'in'
                if "in" in xl.sheet_names:
                    clinical_df = pd.read_excel(clinical_file, sheet_name="in", engine="openpyxl")
            except Exception:
                pass
        
        if clinical_df is None and therapeutic_file:
            try:
                clinical_df = pd.read_excel(therapeutic_file, sheet_name="Clinical Trial", engine="openpyxl")
            except Exception:
                pass
        
        if clinical_df is None:
            return _get_fallback_comparables(species, modality)
        
        # Extract ADA frequency data from clinical trials
        name_col = "Therapeutic Assessed for ADA INN Name"
        ada_col = "Prevalence of ADA+ patients"
        
        if name_col not in clinical_df.columns or ada_col not in clinical_df.columns:
            return _get_fallback_comparables(species, modality)
        
        # Filter to therapeutic-exposed cohorts only
        if "Therapeutic Exposure Status" in clinical_df.columns:
            clinical_df = clinical_df[clinical_df["Therapeutic Exposure Status"] == "Therapeutic Exposed"]
        
        # Group by therapeutic and compute ADA stats
        grouped = clinical_df.groupby(name_col)
        results = []
        
        for name, group in grouped:
            if pd.isna(name):
                continue
            
            ada_vals = pd.to_numeric(group[ada_col], errors="coerce").dropna().tolist()
            if not ada_vals:
                continue
            
            # Get metadata from therapeutics table if available
            drug_species = "Unknown"
            drug_modality = "Unknown"
            drug_target = "Unknown"
            
            if therapeutics_df is not None:
                match = therapeutics_df[therapeutics_df["INN Name"] == name]
                if len(match) > 0:
                    row = match.iloc[0]
                    drug_species = str(row.get("Species", "Unknown")) if pd.notna(row.get("Species")) else "Unknown"
                    drug_modality = str(row.get("Protein Modality", "Unknown")) if pd.notna(row.get("Protein Modality")) else "Unknown"
                    drug_target = str(row.get("Target(s) (Protein/Molecule)", "Unknown")) if pd.notna(row.get("Target(s) (Protein/Molecule)")) else "Unknown"
            
            entry = {
                "name": str(name),
                "median_ada_freq": float(pd.Series(ada_vals).median()),
                "min_ada_freq": float(min(ada_vals)),
                "max_ada_freq": float(max(ada_vals)),
                "n_datapoints": len(ada_vals),
                "species": drug_species,
                "modality": drug_modality,
                "target": drug_target,
            }
            results.append(entry)
        
        # Sort by median ADA descending and return top N
        results.sort(key=lambda x: x["median_ada_freq"], reverse=True)
        return results[:top_n]

    except Exception:
        return _get_fallback_comparables(species, modality)


def _get_fallback_comparables(species: str = None, modality: str = None) -> List[Dict[str, Any]]:
    """Fallback comparable therapeutics when IDC DB isn't available."""
    return [
        {"name": "Adalimumab (Humira)", "median_ada_freq": 39.0, "min_ada_freq": 5.0,
         "max_ada_freq": 93.0, "n_datapoints": 48, "species": "Human",
         "modality": "mAb", "target": "TNF-alpha"},
        {"name": "Bococizumab", "median_ada_freq": 44.0, "min_ada_freq": 15.0,
         "max_ada_freq": 48.0, "n_datapoints": 6, "species": "Humanized",
         "modality": "mAb", "target": "PCSK9"},
        {"name": "Trastuzumab (Herceptin)", "median_ada_freq": 1.0, "min_ada_freq": 0.0,
         "max_ada_freq": 14.0, "n_datapoints": 12, "species": "Humanized",
         "modality": "mAb", "target": "HER2"},
        {"name": "Nivolumab (Opdivo)", "median_ada_freq": 12.0, "min_ada_freq": 3.0,
         "max_ada_freq": 26.0, "n_datapoints": 28, "species": "Human",
         "modality": "mAb", "target": "PD-1"},
        {"name": "Infliximab (Remicade)", "median_ada_freq": 28.0, "min_ada_freq": 7.0,
         "max_ada_freq": 61.0, "n_datapoints": 35, "species": "Chimeric",
         "modality": "mAb", "target": "TNF-alpha"},
    ]


# ── Tamarind Bio API integration ─────────────────────────────

TAMARIND_BASE_URL = "https://app.tamarind.bio/api/"


def submit_tamarind_structure(api_key: str, sequence: str, job_name: str) -> Optional[str]:
    """Submit a structure prediction job to Tamarind Bio.

    Tries ESMFold (fastest, seconds–minutes) first, then falls back to
    AlphaFold2 in single-sequence / 1-model mode.  Returns the job_name
    on success so the caller can poll for it, or None on failure.
    """
    if not api_key:
        return None

    headers = {"x-api-key": api_key}
    candidates = [
        ("esmfold",    {"sequence": sequence}),
        ("alphafold",  {"sequence": sequence, "useMSA": False,
                        "numModels": "1", "numRecycles": 1}),
    ]
    for tool_type, settings in candidates:
        try:
            payload = {"jobName": job_name, "type": tool_type, "settings": settings}
            resp = requests.post(
                TAMARIND_BASE_URL + "submit-job",
                headers=headers, json=payload, timeout=20
            )
            if resp.status_code == 200:
                return job_name
        except requests.exceptions.RequestException:
            continue
    return None


def get_tamarind_job_status(api_key: str, job_name: str) -> str:
    """Return 'running', 'complete', 'failed', or 'not_found'."""
    if not api_key:
        return "not_found"
    headers = {"x-api-key": api_key}
    try:
        resp = requests.get(TAMARIND_BASE_URL + "jobs", headers=headers, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            jobs = data if isinstance(data, list) else data.get("jobs", [])
            for job in jobs:
                name = (
                    job.get("jobName")
                    or job.get("name")
                    or job.get("job_name")
                    or ""
                )
                if name == job_name:
                    status = str(job.get("status", "")).lower()
                    if any(s in status for s in ["complete", "done", "success", "finish"]):
                        return "complete"
                    if any(s in status for s in ["fail", "error", "cancel"]):
                        return "failed"
                    return "running"
    except requests.exceptions.RequestException:
        pass
    return "not_found"


def fetch_tamarind_pdb(api_key: str, job_name: str) -> Optional[str]:
    """Download the PDB from a completed Tamarind structure prediction job.

    Tries the most common output filenames for ESMFold and AlphaFold2.
    """
    if not api_key:
        return None
    headers = {"x-api-key": api_key}
    candidate_files = [
        "esmfold.pdb",
        "output.pdb",
        "rank_001_alphafold2_ptm_model_1_seed_000.pdb",
        "rank_1.pdb",
        f"{job_name}.pdb",
        "prediction.pdb",
        "structure.pdb",
    ]
    for filename in candidate_files:
        try:
            resp = requests.post(
                TAMARIND_BASE_URL + "result",
                headers=headers,
                json={"jobName": job_name, "filename": filename},
                timeout=30,
            )
            if resp.status_code == 200 and len(resp.text) > 200:
                if any(k in resp.text for k in ("ATOM", "HETATM", "MODEL")):
                    return resp.text
        except requests.exceptions.RequestException:
            continue
    return None


# ── Summary generation ───────────────────────────────────────

def generate_risk_summary(report: AssessmentReport) -> str:
    """Generate a text summary of the risk assessment."""
    strong = len([e for e in report.t_cell_epitopes if e.rank < 10])
    return (
        f"{'='*60}\n"
        f"SafeBind AI — Immunogenicity Risk Assessment\n"
        f"{'='*60}\n"
        f"Therapeutic: {report.name}\n"
        f"Sequence length: {len(report.sequence)} aa\n"
        f"Overall risk: {report.overall_risk_score:.1%} ({report.risk_category})\n"
        f"Strong T-cell binders (rank <10%): {strong}\n"
        f"B-cell epitope regions: {len(report.b_cell_epitopes)}\n"
        f"Hotspot regions: {len(report.hotspot_regions)}\n"
        f"{'='*60}\n"
    )


def export_residue_csv(residue_risks: List[ResidueRisk]) -> str:
    """Export residue risks as CSV string."""
    lines = ["Position,Residue,T_cell_Risk,B_cell_Risk,Combined_Risk,Alleles_Binding"]
    for rr in residue_risks:
        lines.append(
            f"{rr.position},{rr.residue},{rr.t_cell_risk:.4f},"
            f"{rr.b_cell_risk:.4f},{rr.combined_risk:.4f},{rr.num_alleles_binding}"
        )
    return "\n".join(lines)


# ── Main assessment pipeline ─────────────────────────────────

def run_immunogenicity_assessment(
    sequence: str,
    name: str = "Query",
    pdb_id: str = None,
    pdb_chain: str = "A",
    idc_data_path: str = "idc_db_v1_table_s4.xlsx",
    species: str = "Humanized",
    modality: str = "Monoclonal antibody",
    verbose: bool = False,
) -> AssessmentReport:
    """Run the full immunogenicity assessment pipeline.

    1. Predict T-cell epitopes via IEDB MHC II API
    2. Predict B-cell epitopes via IEDB Bepipred API
    3. Compute per-residue risk scores
    4. Identify hotspot regions
    5. Fetch PDB structure (if available)
    6. Cross-reference IDC DB V1 for comparable therapeutics
    """

    # Step 1: T-cell epitopes
    if verbose:
        print(f"[1/5] Predicting T-cell epitopes for {name} ({len(sequence)} aa)...")
    t_cell_epitopes = predict_tcell_epitopes(sequence)

    # Step 2: B-cell epitopes
    if verbose:
        print(f"[2/5] Predicting B-cell epitopes...")
    b_cell_scores = predict_bcell_epitopes(sequence)

    # Build B-cell epitope regions (contiguous above-threshold)
    b_cell_epitopes = []
    b_threshold = 0.5
    # Normalize b_cell_scores for threshold comparison
    if b_cell_scores:
        b_max = max(b_cell_scores) if max(b_cell_scores) > 0 else 1.0
        b_min = min(b_cell_scores)
        b_range = b_max - b_min if b_max > b_min else 1.0
        b_norm = [(s - b_min) / b_range for s in b_cell_scores]
    else:
        b_norm = [0.0] * len(sequence)

    current_start = None
    current_scores = []
    for i, score in enumerate(b_norm):
        if score >= b_threshold:
            if current_start is None:
                current_start = i
            current_scores.append(score)
        else:
            if current_scores and len(current_scores) >= 3:
                b_cell_epitopes.append(BCellEpitope(
                    start=current_start + 1,
                    end=current_start + len(current_scores),
                    sequence=sequence[current_start:current_start + len(current_scores)],
                    avg_score=sum(current_scores) / len(current_scores),
                ))
            current_start = None
            current_scores = []
    if current_scores and len(current_scores) >= 3:
        b_cell_epitopes.append(BCellEpitope(
            start=current_start + 1,
            end=current_start + len(current_scores),
            sequence=sequence[current_start:current_start + len(current_scores)],
            avg_score=sum(current_scores) / len(current_scores),
        ))

    # Step 3: Per-residue risk
    if verbose:
        print(f"[3/5] Computing per-residue risk scores...")
    residue_risks = compute_residue_risks(sequence, t_cell_epitopes, b_cell_scores)

    # Step 4: Hotspots
    hotspot_regions = identify_hotspots(residue_risks)

    # Step 5: PDB structure
    pdb_data = None
    if pdb_id:
        if verbose:
            print(f"[4/5] Fetching PDB structure {pdb_id}...")
        pdb_data = fetch_pdb(pdb_id)

    # Step 6: IDC comparables
    if verbose:
        print(f"[5/5] Loading IDC DB V1 comparables...")
    comparables = load_idc_comparables(idc_data_path, species=species, modality=modality)

    # Overall risk score
    if residue_risks:
        overall = sum(rr.combined_risk for rr in residue_risks) / len(residue_risks)
    else:
        overall = 0.0

    risk_category = categorize_risk(overall)

    report = AssessmentReport(
        name=name,
        sequence=sequence,
        overall_risk_score=overall,
        risk_category=risk_category,
        t_cell_epitopes=t_cell_epitopes,
        b_cell_epitopes=b_cell_epitopes,
        residue_risks=residue_risks,
        hotspot_regions=hotspot_regions,
        comparable_therapeutics=comparables,
        pdb_data=pdb_data,
    )

    if verbose:
        print(generate_risk_summary(report))

    return report
