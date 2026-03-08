"""
immunogenicity_core.py — Core immunogenicity prediction engine for SafeBind AI
Bio x AI Hackathon @ YC HQ — March 8, 2026

Provides T-cell and B-cell epitope prediction, risk scoring, and 3D visualization.
Uses simulated predictions for hackathon demo (real IEDB API integration would be added for production).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import random
import math

# HLA-DRB1 alleles covering ~85% of global population
HLA_ALLELES = [
    "HLA-DRB1*01:01", "HLA-DRB1*03:01", "HLA-DRB1*04:01",
    "HLA-DRB1*07:01", "HLA-DRB1*08:01", "HLA-DRB1*11:01",
    "HLA-DRB1*13:01", "HLA-DRB1*15:01", "HLA-DRB1*04:05",
]

# Amino acid properties for prediction heuristics
HYDROPHOBIC = set("AILMFVPWG")
CHARGED = set("DEKRH")
POLAR = set("STYNQC")

# Known immunogenic motifs (simplified)
IMMUNOGENIC_MOTIFS = [
    "FGGGT", "WGQG", "YYCA", "YWGQ", "DYWG", "PLYA", "GFYA",
    "NPYL", "QQLK", "DAEF", "ERLK", "EDTS", "GNLG"
]


@dataclass
class TCellEpitope:
    """Predicted T-cell epitope (MHC-II binding peptide)."""
    start: int
    end: int
    sequence: str
    allele: str
    rank: float  # Percentile rank (lower = stronger binder)
    ic50: float  # Predicted IC50 in nM
    core: str    # 9-mer binding core


@dataclass
class BCellEpitope:
    """Predicted B-cell epitope (linear)."""
    start: int
    end: int
    sequence: str
    score: float  # 0-1 probability


@dataclass
class ResidueRisk:
    """Per-residue immunogenicity risk."""
    position: int
    residue: str
    t_cell_risk: float
    b_cell_risk: float
    combined_risk: float
    num_alleles_binding: int


@dataclass
class ImmunogenicityReport:
    """Complete immunogenicity assessment report."""
    name: str
    sequence: str
    overall_risk_score: float
    risk_category: str
    t_cell_epitopes: List[TCellEpitope]
    b_cell_epitopes: List[BCellEpitope]
    residue_risks: List[ResidueRisk]
    hotspot_regions: List[Dict[str, Any]]
    comparable_therapeutics: List[Dict[str, Any]]
    pdb_data: Optional[str] = None


def _compute_peptide_score(peptide: str, position: int) -> float:
    """Compute a heuristic immunogenicity score for a peptide."""
    score = 0.0
    
    # Hydrophobic anchor positions (typical for MHC-II)
    if len(peptide) >= 9:
        if peptide[0] in HYDROPHOBIC:
            score += 0.15
        if peptide[3] in HYDROPHOBIC:
            score += 0.1
        if peptide[6] in HYDROPHOBIC:
            score += 0.1
        if peptide[8] in HYDROPHOBIC:
            score += 0.15
    
    # Charged residues can be immunogenic
    charged_count = sum(1 for aa in peptide if aa in CHARGED)
    score += charged_count * 0.03
    
    # Known motifs
    for motif in IMMUNOGENIC_MOTIFS:
        if motif in peptide:
            score += 0.2
    
    # Sequence complexity
    unique_aa = len(set(peptide))
    if unique_aa >= 8:
        score += 0.1
    
    # Position-based (CDRs tend to be more immunogenic in antibodies)
    if 25 <= position <= 35 or 50 <= position <= 65 or 95 <= position <= 105:
        score += 0.15
    
    # Add some deterministic variation based on sequence
    hash_val = sum(ord(c) * (i + 1) for i, c in enumerate(peptide))
    score += (hash_val % 20) / 100
    
    return min(score, 1.0)


def predict_tcell_epitopes(sequence: str) -> List[TCellEpitope]:
    """Predict T-cell epitopes using MHC-II binding prediction (simulated)."""
    epitopes = []
    peptide_len = 15  # Standard MHC-II peptide length
    
    random.seed(hash(sequence) % 2**32)  # Deterministic based on sequence
    
    for i in range(len(sequence) - peptide_len + 1):
        peptide = sequence[i:i + peptide_len]
        base_score = _compute_peptide_score(peptide, i)
        
        # Check against each HLA allele
        for allele in HLA_ALLELES:
            # Allele-specific modifier
            allele_mod = (hash(allele + peptide) % 30) / 100
            score = base_score + allele_mod
            
            # Convert to percentile rank (lower = stronger)
            rank = max(1, 100 - score * 100)
            
            # Only keep strong/moderate binders
            if rank < 50:  # Top 50% are potential binders
                # Convert to IC50 estimate
                ic50 = 10 ** (1 + rank / 25)  # Rough conversion
                
                # Extract binding core (9-mer)
                core_start = (hash(peptide) % 7)
                core = peptide[core_start:core_start + 9]
                
                epitopes.append(TCellEpitope(
                    start=i + 1,
                    end=i + peptide_len,
                    sequence=peptide,
                    allele=allele,
                    rank=rank,
                    ic50=ic50,
                    core=core
                ))
    
    return epitopes


def predict_bcell_epitopes(sequence: str) -> List[BCellEpitope]:
    """Predict linear B-cell epitopes (simulated BepiPred-style)."""
    epitopes = []
    window = 12
    
    random.seed(hash(sequence[::-1]) % 2**32)
    
    for i in range(len(sequence) - window + 1):
        peptide = sequence[i:i + window]
        
        # B-cell epitopes favor: hydrophilic, surface-exposed, flexible regions
        score = 0.0
        
        # Hydrophilic residues
        hydrophilic = sum(1 for aa in peptide if aa in POLAR or aa in CHARGED)
        score += hydrophilic / window * 0.4
        
        # Proline/glycine (flexibility)
        flex = sum(1 for aa in peptide if aa in "PG")
        score += flex / window * 0.2
        
        # Avoid hydrophobic stretches
        hydrophobic = sum(1 for aa in peptide if aa in HYDROPHOBIC)
        score -= hydrophobic / window * 0.2
        
        # Add variation
        hash_val = sum(ord(c) * (i + 1) for i, c in enumerate(peptide))
        score += (hash_val % 25) / 100
        
        score = max(0, min(score, 1.0))
        
        if score > 0.4:
            epitopes.append(BCellEpitope(
                start=i + 1,
                end=i + window,
                sequence=peptide,
                score=score
            ))
    
    return epitopes


def compute_residue_risks(
    sequence: str,
    t_epitopes: List[TCellEpitope],
    b_epitopes: List[BCellEpitope]
) -> List[ResidueRisk]:
    """Compute per-residue immunogenicity risk scores."""
    n = len(sequence)
    t_risks = [0.0] * n
    b_risks = [0.0] * n
    allele_counts = [0] * n
    
    # Aggregate T-cell epitope contributions
    for ep in t_epitopes:
        weight = max(0, (50 - ep.rank) / 50)  # Stronger binders = higher weight
        for pos in range(ep.start - 1, ep.end):
            if 0 <= pos < n:
                t_risks[pos] += weight * 0.1
                allele_counts[pos] += 1
    
    # Aggregate B-cell epitope contributions
    for ep in b_epitopes:
        for pos in range(ep.start - 1, ep.end):
            if 0 <= pos < n:
                b_risks[pos] += ep.score * 0.15
    
    # Normalize and combine
    residue_risks = []
    for i in range(n):
        t_risk = min(t_risks[i], 1.0)
        b_risk = min(b_risks[i], 1.0)
        combined = 0.65 * t_risk + 0.35 * b_risk  # T-cell weighted more heavily
        
        residue_risks.append(ResidueRisk(
            position=i + 1,
            residue=sequence[i],
            t_cell_risk=t_risk,
            b_cell_risk=b_risk,
            combined_risk=min(combined, 1.0),
            num_alleles_binding=min(allele_counts[i], len(HLA_ALLELES))
        ))
    
    return residue_risks


def identify_hotspots(
    residue_risks: List[ResidueRisk],
    threshold: float = 0.3,
    min_length: int = 5
) -> List[Dict[str, Any]]:
    """Identify contiguous high-risk regions (hotspots)."""
    hotspots = []
    in_hotspot = False
    start = 0
    
    for i, rr in enumerate(residue_risks):
        if rr.combined_risk >= threshold:
            if not in_hotspot:
                in_hotspot = True
                start = i
        else:
            if in_hotspot:
                length = i - start
                if length >= min_length:
                    region_risks = residue_risks[start:i]
                    hotspots.append({
                        "start": start + 1,
                        "end": i,
                        "length": length,
                        "sequence": "".join(rr.residue for rr in region_risks),
                        "avg_risk": sum(rr.combined_risk for rr in region_risks) / length,
                        "max_risk": max(rr.combined_risk for rr in region_risks),
                        "avg_t_cell": sum(rr.t_cell_risk for rr in region_risks) / length,
                        "avg_b_cell": sum(rr.b_cell_risk for rr in region_risks) / length,
                    })
                in_hotspot = False
    
    # Handle hotspot at end
    if in_hotspot:
        length = len(residue_risks) - start
        if length >= min_length:
            region_risks = residue_risks[start:]
            hotspots.append({
                "start": start + 1,
                "end": len(residue_risks),
                "length": length,
                "sequence": "".join(rr.residue for rr in region_risks),
                "avg_risk": sum(rr.combined_risk for rr in region_risks) / length,
                "max_risk": max(rr.combined_risk for rr in region_risks),
                "avg_t_cell": sum(rr.t_cell_risk for rr in region_risks) / length,
                "avg_b_cell": sum(rr.b_cell_risk for rr in region_risks) / length,
            })
    
    # Sort by average risk
    hotspots.sort(key=lambda x: x["avg_risk"], reverse=True)
    return hotspots[:10]  # Top 10


def get_comparable_therapeutics(species: str, modality: str) -> List[Dict[str, Any]]:
    """Return comparable therapeutics from IDC DB V1 (simulated)."""
    # Real implementation would query the IDC DB V1 dataset
    comparables = [
        {
            "name": "Adalimumab (Humira)",
            "species": "Human",
            "modality": "Monoclonal antibody",
            "target": "TNF-alpha",
            "median_ada_freq": 44,
            "min_ada_freq": 30,
            "max_ada_freq": 93,
            "n_datapoints": 127,
        },
        {
            "name": "Bococizumab",
            "species": "Humanized",
            "modality": "Monoclonal antibody",
            "target": "PCSK9",
            "median_ada_freq": 44,
            "min_ada_freq": 15,
            "max_ada_freq": 48,
            "n_datapoints": 8,
        },
        {
            "name": "Trastuzumab (Herceptin)",
            "species": "Humanized",
            "modality": "Monoclonal antibody",
            "target": "HER2",
            "median_ada_freq": 5,
            "min_ada_freq": 0,
            "max_ada_freq": 14,
            "n_datapoints": 42,
        },
        {
            "name": "Nivolumab (Opdivo)",
            "species": "Human",
            "modality": "Monoclonal antibody",
            "target": "PD-1",
            "median_ada_freq": 12,
            "min_ada_freq": 11,
            "max_ada_freq": 26,
            "n_datapoints": 89,
        },
        {
            "name": "Infliximab (Remicade)",
            "species": "Chimeric",
            "modality": "Monoclonal antibody",
            "target": "TNF-alpha",
            "median_ada_freq": 52,
            "min_ada_freq": 10,
            "max_ada_freq": 90,
            "n_datapoints": 156,
        },
    ]
    
    # Filter by species if not viral/bacterial
    if species in ["Human", "Humanized", "Chimeric", "Mouse"]:
        return [c for c in comparables if c["species"] in [species, "Humanized", "Human"]]
    return comparables


def fetch_pdb_structure(pdb_id: str, chain: str = "A") -> Optional[str]:
    """Fetch PDB structure data (returns None for demo — real impl would call RCSB)."""
    # In production, this would fetch from RCSB PDB API
    return None


def run_immunogenicity_assessment(
    sequence: str,
    name: str = "Query",
    pdb_id: Optional[str] = None,
    pdb_chain: str = "A",
    idc_data_path: Optional[str] = None,
    species: str = "Humanized",
    modality: str = "Monoclonal antibody",
    verbose: bool = False
) -> ImmunogenicityReport:
    """Run complete immunogenicity assessment pipeline."""
    
    # Clean sequence
    sequence = "".join(c for c in sequence.upper() if c.isalpha())
    
    # Predict epitopes
    t_epitopes = predict_tcell_epitopes(sequence)
    b_epitopes = predict_bcell_epitopes(sequence)
    
    # Compute residue risks
    residue_risks = compute_residue_risks(sequence, t_epitopes, b_epitopes)
    
    # Identify hotspots
    hotspots = identify_hotspots(residue_risks)
    
    # Get comparables from IDC DB
    comparables = get_comparable_therapeutics(species, modality)
    
    # Compute overall risk score
    if residue_risks:
        avg_risk = sum(rr.combined_risk for rr in residue_risks) / len(residue_risks)
        max_risk = max(rr.combined_risk for rr in residue_risks)
        n_strong_binders = len([e for e in t_epitopes if e.rank < 10])
        
        # Weighted score
        overall = (
            0.4 * avg_risk +
            0.3 * max_risk +
            0.3 * min(n_strong_binders / 20, 1.0)
        )
    else:
        overall = 0.0
    
    # Categorize risk
    if overall < 0.2:
        category = "LOW"
    elif overall < 0.35:
        category = "MODERATE"
    elif overall < 0.5:
        category = "HIGH"
    else:
        category = "VERY HIGH"
    
    # Fetch PDB if provided
    pdb_data = None
    if pdb_id:
        pdb_data = fetch_pdb_structure(pdb_id, pdb_chain)
    
    return ImmunogenicityReport(
        name=name,
        sequence=sequence,
        overall_risk_score=overall,
        risk_category=category,
        t_cell_epitopes=t_epitopes,
        b_cell_epitopes=b_epitopes,
        residue_risks=residue_risks,
        hotspot_regions=hotspots,
        comparable_therapeutics=comparables,
        pdb_data=pdb_data,
    )


def generate_3d_heatmap_html(
    pdb_data: str,
    residue_risks: List[ResidueRisk],
    chain: str = "A",
    title: str = "Immunogenicity Heatmap"
) -> str:
    """Generate 3Dmol.js HTML for interactive heatmap visualization."""
    
    # Build color mapping
    colors = []
    for rr in residue_risks:
        risk = rr.combined_risk
        if risk > 0.5:
            color = "0xef4444"  # Red
        elif risk > 0.35:
            color = "0xf97316"  # Orange
        elif risk > 0.2:
            color = "0xfacc15"  # Yellow
        else:
            color = "0x3b82f6"  # Blue
        colors.append(f"{{resi: {rr.position}, color: {color}}}")
    
    color_script = ",\n".join(colors)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <style>
            body {{ margin: 0; background: #0a0a0f; }}
            #viewport {{ width: 100%; height: 600px; position: relative; }}
            .title {{ 
                position: absolute; top: 10px; left: 10px; z-index: 100;
                color: #e0e0e0; font-family: 'DM Sans', sans-serif; font-size: 14px;
                background: rgba(0,0,0,0.7); padding: 8px 12px; border-radius: 6px;
            }}
            .legend {{
                position: absolute; bottom: 10px; right: 10px; z-index: 100;
                background: rgba(0,0,0,0.7); padding: 8px 12px; border-radius: 6px;
                font-family: 'DM Sans', sans-serif; font-size: 11px; color: #888;
            }}
            .legend-item {{ display: flex; align-items: center; margin: 4px 0; }}
            .legend-color {{ width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; }}
        </style>
    </head>
    <body>
        <div id="viewport">
            <div class="title">{title}</div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background:#3b82f6;"></div>Low (&lt;20%)</div>
                <div class="legend-item"><div class="legend-color" style="background:#facc15;"></div>Moderate (20-35%)</div>
                <div class="legend-item"><div class="legend-color" style="background:#f97316;"></div>High (35-50%)</div>
                <div class="legend-item"><div class="legend-color" style="background:#ef4444;"></div>Very High (&gt;50%)</div>
            </div>
        </div>
        <script>
            let viewer = $3Dmol.createViewer("viewport", {{backgroundColor: "0x0a0a0f"}});
            let pdbData = `{pdb_data}`;
            viewer.addModel(pdbData, "pdb");
            viewer.setStyle({{}}, {{cartoon: {{color: "0x444444"}}}});
            
            // Apply risk colors
            let colorMap = [{color_script}];
            colorMap.forEach(c => {{
                viewer.setStyle({{resi: c.resi, chain: "{chain}"}}, {{cartoon: {{color: c.color}}}});
            }});
            
            viewer.zoomTo();
            viewer.render();
        </script>
    </body>
    </html>
    """
    return html


def generate_risk_summary(report: ImmunogenicityReport) -> str:
    """Generate a text summary of the risk assessment."""
    n_strong = len([e for e in report.t_cell_epitopes if e.rank < 10])
    
    summary = f"""
IMMUNOGENICITY RISK ASSESSMENT: {report.name}
{'=' * 50}

Overall Risk Score: {report.overall_risk_score:.1%}
Risk Category: {report.risk_category}

Sequence Length: {len(report.sequence)} amino acids
T-cell Epitopes (strong binders, <10% rank): {n_strong}
B-cell Epitope Regions: {len(report.b_cell_epitopes)}
Hotspot Regions: {len(report.hotspot_regions)}

TOP HOTSPOTS:
"""
    for i, hs in enumerate(report.hotspot_regions[:5], 1):
        summary += f"  {i}. Positions {hs['start']}-{hs['end']}: {hs['sequence'][:20]}... ({hs['avg_risk']:.0%} avg risk)\n"
    
    return summary


def export_residue_csv(residue_risks: List[ResidueRisk]) -> str:
    """Export residue risks to CSV format."""
    lines = ["Position,Residue,T-cell Risk,B-cell Risk,Combined Risk,Alleles Binding"]
    for rr in residue_risks:
        lines.append(f"{rr.position},{rr.residue},{rr.t_cell_risk:.4f},{rr.b_cell_risk:.4f},{rr.combined_risk:.4f},{rr.num_alleles_binding}")
    return "\n".join(lines)
