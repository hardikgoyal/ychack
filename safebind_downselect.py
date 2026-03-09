"""
SafeBind Downselect™ — Batch Immunogenicity Comparison Engine
Inspired by EpiVax ISPRI Downselect™ quadrant analysis.

Compares multiple biologic sequences on two axes:
  X-axis: Humanness / Tolerance score (JanusMatrix-equivalent)
  Y-axis: Epitope density / T-cell risk (EpiMatrix-equivalent)

Generates a quadrant plot + ranked table for candidate downselection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import json


# ── Data classes ─────────────────────────────────────────────

@dataclass
class DownselectCandidate:
    """A single candidate in the downselect comparison."""
    name: str
    sequence: str
    species: str = "Humanized"

    # Computed scores (filled after analysis)
    epitope_density: float = 0.0        # Y-axis: 0-1, higher = more epitopes = worse
    humanness_score: float = 0.0        # X-axis: 0-1, higher = more human = better
    composite_score: float = 0.0        # Overall 0-100 risk
    risk_category: str = ""
    quadrant: str = ""                  # "Low Risk", "Monitor", "Engineer", "High Risk"

    # Detail metrics
    t_cell_epitopes: int = 0
    b_cell_epitopes: int = 0
    hotspot_count: int = 0
    treg_fraction: float = 0.0
    strong_binders: int = 0
    seq_length: int = 0

    # Tolerance breakdown
    effector_epitopes: int = 0
    treg_epitopes: int = 0
    tregitope_matches: int = 0

    # Optional: cytotoxic pathway
    cytotoxic_risk: float = 0.0
    cytotoxic_binders: int = 0

    # For the plot
    color: str = "#6b7280"
    label_offset: Tuple[float, float] = (0, 0)


@dataclass
class DownselectResult:
    """Full result of a downselect batch analysis."""
    candidates: List[DownselectCandidate]
    ranked: List[DownselectCandidate]         # sorted best → worst
    best_candidate: Optional[DownselectCandidate] = None
    worst_candidate: Optional[DownselectCandidate] = None
    quadrant_counts: Dict[str, int] = field(default_factory=dict)
    analysis_notes: List[str] = field(default_factory=list)


# ── Quadrant assignment ──────────────────────────────────────

def assign_quadrant(epitope_density: float, humanness_score: float) -> str:
    """
    Assign quadrant based on EpiVax-style axes:
      - High humanness + Low epitopes → "Low Risk" (bottom-right, best)
      - High humanness + High epitopes → "Monitor" (top-right)
      - Low humanness + Low epitopes → "Engineer" (bottom-left)
      - Low humanness + High epitopes → "High Risk" (top-left, worst)

    Thresholds calibrated to SafeBind's scoring:
      epitope_density midpoint: 0.35 (maps to ~35% of residues in epitopes)
      humanness_score midpoint: 0.50 (50% human-like TCR-face content)
    """
    epi_high = epitope_density > 0.35
    human_high = humanness_score > 0.50

    if human_high and not epi_high:
        return "Low Risk"
    elif human_high and epi_high:
        return "Monitor"
    elif not human_high and not epi_high:
        return "Engineer"
    else:
        return "High Risk"


def get_quadrant_color(quadrant: str) -> str:
    return {
        "Low Risk": "#059669",
        "Monitor": "#ca8a04",
        "Engineer": "#2563eb",
        "High Risk": "#dc2626",
    }.get(quadrant, "#6b7280")


# ── Core analysis function ───────────────────────────────────

def run_downselect_analysis(
    candidates: List[Dict],
    run_immunogenicity_fn,
    run_tolerance_fn,
    run_cytotoxic_fn=None,
    idc_data_path: str = "idc_db_v1_table_s4.xlsx",
    progress_callback=None,
    verbose: bool = False,
) -> DownselectResult:
    """
    Run batch immunogenicity analysis on multiple candidates.

    Args:
        candidates: List of dicts with keys: name, sequence, species
        run_immunogenicity_fn: The main run_immunogenicity_assessment function
        run_tolerance_fn: The run_tolerance_analysis function
        run_cytotoxic_fn: Optional run_cytotoxic_assessment function
        idc_data_path: Path to IDC DB V1 data
        progress_callback: Optional fn(step, total, message) for progress updates
        verbose: Print progress to console

    Returns:
        DownselectResult with all candidates scored and ranked.
    """
    results = []
    total = len(candidates)

    for i, cand in enumerate(candidates):
        name = cand["name"]
        sequence = cand["sequence"].upper().replace(" ", "").replace("\n", "")
        sequence = "".join(c for c in sequence if c.isalpha())
        species = cand.get("species", "Humanized")

        if progress_callback:
            progress_callback(i, total, f"Analyzing {name} ({len(sequence)} aa)...")

        if verbose:
            print(f"[{i+1}/{total}] Analyzing {name} ({len(sequence)} aa)")

        dc = DownselectCandidate(name=name, sequence=sequence, species=species, seq_length=len(sequence))

        # ── Step 1: Core immunogenicity assessment ──
        try:
            report = run_immunogenicity_fn(
                sequence=sequence,
                name=name,
                pdb_id=None,
                pdb_chain="A",
                idc_data_path=idc_data_path,
                species=species,
                modality="Monoclonal antibody",
                verbose=False,
            )

            dc.epitope_density = report.overall_risk_score
            dc.t_cell_epitopes = len([e for e in report.t_cell_epitopes if e.rank < 10])
            dc.strong_binders = dc.t_cell_epitopes
            dc.b_cell_epitopes = len(report.b_cell_epitopes)
            dc.hotspot_count = len(report.hotspot_regions)
            dc.risk_category = report.risk_category
        except Exception as e:
            if verbose:
                print(f"  ⚠ Core analysis failed for {name}: {e}")
            dc.epitope_density = 0.5  # fallback
            dc.risk_category = "ERROR"

        # ── Step 2: Tolerance / humanness analysis ──
        try:
            tol = run_tolerance_fn(
                sequence=sequence,
                t_cell_epitopes=report.t_cell_epitopes if 'report' in dir() else [],
                residue_risks=report.residue_risks if 'report' in dir() else [],
                overall_risk=dc.epitope_density,
            )

            dc.humanness_score = tol.treg_fraction  # Treg fraction ≈ humanness
            dc.treg_fraction = tol.treg_fraction
            dc.treg_epitopes = tol.putative_treg_epitopes
            dc.effector_epitopes = tol.putative_effector_epitopes
            dc.tregitope_matches = tol.tregitope_matches
        except Exception as e:
            if verbose:
                print(f"  ⚠ Tolerance analysis failed for {name}: {e}")
            dc.humanness_score = 0.5  # fallback

        # ── Step 3: Optional cytotoxic analysis ──
        if run_cytotoxic_fn:
            try:
                cyto = run_cytotoxic_fn(
                    sequence=sequence,
                    name=name,
                    serotype=None,
                    use_mhcflurry=True,
                    use_iedb=True,
                    verbose=False,
                )
                dc.cytotoxic_risk = cyto.overall_cytotoxic_risk
                dc.cytotoxic_binders = cyto.strong_binders
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Cytotoxic analysis failed for {name}: {e}")

        # ── Assign quadrant and color ──
        dc.quadrant = assign_quadrant(dc.epitope_density, dc.humanness_score)
        dc.color = get_quadrant_color(dc.quadrant)

        # ── Composite score (simple weighted) ──
        # Lower is better: high epitopes bad, low humanness bad
        dc.composite_score = (
            dc.epitope_density * 60          # 60% weight on epitope load
            + (1 - dc.humanness_score) * 30  # 30% weight on lack of humanness
            + dc.cytotoxic_risk * 10         # 10% weight on cytotoxic risk
        )

        results.append(dc)

    if progress_callback:
        progress_callback(total, total, "Ranking candidates...")

    # ── Rank: lower composite_score = better ──
    ranked = sorted(results, key=lambda c: c.composite_score)

    # ── Quadrant counts ──
    qcounts = {}
    for r in results:
        qcounts[r.quadrant] = qcounts.get(r.quadrant, 0) + 1

    # ── Notes ──
    notes = []
    low_risk = [r for r in results if r.quadrant == "Low Risk"]
    high_risk = [r for r in results if r.quadrant == "High Risk"]
    if low_risk:
        names = ", ".join(r.name for r in low_risk)
        notes.append(f"Low-risk candidates for advancement: {names}")
    if high_risk:
        names = ", ".join(r.name for r in high_risk)
        notes.append(f"High-risk candidates requiring engineering or de-selection: {names}")

    return DownselectResult(
        candidates=results,
        ranked=ranked,
        best_candidate=ranked[0] if ranked else None,
        worst_candidate=ranked[-1] if ranked else None,
        quadrant_counts=qcounts,
        analysis_notes=notes,
    )


# ── Quadrant plot HTML generator ─────────────────────────────

def generate_quadrant_plot_html(
    result: DownselectResult,
    width: int = 700,
    height: int = 520,
    title: str = "SafeBind Downselect™ — Immunogenicity Quadrant Plot",
) -> str:
    """
    Generate a standalone HTML/JS quadrant plot.
    X-axis: Humanness (Treg fraction) — higher = better
    Y-axis: Epitope Density — higher = worse
    """
    # Build data points JSON
    points = []
    for c in result.candidates:
        points.append({
            "name": c.name,
            "x": round(c.humanness_score * 100, 1),
            "y": round(c.epitope_density * 100, 1),
            "quadrant": c.quadrant,
            "color": c.color,
            "composite": round(c.composite_score, 1),
            "t_cell": c.t_cell_epitopes,
            "b_cell": c.b_cell_epitopes,
            "hotspots": c.hotspot_count,
            "treg": c.treg_epitopes,
            "effector": c.effector_epitopes,
            "seq_len": c.seq_length,
            "species": c.species,
            "cyto_risk": round(c.cytotoxic_risk * 100, 1),
        })
    data_json = json.dumps(points)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, sans-serif; background: #fff; }}
  .plot-container {{ position: relative; width: {width}px; height: {height}px; margin: 0 auto; }}
  
  .plot-area {{
    position: absolute; left: 60px; top: 40px;
    width: {width - 100}px; height: {height - 100}px;
    border: 1px solid #e5e7eb;
  }}
  
  /* Quadrant backgrounds */
  .q-topleft {{ position: absolute; left: 0; top: 0; width: 50%; height: 50%; background: #fef2f2; }}
  .q-topright {{ position: absolute; right: 0; top: 0; width: 50%; height: 50%; background: #fffbeb; }}
  .q-bottomleft {{ position: absolute; left: 0; bottom: 0; width: 50%; height: 50%; background: #eff6ff; }}
  .q-bottomright {{ position: absolute; right: 0; bottom: 0; width: 50%; height: 50%; background: #ecfdf5; }}
  
  /* Quadrant labels */
  .q-label {{
    position: absolute; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.7;
  }}
  .q-label-tl {{ top: 8px; left: 8px; color: #dc2626; }}
  .q-label-tr {{ top: 8px; right: 8px; color: #ca8a04; text-align: right; }}
  .q-label-bl {{ bottom: 8px; left: 8px; color: #2563eb; }}
  .q-label-br {{ bottom: 8px; right: 8px; color: #059669; text-align: right; }}
  
  /* Midlines */
  .midline-h, .midline-v {{
    position: absolute; background: #d1d5db;
  }}
  .midline-h {{ left: 0; width: 100%; height: 1px; top: 50%; }}
  .midline-v {{ top: 0; height: 100%; width: 1px; left: 50%; }}
  
  /* Data points */
  .point {{
    position: absolute; width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    cursor: pointer; transition: transform 0.15s, box-shadow 0.15s;
    z-index: 10;
  }}
  .point:hover {{
    transform: scale(1.5); box-shadow: 0 2px 8px rgba(0,0,0,0.3); z-index: 20;
  }}
  
  /* Point labels */
  .point-label {{
    position: absolute; font-size: 10px; font-weight: 500; color: #374151;
    white-space: nowrap; pointer-events: none; z-index: 5;
  }}
  
  /* Tooltip */
  .tooltip {{
    display: none; position: absolute; z-index: 100;
    background: #1f2937; color: #fff; border-radius: 8px;
    padding: 12px 16px; font-size: 12px; line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); max-width: 260px;
    pointer-events: none;
  }}
  .tooltip-name {{ font-weight: 600; font-size: 14px; margin-bottom: 6px; }}
  .tooltip-row {{ display: flex; justify-content: space-between; gap: 16px; }}
  .tooltip-label {{ color: #9ca3af; }}
  .tooltip-val {{ font-weight: 500; }}
  
  /* Axes */
  .axis-label {{
    position: absolute; font-size: 12px; font-weight: 600; color: #374151;
  }}
  .x-label {{ bottom: 4px; left: 50%; transform: translateX(-50%); }}
  .y-label {{
    left: 4px; top: 50%; transform: rotate(-90deg) translateX(-50%);
    transform-origin: left center;
  }}
  
  /* Title */
  .plot-title {{
    position: absolute; top: 8px; left: 60px;
    font-size: 14px; font-weight: 600; color: #111827;
  }}
  
  /* Tick marks */
  .tick {{ position: absolute; font-size: 10px; color: #9ca3af; }}
</style>
</head>
<body>
<div class="plot-container">
  <div class="plot-title">{title}</div>
  
  <div class="axis-label x-label">Humanness Score (Treg Fraction %) →</div>
  <div class="axis-label y-label">← Epitope Density (%)</div>
  
  <div class="plot-area" id="plotArea">
    <div class="q-topleft"></div>
    <div class="q-topright"></div>
    <div class="q-bottomleft"></div>
    <div class="q-bottomright"></div>
    
    <div class="midline-h"></div>
    <div class="midline-v"></div>
    
    <div class="q-label q-label-tl">High Risk</div>
    <div class="q-label q-label-tr">Monitor</div>
    <div class="q-label q-label-bl">Engineer</div>
    <div class="q-label q-label-br">Low Risk ✓</div>
  </div>
  
  <!-- Tick marks -->
  <div class="tick" style="left:58px;top:{height-55}px;text-align:right;">0%</div>
  <div class="tick" style="left:{60 + (width-100)//2 - 10}px;top:{height-55}px;">50%</div>
  <div class="tick" style="left:{width-50}px;top:{height-55}px;">100%</div>
  
  <div class="tick" style="left:30px;top:38px;">100%</div>
  <div class="tick" style="left:30px;top:{40 + (height-100)//2 - 5}px;">50%</div>
  <div class="tick" style="left:30px;top:{height-65}px;">0%</div>
  
  <div class="tooltip" id="tooltip"></div>
</div>

<script>
const data = {data_json};
const plotArea = document.getElementById('plotArea');
const tooltip = document.getElementById('tooltip');
const plotW = plotArea.offsetWidth;
const plotH = plotArea.offsetHeight;

data.forEach((d, i) => {{
  // X: humanness 0-100 maps to 0-plotW (left to right)
  // Y: epitope density 0-100 maps to plotH-0 (bottom to top, inverted)
  const px = (d.x / 100) * plotW;
  const py = plotH - (d.y / 100) * plotH;
  
  // Point
  const pt = document.createElement('div');
  pt.className = 'point';
  pt.style.left = (px - 7) + 'px';
  pt.style.top = (py - 7) + 'px';
  pt.style.background = d.color;
  
  // Tooltip
  pt.addEventListener('mouseenter', (e) => {{
    tooltip.innerHTML = `
      <div class="tooltip-name">${{d.name}}</div>
      <div class="tooltip-row"><span class="tooltip-label">Quadrant:</span><span class="tooltip-val" style="color:${{d.color}}">${{d.quadrant}}</span></div>
      <div class="tooltip-row"><span class="tooltip-label">Epitope Density:</span><span class="tooltip-val">${{d.y}}%</span></div>
      <div class="tooltip-row"><span class="tooltip-label">Humanness:</span><span class="tooltip-val">${{d.x}}%</span></div>
      <div class="tooltip-row"><span class="tooltip-label">T-cell Epitopes:</span><span class="tooltip-val">${{d.t_cell}}</span></div>
      <div class="tooltip-row"><span class="tooltip-label">Treg Epitopes:</span><span class="tooltip-val">${{d.treg}}</span></div>
      <div class="tooltip-row"><span class="tooltip-label">Composite Score:</span><span class="tooltip-val">${{d.composite}}/100</span></div>
      ${{d.cyto_risk > 0 ? '<div class="tooltip-row"><span class="tooltip-label">Cytotoxic Risk:</span><span class="tooltip-val">' + d.cyto_risk + '%</span></div>' : ''}}
    `;
    tooltip.style.display = 'block';
    
    // Position tooltip
    let tx = px + 70;  // offset from plot area left edge + axis margin
    let ty = py + 20;
    if (tx + 260 > plotW + 60) tx = px - 200;
    if (ty + 180 > plotH + 40) ty = py - 160;
    tooltip.style.left = tx + 'px';
    tooltip.style.top = ty + 'px';
  }});
  pt.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});
  plotArea.appendChild(pt);
  
  // Label
  const lbl = document.createElement('div');
  lbl.className = 'point-label';
  // Truncate long names
  const shortName = d.name.length > 18 ? d.name.substring(0, 16) + '…' : d.name;
  lbl.textContent = shortName;
  lbl.style.left = (px + 10) + 'px';
  lbl.style.top = (py - 6) + 'px';
  plotArea.appendChild(lbl);
}});
</script>
</body>
</html>"""
    return html


# ── Streamlit tab renderer ───────────────────────────────────

def render_downselect_tab(
    st_module,
    components_module,
    run_immunogenicity_fn,
    run_tolerance_fn,
    run_cytotoxic_fn=None,
    preloaded_sequences: dict = None,
    idc_data_path: str = "idc_db_v1_table_s4.xlsx",
):
    """
    Render the full Downselect tab inside a Streamlit app.
    
    Args:
        st_module: The `st` streamlit module
        components_module: streamlit.components.v1 module
        run_immunogenicity_fn: run_immunogenicity_assessment function
        run_tolerance_fn: run_tolerance_analysis function
        run_cytotoxic_fn: Optional run_cytotoxic_assessment function
        preloaded_sequences: Dict of preloaded sequences (from PRELOADED)
        idc_data_path: Path to IDC DB V1 data
    """
    st = st_module

    st.markdown("**SafeBind Downselect™ — Batch Immunogenicity Comparison**")
    st.caption(
        "Compare multiple candidates side-by-side on a quadrant plot. "
        "X-axis = humanness (tolerance), Y-axis = epitope density (risk). "
        "Inspired by EpiVax ISPRI Downselect™."
    )

    # ── Input mode ──
    input_mode = st.radio(
        "Input method",
        ["Select from preloaded", "Paste sequences (FASTA)"],
        horizontal=True,
    )

    candidates = []

    if input_mode == "Select from preloaded" and preloaded_sequences:
        # Filter out section headers and empty entries
        valid = {k: v for k, v in preloaded_sequences.items()
                 if v is not None and v.get("seq")}
        
        selected_names = st.multiselect(
            "Select candidates to compare (2–20)",
            list(valid.keys()),
            default=list(valid.keys())[:4],
            max_selections=20,
        )
        
        for name in selected_names:
            info = valid[name]
            short_name = name.split("(")[0].strip().split("/")[0].strip()
            candidates.append({
                "name": short_name,
                "sequence": info["seq"],
                "species": info.get("species", "Humanized"),
            })

    else:
        fasta_input = st.text_area(
            "Paste sequences in FASTA format",
            height=200,
            placeholder=""">Candidate_A
EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMH...
>Candidate_B
QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMH...""",
        )
        
        if fasta_input:
            # Parse FASTA
            current_name = None
            current_seq = []
            for line in fasta_input.strip().split("\n"):
                line = line.strip()
                if line.startswith(">"):
                    if current_name and current_seq:
                        candidates.append({
                            "name": current_name,
                            "sequence": "".join(current_seq),
                            "species": "Humanized",
                        })
                    current_name = line[1:].strip().split()[0]  # first word after >
                    current_seq = []
                elif line:
                    current_seq.append(line)
            if current_name and current_seq:
                candidates.append({
                    "name": current_name,
                    "sequence": "".join(current_seq),
                    "species": "Humanized",
                })

    # Show candidate count
    if candidates:
        st.info(f"**{len(candidates)} candidates** ready for analysis")

    # ── Run button ──
    include_cytotoxic = st.checkbox(
        "Include MHC-I cytotoxic pathway (slower but more complete)",
        value=False,
    )

    run_downselect = st.button(
        "🔬 Run Downselect Analysis",
        type="primary",
        use_container_width=True,
        disabled=len(candidates) < 2,
    )

    if len(candidates) < 2 and candidates:
        st.warning("Select at least 2 candidates for comparison.")

    # ── Run analysis ──
    ds_key = "downselect_result"

    if run_downselect and len(candidates) >= 2:
        progress_bar = st.progress(0)
        status = st.empty()

        def progress_cb(step, total, msg):
            progress_bar.progress(step / total if total > 0 else 0)
            status.markdown(f"**[{step}/{total}]** {msg}")

        cyto_fn = run_cytotoxic_fn if include_cytotoxic else None

        result = run_downselect_analysis(
            candidates=candidates,
            run_immunogenicity_fn=run_immunogenicity_fn,
            run_tolerance_fn=run_tolerance_fn,
            run_cytotoxic_fn=cyto_fn,
            idc_data_path=idc_data_path,
            progress_callback=progress_cb,
            verbose=True,
        )

        progress_bar.progress(1.0)
        status.markdown("**Analysis complete!**")
        st.session_state[ds_key] = result

    # ── Display results ──
    if ds_key in st.session_state:
        result = st.session_state[ds_key]

        st.markdown("---")

        # ── Quadrant summary ──
        qcols = st.columns(4)
        q_info = [
            ("Low Risk ✓", "risk-low", result.quadrant_counts.get("Low Risk", 0)),
            ("Monitor", "risk-moderate", result.quadrant_counts.get("Monitor", 0)),
            ("Engineer", "risk-moderate", result.quadrant_counts.get("Engineer", 0)),
            ("High Risk", "risk-high", result.quadrant_counts.get("High Risk", 0)),
        ]
        for col, (label, cls, count) in zip(qcols, q_info):
            with col:
                color = {"Low Risk ✓": "#059669", "Monitor": "#ca8a04",
                         "Engineer": "#2563eb", "High Risk": "#dc2626"}[label]
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{color};">{count}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Quadrant plot ──
        st.markdown("**Quadrant Plot** — hover over points for details")
        plot_html = generate_quadrant_plot_html(result)
        components_module.html(plot_html, height=540, scrolling=False)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Ranked table ──
        st.markdown("**Ranked Candidates** (best → worst)")
        for rank, c in enumerate(result.ranked, 1):
            q_color = get_quadrant_color(c.quadrant)
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
            
            # Risk bar width
            bar_width = min(100, c.composite_score * 1.5)
            bar_color = "#dc2626" if c.composite_score > 40 else "#ea580c" if c.composite_score > 25 else "#ca8a04" if c.composite_score > 15 else "#059669"

            st.markdown(f"""<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
                padding:14px 18px;margin-bottom:8px;border-left:4px solid {q_color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-size:18px;margin-right:8px;">{medal}</span>
                        <span style="font-weight:600;font-size:15px;color:#111827;">{c.name}</span>
                        <span style="background:{q_color};color:white;padding:2px 8px;border-radius:4px;
                            font-size:10px;font-weight:500;margin-left:8px;">{c.quadrant}</span>
                        <span style="color:#6b7280;font-size:12px;margin-left:8px;">
                            {c.species} · {c.seq_length} aa
                        </span>
                    </div>
                    <div style="font-weight:600;font-size:20px;color:{bar_color};">
                        {c.composite_score:.0f}<span style="font-size:12px;color:#6b7280;">/100</span>
                    </div>
                </div>
                <div style="background:#e5e7eb;border-radius:4px;height:6px;margin:10px 0 8px;">
                    <div style="background:{bar_color};width:{bar_width}%;height:100%;border-radius:4px;"></div>
                </div>
                <div style="display:flex;gap:24px;font-size:12px;color:#6b7280;">
                    <span>Epitopes: <b style="color:#ea580c;">{c.t_cell_epitopes}</b> T / <b style="color:#0891b2;">{c.b_cell_epitopes}</b> B</span>
                    <span>Hotspots: <b>{c.hotspot_count}</b></span>
                    <span>Treg: <b style="color:#059669;">{c.treg_epitopes}</b> · Effector: <b style="color:#dc2626;">{c.effector_epitopes}</b></span>
                    <span>Humanness: <b>{c.humanness_score:.0%}</b></span>
                    {"<span>Cyto risk: <b style='color:#7c3aed;'>" + f"{c.cytotoxic_risk:.0%}" + "</b></span>" if c.cytotoxic_risk > 0 else ""}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Analysis notes ──
        if result.analysis_notes:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Recommendations**")
            for note in result.analysis_notes:
                st.markdown(f"- {note}")

        # ── Download ──
        st.markdown("<br>", unsafe_allow_html=True)
        import csv
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "Rank", "Name", "Quadrant", "Composite_Score",
            "Epitope_Density", "Humanness_Score",
            "T_Cell_Epitopes", "B_Cell_Epitopes", "Hotspots",
            "Treg_Epitopes", "Effector_Epitopes", "Tregitope_Matches",
            "Cytotoxic_Risk", "Species", "Seq_Length",
        ])
        for rank, c in enumerate(result.ranked, 1):
            writer.writerow([
                rank, c.name, c.quadrant, f"{c.composite_score:.1f}",
                f"{c.epitope_density:.3f}", f"{c.humanness_score:.3f}",
                c.t_cell_epitopes, c.b_cell_epitopes, c.hotspot_count,
                c.treg_epitopes, c.effector_epitopes, c.tregitope_matches,
                f"{c.cytotoxic_risk:.3f}", c.species, c.seq_length,
            ])
        st.download_button(
            "📥 Download Downselect Report (CSV)",
            buf.getvalue(),
            file_name="safebind_downselect_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("How SafeBind Downselect™ works"):
            st.markdown("""
**Quadrant Plot Axes (inspired by EpiVax ISPRI):**
- **X-axis: Humanness Score** — fraction of T-cell epitopes with human-like TCR-facing residues 
  (JanusMatrix-equivalent). Higher = more tolerance expected.
- **Y-axis: Epitope Density** — fraction of residues in immunogenic MHC-II binding regions 
  (EpiMatrix-equivalent). Higher = more immunogenic.

**Quadrants:**
- **Low Risk** (bottom-right): High humanness + low epitopes → best candidates
- **Monitor** (top-right): High humanness + high epitopes → tolerance may compensate
- **Engineer** (bottom-left): Low humanness + low epitopes → consider humanization
- **High Risk** (top-left): Low humanness + high epitopes → de-select or heavy engineering

**Composite Score:** Weighted combination: epitope density (60%) + inverse humanness (30%) + cytotoxic risk (10%).
            """)
