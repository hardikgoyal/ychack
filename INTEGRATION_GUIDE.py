"""
INTEGRATION GUIDE: Adding SafeBind Downselect™ to your existing app.py
=======================================================================

3 changes needed:
  1. Add the import (top of file)
  2. Add a new tab to the tabs list
  3. Render the tab content

Below is the exact code to add at each location.
"""


# ═══════════════════════════════════════════════════════════════
# CHANGE 1: Add import at the top of app.py (with your other imports)
# ═══════════════════════════════════════════════════════════════

# Add this line near your other safebind imports:

"""
from safebind_downselect import render_downselect_tab
"""


# ═══════════════════════════════════════════════════════════════
# CHANGE 2: Add "Downselect" to the tab list
# ═══════════════════════════════════════════════════════════════

# Find this line in render_results():
#
#   tab1, tab2, tab2b, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
#       "3D Heatmap", "T-cell Hotspots", "B-cell Epitopes", "Residue Plot",
#       "Clinical Context", "AI Report", "Tolerance", "Deimmunize",
#       "Cytotoxic (MHC-I)", "Composite Score", "Advanced"
#   ])
#
# Replace with:

"""
    tab1, tab2, tab2b, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab_ds = st.tabs([
        "3D Heatmap", "T-cell Hotspots", "B-cell Epitopes", "Residue Plot",
        "Clinical Context", "AI Report", "Tolerance", "Deimmunize",
        "Cytotoxic (MHC-I)", "Composite Score", "Advanced", "⚖️ Downselect"
    ])
"""


# ═══════════════════════════════════════════════════════════════
# CHANGE 3: Add the tab content (add after the tab10 block, before
#           the disclaimer div at the end of render_results)
# ═══════════════════════════════════════════════════════════════

"""
    # ── Tab Downselect: Batch comparison ──
    with tab_ds:
        render_downselect_tab(
            st_module=st,
            components_module=components,
            run_immunogenicity_fn=run_immunogenicity_assessment,
            run_tolerance_fn=run_tolerance_analysis,
            run_cytotoxic_fn=run_cytotoxic_assessment,  # or None to skip
            preloaded_sequences=PRELOADED,
            idc_data_path="idc_db_v1_table_s4.xlsx",
        )
"""


# ═══════════════════════════════════════════════════════════════
# OPTIONAL: Add as a standalone page instead of a tab
# ═══════════════════════════════════════════════════════════════
# If you'd prefer the Downselect as its own page (not nested inside
# the results view), you can add it to the main content area.
# Replace the landing state section with a page selector:

"""
# At the bottom of app.py, replace the landing state with:

if "report" in st.session_state:
    render_results(...)
elif not run_clicked:
    # Add a toggle between landing page and downselect mode
    mode = st.radio("", ["Single Analysis", "Batch Downselect"], horizontal=True, label_visibility="collapsed")
    
    if mode == "Batch Downselect":
        render_downselect_tab(
            st_module=st,
            components_module=components,
            run_immunogenicity_fn=run_immunogenicity_assessment,
            run_tolerance_fn=run_tolerance_analysis,
            run_cytotoxic_fn=run_cytotoxic_assessment,
            preloaded_sequences=PRELOADED,
            idc_data_path="idc_db_v1_table_s4.xlsx",
        )
    else:
        # ... existing landing page HTML ...
"""
