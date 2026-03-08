"""
safebind_new_tabs.py — Integration code for app.py
====================================================

This file contains the exact code to add to app.py to integrate:
1. MHC-I / Cytotoxic T-cell tab
2. Composite Scoring Engine tab

INSTRUCTIONS:
1. Drop safebind_mhc1_cytotoxic.py and safebind_composite_scorer.py
   into same directory as app.py
2. Add imports (see STEP 1 below)
3. Update tabs list (see STEP 2)
4. Add the two new tab blocks (see STEP 3 and STEP 4)
"""

# ═══════════════════════════════════════════════════════════════
# STEP 1: Add these imports at the top of app.py (after existing imports)
# ═══════════════════════════════════════════════════════════════
IMPORTS = '''
from safebind_mhc1_cytotoxic import (
    run_cytotoxic_assessment,
    CytotoxicReport,
    HLA_CLASS_I_ALLELES,
    AAV_VALIDATED_EPITOPES_HUI_2015,
    AAV_IMMUNOPEPTIDOMICS_2023,
    GENER8_1_REFERENCE,
)

from safebind_composite_scorer import (
    compute_composite_score,
    CompositeScore,
)
'''

# ═══════════════════════════════════════════════════════════════
# STEP 2: Update the tabs list in render_results() 
# Change from 8 tabs to 10 tabs:
# ═══════════════════════════════════════════════════════════════
TABS_LINE = '''
    tab1, tab2, tab2b, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🧬 3D Heatmap", "🔥 T-cell Hotspots", "🧫 B-cell Epitopes", "📊 Residue Plot", 
        "🏥 Clinical Context", "🤖 AI Report", "🧪 Tolerance", "🔧 Deimmunize",
        "⚡ Cytotoxic (MHC-I)", "🎯 Composite Score"
    ])
'''

# ═══════════════════════════════════════════════════════════════
# STEP 3: Add Tab 8 (Cytotoxic MHC-I) — paste after tab7 block
# ═══════════════════════════════════════════════════════════════
TAB8_CODE = '''
    # ── Tab 8: MHC-I / Cytotoxic T-cell Analysis ──
    with tab8:
        st.markdown("**MHC Class I / CD8+ Cytotoxic T-cell Analysis**")
        st.caption(
            "Predicts where CD8+ cytotoxic T cells will attack — the pathway that causes "
            "liver toxicity in gene therapy and cell rejection in CAR-T. Uses IEDB NetMHCpan 4.1 "
            "(trained on 185,985 peptide-MHC-I pairs from TDC) across 12 HLA-A/B supertypes."
        )
        
        # Run cytotoxic assessment (cache in session state)
        cyto_key = f"cytotoxic_{seq_name}"
        if cyto_key not in st.session_state:
            # Determine serotype for AAV validation
            ctx = st.session_state.get("clinical_context", {})
            serotype = ctx.get("serotype")
            
            with st.spinner("Running MHC-I predictions across 12 HLA Class I alleles..."):
                st.session_state[cyto_key] = run_cytotoxic_assessment(
                    sequence=seq_clean,
                    name=seq_name or "Query",
                    serotype=serotype,
                    use_mhcflurry=True,  # Will gracefully skip if not installed
                    use_iedb=True,
                    verbose=False,
                )
        cyto = st.session_state[cyto_key]
        
        # ── Summary metrics ──
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        
        risk_color = "#dc2626" if cyto.overall_cytotoxic_risk > 0.4 else "#ea580c" if cyto.overall_cytotoxic_risk > 0.25 else "#ca8a04" if cyto.overall_cytotoxic_risk > 0.12 else "#0891b2"
        
        with cc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{risk_color};">{cyto.overall_cytotoxic_risk:.0%}</div>
                <div class="metric-label">Cytotoxic Risk</div>
            </div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#111827;">{cyto.risk_category}</div>
                <div class="metric-label">Risk Category</div>
            </div>""", unsafe_allow_html=True)
        with cc3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#dc2626;">{cyto.strong_binders}</div>
                <div class="metric-label">Strong Binders (&lt;2%)</div>
            </div>""", unsafe_allow_html=True)
        with cc4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#7c3aed;">{len(cyto.hotspot_regions)}</div>
                <div class="metric-label">CTL Hotspots</div>
            </div>""", unsafe_allow_html=True)
        with cc5:
            val_color = "#059669" if cyto.validated_hits > 0 else "#6b7280"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{val_color};">{cyto.validated_hits}</div>
                <div class="metric-label">Validated Hits</div>
            </div>""", unsafe_allow_html=True)
        
        # ── Prediction sources ──
        src_badges = " ".join(
            f'<span style="background:#eff6ff;color:#2563eb;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:500;">{s}</span>'
            for s in cyto.prediction_sources
        ) if cyto.prediction_sources else '<span style="color:#6b7280;font-size:12px;">API predictions pending — showing cached/mock data</span>'
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
        
        # ── AAV Validated Epitope Recovery (if applicable) ──
        if cyto.aav_epitope_recovery is not None and cyto.validated_details:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🧬 AAV Validated Epitope Cross-Reference**")
            st.caption(
                f"Cross-checked against {len(AAV_VALIDATED_EPITOPES_HUI_2015)} validated CD8+ epitopes "
                f"(Hui et al. 2015) + {len(AAV_IMMUNOPEPTIDOMICS_2023)} immunopeptidomics peptides (2023)."
            )
            
            # Recovery rate badge
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
            st.markdown(f"**⚡ Cytotoxic Hotspot Regions ({len(cyto.hotspot_regions)})**")
            st.caption("Regions with high MHC-I binding density — CD8+ T cells will target these areas")
            
            for i, hs in enumerate(cyto.hotspot_regions[:8], 1):
                bar_color = "#7c3aed" if hs['avg_risk'] > 0.4 else "#a855f7" if hs['avg_risk'] > 0.25 else "#c084fc"
                val_badge = ' <span style="background:#fef3c7;color:#92400e;padding:1px 5px;border-radius:3px;font-size:10px;">VALIDATED</span>' if hs.get("has_validated") else ""
                st.markdown(f"""<div class="hotspot-row" style="border-left:3px solid {bar_color};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-weight:600;font-size:15px;color:#111827;">CTL Hotspot {i}</span>
                            <span style="color:#6b7280;font-size:13px;margin-left:8px;">Pos {hs['start']}–{hs['end']} ({hs['length']} aa){val_badge}</span>
                        </div>
                        <div style="font-weight:600;font-size:18px;color:{bar_color};">{hs['avg_risk']:.0%}</div>
                    </div>
                    <div class="hotspot-seq" style="margin-top:8px;color:#7c3aed;">{hs['sequence']}</div>
                    <div style="display:flex;gap:24px;margin-top:8px;font-size:12px;color:#6b7280;">
                        <span>Max alleles: <b>{hs['max_alleles']}</b></span>
                        <span>Max risk: <b style="color:#7c3aed;">{hs['max_risk']:.0%}</b></span>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No significant MHC-I hotspot regions detected.")
        
        # ── Per-residue cytotoxic risk chart ──
        if cyto.residue_risks:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Per-residue MHC-I risk profile**")
            import pandas as pd
            cyto_df = pd.DataFrame([
                {"Position": rr.position, "MHC-I Risk": rr.mhc1_risk, "Alleles": rr.num_alleles_binding}
                for rr in cyto.residue_risks
            ])
            st.area_chart(cyto_df.set_index("Position")["MHC-I Risk"], color="#7c3aed", height=250)
        
        # ── Data references ──
        with st.expander("ℹ️ Data sources & methodology"):
            st.markdown("""
**MHC-I Prediction:** IEDB NetMHCpan 4.1 across 12 HLA-A/B supertypes covering ~95% of the global population. Training data: TDC MHC1_IEDB-IMGT_Nielsen dataset (185,985 peptide-MHC-I pairs across 150 HLA alleles, CC BY 4.0).

**Cross-check:** MHCflurry 2.0 (O'Donnell et al., Cell Systems 2020) — open-source local predictor integrating binding affinity, antigen processing, and presentation scores.

**AAV Validation:** Hui et al. 2015 (21 validated CD8+ epitopes, PMC4588448) + 2023 immunopeptidomics (65 HLA-I peptides from AAV capsids, PMC10469481). Clinical correlation: GENEr8-1 Phase 3 trial, 134 patients (PMID 38796703).

**Additional references:** TANTIGEN 2.0 (4,296 tumor antigens, Olsen et al. 2021), AbImmPred (199 therapeutic antibodies, 2024).

**Biology:** MHC-I presents intracellular peptides (8-11mers) to CD8+ cytotoxic T cells, triggering direct cell killing. This is the primary pathway driving hepatotoxicity in AAV gene therapy (AT132 patient deaths, Elevidys liver failures, Roctavian withdrawal).
            """)
'''

# ═══════════════════════════════════════════════════════════════
# STEP 4: Add Tab 9 (Composite Score) — paste after tab8 block
# ═══════════════════════════════════════════════════════════════
TAB9_CODE = '''
    # ── Tab 9: Composite Scoring Engine ──
    with tab9:
        st.markdown("**Composite Immunogenicity Scoring Engine**")
        st.caption(
            "Fuses 4 orthogonal signals: clinical benchmarking (3,334 datapoints), "
            "sequence similarity (222 references), dual-pathway epitope load (MHC-I + MHC-II), "
            "and AI synthesis — into a single clinically-meaningful risk prediction."
        )
        
        # Run composite scoring (cache)
        comp_key = f"composite_{seq_name}"
        if comp_key not in st.session_state:
            ctx = st.session_state.get("clinical_context", {})
            
            # Get MHC-I data if available
            cyto = st.session_state.get(f"cytotoxic_{seq_name}")
            mhc1_eps = cyto.strong_binders if cyto else 0
            mhc1_hs = len(cyto.hotspot_regions) if cyto else 0
            mhc1_risk = cyto.overall_cytotoxic_risk if cyto else 0.0
            
            # Get MHC-II data
            mhc2_eps = len([e for e in report.t_cell_epitopes if e.rank < 10])
            mhc2_hs = len(report.hotspot_regions)
            
            # Get tolerance data if available
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
        score_color = "#dc2626" if comp.composite_score > 60 else "#ea580c" if comp.composite_score > 40 else "#ca8a04" if comp.composite_score > 20 else "#0891b2"
        
        st.markdown(f"""<div style="background:linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);border:2px solid {score_color};border-radius:12px;padding:24px;text-align:center;margin-bottom:20px;">
            <div style="font-size:56px;font-weight:700;color:{score_color};">{comp.composite_score:.0f}<span style="font-size:24px;color:#6b7280;">/100</span></div>
            <div style="font-size:18px;font-weight:600;color:#111827;margin-top:4px;">{comp.composite_category}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:4px;">
                95% CI: [{comp.confidence_interval[0]:.0f}, {comp.confidence_interval[1]:.0f}] · 
                {comp.total_data_sources} data sources
            </div>
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
                icon = "🚨" if is_critical else "⚠️"
                st.markdown(f"""<div style="background:{bg};border:1px solid {border};border-radius:6px;padding:10px 14px;margin-bottom:6px;">
                    {icon} <span style="font-weight:600;font-size:12px;color:#374151;">{flag_name}</span>
                    <span style="font-size:12px;color:#6b7280;margin-left:6px;">{flag_desc}</span>
                </div>""", unsafe_allow_html=True)
        
        # ── Signal breakdown ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Signal Breakdown**")
        
        signals = [
            (comp.benchmark_signal, "#2563eb", "📊"),
            (comp.similarity_signal, "#059669", "🔗"),
            (comp.epitope_signal, "#dc2626", "🧬"),
        ]
        
        for sig, color, icon in signals:
            bar_width = min(100, sig.score)
            conf_pct = sig.confidence * 100
            st.markdown(f"""<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <div>
                        <span style="font-size:16px;">{icon}</span>
                        <span style="font-weight:600;font-size:14px;color:#111827;margin-left:6px;">
                            {sig.name}
                        </span>
                        <span style="font-size:11px;color:#6b7280;margin-left:8px;">
                            weight {sig.weight:.0%} · confidence {conf_pct:.0f}%
                        </span>
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
        st.markdown("**🤖 Signal 4: AI Synthesis**")
        
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
                    <div class="claude-badge">Composite Synthesis · Claude Sonnet 4.6</div>
                    <div>{synthesis_text}</div>
                </div>""", unsafe_allow_html=True)
            elif synthesis_text.startswith("Claude API error"):
                st.error(synthesis_text)
        else:
            st.info("Add an Anthropic API key to enable AI synthesis of the composite score.")
            with st.expander("View synthesis prompt"):
                st.code(comp.synthesis_prompt, language=None)
        
        # ── Data sources ──
        with st.expander("ℹ️ All data sources & references"):
            st.markdown(f"**{comp.total_data_sources} data sources referenced:**")
            for key, desc in comp.data_references.items():
                st.markdown(f"- **{key}**: {desc}")
            
            st.markdown("""
**Composite scoring methodology:**
The composite score fuses three quantitative signals weighted by confidence:
1. **Clinical Benchmark (30%):** Empirical ADA rates from IDC DB V1 for the drug's class/route/disease
2. **Sequence Similarity (20%):** 5-mer Jaccard similarity against 222 reference therapeutics with known outcomes
3. **Epitope Load (35%):** Combined MHC-II (ADA pathway, 70%) + MHC-I (cytotoxic pathway, 30%) epitope density vs benchmarks
4. **AI Synthesis (15%):** Claude integrates all signals, identifies conflicting evidence, and recommends next steps

Confidence intervals are computed from signal agreement — wide intervals indicate signals disagree (high uncertainty).
            """)
'''


# ═══════════════════════════════════════════════════════════════
# Print integration instructions
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("SafeBind AI — New Tab Integration Guide")
    print("=" * 60)
    print()
    print("FILES NEEDED (place in same directory as app.py):")
    print("  1. safebind_mhc1_cytotoxic.py      — MHC-I prediction module")
    print("  2. safebind_composite_scorer.py     — Composite scoring engine")
    print("  3. safebind_tolerance_deimmunization.py — (already have this)")
    print()
    print("CHANGES TO app.py:")
    print()
    print("STEP 1: Add imports at the top:")
    print(IMPORTS)
    print()
    print("STEP 2: Change tabs list to 10 tabs:")
    print(TABS_LINE)
    print()
    print("STEP 3: Add tab8 block (MHC-I) after tab7")
    print("STEP 4: Add tab9 block (Composite) after tab8")
    print()
    print("See TAB8_CODE and TAB9_CODE in this file for the full code.")
    print()
    print("✅ Integration guide complete!")
