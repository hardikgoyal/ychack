#!/usr/bin/env python3
"""
SafeBind AI walkthrough script - captures screenshots at each step.
Run: python safebind_walkthrough.py
Requires: pip install playwright && playwright install chromium
"""

import asyncio
import os
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing playwright...")
    os.system("pip install playwright")
    from playwright.async_api import async_playwright


SCREENSHOT_DIR = Path(__file__).parent / "walkthrough_screenshots"
BASE_URL = "http://localhost:8501"


async def main():
    Path(SCREENSHOT_DIR).mkdir(exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
            ignore_https_errors=True,
        )
        page = await context.new_page()
        
        try:
            # Step 1: Navigate and screenshot landing page
            print("Step 1: Navigating to landing page...")
            await page.goto(BASE_URL, wait_until="networkidle", timeout=15000)
            await asyncio.sleep(2)
            await page.screenshot(path=SCREENSHOT_DIR / "01_landing_page.png", full_page=True)
            print("  -> Saved 01_landing_page.png")
            
            # Step 2: Screenshot sidebar
            print("Step 2: Capturing sidebar...")
            sidebar = await page.locator('[data-testid="stSidebar"]')
            await sidebar.screenshot(path=SCREENSHOT_DIR / "02_sidebar.png")
            print("  -> Saved 02_sidebar.png")
            
            # Step 3: Select Bococizumab from dropdown
            print("Step 3: Selecting Bococizumab from dropdown...")
            selectbox = page.locator('[data-testid="stSelectbox"]').first
            await selectbox.click()
            await asyncio.sleep(0.5)
            # Select the option by text
            await page.get_by_role("option", name="Bococizumab (Pfizer — TERMINATED, 44% ADA)").click()
            await asyncio.sleep(1)
            
            # Step 4: Click Analyze Immunogenicity
            print("Step 4: Clicking Analyze Immunogenicity (waiting 30-60s for IEDB APIs)...")
            analyze_btn = page.get_by_role("button", name="🔬 Analyze Immunogenicity")
            await analyze_btn.click()
            
            # Wait for analysis to complete - look for metric cards or progress to finish
            await asyncio.sleep(5)
            for i in range(12):  # Up to 60 more seconds
                # Check if we have results (metric cards with "Overall Risk")
                if await page.locator("text=Overall Risk").count() > 0:
                    print(f"  -> Analysis complete after ~{5 + i*5}s")
                    break
                await asyncio.sleep(5)
            else:
                print("  -> Timeout waiting - taking screenshot anyway")
            
            await asyncio.sleep(2)
            
            # Step 5: Screenshot 4 metric cards
            print("Step 5: Capturing metric cards...")
            await page.screenshot(path=SCREENSHOT_DIR / "05_metric_cards.png", full_page=False)
            print("  -> Saved 05_metric_cards.png")
            
            # Step 6: Click Hotspots tab
            print("Step 6: Clicking Hotspots tab...")
            hotspots_tab = page.get_by_role("tab", name="🔥 Hotspots")
            await hotspots_tab.click()
            await asyncio.sleep(1.5)
            await page.screenshot(path=SCREENSHOT_DIR / "06_hotspots_tab.png", full_page=True)
            print("  -> Saved 06_hotspots_tab.png")
            
            # Step 7: Click Residue Plot tab
            print("Step 7: Clicking Residue Plot tab...")
            residue_tab = page.get_by_role("tab", name="📊 Residue Plot")
            await residue_tab.click()
            await asyncio.sleep(1.5)
            await page.screenshot(path=SCREENSHOT_DIR / "07_residue_plot_tab.png", full_page=True)
            print("  -> Saved 07_residue_plot_tab.png")
            
            # Step 8: Click Clinical Context tab
            print("Step 8: Clicking Clinical Context tab...")
            clinical_tab = page.get_by_role("tab", name="🏥 Clinical Context")
            await clinical_tab.click()
            await asyncio.sleep(1.5)
            await page.screenshot(path=SCREENSHOT_DIR / "08_clinical_context_tab.png", full_page=True)
            print("  -> Saved 08_clinical_context_tab.png")
            
            # Step 9: Go back to 3D Heatmap tab
            print("Step 9: Going back to 3D Heatmap tab...")
            heatmap_tab = page.get_by_role("tab", name="🧬 3D Heatmap")
            await heatmap_tab.click()
            await asyncio.sleep(2)
            await page.screenshot(path=SCREENSHOT_DIR / "09_3d_heatmap_tamarind.png", full_page=True)
            print("  -> Saved 09_3d_heatmap_tamarind.png")
            
        except Exception as e:
            print(f"Error: {e}")
            await page.screenshot(path=SCREENSHOT_DIR / "error_state.png", full_page=True)
            print("  -> Saved error_state.png")
        finally:
            await browser.close()
    
    print(f"\nDone! Screenshots saved to {SCREENSHOT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
