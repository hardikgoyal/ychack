"""
Quick end-to-end test of the Tamarind Bio API integration.
Usage:
    python3 test_tamarind.py <YOUR_TAMARIND_API_KEY>

Tests:
  1. Submit ESMFold job for a short test sequence
  2. Poll until complete (or timeout)
  3. Download + validate PDB
"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from immunogenicity_core_2 import (
    submit_tamarind_structure,
    get_tamarind_job_status,
    fetch_tamarind_pdb,
    TAMARIND_BASE_URL,
)

# Short VH domain — Trastuzumab CDR loop region (fast to fold)
TEST_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
TEST_JOB = f"safebind_test_{int(time.time())}"


def run_test(api_key: str):
    print(f"\n{'='*60}")
    print("SafeBind AI — Tamarind Bio API Integration Test")
    print(f"{'='*60}")
    print(f"API base:  {TAMARIND_BASE_URL}")
    print(f"Job name:  {TEST_JOB}")
    print(f"Sequence:  {TEST_SEQ[:40]}... ({len(TEST_SEQ)} aa)")

    # ── Step 1: Submit ───────────────────────────────────────
    print("\n[1/3] Submitting structure prediction job...")
    job = submit_tamarind_structure(api_key, TEST_SEQ, TEST_JOB)
    if not job:
        print("  ✗ Submission failed — check your API key and network.")
        return False
    print(f"  ✓ Job submitted: {job}")

    # ── Step 2: Poll ─────────────────────────────────────────
    print("\n[2/3] Polling for completion (max 5 min)...")
    max_polls = 30
    for i in range(max_polls):
        status = get_tamarind_job_status(api_key, TEST_JOB)
        elapsed = (i + 1) * 10
        print(f"  [{elapsed:3d}s] status = {status}")
        if status == "complete":
            print("  ✓ Job complete!")
            break
        elif status == "failed":
            print("  ✗ Job failed on Tamarind side.")
            return False
        elif i == max_polls - 1:
            print("  ✗ Timed out after 5 min.")
            return False
        time.sleep(10)

    # ── Step 3: Fetch PDB ─────────────────────────────────────
    print("\n[3/3] Downloading PDB result...")
    pdb = fetch_tamarind_pdb(api_key, TEST_JOB)
    if not pdb:
        print("  ✗ PDB download failed — filename mismatch?")
        return False

    atom_lines = [l for l in pdb.splitlines() if l.startswith("ATOM")]
    print(f"  ✓ PDB received: {len(pdb):,} chars, {len(atom_lines):,} ATOM records")

    out_path = f"/Users/kleung/hackathon/{TEST_JOB}.pdb"
    with open(out_path, "w") as f:
        f.write(pdb)
    print(f"  ✓ Saved to: {out_path}")

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED ✓")
    print(f"{'='*60}\n")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_tamarind.py <TAMARIND_API_KEY>")
        sys.exit(1)
    success = run_test(sys.argv[1])
    sys.exit(0 if success else 1)
