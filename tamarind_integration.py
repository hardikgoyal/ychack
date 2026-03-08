"""SafeBind Risk — Tamarind Bio API integration for ESMFold + ProteinMPNN.

Requires TAMARIND_API_KEY environment variable.
Tamarind Bio API docs: https://docs.tamarind.bio
"""

import time
import requests
import streamlit as st

TAMARIND_BASE_URL = "https://api.tamarind.bio/v1"
POLL_INTERVAL = 5  # seconds
MAX_POLL_TIME = 300  # 5 minutes


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _submit_job(payload: dict, api_key: str) -> str:
    """Submit a job and return the job ID."""
    resp = requests.post(
        f"{TAMARIND_BASE_URL}/submit-job",
        json=payload,
        headers=_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["job_id"]


def _poll_job(job_id: str, api_key: str) -> dict:
    """Poll until job completes or times out."""
    elapsed = 0
    while elapsed < MAX_POLL_TIME:
        resp = requests.get(
            f"{TAMARIND_BASE_URL}/jobs/{job_id}",
            headers=_headers(api_key),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status == "completed":
            return data
        if status in ("failed", "error"):
            raise RuntimeError(f"Job {job_id} failed: {data.get('error', 'unknown')}")
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    raise TimeoutError(f"Job {job_id} timed out after {MAX_POLL_TIME}s")


def _get_result(job_id: str, api_key: str) -> str:
    """Download the result file (PDB string)."""
    resp = requests.get(
        f"{TAMARIND_BASE_URL}/jobs/{job_id}/result",
        headers=_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    # Result may be JSON with a file URL or direct PDB text
    content_type = resp.headers.get("Content-Type", "")
    if "application/json" in content_type:
        data = resp.json()
        # If it's a URL, fetch the PDB
        if "url" in data:
            pdb_resp = requests.get(data["url"], timeout=30)
            pdb_resp.raise_for_status()
            return pdb_resp.text
        elif "pdb" in data:
            return data["pdb"]
        elif "result" in data:
            return data["result"]
        return str(data)
    return resp.text


@st.cache_data(show_spinner="Folding protein with ESMFold...")
def fold_protein(sequence: str, api_key: str) -> str:
    """Fold a protein sequence using Tamarind Bio ESMFold API.

    Args:
        sequence: Amino acid sequence
        api_key: Tamarind Bio API key

    Returns:
        PDB format string
    """
    # Truncate very long sequences for ESMFold
    seq = sequence[:1000]

    job_id = _submit_job(
        {
            "job_type": "esmfold",
            "params": {
                "sequence": seq,
            },
        },
        api_key,
    )

    _poll_job(job_id, api_key)
    pdb_data = _get_result(job_id, api_key)
    return pdb_data


def suggest_redesigns(
    sequence: str,
    hotspot_positions: list[int],
    api_key: str,
    n_designs: int = 5,
) -> list[str]:
    """Use ProteinMPNN to suggest redesigned sequences at epitope hotspots.

    Args:
        sequence: Original amino acid sequence
        hotspot_positions: 1-indexed positions to allow mutations
        api_key: Tamarind Bio API key
        n_designs: Number of redesign variants to generate

    Returns:
        List of redesigned sequences
    """
    if not hotspot_positions:
        return []

    # Build fixed positions mask: 1 = fixed, 0 = designable
    fixed_mask = [1] * len(sequence)
    for pos in hotspot_positions:
        if 1 <= pos <= len(sequence):
            fixed_mask[pos - 1] = 0

    # First we need a PDB structure — fold the protein
    pdb_data = fold_protein(sequence, api_key)

    job_id = _submit_job(
        {
            "job_type": "proteinmpnn",
            "params": {
                "pdb": pdb_data,
                "fixed_positions": fixed_mask,
                "num_sequences": n_designs,
                "temperature": 0.1,
            },
        },
        api_key,
    )

    result = _poll_job(job_id, api_key)

    # Parse redesigned sequences from result
    result_data = _get_result(job_id, api_key)

    sequences = []
    try:
        import json
        parsed = json.loads(result_data)
        if isinstance(parsed, list):
            sequences = [s.get("sequence", s) if isinstance(s, dict) else str(s) for s in parsed]
        elif isinstance(parsed, dict) and "sequences" in parsed:
            sequences = parsed["sequences"]
    except (json.JSONDecodeError, TypeError):
        # Try parsing as FASTA
        current_seq = []
        for line in result_data.strip().split("\n"):
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.strip())
        if current_seq:
            sequences.append("".join(current_seq))

    return sequences[:n_designs]
