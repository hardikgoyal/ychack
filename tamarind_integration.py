"""SafeBind Risk — Tamarind Bio API integration for ESMFold + ProteinMPNN.

Uses the Tamarind Bio REST API:
  - ESMFold: Protein structure prediction from sequence
  - ProteinMPNN: Sequence design given a structure (for deimmunization)

API docs: https://docs.tamarind.bio
Requires TAMARIND_API_KEY set in environment or Streamlit secrets.
"""

import io
import json
import time
import zipfile
import requests
import streamlit as st

BASE_URL = "https://app.tamarind.bio/api"
POLL_INTERVAL = 5  # seconds
MAX_POLL_TIME = 300  # 5 minutes


def _get_api_key():
    """Get Tamarind API key from env or Streamlit secrets."""
    import os
    key = os.environ.get("TAMARIND_API_KEY")
    if not key:
        try:
            key = st.secrets.get("TAMARIND_API_KEY")
        except Exception:
            pass
    return key


def _headers(api_key: str) -> dict:
    return {"x-api-key": api_key}


def _submit_job(job_name: str, job_type: str, settings: dict, api_key: str) -> str:
    """Submit a job and return the job name."""
    resp = requests.post(
        f"{BASE_URL}/submit-job",
        json={
            "jobName": job_name,
            "type": job_type,
            "settings": settings,
        },
        headers=_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    return job_name


def _poll_job(job_name: str, api_key: str) -> dict:
    """Poll until job completes or times out. Returns job info dict."""
    elapsed = 0
    while elapsed < MAX_POLL_TIME:
        resp = requests.get(
            f"{BASE_URL}/jobs",
            params={"jobName": job_name},
            headers=_headers(api_key),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("JobStatus", "")
        if status == "Complete":
            return data
        if status in ("Stopped", "Deleted"):
            error = data.get("error", "Job stopped or deleted")
            raise RuntimeError(f"Job {job_name} failed: {error}")

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise TimeoutError(f"Job {job_name} timed out after {MAX_POLL_TIME}s")


def _get_result_url(job_name: str, api_key: str, file_name: str | None = None) -> str:
    """Get presigned S3 URL for job results."""
    payload = {"jobName": job_name}
    if file_name:
        payload["fileName"] = file_name
    resp = requests.post(
        f"{BASE_URL}/result",
        json=payload,
        headers=_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    # Response is a presigned URL string or JSON with URL
    content_type = resp.headers.get("Content-Type", "")
    if "application/json" in content_type:
        data = resp.json()
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get("url", data.get("signedUrl", str(data)))
    return resp.text.strip().strip('"')


def _download_result(url: str) -> bytes:
    """Download file from presigned URL."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _upload_file(filename: str, content: bytes, api_key: str) -> str:
    """Upload a file to Tamarind. Returns the filename for use in job settings."""
    resp = requests.put(
        f"{BASE_URL}/upload/{filename}",
        data=content,
        headers={
            **_headers(api_key),
            "Content-Type": "application/octet-stream",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return filename


def _extract_pdb_from_result(result_bytes: bytes) -> str | None:
    """Extract PDB content from result (may be zip, raw PDB, or JSON)."""
    # Try as zip first
    try:
        with zipfile.ZipFile(io.BytesIO(result_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith(".pdb"):
                    return zf.read(name).decode("utf-8")
            # If no .pdb found, try first file
            if zf.namelist():
                return zf.read(zf.namelist()[0]).decode("utf-8")
    except zipfile.BadZipFile:
        pass

    # Try as raw text
    text = result_bytes.decode("utf-8", errors="ignore")
    if "ATOM" in text and ("END" in text or "TER" in text):
        return text

    # Try as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("pdb", "result", "structure"):
                if key in data:
                    return data[key]
    except (json.JSONDecodeError, TypeError):
        pass

    return text if text.strip() else None


def _make_job_name(prefix: str) -> str:
    """Generate a unique job name."""
    import hashlib
    ts = str(time.time()).encode()
    h = hashlib.md5(ts).hexdigest()[:8]
    return f"safebind-{prefix}-{h}"


@st.cache_data(show_spinner="Folding protein with ESMFold...", ttl=3600)
def fold_protein(sequence: str, api_key: str) -> str | None:
    """Fold a protein sequence using Tamarind Bio ESMFold.

    Args:
        sequence: Amino acid sequence (truncated to 1000 residues)
        api_key: Tamarind Bio API key

    Returns:
        PDB format string, or None on failure
    """
    seq = sequence[:1000]  # ESMFold limit
    job_name = _make_job_name("esmfold")

    try:
        _submit_job(job_name, "esmfold", {"sequence": seq}, api_key)
        _poll_job(job_name, api_key)
        url = _get_result_url(job_name, api_key)
        result_bytes = _download_result(url)
        pdb_data = _extract_pdb_from_result(result_bytes)
        return pdb_data
    except Exception as e:
        st.warning(f"ESMFold failed: {e}")
        return None


def suggest_redesigns(
    sequence: str,
    hotspot_positions: list[int],
    api_key: str,
    n_designs: int = 3,
    pdb_data: str | None = None,
) -> list[dict]:
    """Use ProteinMPNN to suggest redesigned sequences at epitope hotspots.

    Args:
        sequence: Original amino acid sequence
        hotspot_positions: 1-indexed positions to allow mutations
        api_key: Tamarind Bio API key
        n_designs: Number of redesign variants
        pdb_data: Pre-folded PDB string (optional, will fold if not provided)

    Returns:
        List of dicts with 'sequence' and 'score' keys
    """
    if not hotspot_positions:
        return []

    # Need a PDB structure first
    if not pdb_data:
        pdb_data = fold_protein(sequence, api_key)
    if not pdb_data:
        return []

    # Upload PDB file
    pdb_filename = _make_job_name("pdb") + ".pdb"
    try:
        _upload_file(pdb_filename, pdb_data.encode("utf-8"), api_key)
    except Exception as e:
        st.warning(f"PDB upload failed: {e}")
        return []

    # Build designedResidues — specify which residues to redesign on chain A
    # ProteinMPNN format: {"A": "26 27 28 29 30"}
    residue_str = " ".join(str(p) for p in sorted(hotspot_positions) if 1 <= p <= len(sequence))
    designed_residues = json.dumps({"A": residue_str})

    job_name = _make_job_name("mpnn")

    try:
        _submit_job(
            job_name,
            "proteinmpnn",
            {
                "pdbFile": pdb_filename,
                "designedResidues": designed_residues,
                "numSequences": n_designs,
                "temperature": 0.1,
                "modelType": "proteinmpnn",
                "omitAAs": "C",  # Avoid introducing new cysteines
            },
            api_key,
        )

        with st.spinner("Running ProteinMPNN sequence design..."):
            _poll_job(job_name, api_key)

        url = _get_result_url(job_name, api_key)
        result_bytes = _download_result(url)
        return _parse_mpnn_results(result_bytes, sequence)

    except Exception as e:
        st.warning(f"ProteinMPNN failed: {e}")
        return []


def _parse_mpnn_results(result_bytes: bytes, original_seq: str) -> list[dict]:
    """Parse ProteinMPNN output into list of redesigned sequences."""
    results = []

    # Try as zip
    try:
        with zipfile.ZipFile(io.BytesIO(result_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith(".fa") or name.endswith(".fasta"):
                    content = zf.read(name).decode("utf-8")
                    results.extend(_parse_fasta_results(content))
                elif name.endswith(".json"):
                    content = zf.read(name).decode("utf-8")
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "sequence" in item:
                                    results.append(item)
                    except json.JSONDecodeError:
                        pass
            if results:
                return results
    except zipfile.BadZipFile:
        pass

    # Try as raw text (FASTA)
    text = result_bytes.decode("utf-8", errors="ignore")
    results = _parse_fasta_results(text)
    if results:
        return results

    # Try as JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [{"sequence": s.get("sequence", str(s)), "score": s.get("score", 0)}
                    if isinstance(s, dict) else {"sequence": str(s), "score": 0}
                    for s in data]
        if isinstance(data, dict) and "sequences" in data:
            return [{"sequence": s, "score": 0} for s in data["sequences"]]
    except (json.JSONDecodeError, TypeError):
        pass

    return results


def _parse_fasta_results(text: str) -> list[dict]:
    """Parse FASTA-format output from ProteinMPNN."""
    results = []
    current_header = ""
    current_seq = []

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                seq = "".join(current_seq)
                score = _extract_score_from_header(current_header)
                results.append({"sequence": seq, "score": score, "header": current_header})
                current_seq = []
            current_header = line[1:]
        elif line:
            current_seq.append(line)

    if current_seq:
        seq = "".join(current_seq)
        score = _extract_score_from_header(current_header)
        results.append({"sequence": seq, "score": score, "header": current_header})

    # Skip the first entry if it's the original sequence
    if len(results) > 1 and results[0].get("header", "").startswith("T="):
        results = results[1:]

    return results


def _extract_score_from_header(header: str) -> float:
    """Extract score from ProteinMPNN FASTA header."""
    # Headers often look like: "T=0.1, sample=1, score=1.234, ..."
    import re
    match = re.search(r"score=([\d.]+)", header)
    if match:
        return float(match.group(1))
    match = re.search(r"global_score=([\d.]+)", header)
    if match:
        return float(match.group(1))
    return 0.0
