"""SafeBind Risk — Sequence alignment, epitope prediction, and structural analysis."""
from __future__ import annotations

import re
import requests
import io
import csv
from collections import Counter
from dataclasses import dataclass, field

from Bio.Align import PairwiseAligner, substitution_matrices
import streamlit as st

from config import (
    AMINO_ACIDS, KMER_SIZE, KMER_PREFILTER_TOP_N, ALIGNMENT_TOP_K,
    IEDB_API_URL, IEDB_ALLELES, IEDB_EPITOPE_PERCENTILE_CUTOFF,
    IEDB_BCELL_API_URL, CDR_DEFINITIONS, CDR_MOTIFS,
)


@dataclass
class AlignmentResult:
    inn_name: str
    chain_descriptor: str
    pct_identity: float
    score: float
    therapeutic_id: str
    ref_sequence: str


@dataclass
class EpitopeResult:
    start: int
    end: int
    peptide: str
    percentile_rank: float
    allele: str


@dataclass
class BCellEpitope:
    """A predicted B-cell epitope region."""
    start: int
    end: int
    sequence: str
    avg_score: float
    surface_exposed: bool | None = None
    avg_sasa: float | None = None


def _clean_seq(seq: str) -> str:
    """Remove whitespace, numbers, dashes and validate."""
    seq = re.sub(r"[\s\d\-]", "", seq).upper()
    invalid = set(seq) - AMINO_ACIDS
    if invalid:
        raise ValueError(f"Invalid amino acid characters: {', '.join(sorted(invalid))}")
    if len(seq) < 10:
        raise ValueError(f"Sequence too short ({len(seq)} residues, minimum 10)")
    return seq


def parse_fasta(text: str) -> str:
    """Parse raw sequence or FASTA format (single chain). Validate amino acid alphabet."""
    lines = text.strip().split("\n")
    seq_lines = [l.strip() for l in lines if not l.startswith(">")]
    seq = "".join(seq_lines)
    return _clean_seq(seq)


def parse_multi_fasta(text: str) -> dict[str, str]:
    """Parse multi-chain FASTA input.

    Supports:
        - Multiple FASTA entries with >headers
        - Single raw sequence (returned as {"Chain 1": seq})

    Returns:
        dict of {chain_name: sequence}
    """
    text = text.strip()
    if not text:
        return {}

    lines = text.split("\n")

    # Check if any FASTA headers exist
    has_headers = any(l.strip().startswith(">") for l in lines)

    if not has_headers:
        # Single raw sequence
        seq = _clean_seq("".join(l.strip() for l in lines))
        return {"Chain 1": seq}

    # Parse multi-FASTA
    chains = {}
    current_name = None
    current_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            # Save previous chain
            if current_name and current_lines:
                chains[current_name] = _clean_seq("".join(current_lines))
            # Start new chain
            current_name = line[1:].strip()
            # Clean up common prefixes
            if not current_name:
                current_name = f"Chain {len(chains) + 1}"
            current_lines = []
        else:
            current_lines.append(line)

    # Save last chain
    if current_name and current_lines:
        chains[current_name] = _clean_seq("".join(current_lines))

    if not chains:
        raise ValueError("No valid sequences found in input")

    return chains


def _get_kmers(seq: str, k: int = KMER_SIZE) -> set:
    """Extract k-mers from a sequence."""
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


@st.cache_data(show_spinner="Aligning sequences...")
def align_to_references(query_seq: str, _ref_df, top_k: int = ALIGNMENT_TOP_K):
    """Align query to reference sequences using k-mer prefilter + pairwise alignment.

    Args:
        query_seq: Cleaned amino acid sequence
        _ref_df: DataFrame with columns [INN Name, Chain Descriptor, Amino Acid Sequence,
                  Parental Molecule Therapeutic ID]
        top_k: Number of top matches to return

    Returns:
        List of AlignmentResult
    """
    ref_df = _ref_df
    query_kmers = _get_kmers(query_seq)

    # Step 1: K-mer Jaccard pre-filter
    scores = []
    for idx, row in ref_df.iterrows():
        ref_seq = str(row["Amino Acid Sequence"]).strip()
        if len(ref_seq) < 10:
            continue
        ref_kmers = _get_kmers(ref_seq)
        jac = _jaccard(query_kmers, ref_kmers)
        scores.append((idx, jac, ref_seq, row))

    scores.sort(key=lambda x: x[1], reverse=True)
    candidates = scores[:KMER_PREFILTER_TOP_N]

    # Step 2: Full pairwise alignment on top candidates
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.mode = "local"

    results = []
    for idx, jac, ref_seq, row in candidates:
        try:
            alignment = aligner.align(query_seq, ref_seq)[0]
            score = alignment.score

            # Compute percent identity from alignment
            aligned_q = alignment[0]
            aligned_r = alignment[1]
            matches = sum(1 for a, b in zip(aligned_q, aligned_r) if a == b and a != "-")
            align_len = max(len(aligned_q), 1)
            pct_identity = matches / align_len

            results.append(AlignmentResult(
                inn_name=row["INN Name"],
                chain_descriptor=row["Chain Descriptor"],
                pct_identity=pct_identity,
                score=score,
                therapeutic_id=row["Parental Molecule Therapeutic ID"],
                ref_sequence=ref_seq,
            ))
        except Exception:
            continue

    results.sort(key=lambda x: x.score, reverse=True)

    # Deduplicate by INN name (keep best per drug)
    seen = set()
    deduped = []
    for r in results:
        if r.inn_name not in seen:
            seen.add(r.inn_name)
            deduped.append(r)
        if len(deduped) >= top_k:
            break

    return deduped


def get_sequence_diffs(query: str, ref_seq: str) -> list:
    """Find positions where query differs from reference.

    Returns list of (position, query_aa, ref_aa).
    """
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"

    try:
        alignment = aligner.align(query, ref_seq)[0]
        aligned_q = alignment[0]
        aligned_r = alignment[1]
    except Exception:
        return []

    diffs = []
    pos = 0
    for q_aa, r_aa in zip(aligned_q, aligned_r):
        pos += 1
        if q_aa != r_aa and q_aa != "-" and r_aa != "-":
            diffs.append((pos, q_aa, r_aa))

    return diffs


@st.cache_data(show_spinner="Predicting T-cell epitopes via IEDB...", ttl=3600)
def predict_epitopes(sequence: str, timeout: int = 120) -> list:
    """Call IEDB MHC-II binding API for T-cell epitope prediction.

    Returns list of EpitopeResult for high-affinity binders.
    Retries on 403 with exponential backoff.
    """
    seq = sequence[:2000]

    import time as _time
    resp = None
    last_err = None
    for attempt in range(5):
        try:
            resp = requests.post(
                IEDB_API_URL,
                data={
                    "method": "recommended",
                    "sequence_text": seq,
                    "allele": IEDB_ALLELES,
                    "length": "15",
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                break
            if resp.status_code == 403:
                last_err = f"403 Forbidden (attempt {attempt + 1}/5)"
                delay = 3 * (attempt + 1)
                _time.sleep(delay)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            _time.sleep(3)
            continue

    if resp is None or resp.status_code != 200:
        st.warning(f"IEDB T-cell API unavailable after 5 attempts: {last_err}. Epitope prediction skipped.")
        return []

    results = []
    try:
        reader = csv.reader(io.StringIO(resp.text), delimiter="\t")
        header = next(reader)
        # Find column indices
        col_map = {name.strip().lower(): i for i, name in enumerate(header)}
        start_idx = col_map.get("start", None)
        end_idx = col_map.get("end", None)
        peptide_idx = col_map.get("peptide", None)
        rank_idx = col_map.get("percentile_rank", col_map.get("rank", None))
        allele_idx = col_map.get("allele", None)

        if any(x is None for x in [start_idx, end_idx, peptide_idx, rank_idx]):
            return []

        for row in reader:
            if len(row) <= max(start_idx, end_idx, peptide_idx, rank_idx):
                continue
            try:
                rank = float(row[rank_idx])
                if rank < IEDB_EPITOPE_PERCENTILE_CUTOFF:
                    results.append(EpitopeResult(
                        start=int(row[start_idx]),
                        end=int(row[end_idx]),
                        peptide=row[peptide_idx],
                        percentile_rank=rank,
                        allele=row[allele_idx] if allele_idx is not None else "",
                    ))
            except (ValueError, IndexError):
                continue
    except Exception:
        pass

    # Sort by percentile rank (lower = stronger binder)
    results.sort(key=lambda x: x.percentile_rank)
    return results


def compute_epitope_density(epitope_list: list, seq_length: int) -> float:
    """Count of high-affinity epitopes per 100 residues."""
    if seq_length == 0:
        return 0.0
    return len(epitope_list) / seq_length * 100


# ── B-cell epitope prediction ──────────────────────────────────


def predict_bcell_epitopes(sequence: str, timeout: int = 60) -> list[BCellEpitope]:
    """Call IEDB Bepipred Linear Epitope API for B-cell epitope prediction.

    Returns list of BCellEpitope for regions with above-threshold scores.
    Retries on 403 (IEDB rate-limiting) with exponential backoff.
    """
    seq = sequence[:2000]

    import time as _time
    resp = None
    last_err = None
    for attempt in range(5):
        try:
            resp = requests.post(
                IEDB_BCELL_API_URL,
                data={
                    "method": "Bepipred",
                    "sequence_text": seq,
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                break
            if resp.status_code == 403:
                last_err = f"403 Forbidden (attempt {attempt + 1}/5)"
                delay = 3 * (attempt + 1)  # 3, 6, 9, 12, 15s
                _time.sleep(delay)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            _time.sleep(3)
            continue

    if resp is None or resp.status_code != 200:
        st.warning(f"IEDB B-cell API unavailable after 5 attempts: {last_err}. B-cell prediction skipped.")
        return []

    # Parse per-residue scores from the response
    per_residue_scores = [0.0] * len(seq)
    try:
        lines = resp.text.strip().split("\n")
        if len(lines) > 1:
            for line in lines[1:]:
                cols = line.split("\t")
                if len(cols) >= 3:
                    try:
                        pos = int(cols[0]) - 1  # Convert to 0-indexed
                        score = float(cols[2])
                        if 0 <= pos < len(seq):
                            per_residue_scores[pos] = score
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass

    # Identify contiguous epitope regions (score > 0.5 threshold)
    epitopes = []
    threshold = 0.5
    in_epitope = False
    ep_start = 0

    for i, score in enumerate(per_residue_scores):
        if score > threshold and not in_epitope:
            in_epitope = True
            ep_start = i
        elif score <= threshold and in_epitope:
            in_epitope = False
            ep_end = i - 1
            if ep_end - ep_start >= 4:  # Minimum 5 residues
                region_scores = per_residue_scores[ep_start:ep_end + 1]
                epitopes.append(BCellEpitope(
                    start=ep_start + 1,  # 1-indexed
                    end=ep_end + 1,
                    sequence=seq[ep_start:ep_end + 1],
                    avg_score=sum(region_scores) / len(region_scores),
                ))

    # Handle trailing epitope
    if in_epitope:
        ep_end = len(per_residue_scores) - 1
        if ep_end - ep_start >= 4:
            region_scores = per_residue_scores[ep_start:ep_end + 1]
            epitopes.append(BCellEpitope(
                start=ep_start + 1,
                end=ep_end + 1,
                sequence=seq[ep_start:ep_end + 1],
                avg_score=sum(region_scores) / len(region_scores),
            ))

    # Sort by average score descending (strongest epitopes first)
    epitopes.sort(key=lambda x: x.avg_score, reverse=True)
    return epitopes


# ── Solvent Accessibility (SASA) ────────────────────────────────


def calculate_sasa_from_pdb(pdb_data: str, chain_id: str = "A") -> dict[int, float]:
    """Calculate per-residue SASA approximation from PDB data.

    Uses a simplified neighbor-counting approach on CA atoms:
    residues with fewer neighbors within 10 Angstrom are more surface-exposed.

    Args:
        pdb_data: Raw PDB file content.
        chain_id: Chain to analyse (default "A").

    Returns:
        Dict of {residue_number: sasa_score} scaled 0-100.
    """
    if not pdb_data:
        return {}

    # Parse CA atoms from PDB
    ca_atoms: list[tuple[int, float, float, float]] = []
    try:
        for line in pdb_data.split("\n"):
            if line.startswith("ATOM") and " CA " in line:
                try:
                    atom_chain = line[21].strip()
                    if atom_chain != chain_id and chain_id != "":
                        continue
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    ca_atoms.append((resnum, x, y, z))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        st.warning(f"PDB parsing failed: {e}. SASA calculation skipped.")
        return {}

    if not ca_atoms:
        return {}

    # Neighbor counting: fewer neighbors within probe radius -> more exposed
    probe_radius = 10.0  # Angstrom
    max_neighbors = 15
    sasa_scores: dict[int, float] = {}

    for i, (resnum, x1, y1, z1) in enumerate(ca_atoms):
        neighbor_count = 0
        for j, (_, x2, y2, z2) in enumerate(ca_atoms):
            if i == j:
                continue
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
            if dist < probe_radius:
                neighbor_count += 1
        exposure = max(0, 1.0 - (neighbor_count / max_neighbors))
        sasa_scores[resnum] = exposure * 100  # Scale to 0-100 equivalent

    return sasa_scores


def is_surface_exposed(sasa_value: float, threshold: float = 25.0) -> bool:
    """Return True if a residue's SASA exceeds the exposure threshold."""
    return sasa_value >= threshold


def filter_surface_epitopes(
    epitopes: list[BCellEpitope],
    sasa_scores: dict[int, float],
    threshold: float = 25.0,
) -> list[BCellEpitope]:
    """Filter B-cell epitopes to only surface-exposed regions.

    An epitope is kept when >50 % of its residues are surface-exposed.
    """
    surface_epitopes: list[BCellEpitope] = []

    for epitope in epitopes:
        surface_count = 0
        total_sasa = 0.0
        valid_residues = 0

        for pos in range(epitope.start, epitope.end + 1):
            sasa = sasa_scores.get(pos, 0)
            total_sasa += sasa
            valid_residues += 1
            if is_surface_exposed(sasa, threshold):
                surface_count += 1

        if valid_residues > 0 and (surface_count / valid_residues) > 0.5:
            avg_sasa = total_sasa / valid_residues
            surface_epitopes.append(BCellEpitope(
                start=epitope.start,
                end=epitope.end,
                sequence=epitope.sequence,
                avg_score=epitope.avg_score,
                surface_exposed=True,
                avg_sasa=avg_sasa,
            ))

    return surface_epitopes


# ── CDR Detection ───────────────────────────────────────────────


def detect_cdr_regions(sequence: str, chain_type: str = "heavy") -> list[dict]:
    """Detect CDR regions in an antibody sequence.

    Uses a combination of:
      1. Kabat position-based detection (assuming standard numbering).
      2. Motif-based detection for CDR-H3 (most variable region).

    Args:
        sequence: Amino acid sequence of the antibody chain.
        chain_type: "heavy" or "light".

    Returns:
        List of dicts with keys: label, start, end, sequence, method.
    """
    cdrs: list[dict] = []
    seq_len = len(sequence)

    try:
        # Position-based detection
        definitions = CDR_DEFINITIONS.get(chain_type, CDR_DEFINITIONS["heavy"])
        for cdr_name, (start, end) in definitions.items():
            if start <= seq_len and end <= seq_len:
                cdr_seq = sequence[start - 1:end]  # Convert to 0-indexed
                cdrs.append({
                    "label": cdr_name,
                    "start": start,
                    "end": end,
                    "sequence": cdr_seq,
                    "method": "position",
                })

        # Motif-based detection for CDR-H3 (most important for nADA)
        for motif in CDR_MOTIFS.get("CDR-H3", []):
            idx = sequence.find(motif)
            if idx != -1:
                h3_start = idx + len(motif)
                for end_motif in ["WG", "FG"]:
                    end_idx = sequence.find(end_motif, h3_start)
                    if end_idx != -1 and end_idx - h3_start < 25:
                        cdrs.append({
                            "label": "CDR-H3",
                            "start": h3_start + 1,  # 1-indexed
                            "end": end_idx,
                            "sequence": sequence[h3_start:end_idx],
                            "method": "motif",
                        })
                        break

        # Deduplicate — prefer motif-detected over position-detected
        seen_labels: set[str] = set()
        unique_cdrs: list[dict] = []
        for cdr in sorted(cdrs, key=lambda x: (x["label"], x["method"] == "position")):
            if cdr["label"] not in seen_labels:
                seen_labels.add(cdr["label"])
                unique_cdrs.append(cdr)

        return unique_cdrs

    except Exception as e:
        st.warning(f"CDR detection failed: {e}.")
        return []


def check_cdr_epitope_overlap(
    cdr_regions: list[dict],
    epitope_start: int,
    epitope_end: int,
) -> dict | None:
    """Check if an epitope overlaps any CDR region.

    Returns overlap info dict with nADA risk level, or None if no overlap.
    CDR-H3 overlap is flagged as HIGH risk; other CDRs as MODERATE.
    """
    try:
        for cdr in cdr_regions:
            cdr_start, cdr_end = cdr["start"], cdr["end"]
            if not (epitope_end < cdr_start or epitope_start > cdr_end):
                overlap_start = max(epitope_start, cdr_start)
                overlap_end = min(epitope_end, cdr_end)
                return {
                    "cdr": cdr["label"],
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "overlap_length": overlap_end - overlap_start + 1,
                    "nada_risk": "HIGH" if cdr["label"] == "CDR-H3" else "MODERATE",
                }
    except Exception as e:
        st.warning(f"CDR overlap check failed: {e}.")
    return None
