"""SafeBind Risk — Sequence alignment, epitope prediction, and structural analysis."""

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


def parse_fasta(text: str) -> str:
    """Parse raw sequence or FASTA format. Validate amino acid alphabet."""
    lines = text.strip().split("\n")
    seq_lines = [l.strip() for l in lines if not l.startswith(">")]
    seq = "".join(seq_lines)
    # Remove whitespace, numbers, dashes
    seq = re.sub(r"[\s\d\-]", "", seq).upper()
    # Validate
    invalid = set(seq) - AMINO_ACIDS
    if invalid:
        raise ValueError(f"Invalid amino acid characters: {', '.join(sorted(invalid))}")
    if len(seq) < 10:
        raise ValueError("Sequence too short (minimum 10 residues)")
    return seq


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


@st.cache_data(show_spinner="Predicting T-cell epitopes via IEDB...")
def predict_epitopes(sequence: str, timeout: int = 60) -> list:
    """Call IEDB MHC-II binding API for T-cell epitope prediction.

    Returns list of EpitopeResult for high-affinity binders.
    """
    # IEDB has a max sequence length; truncate if needed
    seq = sequence[:2000]

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
        resp.raise_for_status()
    except Exception as e:
        st.warning(f"IEDB API unavailable: {e}. Epitope prediction skipped.")
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
