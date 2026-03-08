"""SafeBind Risk — Data loading, normalization, and lookup table construction."""

import os
import pandas as pd
import streamlit as st

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_therapeutic():
    """Load and normalize the Therapeutic table (218 drugs)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "media-1__Therapeutic.csv"))
    # Drop empty trailing columns (24-38)
    df = df.iloc[:, :24]
    # Filter to rows with valid Therapeutic ID
    df = df[df["Therapeutic ID"].notna()].copy()
    # Normalize modality casing
    df["Protein Modality"] = df["Protein Modality"].str.strip().str.title()
    # Fix specific known values to match config
    modality_map = {
        "Antibody-Drug Conjugate": "Antibody-drug Conjugate",
        "Monoclonal Antibody": "Monoclonal Antibody",
        "Fc-Fusion": "Fc-Fusion",
        "Single-Domain Antibody": "Single-Domain Antibody",
        "Igg-Like Bispecific": "IgG-like Bispecific",
    }
    df["Protein Modality"] = df["Protein Modality"].replace(modality_map)
    # Normalize species
    df["Species"] = df["Species"].str.strip()
    return df


@st.cache_data
def load_sequences():
    """Load the Sequence table (222 rows)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "media-1__Sequence.csv"))
    df = df[["Sequence ID", "Parental Molecule Therapeutic ID", "INN Name",
             "Chain Multiplicity", "Chain Identifier", "Chain Descriptor",
             "Amino Acid Sequence"]].copy()
    df = df[df["Amino Acid Sequence"].notna()]
    return df


@st.cache_data
def load_clinical():
    """Load the clinical trial table (3,334 rows)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "media-2__in.csv"))
    # Convert key columns to numeric
    for col in ["Prevalence of ADA+ patients", "Number of Patients analyzed for ADA",
                 "INN_ADA", "N_of_INN_ADA", "cohort_ADA", "PRID_ADA"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def build_lookup_table():
    """Build hierarchical lookup tables for ADA prediction.

    Returns dict with keys:
        'route_disease_modality' — most specific
        'route_modality' — fallback
        'modality' — least specific
    """
    therapeutic = load_therapeutic()
    clinical = load_clinical()

    # Merge on Therapeutic ID
    merged = clinical.merge(
        therapeutic[["Therapeutic ID", "Protein Modality", "Species", "Conjugate Modification"]],
        left_on="Therapeutic Assessed for ADA ID",
        right_on="Therapeutic ID",
        how="left",
    )

    # Filter to Therapeutic Exposed rows with valid ADA data
    mask = (
        (merged["Therapeutic Exposure Status"] == "Therapeutic Exposed")
        & merged["Prevalence of ADA+ patients"].notna()
        & merged["Number of Patients analyzed for ADA"].notna()
        & (merged["Number of Patients analyzed for ADA"] > 0)
    )
    df = merged[mask].copy()

    route_col = "Therapeutic Route of Administration"
    disease_col = "Disease Indication Category"
    modality_col = "Protein Modality"

    def weighted_mean(group):
        weights = group["Number of Patients analyzed for ADA"]
        values = group["Prevalence of ADA+ patients"]
        total_w = weights.sum()
        if total_w == 0:
            return pd.Series({"weighted_ada": 0, "total_patients": 0, "n_cohorts": 0})
        return pd.Series({
            "weighted_ada": (values * weights).sum() / total_w,
            "total_patients": total_w,
            "n_cohorts": len(group),
        })

    # Level 1: Route + Disease + Modality
    l1 = df.groupby([route_col, disease_col, modality_col]).apply(
        weighted_mean, include_groups=False
    ).reset_index()

    # Level 2: Route + Modality
    l2 = df.groupby([route_col, modality_col]).apply(
        weighted_mean, include_groups=False
    ).reset_index()

    # Level 3: Modality only
    l3 = df.groupby([modality_col]).apply(
        weighted_mean, include_groups=False
    ).reset_index()

    return {"route_disease_modality": l1, "route_modality": l2, "modality": l3}


@st.cache_data
def build_drug_ada_map():
    """For drugs with both sequence + ADA data, compute per-INN median ADA.

    Returns dict of {inn_name: ada_pct}.
    """
    sequences = load_sequences()
    clinical = load_clinical()

    # Get unique INN names from sequences
    seq_inns = set(sequences["INN Name"].unique())

    # Get INN-level ADA from clinical data
    inn_ada = clinical[
        (clinical["INN_ADA"].notna())
        & (clinical["Therapeutic Exposure Status"] == "Therapeutic Exposed")
    ].drop_duplicates(subset=["INN_group_id"])[["Therapeutic Assessed for ADA INN Name", "INN_ADA"]]

    # Filter to drugs that have sequences
    inn_ada = inn_ada[inn_ada["Therapeutic Assessed for ADA INN Name"].isin(seq_inns)]

    result = {}
    for _, row in inn_ada.iterrows():
        name = row["Therapeutic Assessed for ADA INN Name"]
        ada = row["INN_ADA"]
        if pd.notna(ada):
            result[name] = float(ada)

    return result


@st.cache_data
def get_historical_precedents(route, disease, modality, top_n=10):
    """Get closest historical drugs for benchmarking display."""
    therapeutic = load_therapeutic()
    clinical = load_clinical()

    merged = clinical.merge(
        therapeutic[["Therapeutic ID", "Protein Modality", "Species"]],
        left_on="Therapeutic Assessed for ADA ID",
        right_on="Therapeutic ID",
        how="left",
    )

    mask = (
        (merged["Therapeutic Exposure Status"] == "Therapeutic Exposed")
        & merged["Prevalence of ADA+ patients"].notna()
        & (merged["Number of Patients analyzed for ADA"] > 0)
    )
    df = merged[mask].copy()

    route_col = "Therapeutic Route of Administration"
    disease_col = "Disease Indication Category"
    modality_col = "Protein Modality"

    # Score relevance: exact match on each dimension
    df["relevance"] = (
        (df[route_col] == route).astype(int)
        + (df[disease_col] == disease).astype(int)
        + (df[modality_col] == modality).astype(int)
    )

    df = df.sort_values("relevance", ascending=False)

    # Group by INN name, take the weighted mean
    grouped = (
        df.head(200)
        .groupby("Therapeutic Assessed for ADA INN Name")
        .agg(
            ada_pct=("Prevalence of ADA+ patients", "mean"),
            total_patients=("Number of Patients analyzed for ADA", "sum"),
            n_cohorts=("Prevalence of ADA+ patients", "count"),
            relevance=("relevance", "max"),
            route=(route_col, "first"),
            disease=(disease_col, "first"),
            modality=(modality_col, "first"),
        )
        .reset_index()
        .sort_values("relevance", ascending=False)
        .head(top_n)
    )

    return grouped
