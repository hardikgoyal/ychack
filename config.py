"""SafeBind Risk — Configuration constants and risk maps."""

# Composite score weights
W_LOOKUP = 0.40
W_SEQUENCE = 0.35
W_FEATURE = 0.25

# Global baseline ADA rate (%)
GLOBAL_BASELINE_ADA = 12.0

# Species baseline ADA rates (%)
SPECIES_ADA = {
    "Bacterial": 64,
    "Chimeric": 15,
    "Human": 12,
    "Humanized": 10,
    "Mouse": 0.2,
    "Camelid": 20,
    "Rabbit": 25,
    "Goat": 30,
    "Horse": 30,
    "Guinea Pig": 25,
    "Shark": 15,
    "Viral": 20,
}

# Conjugate modification risk (%)
CONJUGATE_ADA = {
    "PEG Conjugate": 71,
    "Unconjugated": 12,
    "Drug Conjugate": 5,
    "Radioisotope Conjugate": 15,
    "Lipid Conjugate": 20,
    "Not Applicable": 12,
}

# Route of administration multipliers (relative to IV = 1.0)
ROUTE_MULTIPLIER = {
    "Subcutaneous": 1.6,
    "Intravenous": 1.0,
    "Intramuscular": 0.1,
    "Ophthalmic": 1.9,
    "Oral": 0.5,
    "Other": 1.0,
}

# Modality baseline ADA rates (%)
MODALITY_ADA = {
    "Immunotoxin": 52,
    "Enzyme": 49,
    "Single-Domain Antibody": 62,
    "Monoclonal Antibody": 12,
    "Antibody-drug Conjugate": 6,
    "Fc-Fusion": 6,
    "Antibody Fragments": 15,
    "Blood Clotting Factors": 30,
    "Cytokine": 35,
    "Hormone and Growth Factors": 20,
    "IgG-like Bispecific": 10,
    "Peptide": 40,
    "All Other Multispecifics": 15,
}

# Risk tier thresholds
RISK_TIERS = [
    (10, "Low", "#2ecc71"),
    (30, "Moderate", "#f39c12"),
    (60, "High", "#e74c3c"),
    (100, "Very High", "#8e44ad"),
]

# Dropdown options (from Controlled_Language.csv)
MODALITY_OPTIONS = [
    "Monoclonal Antibody",
    "Antibody-drug Conjugate",
    "Fc-Fusion",
    "Enzyme",
    "Antibody Fragments",
    "Blood Clotting Factors",
    "Cytokine",
    "Hormone and Growth Factors",
    "IgG-like Bispecific",
    "Immunotoxin",
    "Peptide",
    "Single-Domain Antibody",
    "All Other Multispecifics",
]

SPECIES_OPTIONS = [
    "Human",
    "Humanized",
    "Chimeric",
    "Bacterial",
    "Mouse",
    "Camelid",
    "Rabbit",
    "Goat",
    "Horse",
    "Guinea Pig",
    "Shark",
    "Viral",
]

ROUTE_OPTIONS = [
    "Subcutaneous",
    "Intravenous",
    "Intramuscular",
    "Ophthalmic",
    "Oral",
    "Other",
]

DISEASE_OPTIONS = [
    "Cancer and neoplasms",
    "Inflammation and autoimmunity",
    "Blood Disorders",
    "Cardiovascular",
    "Infectious diseases",
    "Metobolic and endocrine disorders",
    "Musculoskeletal",
    "Neurological",
    "Renal and urogenital",
    "Reproductive health",
    "Non-inflammatory respiratory diseases",
    "Non-inflammatory skin diseases",
    "Healthy Volunteer",
    "Unspecified",
]

CONJUGATE_OPTIONS = [
    "Unconjugated",
    "Drug Conjugate",
    "PEG Conjugate",
    "Radioisotope Conjugate",
    "Lipid Conjugate",
    "Not Applicable",
]

BACKBONE_OPTIONS = [
    "human IgG1",
    "Human IgG4",
    "human IgG2",
    "Human IgG2(CH1-hinge)/IgG4(CH2-CH3)",
    "Human Fab (CH1 IgG1)",
    "Murine IgG1",
    "Murine IgG2",
    "VHH",
    "ScFv",
    "Not Applicable",
]

# Sequence alignment parameters
MIN_IDENTITY_THRESHOLD = 0.40  # Below this, sequence match is not used
KMER_SIZE = 3
KMER_PREFILTER_TOP_N = 20
ALIGNMENT_TOP_K = 5

# IEDB API
IEDB_API_URL = "https://tools-cluster-interface.iedb.org/tools_api/mhcii/"
IEDB_ALLELES = "HLA-DRB1*01:01,HLA-DRB1*03:01,HLA-DRB1*07:01"
IEDB_EPITOPE_PERCENTILE_CUTOFF = 10  # Top binders below this percentile

# Valid amino acid alphabet
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWXY")
