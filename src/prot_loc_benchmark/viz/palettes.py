"""ClinVar clinical significance color palettes and ordering."""

from __future__ import annotations

# ── ClinVar clinical significance palettes ────────────────────────────────
CLINVAR_PALETTE: dict[str, dict[str, str]] = {
    "clinvar_clnsig_clean": {
        "Pathogenic": "#CA7682",
        "Benign": "#1D7AAB",
        "Conflicting": "#505050",
        "VUS": "#A0A0A0",
        "Others": "#E0E0E0",
    },
    "clinvar_clnsig_clean_pp_strict": {
        "PLP": "#CA7682",
        "BLB": "#1D7AAB",
        "Pathogenic": "#CA7682",
        "Likely pathogenic": "#E6B1B8",
        "Benign": "#1D7AAB",
        "Likely benign": "#63A1C4",
        "Conflicting": "#505050",
        "VUS": "#A0A0A0",
        "Others": "#E0E0E0",
    },
}

CLINVAR_ORDER: dict[str, list[str]] = {
    "clinvar_clnsig_clean": ["Pathogenic", "Benign", "VUS", "Conflicting", "Others"],
    "clinvar_clnsig_clean_pp_strict": [
        "Pathogenic",
        "Likely pathogenic",
        "Benign",
        "Likely benign",
        "VUS",
        "Conflicting",
        "Others",
    ],
    # Collapsed PLP/BLB grouping (derived from pp_strict)
    "clinvar_plp_blb": ["PLP", "BLB", "VUS", "Conflicting", "Others"],
}
