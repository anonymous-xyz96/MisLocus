"""Pathogenicity-predictor annotations for per-allele benchmarks.

Provides a uniform loader for ClinVar / AlphaMissense / ESM-1b /
REVEL / MutPred predictor annotations. ``scripts/11_summarize_across_reps.py``
imports ``PREDICTORS`` to label alleles in cross-rep correlation plots.

Each predictor exposes two label columns:

* ``{predictor}_label``        - coarse split (Pathogenic / Benign / ...)
* ``{predictor}_label_strict`` - fine split when the predictor supports
                                 it (e.g. AlphaMissense P/LP/LB/B/A,
                                 ClinVar pp_strict). Equals the coarse
                                 column when no fine split exists
                                 (e.g. ESM1b's D/T).

Supported predictors:

* ``clinvar``       - ClinVar clinical-significance, harmonized into
                      ``clinvar_clnsig_clean`` and
                      ``clinvar_clnsig_clean_pp_strict`` upstream.
* ``alphamissense`` - dbNSFP AlphaMissense_pred. P/LP map to Pathogenic
                      and B/LB map to Benign in the coarse view; the
                      strict view keeps all five classes.
* ``esm1b``         - dbNSFP ESM1b_pred. D (Damaging) → Pathogenic,
                      T (Tolerated) → Benign. Coarse == strict.
* ``revel``         - dbNSFP REVEL_score (continuous, 0–1). Coarse split
                      at 0.5; strict tiers use the ClinGen 2022
                      calibration (≥0.773 P, ≥0.644 LP, ≤0.290 LB,
                      ≤0.183 B, intermediate → Ambiguous).
* ``mutpred``       - dbNSFP MutPred_score (continuous, 0–1). Coarse
                      split at 0.5; strict tiers ≥0.75 P, ≥0.5 LP,
                      ≥0.25 LB, <0.25 B.
* ``eve``           - NOT in the allele collection. Stub raises so
                      ``--predictor eve`` fails loudly until annotations
                      are added (drop a parquet at ``data/raw/eve/`` and
                      fill in ``_load_eve``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from prot_loc_benchmark.config import ALLELE_COLLECTION_PATH

log = logging.getLogger(__name__)

# dbNSFP missing-value sentinels (treated alongside pl null)
_DBNSFP_NA = ("", ".")


@dataclass(frozen=True)
class PredictorTest:
    """Single Wilcoxon comparison spec."""
    name: str           # e.g. "Pathogenic_vs_Benign"
    column: str         # which label column to filter on
    group_a: str        # numerator group (typically "Pathogenic")
    group_b: str        # denominator group (typically "Benign")


@dataclass(frozen=True)
class PredictorConfig:
    name: str                                   # CLI key
    title: str                                  # human-readable name for plot titles
    label_col: str                              # coarse label column on the joined df
    label_strict_col: str                       # strict label column on the joined df
    palette: dict[str, dict[str, str]]          # column → {category: hex}
    order: dict[str, list[str]]                 # column → list[category] (display order)
    tests: tuple[PredictorTest, ...]            # Wilcoxon comparisons to run


# Shared Pathogenic/Benign palette so violins / heatmaps look identical
# across predictors. Categories not present for a given predictor are
# simply unused.
_PB_PALETTE: dict[str, str] = {
    "Pathogenic":         "#CA7682",
    "Likely pathogenic":  "#E6B1B8",
    "Benign":             "#1D7AAB",
    "Likely benign":      "#63A1C4",
    "Ambiguous":          "#A0A0A0",
    "VUS":                "#A0A0A0",
    "Conflicting":        "#505050",
    "Others":             "#E0E0E0",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_unique(columns: list[str]) -> pl.DataFrame:
    """Read selected columns from the merged allele collection, deduped on
    ``gene_variant``.
    """
    return (
        pl.read_parquet(ALLELE_COLLECTION_PATH, columns=columns)
        .unique(subset=["gene_variant"])
    )


def _na_to_null(col: str) -> pl.Expr:
    """Map dbNSFP NA sentinels (``''`` / ``'.'``) and actual nulls to null,
    keeping every other string value as-is.
    """
    e = pl.col(col)
    return pl.when(e.is_null() | e.is_in(list(_DBNSFP_NA))).then(None).otherwise(e)


def _load_clinvar() -> pl.DataFrame:
    df = _read_unique(
        ["gene_variant", "clinvar_clnsig_clean", "clinvar_clnsig_clean_pp_strict"]
    )
    return df.rename(
        {
            "clinvar_clnsig_clean":           "clinvar_label",
            "clinvar_clnsig_clean_pp_strict": "clinvar_label_strict",
        }
    )


# AlphaMissense dbNSFP categories: P (Pathogenic), LP (Likely pathogenic),
# B (Benign), LB (Likely benign), A (Ambiguous). The coarse map collapses
# the "likely" variants into the parent class so the Pathogenic/Benign
# violin is comparable to ClinVar's clean column.
_AM_COARSE_MAP = {"P":  "Pathogenic", "LP": "Pathogenic",
                  "B":  "Benign",     "LB": "Benign",
                  "A":  "Ambiguous"}
_AM_STRICT_MAP = {"P":  "Pathogenic",
                  "LP": "Likely pathogenic",
                  "B":  "Benign",
                  "LB": "Likely benign",
                  "A":  "Ambiguous"}


def _load_alphamissense() -> pl.DataFrame:
    df = _read_unique(["gene_variant", "AlphaMissense_pred"])
    pred = _na_to_null("AlphaMissense_pred")
    return df.with_columns(
        alphamissense_label=pred.replace_strict(
            _AM_COARSE_MAP, default=None, return_dtype=pl.String
        ),
        alphamissense_label_strict=pred.replace_strict(
            _AM_STRICT_MAP, default=None, return_dtype=pl.String
        ),
    ).drop("AlphaMissense_pred")


# ESM1b dbNSFP: D (Damaging) → Pathogenic, T (Tolerated) → Benign.
# No likely-pathogenic / likely-benign tier, so coarse == strict.
_ESM1B_MAP = {"D": "Pathogenic", "T": "Benign"}


def _load_esm1b() -> pl.DataFrame:
    # ESM1b has no LP/LB tier, so coarse and strict are identical. We emit
    # a single column and the PredictorConfig points label_col and
    # label_strict_col at the same name to avoid rendering duplicate plots.
    df = _read_unique(["gene_variant", "ESM1b_pred"])
    pred = _na_to_null("ESM1b_pred")
    return df.with_columns(
        esm1b_label=pred.replace_strict(
            _ESM1B_MAP, default=None, return_dtype=pl.String
        ),
    ).drop("ESM1b_pred")


# REVEL: continuous 0–1 score. Coarse split at 0.5 (the long-standing
# binary convention). Strict tiers use the ClinGen 2022 calibration
# (Pejaver et al., AJHG 2022): ≥0.773 PP3_Moderate, ≥0.644 PP3,
# ≤0.290 BP4, ≤0.183 BP4_Moderate, otherwise Ambiguous.
_REVEL_COARSE_CUT = 0.5
_REVEL_STRICT_BINS = (
    # (threshold, label) — applied top-down with `>=` semantics.
    (0.773, "Pathogenic"),
    (0.644, "Likely pathogenic"),
    (0.290, "Ambiguous"),
    (0.183, "Likely benign"),
    (0.0,   "Benign"),
)


# MutPred (MutPred2): continuous 0–1 score. Default cutoff for
# "harmful" prediction is 0.5; ≥0.75 is widely used as a high-confidence
# pathogenic threshold. Strict tiers mirror that: ≥0.75 P, ≥0.5 LP,
# ≥0.25 LB, <0.25 B.
_MUTPRED_COARSE_CUT = 0.5
_MUTPRED_STRICT_BINS = (
    (0.75, "Pathogenic"),
    (0.50, "Likely pathogenic"),
    (0.25, "Likely benign"),
    (0.0,  "Benign"),
)


def _score_to_float(col: str) -> pl.Expr:
    """Cast a dbNSFP string score column to Float64, mapping NA sentinels
    (``''`` / ``'.'``) to null.
    """
    return (
        pl.col(col)
        .replace({s: None for s in _DBNSFP_NA})
        .cast(pl.Float64, strict=False)
    )


def _bin_score_coarse(score: pl.Expr, cutoff: float) -> pl.Expr:
    """``score >= cutoff`` → Pathogenic, ``< cutoff`` → Benign, null → null."""
    return (
        pl.when(score.is_null())
        .then(None)
        .when(score >= cutoff)
        .then(pl.lit("Pathogenic"))
        .otherwise(pl.lit("Benign"))
    )


def _bin_score_strict(score: pl.Expr, bins: tuple[tuple[float, str], ...]) -> pl.Expr:
    """Bin a continuous score into strict tiers using ``>=`` thresholds
    walked top-down. ``bins`` must be ordered by descending threshold.
    """
    expr = pl.when(score.is_null()).then(None)
    for threshold, label in bins:
        expr = expr.when(score >= threshold).then(pl.lit(label))
    return expr.otherwise(None)


def _load_revel() -> pl.DataFrame:
    df = _read_unique(["gene_variant", "REVEL_score"])
    score = _score_to_float("REVEL_score")
    return df.with_columns(
        revel_label=_bin_score_coarse(score, _REVEL_COARSE_CUT),
        revel_label_strict=_bin_score_strict(score, _REVEL_STRICT_BINS),
    ).drop("REVEL_score")


def _load_mutpred() -> pl.DataFrame:
    df = _read_unique(["gene_variant", "MutPred_score"])
    score = _score_to_float("MutPred_score")
    return df.with_columns(
        mutpred_label=_bin_score_coarse(score, _MUTPRED_COARSE_CUT),
        mutpred_label_strict=_bin_score_strict(score, _MUTPRED_STRICT_BINS),
    ).drop("MutPred_score")


def _load_eve() -> pl.DataFrame:
    raise NotImplementedError(
        "EVE annotations are not in the allele collection. To enable: "
        "drop a parquet with columns (gene_variant, eve_label, "
        "eve_label_strict) at data/raw/eve/eve_predictions.parquet and "
        "replace this stub with a join. Categories should match the "
        "shared Pathogenic/Benign vocabulary used by the other predictors."
    )


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------


PREDICTORS: dict[str, PredictorConfig] = {
    "clinvar": PredictorConfig(
        name="clinvar",
        title="ClinVar",
        label_col="clinvar_label",
        label_strict_col="clinvar_label_strict",
        palette={
            "clinvar_label":        _PB_PALETTE,
            "clinvar_label_strict": _PB_PALETTE,
        },
        order={
            "clinvar_label":        ["Pathogenic", "Benign", "VUS", "Conflicting", "Others"],
            "clinvar_label_strict": [
                "Pathogenic", "Likely pathogenic",
                "Benign",     "Likely benign",
                "VUS", "Conflicting", "Others",
            ],
        },
        tests=(
            PredictorTest("Pathogenic_vs_Benign",        "clinvar_label",        "Pathogenic", "Benign"),
            PredictorTest("Pathogenic_vs_Benign_strict", "clinvar_label_strict", "Pathogenic", "Benign"),
        ),
    ),
    "alphamissense": PredictorConfig(
        name="alphamissense",
        title="AlphaMissense",
        label_col="alphamissense_label",
        label_strict_col="alphamissense_label_strict",
        palette={
            "alphamissense_label":        _PB_PALETTE,
            "alphamissense_label_strict": _PB_PALETTE,
        },
        order={
            "alphamissense_label":        ["Pathogenic", "Benign", "Ambiguous"],
            "alphamissense_label_strict": [
                "Pathogenic", "Likely pathogenic",
                "Benign",     "Likely benign",
                "Ambiguous",
            ],
        },
        tests=(
            PredictorTest("Pathogenic_vs_Benign",        "alphamissense_label",        "Pathogenic", "Benign"),
            PredictorTest("Pathogenic_vs_Benign_strict", "alphamissense_label_strict", "Pathogenic", "Benign"),
        ),
    ),
    "esm1b": PredictorConfig(
        name="esm1b",
        title="ESM1b",
        label_col="esm1b_label",
        # Coarse == strict for ESM1b (no LP/LB tier in dbNSFP).
        label_strict_col="esm1b_label",
        palette={
            "esm1b_label": _PB_PALETTE,
        },
        order={
            "esm1b_label": ["Pathogenic", "Benign"],
        },
        tests=(
            PredictorTest("Pathogenic_vs_Benign", "esm1b_label", "Pathogenic", "Benign"),
        ),
    ),
    "revel": PredictorConfig(
        name="revel",
        title="REVEL",
        label_col="revel_label",
        label_strict_col="revel_label_strict",
        palette={
            "revel_label":        _PB_PALETTE,
            "revel_label_strict": _PB_PALETTE,
        },
        order={
            "revel_label":        ["Pathogenic", "Benign"],
            "revel_label_strict": [
                "Pathogenic", "Likely pathogenic",
                "Benign",     "Likely benign",
                "Ambiguous",
            ],
        },
        tests=(
            PredictorTest("Pathogenic_vs_Benign",        "revel_label",        "Pathogenic", "Benign"),
            PredictorTest("Pathogenic_vs_Benign_strict", "revel_label_strict", "Pathogenic", "Benign"),
        ),
    ),
    "mutpred": PredictorConfig(
        name="mutpred",
        title="MutPred",
        label_col="mutpred_label",
        label_strict_col="mutpred_label_strict",
        palette={
            "mutpred_label":        _PB_PALETTE,
            "mutpred_label_strict": _PB_PALETTE,
        },
        order={
            "mutpred_label":        ["Pathogenic", "Benign"],
            "mutpred_label_strict": [
                "Pathogenic", "Likely pathogenic",
                "Benign",     "Likely benign",
            ],
        },
        tests=(
            PredictorTest("Pathogenic_vs_Benign",        "mutpred_label",        "Pathogenic", "Benign"),
            PredictorTest("Pathogenic_vs_Benign_strict", "mutpred_label_strict", "Pathogenic", "Benign"),
        ),
    ),
    "eve": PredictorConfig(
        name="eve",
        title="EVE",
        label_col="eve_label",
        # Coarse == strict until the EVE source provides an LP/LB tier.
        label_strict_col="eve_label",
        palette={
            "eve_label": _PB_PALETTE,
        },
        order={
            "eve_label": ["Pathogenic", "Benign"],
        },
        tests=(
            PredictorTest("Pathogenic_vs_Benign", "eve_label", "Pathogenic", "Benign"),
        ),
    ),
}


_LOADERS = {
    "clinvar":       _load_clinvar,
    "alphamissense": _load_alphamissense,
    "esm1b":         _load_esm1b,
    "revel":         _load_revel,
    "mutpred":       _load_mutpred,
    "eve":           _load_eve,
}


def load_predictor_annotations(predictor: str) -> tuple[pl.DataFrame, PredictorConfig]:
    """Load (gene_variant, *_label, *_label_strict) for one predictor.

    Returns the annotation DataFrame and the matching PredictorConfig.
    """
    if predictor not in PREDICTORS:
        raise ValueError(
            f"Unknown predictor: {predictor!r}. "
            f"Choose from {sorted(PREDICTORS)}."
        )
    df = _LOADERS[predictor]()
    cfg = PREDICTORS[predictor]
    n_with_coarse = df.filter(pl.col(cfg.label_col).is_not_null()).height
    n_with_strict = df.filter(pl.col(cfg.label_strict_col).is_not_null()).height
    log.info(
        "Loaded %s: %d rows, %d non-null %s, %d non-null %s",
        predictor, len(df),
        n_with_coarse, cfg.label_col,
        n_with_strict, cfg.label_strict_col,
    )
    return df, cfg
