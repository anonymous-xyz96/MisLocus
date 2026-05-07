"""Generate classification pairs: experimental (ref vs variant) and control (null)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationPair:
    """A single classification task: two groups of cells to distinguish."""

    pair_id: str
    gene: str
    allele_ref: str
    allele_var: str
    is_control: bool
    category: str  # "Exp", "cPC", "NC", "PC"


def _build_ref_var_pairs(
    lf: pl.LazyFrame,
    control_filter: str,
    category: str,
    min_cells: int,
) -> list[ClassificationPair]:
    """Build ref-vs-variant pairs for cells matching a Metadata_Control value.

    Used for both Exp and cPC categories. For each gene (Metadata_symbol):
    - Reference = cells with Metadata_node_type == "disease_wt"
    - Variant = each distinct allele with Metadata_node_type == "allele"
    """
    counts = (
        lf.filter(
            (pl.col("Metadata_Control") == control_filter)
            & pl.col("Metadata_node_type").is_in(["disease_wt", "allele"])
        )
        .group_by("Metadata_symbol", "Metadata_gene_allele", "Metadata_node_type")
        .agg(pl.len().alias("n"))
        .collect()
    )

    pairs: list[ClassificationPair] = []

    for gene in counts.filter(pl.col("Metadata_node_type") == "disease_wt")[
        "Metadata_symbol"
    ].unique():
        ref_rows = counts.filter(
            (pl.col("Metadata_symbol") == gene)
            & (pl.col("Metadata_node_type") == "disease_wt")
        )
        if ref_rows.is_empty():
            continue
        ref_allele = ref_rows["Metadata_gene_allele"][0]
        ref_count = ref_rows["n"].sum()

        if ref_count < min_cells:
            logger.debug("Skip gene %s: ref count %d < %d", gene, ref_count, min_cells)
            continue

        var_rows = counts.filter(
            (pl.col("Metadata_symbol") == gene)
            & (pl.col("Metadata_node_type") == "allele")
        )
        for row in var_rows.iter_rows(named=True):
            if row["n"] < min_cells:
                logger.debug(
                    "Skip %s: variant count %d < %d",
                    row["Metadata_gene_allele"],
                    row["n"],
                    min_cells,
                )
                continue

            pair_id = f"{gene}__{row['Metadata_gene_allele']}"
            pairs.append(
                ClassificationPair(
                    pair_id=pair_id,
                    gene=gene,
                    allele_ref=ref_allele,
                    allele_var=row["Metadata_gene_allele"],
                    is_control=False,
                    category=category,
                )
            )

    return pairs


def build_experimental_pairs(
    lf: pl.LazyFrame,
    min_cells: int = 50,
) -> list[ClassificationPair]:
    """Build ref-vs-variant pairs for Exp alleles (Metadata_Control == 'Exp')."""
    pairs = _build_ref_var_pairs(lf, "Exp", "Exp", min_cells)
    logger.info("Built %d experimental (Exp) pairs", len(pairs))
    return pairs


def build_cpc_pairs(
    lf: pl.LazyFrame,
    min_cells: int = 50,
) -> list[ClassificationPair]:
    """Build ref-vs-variant pairs for cPC alleles (Metadata_Control == 'cPC').

    cPC (confirmed positive controls) are classified exactly like Exp
    (ref vs variant, same CV), but tracked separately in outputs.
    """
    pairs = _build_ref_var_pairs(lf, "cPC", "cPC", min_cells)
    logger.info("Built %d cPC pairs", len(pairs))
    return pairs


def build_control_pairs(
    lf: pl.LazyFrame,
    control_types: list[str] | None = None,
    min_cells: int = 50,
) -> list[ClassificationPair]:
    """Build same-allele different-well control pairs for null distribution.

    Uses ``Metadata_Control`` column (not ``Metadata_node_type``) to identify
    control alleles. For each control allele (NC, PC):
    - Group cells by (Metadata_gene_allele, Metadata_plate_map_name)
    - For each pair of wells with the same allele on the same platemap,
      create a control pair (arbitrary label 0/1 assignment).
    """
    if control_types is None:
        control_types = ["NC", "PC"]

    ctrl = (
        lf.filter(pl.col("Metadata_Control").is_in(control_types))
        .group_by(
            "Metadata_gene_allele",
            "Metadata_plate_map_name",
            "Metadata_well_position",
            "Metadata_Control",
        )
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_cells)
        .collect()
    )

    pairs: list[ClassificationPair] = []

    for (allele, platemap, ctrl_type), grp in ctrl.group_by(
        "Metadata_gene_allele", "Metadata_plate_map_name", "Metadata_Control"
    ):
        wells = grp["Metadata_well_position"].to_list()
        if len(wells) < 2:
            continue

        for w1, w2 in combinations(sorted(wells), 2):
            pair_id = f"ctrl__{allele}__{platemap}__{w1}__{w2}"
            pairs.append(
                ClassificationPair(
                    pair_id=pair_id,
                    gene=str(allele),
                    allele_ref=f"{allele}_{w1}",
                    allele_var=f"{allele}_{w2}",
                    is_control=True,
                    category=str(ctrl_type),
                )
            )

    logger.info("Built %d control null pairs (NC+PC)", len(pairs))
    return pairs


def get_pair_data(
    data: pl.LazyFrame | pl.DataFrame,
    pair: ClassificationPair,
) -> pl.DataFrame:
    """Extract and label cells for one classification pair.

    Accepts either a LazyFrame (will be collected) or an eager DataFrame
    (filtered in-memory — much faster when called repeatedly).

    Returns DataFrame with all columns plus a ``Label`` column:
    - Label=1 for reference (disease_wt or first control well)
    - Label=0 for variant (allele or second control well)
    """
    is_lazy = isinstance(data, pl.LazyFrame)

    if not pair.is_control:
        # Experimental or cPC: ref (disease_wt) vs variant (allele)
        ref_filter = (
            (pl.col("Metadata_gene_allele") == pair.allele_ref)
            & (pl.col("Metadata_node_type") == "disease_wt")
        )
        var_filter = (
            (pl.col("Metadata_gene_allele") == pair.allele_var)
            & (pl.col("Metadata_node_type") == "allele")
        )

        if is_lazy:
            ref = data.filter(ref_filter).with_columns(pl.lit(1).alias("Label")).collect()
            var = data.filter(var_filter).with_columns(pl.lit(0).alias("Label")).collect()
        else:
            ref = data.filter(ref_filter).with_columns(pl.lit(1).alias("Label"))
            var = data.filter(var_filter).with_columns(pl.lit(0).alias("Label"))

        return pl.concat([ref, var])

    # Control pair: same allele, different wells
    allele = pair.gene
    well_ref = pair.allele_ref.rsplit("_", 1)[-1]
    well_var = pair.allele_var.rsplit("_", 1)[-1]
    parts = pair.pair_id.split("__")
    platemap = parts[2] if len(parts) > 2 else ""

    base_filter = pl.col("Metadata_gene_allele") == allele
    if platemap:
        base_filter = base_filter & (pl.col("Metadata_plate_map_name") == platemap)

    ref_filter = base_filter & (pl.col("Metadata_well_position") == well_ref)
    var_filter = base_filter & (pl.col("Metadata_well_position") == well_var)

    if is_lazy:
        ref = data.filter(ref_filter).with_columns(pl.lit(1).alias("Label")).collect()
        var = data.filter(var_filter).with_columns(pl.lit(0).alias("Label")).collect()
    else:
        ref = data.filter(ref_filter).with_columns(pl.lit(1).alias("Label"))
        var = data.filter(var_filter).with_columns(pl.lit(0).alias("Label"))

    return pl.concat([ref, var])


_SCOPE_KEYWORDS = {"all", "exp", "cpc", "control"}


def filter_pairs_by_scope(
    pairs: list[ClassificationPair],
    scope: str,
) -> list[ClassificationPair]:
    """Filter pairs based on ``--scope`` CLI argument.

    Scope values (can be combined with commas):
    - "all": return all pairs unchanged
    - "exp": Exp pairs only (category="Exp")
    - "cpc": cPC pairs only (category="cPC")
    - "control": control null pairs only (NC + PC)
    - Allele names: matched against allele_var or gene
    - Comma-separated mix: e.g., "CCM2_Ile432Thr,control" selects that
      allele plus all control pairs
    """
    if scope == "all":
        return pairs

    # Parse comma-separated tokens into keywords and allele names
    tokens = {t.strip() for t in scope.split(",")}
    keywords = tokens & _SCOPE_KEYWORDS
    allele_names = tokens - _SCOPE_KEYWORDS

    filtered: list[ClassificationPair] = []
    seen_ids: set[str] = set()

    for p in pairs:
        match = False
        if "exp" in keywords and p.category == "Exp":
            match = True
        if "cpc" in keywords and p.category == "cPC":
            match = True
        if "control" in keywords and p.is_control:
            match = True
        if allele_names and (p.allele_var in allele_names or p.gene in allele_names):
            match = True

        if match and p.pair_id not in seen_ids:
            filtered.append(p)
            seen_ids.add(p.pair_id)

    logger.info(
        "Scope '%s': %d / %d pairs selected", scope, len(filtered), len(pairs)
    )
    if not filtered:
        logger.warning("No pairs matched scope '%s'", scope)

    return filtered
