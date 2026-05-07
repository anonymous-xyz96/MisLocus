"""Control type annotation."""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def annotate_controls(
    lf: pl.LazyFrame,
    tc: list[str],
    nc: list[str],
    pc: list[str],
    cpc_gene_alleles: list[str],
) -> pl.LazyFrame:
    """Annotate Metadata_node_type and Metadata_Control.

    Two orthogonal annotations:

    1. Metadata_node_type — classification role:
       - "disease_wt": WT reference (Metadata_symbol == Metadata_gene_allele)
       - "allele": variant
       - "TC" / "NC" / "PC": true controls (allele-level exact match)

    2. Metadata_Control — control status for filtering:
       - "NC", "PC", "cPC", "TC": control type
       - "experimental": not a control

    This separation matters because cPC genes (e.g., KRAS) still need their
    WT/allele distinction for variant-vs-reference classification, but should
    be flagged as complementary positive controls for downstream analysis.

    cPC matching: each entry in cpc_gene_alleles is matched BOTH as a gene
    symbol (Metadata_symbol) AND as an exact allele (Metadata_gene_allele).

    Args:
        tc: Transfection control allele names (e.g., ["eGFP"])
        nc: Negative control allele names (e.g., ["RHEB", "MAPK9", ...])
        pc: Positive control allele names (e.g., ["ALK", "ALK_Arg1275Gln", ...])
        cpc_gene_alleles: Complementary positive control gene symbols AND/OR
            specific allele names.
    """
    cols = lf.collect_schema().names()

    # ── Metadata_node_type: classification role ──────────────────────────
    if "Metadata_node_type" in cols:
        lf = lf.drop("Metadata_node_type")

    # Base: disease_wt vs allele
    lf = lf.with_columns(
        pl.when(pl.col("Metadata_symbol") == pl.col("Metadata_gene_allele"))
        .then(pl.lit("disease_wt"))
        .otherwise(pl.lit("allele"))
        .alias("Metadata_node_type")
    )

    # TC/NC/PC override (allele-level exact match)
    for label, allele_list in [("TC", tc), ("NC", nc), ("PC", pc)]:
        if allele_list:
            lf = lf.with_columns(
                pl.when(pl.col("Metadata_gene_allele").is_in(allele_list))
                .then(pl.lit(label))
                .otherwise(pl.col("Metadata_node_type"))
                .alias("Metadata_node_type")
            )

    # ── Metadata_Control: control status ─────────────────────────────────
    if "Metadata_Control" in cols:
        lf = lf.drop("Metadata_Control")

    # Start with "experimental"
    lf = lf.with_columns(pl.lit("Exp").alias("Metadata_Control"))

    # cPC: gene-level (symbol) OR allele-level (gene_allele) match
    if cpc_gene_alleles:
        lf = lf.with_columns(
            pl.when(
                pl.col("Metadata_symbol").is_in(cpc_gene_alleles)
                | pl.col("Metadata_gene_allele").is_in(cpc_gene_alleles)
            )
            .then(pl.lit("cPC"))
            .otherwise(pl.col("Metadata_Control"))
            .alias("Metadata_Control")
        )

    # TC/NC/PC override cPC in control status too
    for label, allele_list in [("TC", tc), ("NC", nc), ("PC", pc)]:
        if allele_list:
            lf = lf.with_columns(
                pl.when(pl.col("Metadata_gene_allele").is_in(allele_list))
                .then(pl.lit(label))
                .otherwise(pl.col("Metadata_Control"))
                .alias("Metadata_Control")
            )

    # ── Log distributions ────────────────────────────────────────────────
    logger.info("Metadata_node_type:")
    counts = (
        lf.group_by("Metadata_node_type")
        .agg(pl.len().alias("count"))
        .sort("Metadata_node_type")
        .collect()
    )
    for row in counts.iter_rows(named=True):
        logger.info("  %s: %d cells", row["Metadata_node_type"], row["count"])

    logger.info("Metadata_Control:")
    counts = (
        lf.group_by("Metadata_Control")
        .agg(pl.len().alias("count"))
        .sort("Metadata_Control")
        .collect()
    )
    for row in counts.iter_rows(named=True):
        logger.info("  %s: %d cells", row["Metadata_Control"], row["count"])

    return lf
