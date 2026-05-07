"""Shared plotting functions for ClinVar benchmark outputs."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats

from prot_loc_benchmark.viz.palettes import CLINVAR_ORDER, CLINVAR_PALETTE

log = logging.getLogger(__name__)


# ============================================================================
# SIGNIFICANCE HELPERS
# ============================================================================


def significance_stars(p: float) -> str:
    """Convert BH-corrected p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _add_significance_bracket(
    ax: plt.Axes,
    x0: int,
    x1: int,
    pval: float,
    y: float,
    h: float = 0.015,
) -> None:
    """Draw a bracket with p-value between two x positions."""
    if pval < 0.001:
        p_text = f"p={pval:.1e}"
    elif pval < 0.01:
        p_text = f"p={pval:.4f}"
    else:
        p_text = f"p={pval:.3f}"

    color = "black" if pval < 0.05 else "grey"
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=1.0, color=color)
    ax.text(
        (x0 + x1) / 2, y + h + 0.005, p_text,
        ha="center", va="bottom", fontsize=7, color=color, fontstyle="italic",
    )


# ============================================================================
# HEATMAP
# ============================================================================


def plot_summary_heatmap(
    stat_results: pl.DataFrame,
    output_dir: Path,
    rep_order: list[str] | None = None,
    stat: str = "median",
    title: str = "ClinVar benchmark — Pathogenic vs Benign AUROC",
    filename: str = "clinvar_significance_heatmap.png",
) -> None:
    """Heatmap of AUROC/mAP difference (Pathogenic - Benign) with significance stars.

    Cell color encodes the effect size (diverging blue-white-red).
    Text shows the numeric difference plus significance stars from BH-corrected p-values.

    Args:
        stat: "median" (default) or "mean". Use "mean" for PA norm mAP where
              many values are floored at 0 and medians are uninformative.
        title: Top-line plot title (the BH-corrected legend line is appended
            automatically). Override per predictor (AlphaMissense / ESM1b /
            EVE) to keep the subtitle accurate.
        filename: Output PNG filename. Default keeps backward compatibility
            with the ClinVar pipeline.

    If *rep_order* is given, rows are sorted so that representations appear in
    that order (channels within each representation stay alphabetical).
    """
    if len(stat_results) == 0:
        log.warning("No test results for heatmap")
        return

    a_col = f"{stat}_a"
    b_col = f"{stat}_b"
    if a_col not in stat_results.columns or b_col not in stat_results.columns:
        log.warning("Heatmap: %s columns missing, falling back to median", stat)
        a_col, b_col, stat = "median_a", "median_b", "median"

    df = stat_results.with_columns(
        median_diff=(pl.col(a_col) - pl.col(b_col)),
        label=pl.col("representation") + " / " + pl.col("channel"),
    ).to_pandas()

    comparisons = sorted(df["comparison"].unique())

    if rep_order:
        # Build a sort key: (rep_index, channel) so rows follow rep_order
        rep_rank = {r: i for i, r in enumerate(rep_order)}
        all_labels = df[["representation", "channel", "label"]].drop_duplicates()
        all_labels["_sort"] = all_labels["representation"].map(
            lambda r: rep_rank.get(r, len(rep_rank))
        )
        all_labels = all_labels.sort_values(["_sort", "channel"])
        labels = all_labels["label"].tolist()
    else:
        labels = sorted(df["label"].unique())

    # Build numeric matrix and annotation matrix
    diff_data = pd.DataFrame(index=labels, columns=comparisons, dtype=float)
    annot_data = pd.DataFrame(index=labels, columns=comparisons, dtype=str)
    for _, row in df.iterrows():
        diff = row["median_diff"]
        stars = significance_stars(row["pvalue_bh"])
        diff_data.loc[row["label"], row["comparison"]] = diff
        annot_data.loc[row["label"], row["comparison"]] = f"{diff:+.3f}{stars}"

    # Symmetric color range centered at 0
    vmax = max(0.05, diff_data.abs().max().max())

    col_width = max(3.0, max(len(c) for c in comparisons) * 0.12)
    fig, ax = plt.subplots(figsize=(len(comparisons) * col_width, max(3, len(labels) * 0.45 + 1.5)))
    sns.heatmap(
        diff_data.astype(float),
        annot=annot_data.values,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"{stat.capitalize()} diff (Pathogenic \u2212 Benign)"},
        annot_kws={"fontsize": 9},
    )
    ax.set_title(
        f"{title}\n(* p<0.05, ** p<0.01, *** p<0.001, BH-corrected)",
        fontsize=11, pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", filename)


# ============================================================================
# VIOLIN PLOT
# ============================================================================


def plot_clinvar_violin(
    data: pl.DataFrame,
    clinvar_col: str,
    palette: dict[str, str],
    category_order: list[str],
    representation: str,
    output_dir: Path,
    stat_results: pl.DataFrame | None = None,
    ylabel: str = "AUROC (bio-rep avg)",
    ylim: tuple[float, float] = (0.3, 1.12),
    yref: float | None = 0.5,
    title: str | None = None,
) -> None:
    """Violin + boxplot + strip plot by ClinVar category, faceted by channel.

    ``title`` overrides the suptitle. If None (default), uses the legacy
    ``"{representation} — {clinvar_col}"`` form.
    """
    rep_data = data.filter(pl.col("representation") == representation)
    channels = sorted(rep_data["channel"].unique().to_list())
    n_ch = len(channels)
    if n_ch == 0:
        return

    # Filter category_order to only categories present in data
    present_cats = set(rep_data[clinvar_col].unique().to_list())
    order = [c for c in category_order if c in present_cats]

    fig, axes = plt.subplots(1, n_ch, figsize=(n_ch * 3.2, 5.0), sharey=True)
    if n_ch == 1:
        axes = [axes]

    pdf = rep_data.to_pandas()

    for ax, channel in zip(axes, channels):
        ch_df = pdf[pdf["channel"] == channel]

        # Violin
        sns.violinplot(
            data=ch_df, x=clinvar_col, y="auroc_avg",
            hue=clinvar_col, order=order, hue_order=order,
            palette=palette, inner=None, linewidth=0.8,
            saturation=0.8, ax=ax, cut=0, legend=False,
        )
        # Boxplot inside violin
        sns.boxplot(
            data=ch_df, x=clinvar_col, y="auroc_avg",
            order=order, width=0.15, showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.8),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            medianprops=dict(color="black", linewidth=1.0),
            ax=ax,
        )
        # Strip (jittered points)
        sns.stripplot(
            data=ch_df, x=clinvar_col, y="auroc_avg",
            order=order, color="black", alpha=0.3, size=2.0,
            jitter=0.12, ax=ax,
        )

        # Red line for mean per category
        for i, cat in enumerate(order):
            cat_vals = ch_df.loc[ch_df[clinvar_col] == cat, "auroc_avg"].dropna()
            if len(cat_vals) > 0:
                mean_val = cat_vals.mean()
                ax.hlines(mean_val, i - 0.2, i + 0.2, color="red", lw=2.0, zorder=10)

        ax.set_title(channel, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax == axes[0] else "")
        ax.set_ylim(*ylim)
        if yref is not None:
            ax.axhline(yref, color="grey", ls="--", lw=0.7, alpha=0.5)

        # Rotate x labels
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")

        # Sample sizes
        for i, cat in enumerate(order):
            n = len(ch_df[ch_df[clinvar_col] == cat])
            ax.text(i, ylim[0] + 0.03, f"n={n}", ha="center", va="top", fontsize=7, color="grey")

        # Significance bracket between Pathogenic and Benign
        if stat_results is not None and len(stat_results) > 0:
            comp = stat_results.filter(
                (pl.col("representation") == representation)
                & (pl.col("channel") == channel)
                & (pl.col("clinvar_col") == clinvar_col)
            )
            if len(comp) == 1:
                pval = comp["pvalue_bh"][0]
                group_a = comp["group_a"][0]
                group_b = comp["group_b"][0]
                if group_a in order and group_b in order:
                    x0 = order.index(group_a)
                    x1 = order.index(group_b)
                    _add_significance_bracket(ax, x0, x1, pval, y=ylim[1] - 0.09)

    fig.suptitle(
        title if title is not None else f"{representation} \u2014 {clinvar_col}",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()

    stem = f"{representation}_{clinvar_col}_violin"
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s.png", stem)


def plot_clinvar_violin_by_organelle(
    data: pl.DataFrame,
    clinvar_col: str,
    palette: dict[str, str],
    category_order: list[str],
    representation: str,
    channel: str,
    gene_to_locations: dict[str, list[str]],
    output_dir: Path,
    ylabel: str = "Norm. mAP (bio-rep avg)",
    ylim: tuple[float, float] = (-0.15, 0.8),
    yref: float | None = 0.0,
    min_genes_per_organelle: int = 3,
) -> None:
    """Grid of ClinVar-split violins, one subplot per HPA organelle.

    Variants are assigned to every organelle their parent gene is annotated
    to (so a variant can appear in multiple subplots). Organelles with
    fewer than ``min_genes_per_organelle`` matching reference genes in the
    pool are skipped.
    """
    rep_data = data.filter(
        (pl.col("representation") == representation)
        & (pl.col("channel") == channel)
    ).to_pandas()
    if rep_data.empty:
        return

    # Shorten organelle labels (same convention as per_organelle_heatmap in 10b)
    gene_to_short = {g: sorted({loc.split("_")[0] for loc in locs}) for g, locs in gene_to_locations.items()}

    # {organelle → all HPA-labeled genes in that organelle} — match 10b's filter
    # (count uses the full HPA reference gene roster, not just genes with variants here)
    org_to_genes_all: dict[str, set[str]] = defaultdict(set)
    for g, orgs in gene_to_short.items():
        for o in orgs:
            org_to_genes_all[o].add(g)

    organelles = sorted(
        [o for o, gs in org_to_genes_all.items() if len(gs) >= min_genes_per_organelle],
        key=lambda o: -len(org_to_genes_all[o]),
    )
    if not organelles:
        return

    present_cats = set(rep_data[clinvar_col].dropna().unique())
    order = [c for c in category_order if c in present_cats]

    ncols = 4
    nrows = (len(organelles) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.2, nrows * 3.5),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, organelle in zip(axes_flat, organelles):
        org_genes = org_to_genes_all[organelle]
        sub = rep_data[rep_data["gene"].isin(org_genes)]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.violinplot(
            data=sub, x=clinvar_col, y="auroc_avg",
            hue=clinvar_col, order=order, hue_order=order,
            palette=palette, inner=None, linewidth=0.6,
            saturation=0.8, ax=ax, cut=0, legend=False,
        )
        sns.boxplot(
            data=sub, x=clinvar_col, y="auroc_avg",
            order=order, width=0.12, showfliers=False,
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.6),
            whiskerprops=dict(color="black", linewidth=0.6),
            capprops=dict(color="black", linewidth=0.6),
            medianprops=dict(color="black", linewidth=0.8),
            ax=ax,
        )
        sns.stripplot(
            data=sub, x=clinvar_col, y="auroc_avg",
            order=order, color="black", alpha=0.3, size=1.5,
            jitter=0.12, ax=ax,
        )
        for i, cat in enumerate(order):
            vals = sub.loc[sub[clinvar_col] == cat, "auroc_avg"].dropna()
            if len(vals):
                ax.hlines(vals.mean(), i - 0.2, i + 0.2, color="red", lw=1.5, zorder=10)

        n_genes_hpa = len(org_genes)
        n_genes_with_variants = sub["gene"].nunique()
        n_variants = len(sub)
        ax.set_title(
            f"{organelle}\n(n_HPA={n_genes_hpa}, n_gene={n_genes_with_variants}, n_var={n_variants})",
            fontsize=9,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax in axes[:, 0] else "")
        ax.set_ylim(*ylim)
        if yref is not None:
            ax.axhline(yref, color="grey", ls="--", lw=0.6, alpha=0.5)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes_flat[len(organelles):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{representation} / {channel} — {clinvar_col} by HPA organelle "
        f"(≥{min_genes_per_organelle} genes/organelle)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    stem = f"{representation}_{clinvar_col}_{channel}_violin_by_organelle"
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s.png", stem)


def plot_pa_pb_diff_heatmap(
    data: pl.DataFrame,
    clinvar_col: str,
    gene_to_locations: dict[str, list[str]],
    output_dir: Path,
    filename: str = "pa_pb_diff_heatmap.png",
    title_suffix: str = "",
    min_genes_per_organelle: int = 3,
    min_samples_per_category: int = 3,
    pathogenic_label: str = "Pathogenic",
    benign_label: str = "Benign",
) -> None:
    """Heatmap of (mean norm-mAP Pathogenic − mean norm-mAP Benign) per
    (rep × channel) row × HPA organelle column.

    Layout mirrors ``plot_per_organelle_heatmap`` in ``10b_benchmark_hpa.py``:
    rows = ``{representation} / {channel}``, columns sorted left-to-right by
    mean diff across rows, secondary top x-axis shows ``n=N`` reference-gene
    counts per organelle. Organelles with fewer than
    ``min_genes_per_organelle`` annotated genes in the HPA roster are dropped.

    Gene pooling: for each organelle we collect all variant alleles whose
    gene is in the HPA label set for that organelle (gene_to_locations
    comes in at whatever threshold the caller chose). Mean mAP is taken
    over all such (gene, variant) rows; the diff is P_mean − B_mean.
    Cell annotation includes a Mann-Whitney U stars on the two subsets.
    """
    pdf = data.to_pandas()
    # Shortened organelle labels, same convention as 10b
    gene_to_short = {
        g: sorted({loc.split("_")[0] for loc in locs}) for g, locs in gene_to_locations.items()
    }
    org_to_genes_all: dict[str, set[str]] = defaultdict(set)
    for g, orgs in gene_to_short.items():
        for o in orgs:
            org_to_genes_all[o].add(g)
    organelles = sorted(
        [o for o, gs in org_to_genes_all.items() if len(gs) >= min_genes_per_organelle]
    )
    if not organelles:
        log.warning("No organelles with >= %d genes — skipping %s", min_genes_per_organelle, filename)
        return
    loc_counts_short = {o: len(org_to_genes_all[o]) for o in organelles}

    # Build long-format rows (rep_label, organelle, diff, p_value, n_p, n_b)
    records: list[dict] = []
    for (rep, ch), grp in pdf.groupby(["representation", "channel"], dropna=False):
        for organelle in organelles:
            org_genes = org_to_genes_all[organelle]
            sub = grp[grp["gene"].isin(org_genes)]
            vals_p = sub.loc[sub[clinvar_col] == pathogenic_label, "auroc_avg"].dropna()
            vals_b = sub.loc[sub[clinvar_col] == benign_label, "auroc_avg"].dropna()
            if len(vals_p) < min_samples_per_category or len(vals_b) < min_samples_per_category:
                diff, pval = float("nan"), float("nan")
            else:
                diff = float(vals_p.mean() - vals_b.mean())
                try:
                    pval = float(stats.mannwhitneyu(vals_p, vals_b, alternative="two-sided").pvalue)
                except ValueError:
                    pval = float("nan")
            records.append({
                "label": f"{rep} / {ch}",
                "organelle": organelle,
                "diff": diff,
                "pvalue": pval,
                "n_p": int(len(vals_p)),
                "n_b": int(len(vals_b)),
            })
    diff_df = pd.DataFrame(records)
    if diff_df.empty:
        return

    diff_mat = diff_df.pivot_table(index="label", columns="organelle", values="diff", aggfunc="mean")
    pmat = diff_df.pivot_table(index="label", columns="organelle", values="pvalue", aggfunc="min")
    # Sort columns by mean diff across rows (best-P-minus-B on the left)
    col_order = diff_mat.mean(axis=0).sort_values(ascending=False).index
    diff_mat = diff_mat[col_order]
    pmat = pmat.reindex(index=diff_mat.index, columns=col_order)

    annot = diff_mat.copy().astype(object)
    for r in diff_mat.index:
        for c in diff_mat.columns:
            v = diff_mat.loc[r, c]
            p = pmat.loc[r, c]
            stars = significance_stars(p) if pd.notna(p) else ""
            annot.loc[r, c] = f"{v:+.2f}{stars}" if pd.notna(v) else ""

    absmax = max(0.05, float(diff_mat.abs().max().max()))
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.4 * len(diff_mat.columns) + 3), max(3.0, len(diff_mat) * 0.35 + 1.5)),
    )
    sns.heatmap(
        diff_mat, annot=annot.values, fmt="", cmap="RdBu_r",
        vmin=-absmax, vmax=absmax, center=0,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Norm. mAP diff (Pathogenic − Benign)", "shrink": 0.7},
        annot_kws={"fontsize": 5}, ax=ax,
    )
    ax.set_title(
        f"ClinVar PA — norm-mAP(Pathogenic) − norm-mAP(Benign) per organelle{title_suffix}\n"
        f"(≥{min_samples_per_category} P + ≥{min_samples_per_category} B required; "
        "* p<0.05, ** p<0.01, *** p<0.001, Mann-Whitney U)",
        fontsize=11, pad=28,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(np.arange(len(diff_mat.columns)) + 0.5)
    top_ax.set_xticklabels(
        [f"n={loc_counts_short.get(c, 0)}" for c in diff_mat.columns],
        fontsize=7, rotation=0,
    )
    top_ax.tick_params(axis="x", length=0, pad=2)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", filename)


# ============================================================================
# CORRELATION PLOT
# ============================================================================


def plot_auroc_correlation(
    data: pl.DataFrame,
    rep_a: str,
    channel_a: str,
    rep_b: str,
    channel_b: str,
    output_dir: Path,
    label_alleles: list[str] | None = None,
    label_col: str = "clinvar_clnsig_clean",
    palette: dict[str, str] | None = None,
    pathogenic_label: str = "Pathogenic",
    benign_label: str = "Benign",
) -> None:
    """Scatter plot of AUROC: rep_a/channel_a vs rep_b/channel_b, colored by
    a Pathogenic/Benign label column.

    ``label_col`` defaults to the ClinVar coarse column for backward
    compatibility. Pass e.g. ``alphamissense_label`` for other predictors.
    """
    if palette is None:
        palette = CLINVAR_PALETTE["clinvar_clnsig_clean"]

    df_a = (
        data.filter(
            (pl.col("representation") == rep_a) & (pl.col("channel") == channel_a)
        )
        .select("allele_var", "gene", "auroc_avg", label_col)
        .rename({"auroc_avg": "auroc_x"})
    )
    df_b = (
        data.filter(
            (pl.col("representation") == rep_b) & (pl.col("channel") == channel_b)
        )
        .select("allele_var", pl.col("auroc_avg").alias("auroc_y"))
    )
    merged = df_a.join(df_b, on="allele_var", how="inner")
    log.info(
        "Correlation plot: %d alleles (%s/%s vs %s/%s)",
        len(merged), rep_a, channel_a, rep_b, channel_b,
    )

    pdf = merged.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, clinvar_cat, title in zip(
        axes,
        [pathogenic_label, benign_label],
        [f"{pathogenic_label} variants", f"{benign_label} variants"],
    ):
        sub = pdf[pdf[label_col] == clinvar_cat]
        others = pdf[pdf[label_col] != clinvar_cat]
        ax.scatter(
            others["auroc_x"], others["auroc_y"],
            c="#E0E0E0", s=12, alpha=0.3, edgecolors="none", zorder=1,
        )
        ax.scatter(
            sub["auroc_x"], sub["auroc_y"],
            c=palette[clinvar_cat], s=18, alpha=0.6, edgecolors="none", zorder=2,
            label=f"{clinvar_cat} (n={len(sub)})",
        )

        if label_alleles and clinvar_cat == pathogenic_label:
            for allele in label_alleles:
                row = sub[sub["allele_var"] == allele]
                if len(row) == 0:
                    log.warning("Label allele %s not found in %s", allele, clinvar_cat)
                    continue
                ax.annotate(
                    allele,
                    xy=(row["auroc_x"].values[0], row["auroc_y"].values[0]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7.5, fontstyle="italic", color="black",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6),
                    zorder=3,
                )

        ax.plot([0.3, 1.0], [0.3, 1.0], ls="--", lw=0.7, color="grey", alpha=0.5)
        ax.axhline(0.5, color="grey", ls=":", lw=0.5, alpha=0.4)
        ax.axvline(0.5, color="grey", ls=":", lw=0.5, alpha=0.4)

        corr = sub[["auroc_x", "auroc_y"]].corr().iloc[0, 1] if len(sub) > 2 else float("nan")
        ax.text(
            0.05, 0.95, f"r = {corr:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(f"{rep_a} {channel_a} AUROC", fontsize=10)
        ax.set_ylabel(f"{rep_b} {channel_b} AUROC", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
        ax.set_xlim(0.3, 1.05)
        ax.set_ylim(0.3, 1.05)
        ax.set_aspect("equal")

    fig.suptitle(
        f"{rep_a} {channel_a} vs {rep_b} {channel_b}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    stem = f"correlation_{rep_a}_{channel_a}_vs_{rep_b}_{channel_b}"
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s.png", stem)
