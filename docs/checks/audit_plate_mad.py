"""Read only small, explicitly named plate_stats files; never read/modify features.

Usage: python docs/checks/audit_plate_mad.py /path/to/plate_stats.parquet [...]
Numbers describe the historical input to normalization, not a new export.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from prot_loc_benchmark.provenance import sha256


def audit(path):
    stats = pl.read_parquet(path)
    if stats.select("Metadata_Plate", "feature").unique().height != stats.height:
        raise ValueError(f"Duplicate plate-feature statistics: {path}")
    invalid = stats.filter(
        ~pl.all_horizontal(pl.col("median", "mad", "min", "max").is_finite().fill_null(False)) | (pl.col("mad") < 0)
    )
    valid = stats.filter(
        pl.all_horizontal(pl.col("median", "mad", "min", "max").is_finite().fill_null(False)) & (pl.col("mad") >= 0)
    )
    zero = valid.filter(pl.col("mad") == 0)
    variable_zero = zero.filter(pl.col("max") > pl.col("min"))
    # Bounds under the EXISTING normalization rule. This is not a count of clipped cells.
    bounds = valid.with_columns(
        [
            pl.when(pl.col("mad") > 0)
            .then((pl.col(c) - pl.col("median")) / pl.col("mad"))
            .otherwise(pl.col(c))
            .alias(f"normalized_{c}")
            for c in ("min", "max")
        ]
    )
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "plates": stats["Metadata_Plate"].n_unique(),
        "dimensions": stats["feature"].n_unique(),
        "plate_feature_pairs": stats.height,
        "invalid_stat_pairs": invalid.height,
        "zero_mad_pairs": zero.height,
        "zero_mad_dimensions_any_plate": zero["feature"].n_unique(),
        "zero_mad_constant_pairs": zero.filter(pl.col("min") == pl.col("max")).height,
        "zero_mad_nonconstant_pairs": variable_zero.height,
        "zero_mad_nonconstant_nonzero_median_pairs": variable_zero.filter(pl.col("median") != 0).height,
        "smallest_positive_mad": valid.filter(pl.col("mad") > 0)["mad"].min(),
        "largest_abs_normalized_extremum": bounds.select(
            pl.max_horizontal(pl.col("normalized_min").abs(), pl.col("normalized_max").abs()).max()
        ).item(),
        "pairs_with_extrema_outside_100_after_normalization": bounds.filter(
            (pl.col("normalized_min") < -100) | (pl.col("normalized_max") > 100)
        ).height,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(p) for p in args.paths], indent=2, allow_nan=False))
