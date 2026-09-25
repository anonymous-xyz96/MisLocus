"""Read-only one-batch MAD diagnostic on independently verified SubCell exports.

Writes ONLY small statistics/evidence into a fresh diagnostic directory, never
normalized features. Uses the existing well-QC and plate-statistics functions.
"""

import argparse
import json
from pathlib import Path

import polars as pl
from audit_plate_mad import audit

from prot_loc_benchmark.config import PREPROCESS_CC_THRESHOLD, REPO_ROOT
from prot_loc_benchmark.preprocessing.normalize import compute_plate_stats
from prot_loc_benchmark.preprocessing.qc import drop_low_cell_count_wells
from prot_loc_benchmark.provenance import save_json, sha256


def compute_export_stats(path, features, expected_cells):
    """Project 64 features at a time; per-plate statistics are column-independent."""
    lf = pl.scan_parquet(path, low_memory=True)
    if not features:
        raise ValueError("No features requested")
    parts = []
    for start in range(0, len(features), 64):
        block = features[start : start + 64]
        frame = lf.select(["Metadata_Plate", "Metadata_Well"] + block).collect()
        if (
            frame.height != expected_cells
            or frame["Metadata_Plate"].null_count()
            or not frame.select(pl.all_horizontal(pl.col(block).is_finite().fill_null(False)).all()).item()
        ):
            raise ValueError("Invalid raw embedding rows/features/plate identities")
        filtered = drop_low_cell_count_wells(frame.lazy(), cc_threshold=PREPROCESS_CC_THRESHOLD)
        parts.append(compute_plate_stats(filtered))
        print(f"  features {start + 1}–{start + len(block)} / {len(features)}", flush=True)
        del frame, filtered
    return pl.concat(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    expected_spec = args.spec.with_suffix(".sha256").read_text().split()[0]
    if sha256(args.spec) != expected_spec:
        raise ValueError("Changed extraction specification")
    control = Path(spec["control"])
    status = json.loads((control / "production-status.json").read_text())
    if status["status"] != "complete" or status["spec_sha256"] != expected_spec:
        raise ValueError("Extraction is not complete and verified")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for family in ("mae", "vit"):
        model = spec["models"][family]
        verification_path = control / f"verified-production-{family}.json"
        if sha256(verification_path) != status["jobs"][family]["verified_receipt_sha256"]:
            raise ValueError("Changed independent verification report")
        verified = json.loads(verification_path.read_text())
        export = Path(spec["export_root"]) / model["representation"]
        receipt_path = export / "extraction.json"
        if sha256(receipt_path) != verified["receipt_sha256"]:
            raise ValueError("Changed extraction receipt")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["status"] != "complete"
            or receipt["split"] != "all"
            or receipt["artifact_kind"] != "raw_embeddings"
            or receipt["checkpoint_sha256"] != model["checkpoint_sha256"]
            or receipt["outputs"] != verified["outputs"]
        ):
            raise ValueError("Export bindings differ from verified specification")
        path = export / args.batch / "embeddings.parquet"
        binding = verified["outputs"][args.batch]
        print(f"{family}: verifying {path}", flush=True)
        if sha256(path) != binding["sha256"]:
            raise ValueError("Changed raw embeddings")
        features = [f"SubCell_{i}" for i in range(1536)]
        if receipt["feature_columns"] != features:
            raise ValueError("Unexpected feature ordering")
        print(f"{family}: computing diagnostics on {binding['cells']} cells", flush=True)
        stats = compute_export_stats(path, features, binding["cells"])
        destination = args.output / f"{family}_plate_stats.parquet"
        stats.write_parquet(destination)
        result = audit(destination)
        result.update(
            representation=model["representation"],
            batch=args.batch,
            raw_cells=binding["cells"],
            retained_cells=int(stats.group_by("Metadata_Plate").agg(pl.col("count").first())["count"].sum()),
            input_path=str(path),
            input_sha256=binding["sha256"],
            verification_sha256=sha256(verification_path),
            receipt_sha256=sha256(receipt_path),
        )
        if sha256(path) != binding["sha256"]:
            raise ValueError("Raw embeddings changed during audit")
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        del stats
    save_json(
        args.output / "audit.json",
        {
            "kind": "read_only_diagnostic_not_preprocessed_features",
            "status": "complete",
            "spec_sha256": expected_spec,
            "batch": args.batch,
            "source_sha256": {
                str(p): sha256(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("audit_plate_mad.py"),
                    REPO_ROOT / "src/prot_loc_benchmark/preprocessing/normalize.py",
                    REPO_ROOT / "src/prot_loc_benchmark/preprocessing/qc.py",
                )
            },
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
