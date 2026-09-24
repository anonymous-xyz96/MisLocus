"""Consume the completed producer contract without importing its Torch environment."""

import json
from pathlib import Path

import polars as pl

from prot_loc_benchmark.config import DATA_DIR
from prot_loc_benchmark.identity import identify_cells, ordered_id_hash
from prot_loc_benchmark.provenance import sha256, verify_source
from prot_loc_benchmark.stages import require_bounded_execution


def export_inputs(spec_path, representation):
    """Declare files before entering stage(); verify their contents inside that transaction."""
    if spec_path is None:
        raise ValueError("New SubCell preprocessing requires --extraction-spec")
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    matches = [(name, m) for name, m in spec["models"].items() if m["representation"] == representation]
    if len(matches) != 1:
        raise ValueError("Representation absent or duplicated in extraction specification")
    family, model = matches[0]
    export, control = Path(spec["export_root"]) / representation, Path(spec["control"])
    return [
        spec_path,
        spec_path.with_suffix(".sha256"),
        control / "production-status.json",
        control / f"verified-production-{family}.json",
        export / "extraction.json",
        export / "source.json",
        export / "source.tar.gz",
        Path(model["checkpoint"]),
        Path(model["selection"]),
        Path(spec["preflight"]) / "preflight.json",
        Path(spec["crop_verification"]),
    ]


def verify_export(spec_path, representation, batch, input_path):
    if spec_path is None:
        raise ValueError("New SubCell preprocessing requires --extraction-spec")
    require_bounded_execution(representation)
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    spec_digest = sha256(spec_path)
    if spec_path.with_suffix(".sha256").read_text().split()[0] != spec_digest:
        raise ValueError("Extraction specification checksum mismatch")
    matches = [(name, model) for name, model in spec["models"].items() if model["representation"] == representation]
    if len(matches) != 1 or batch not in spec["expected_batch_split_counts"]:
        raise ValueError("Representation/batch absent from extraction specification")
    family, model = matches[0]
    control = Path(spec["control"])
    status_path = control / "production-status.json"
    status = json.loads(status_path.read_text())
    if status["status"] != "complete" or status["spec_sha256"] != spec_digest:
        raise ValueError("Producer has not completed independent verification")
    verification_path = control / f"verified-production-{family}.json"
    if sha256(verification_path) != status["jobs"][family]["verified_receipt_sha256"]:
        raise ValueError("Changed independent verification report")
    verified = json.loads(verification_path.read_text())
    export = Path(spec["export_root"]) / representation
    if DATA_DIR.resolve().is_relative_to(Path(spec["export_root"]).resolve()):
        raise ValueError("Analysis root must not be inside producer exports")
    receipt_path = export / "extraction.json"
    if sha256(receipt_path) != verified["receipt_sha256"]:
        raise ValueError("Changed extraction receipt")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["status"] != "complete"
        or receipt["artifact_kind"] != "raw_embeddings"
        or receipt["split"] != "all"
        or receipt["family"] != family
        or receipt["checkpoint_sha256"] != model["checkpoint_sha256"]
        or receipt["selected_pass"] != model["selected_pass"]
        or receipt["outputs"] != verified["outputs"]
        or set(receipt["outputs"]) != set(spec["expected_batch_split_counts"])
        or receipt["training"]["code_sha256"] != spec["training_code_sha256"]
        or receipt["invocation"]["code_sha256"] != spec["code_sha256"]
    ):
        raise ValueError("Wrong checkpoint, split, source or incomplete export bindings")
    verify_source(export, spec["code_sha256"])
    if (
        sha256(export / "source.tar.gz") != receipt["source_archive_sha256"]
        or json.loads((export / "source.json").read_text())["git_head"] != spec["source_commit"]
    ):
        raise ValueError("Extraction source identity mismatch")
    dependencies = [
        (Path(model["checkpoint"]), model["checkpoint_sha256"]),
        (Path(model["selection"]), model["selection_sha256"]),
        (Path(spec["preflight"]) / "preflight.json", spec["preflight_sha256"]),
        (Path(spec["crop_verification"]), spec["crop_verification_sha256"]),
    ]
    for path, digest in dependencies:
        if sha256(path) != digest:
            raise ValueError(f"Changed producer dependency: {path}")
    features = [f"SubCell_{i}" for i in range(1536)]
    schema = pl.read_parquet_schema(input_path)
    if (
        receipt["feature_columns"] != features
        or [c for c in schema if not c.startswith("Metadata_")] != features
        or any(schema[c] != pl.Float32 for c in features)
    ):
        raise ValueError("Raw SubCell feature order/dtype/shape mismatch")
    binding = receipt["outputs"][batch]
    if sha256(input_path) != binding["sha256"]:
        raise ValueError("Raw embedding checksum mismatch")
    cells = identify_cells(
        pl.read_parquet(input_path, columns=[c for c in schema if c.startswith("Metadata_")]), batch, canonical=True
    )
    counts = dict(cells.group_by("Metadata_Split").len().iter_rows())
    expected_split = (
        pl.when(pl.col("Metadata_Plate").str.ends_with("T4"))
        .then(pl.lit("test"))
        .when(pl.col("Metadata_Plate").str.ends_with("T3"))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
    )
    if (
        cells.height != binding["cells"]
        or ordered_id_hash(cells) != binding["ordered_cell_ids_sha256"]
        or counts != binding["split_counts"]
        or counts != spec["expected_batch_split_counts"][batch]
        or set(cells["Metadata_Plate"].str.extract(r"(T[1-4])$", 1)) != {"T1", "T2", "T3", "T4"}
        or not cells.select((pl.col("Metadata_Split") == expected_split).fill_null(False).all()).item()
    ):
        raise ValueError("Export cell identities/split coverage mismatch")
    if (
        not pl.scan_parquet(input_path)
        .select(pl.all_horizontal(pl.col(features).is_finite().fill_null(False)).all())
        .collect(engine="streaming")
        .item()
    ):
        raise ValueError("Raw SubCell embeddings contain null/nonfinite features")
    return export_inputs(spec_path, representation)
