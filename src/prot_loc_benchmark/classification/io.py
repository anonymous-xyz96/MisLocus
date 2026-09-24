"""Streaming Parquet output and schema definitions for classification results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from prot_loc_benchmark.identity import CELL_ID

logger = logging.getLogger(__name__)

# Schema for cell-level predictions (streamed to Parquet)
PREDICTIONS_SCHEMA = pa.schema(
    [
        ("Classifier_ID", pa.string()),
        ("AnalysisStageID", pa.string()),
        ("Representation", pa.string()),
        ("Metadata_BatchQualifiedCellID", pa.string()),
        ("Metadata_CellID", pa.string()),
        ("Metadata_Batch", pa.string()),
        ("Metadata_ImageNumber", pa.int64()),
        ("Metadata_Site", pa.string()),
        ("Metadata_Split", pa.string()),
        ("Metadata_Plate", pa.string()),
        ("Metadata_well_position", pa.string()),
        ("Metadata_ObjectNumber", pa.int64()),
        ("Label", pa.int8()),
        ("Prediction", pa.float32()),
        ("Feature_Type", pa.string()),
        ("Control", pa.bool_()),
    ]
)


class ClassificationWriter:
    """Streaming ParquetWriter for cell-level predictions.

    Avoids accumulating all predictions in memory by appending
    batches incrementally to a Parquet file.

    Usage::

        with ClassificationWriter(output_path, stage_id=stage_id, representation=rep) as writer:
            writer.write_predictions(...)
    """

    def __init__(self, path: Path | str, *, stage_id: str, representation: str) -> None:
        self._path = Path(path)
        if not stage_id or not representation:
            raise ValueError("Predictions require analysis stage and representation identity")
        self.stage_id, self.representation = stage_id, representation
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: pq.ParquetWriter | None = None
        self._membership_writer: pq.ParquetWriter | None = None

    def __enter__(self) -> ClassificationWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def write_predictions(
        self,
        classifier_id: str,
        labels: np.ndarray,
        predictions: np.ndarray,
        channel: str,
        is_control: bool,
        identities: pl.DataFrame,
    ) -> None:
        """Append one classifier's predictions to the Parquet file."""
        n = len(labels)
        if identities.height != n or identities[CELL_ID].null_count() or identities[CELL_ID].n_unique() != n:
            raise ValueError("Predictions require one unique non-null cell ID per row")
        table = pa.table(
            {
                "Classifier_ID": pa.array([classifier_id] * n, type=pa.string()),
                "AnalysisStageID": pa.array([self.stage_id] * n, type=pa.string()),
                "Representation": pa.array([self.representation] * n, type=pa.string()),
                **{
                    c: pa.array(identities[c].to_list(), type=pa.string())
                    for c in (CELL_ID, "Metadata_CellID", "Metadata_Batch")
                },
                "Metadata_ImageNumber": pa.array(identities["Metadata_ImageNumber"].to_list(), type=pa.int64()),
                "Metadata_Site": pa.array(
                    identities["Metadata_Site"].cast(pl.String).to_list()
                    if "Metadata_Site" in identities.columns
                    else [None] * n,
                    type=pa.string(),
                ),
                "Metadata_Split": pa.array(
                    identities["Metadata_Split"].to_list() if "Metadata_Split" in identities.columns else [None] * n,
                    type=pa.string(),
                ),
                "Metadata_Plate": pa.array(identities["Metadata_Plate"].to_list(), type=pa.string()),
                "Metadata_well_position": pa.array(identities["Metadata_Well"].to_list(), type=pa.string()),
                "Metadata_ObjectNumber": pa.array(identities["Metadata_ObjectNumber"].to_list(), type=pa.int64()),
                "Label": pa.array(labels, type=pa.int8()),
                "Prediction": pa.array(predictions, type=pa.float32()),
                "Feature_Type": pa.array([channel] * n, type=pa.string()),
                "Control": pa.array([is_control] * n, type=pa.bool_()),
            },
            schema=PREDICTIONS_SCHEMA,
        )
        if self._writer is None:
            self._writer = pq.ParquetWriter(str(self._path), PREDICTIONS_SCHEMA, compression="zstd")
        self._writer.write_table(table)

    def write_membership(self, classifier_id, pair_id, fold_id, frame, role):
        table = (
            frame.select([c for c in frame.columns if c.startswith("Metadata_")] + ["Label"])
            .with_columns(
                pl.lit(classifier_id).alias("Classifier_ID"),
                pl.lit(pair_id).alias("pair_id"),
                pl.lit(fold_id).alias("fold_id"),
                pl.lit(role).alias("role"),
                pl.lit(self.stage_id).alias("AnalysisStageID"),
                pl.lit(self.representation).alias("Representation"),
            )
            .to_arrow()
        )
        if self._membership_writer is None:
            self._membership_writer = pq.ParquetWriter(
                self._path.parent / "fold_membership.parquet", table.schema, compression="zstd"
            )
        self._membership_writer.write_table(table)

    def close(self) -> None:
        """Close both streaming Parquet writers."""
        if self._membership_writer is not None:
            self._membership_writer.close()
            self._membership_writer = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.info("Predictions written to %s", self._path)
