"""Streaming Parquet output and schema definitions for classification results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Schema for cell-level predictions (streamed to Parquet)
PREDICTIONS_SCHEMA = pa.schema(
    [
        ("Classifier_ID", pa.string()),
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

        with ClassificationWriter(output_path) as writer:
            writer.write_predictions(...)
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: pq.ParquetWriter | None = None

    def __enter__(self) -> ClassificationWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def write_predictions(
        self,
        classifier_id: str,
        plates: np.ndarray,
        wells: np.ndarray,
        object_numbers: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        channel: str,
        is_control: bool,
    ) -> None:
        """Append one classifier's predictions to the Parquet file."""
        n = len(labels)
        table = pa.table(
            {
                "Classifier_ID": pa.array([classifier_id] * n, type=pa.string()),
                "Metadata_Plate": pa.array(plates, type=pa.string()),
                "Metadata_well_position": pa.array(wells, type=pa.string()),
                "Metadata_ObjectNumber": pa.array(
                    object_numbers, type=pa.int64()
                ),
                "Label": pa.array(labels, type=pa.int8()),
                "Prediction": pa.array(predictions, type=pa.float32()),
                "Feature_Type": pa.array([channel] * n, type=pa.string()),
                "Control": pa.array([is_control] * n, type=pa.bool_()),
            },
            schema=PREDICTIONS_SCHEMA,
        )
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                str(self._path), PREDICTIONS_SCHEMA, compression="zstd"
            )
        self._writer.write_table(table)

    def close(self) -> None:
        """Close the underlying ParquetWriter."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.info("Predictions written to %s", self._path)
