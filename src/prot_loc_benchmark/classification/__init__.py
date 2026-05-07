"""XGBoost classification for variant mislocalization prediction."""

from .channels import get_feature_channels
from .cv import CVFold, generate_folds, split_fold
from .executor import run_classifier_tasks
from .io import ClassificationWriter
from .metrics import (
    aggregate_allele_metrics,
    compute_classifier_metrics,
    compute_null_threshold,
    load_single_fold_metrics,
)
from .pairs import (
    ClassificationPair,
    build_control_pairs,
    build_cpc_pairs,
    build_experimental_pairs,
    filter_pairs_by_scope,
    get_pair_data,
)
from .reporting import plot_auroc_distributions, write_wide_summary
from .train import select_device, train_and_predict

__all__ = [
    "ClassificationPair",
    "ClassificationWriter",
    "CVFold",
    "aggregate_allele_metrics",
    "build_control_pairs",
    "build_cpc_pairs",
    "build_experimental_pairs",
    "compute_classifier_metrics",
    "compute_null_threshold",
    "filter_pairs_by_scope",
    "generate_folds",
    "get_feature_channels",
    "get_pair_data",
    "load_single_fold_metrics",
    "plot_auroc_distributions",
    "run_classifier_tasks",
    "select_device",
    "split_fold",
    "train_and_predict",
    "write_wide_summary",
]
