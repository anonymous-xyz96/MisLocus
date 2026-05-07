"""CellProfiler profile preprocessing for variant classification.

Top-level attributes are lazily imported (PEP 562) so environments that don't
have heavier dependencies (e.g. the `subcell` env has no pycytominer) can still
`import prot_loc_benchmark.preprocessing.subcell` without triggering those
deps. Accessing a package-level attribute that requires the missing dep will
raise the usual ImportError only at that access.
"""

_LAZY_ATTRS = {
    # submodule → attrs
    ".annotate": ("annotate_controls",),
    ".clean": ("clip_outliers", "drop_nan_features"),
    ".feature_select": (
        "apply_blocklists",
        "apply_variance_threshold",
        "decorrelate_features",
        "prefilter_features",
    ),
    ".normalize": (
        "compute_plate_stats",
        "drop_dead_features",
        "robustmad",
        "select_variant_features",
    ),
    ".qc": ("drop_low_cell_count_wells", "filter_to_manifest"),
}
_ATTR_TO_MODULE = {name: mod for mod, names in _LAZY_ATTRS.items() for name in names}


def __getattr__(name: str):
    mod_name = _ATTR_TO_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(f"module 'prot_loc_benchmark.preprocessing' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(mod_name, __name__)
    attr = getattr(module, name)
    globals()[name] = attr  # cache so subsequent lookups skip the dispatch
    return attr


__all__ = sorted(_ATTR_TO_MODULE.keys())
