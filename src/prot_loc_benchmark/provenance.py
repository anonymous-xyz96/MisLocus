"""
Data provenance tracking for pipeline outputs.

Hybrid system:
  1. Per-directory ``_provenance.json`` sidecars with detailed file metadata
  2. Central ``data/provenance_log.json`` (git-tracked) with one entry per
     script invocation

Usage in scripts::

    from prot_loc_benchmark.provenance import record

    record(
        output_dirs=[out_dir],
        input_paths=[profiles_path, manifest_path],
        duration_seconds=elapsed,
    )
"""

from __future__ import annotations

import fcntl
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from prot_loc_benchmark.config import DATA_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

PROVENANCE_LOG = DATA_DIR / "provenance_log.json"
SIDECAR_NAME = "_provenance.json"
SCHEMA_VERSION = 1

# Default file suffixes to skip when scanning output directories
SKIP_SUFFIXES = frozenset({".png", ".pdf", ".log"})

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

_git_cache: dict | None = None


def get_git_info() -> dict:
    """Return ``{"commit": "<short-hash>", "dirty": bool}``; cached per process."""
    global _git_cache
    if _git_cache is not None:
        return _git_cache

    import subprocess

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            .stdout.strip()
        )
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet"],
                capture_output=True,
                cwd=REPO_ROOT,
            ).returncode
            != 0
        )
    except FileNotFoundError:
        commit = "unknown"
        dirty = False

    _git_cache = {"commit": commit, "dirty": dirty}
    return _git_cache


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def rel_path(p: Path) -> str:
    """Return *p* relative to the repo root, falling back to str(p)."""
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# File metadata helpers
# ---------------------------------------------------------------------------


def file_meta(path: Path) -> dict:
    """Return size + parquet metadata if applicable."""
    info: dict = {"size_bytes": path.stat().st_size}
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            meta = pq.read_metadata(str(path))
            info["rows"] = meta.num_rows
            info["columns"] = meta.num_columns
        except Exception:
            logger.warning("provenance: failed to read parquet metadata: %s", path)
    return info


def scan_dir_files(
    directory: Path,
    extra_skip_suffixes: frozenset[str] = frozenset(),
) -> dict[str, dict]:
    """Collect metadata for trackable files in *directory* (non-recursive)."""
    skip = SKIP_SUFFIXES | extra_skip_suffixes
    files: dict[str, dict] = {}
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.suffix in skip or p.name == SIDECAR_NAME:
            continue
        files[p.name] = file_meta(p)
    return files


# ---------------------------------------------------------------------------
# Central log helpers
# ---------------------------------------------------------------------------


def read_log() -> dict:
    """Read the central provenance log, returning empty structure if absent."""
    if PROVENANCE_LOG.exists():
        return json.loads(PROVENANCE_LOG.read_text())
    return {"schema_version": SCHEMA_VERSION, "runs": []}


def write_log(data: dict) -> None:
    """Atomically write the central provenance log."""
    tmp = PROVENANCE_LOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.rename(PROVENANCE_LOG)


def _make_run_id(script_name: str, timestamp: str) -> str:
    """Generate a human-readable run ID from timestamp + script."""
    # Strip all non-digit characters, keep first 14 digits (YYYYMMDDHHmmss)
    digits = "".join(c for c in timestamp if c.isdigit())[:14]
    stem = Path(script_name).stem
    return f"{digits}_{stem}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record(
    output_dirs: list[Path],
    input_paths: list[Path] | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Record provenance for a script run.

    Writes both per-directory sidecars and one entry in the central log.

    Parameters
    ----------
    output_dirs
        Directories where outputs were written (each gets a sidecar).
    input_paths
        Key input files consumed by this script (optional).
    duration_seconds
        Wall-clock duration of the run (optional).
    """
    git = get_git_info()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    argv = sys.argv
    script = argv[0] if argv else "unknown"

    rel_output_dirs = [rel_path(d) for d in output_dirs]
    rel_inputs = [rel_path(p) for p in (input_paths or [])]

    # --- Per-directory sidecars ---
    for out_dir in output_dirs:
        if not out_dir.is_dir():
            logger.warning("provenance: output dir does not exist: %s", out_dir)
            continue
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "script": rel_path(Path(script)),
            "cli_args": argv[1:],
            "timestamp": now,
            "inputs": rel_inputs,
            "files": scan_dir_files(out_dir),
        }
        if duration_seconds is not None:
            sidecar["duration_seconds"] = round(duration_seconds, 1)

        sidecar_path = out_dir / SIDECAR_NAME
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    # --- Central log ---
    run_id = _make_run_id(script, now)
    entry: dict = {
        "id": run_id,
        "script": rel_path(Path(script)),
        "cli_args": argv[1:],
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "timestamp": now,
        "output_dirs": rel_output_dirs,
    }
    if duration_seconds is not None:
        entry["duration_seconds"] = round(duration_seconds, 1)

    # Use file locking to prevent concurrent writes from losing entries
    PROVENANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    lock_path = PROVENANCE_LOG.with_suffix(".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        log = read_log()
        log["runs"].append(entry)
        write_log(log)

    logger.info(
        "provenance: recorded %d output dir(s) → %s (run %s)",
        len(output_dirs),
        PROVENANCE_LOG.name,
        run_id,
    )
