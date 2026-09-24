"""Fail-closed, input-bound stages for downstream analyses (not producer runs)."""

import json
import os
import platform
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from prot_loc_benchmark.config import DATA_DIR, REPO_ROOT
from prot_loc_benchmark.provenance import (
    capture_source,
    code_fingerprint,
    record,
    save_json,
    sha256,
    verify_source,
)


def bound_environment():
    """Call before importing array libraries; explicit caller settings are retained."""
    for key, default in {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        value = os.environ.setdefault(key, default)
        if not value.isdecimal() or int(value) < 1:
            raise ValueError(f"{key} must be an explicit positive thread count")


def xgboost_identity():
    import xgboost

    build = xgboost.build_info()
    return {**build, "library_sha256": sha256(build["libxgboost"])}


def cgroup_limits():
    path = next(
        (
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    limits = {}
    if path is not None:
        directory = Path("/sys/fs/cgroup") / path.lstrip("/")
        while directory.is_relative_to("/sys/fs/cgroup"):
            limits[str(directory)] = {
                name: (directory / name).read_text().strip()
                for name in ("cpu.max", "memory.max", "memory.high", "pids.max", "memory.events")
                if (directory / name).is_file()
            }
            directory = directory.parent
    return limits


def require_bounded_execution(representation):
    limits = cgroup_limits()
    if representation.startswith("subcell_allele_rybg_v2_"):
        name = os.environ.get("MISLOCUS_CAMPAIGN_SLICE", "")
        campaign = next((v for p, v in limits.items() if Path(p).name == name), {})
        if (
            not name.startswith("mislocus-downstream-")
            or not name.endswith(".slice")
            or any(
                campaign.get(k, "max").split()[0] == "max" for k in ("cpu.max", "memory.max", "memory.high", "pids.max")
            )
        ):
            raise ValueError("Production SubCell stages require a bounded shared campaign cgroup slice")
        job = next(iter(limits.values()), {})
        if any(job.get(k, "max").split()[0] == "max" for k in ("cpu.max", "memory.max", "pids.max")):
            raise ValueError("Each production stage also requires finite job CPU/RAM/task cgroups")
        quota, period = map(int, campaign["cpu.max"].split())
        if (
            quota / period > 64
            or int(campaign["memory.max"]) > 512 * 2**30
            or int(campaign["memory.high"]) > 384 * 2**30
            or int(campaign["pids.max"]) > 2048
        ):
            raise ValueError("Campaign cgroup exceeds approved CPU/RAM/task ceilings")
    return limits


def require_stage(directory, artifact=None, *, representation=None, batch=None):
    directory = Path(directory)
    path = directory / "stage.json"
    if not path.is_file():
        raise ValueError(f"Missing completed parent stage: {path}")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "complete":
        raise ValueError(f"Incomplete parent stage: {path}")
    parameters = receipt["parameters"]
    identity = parameters.get("context", parameters)
    if (representation is not None and identity.get("representation") != representation) or (
        batch is not None and identity.get("batch") != batch
    ):
        raise ValueError("Parent stage representation/batch mismatch")
    verify_source(directory, receipt["code_sha256"])
    selected = receipt["outputs"] if artifact is None else {artifact: receipt["outputs"].get(artifact)}
    for name, digest in selected.items():
        candidate = directory / name
        if (
            not digest
            or not candidate.resolve().is_relative_to(directory.resolve())
            or not candidate.is_file()
            or sha256(candidate) != digest
        ):
            raise ValueError(f"Changed or missing stage output: {candidate}")
    for parent, digest in receipt["parents"].items():
        if sha256(parent) != digest or json.loads(Path(parent).read_text()).get("status") != "complete":
            raise ValueError(f"Changed or incomplete stage parent: {parent}")
    return path


@contextmanager
def stage(directory, inputs, parameters, *, parents=(), allowed=()):
    """Publish stage.json last. A start/failed marker is never a completion receipt.

    Inputs may be file-only symlinks. Output directories/files may not escape
    the explicit analysis root or overwrite any artifact of a previous attempt.
    """
    representation = parameters.get("representation", parameters.get("context", {}).get("representation", ""))
    limits = require_bounded_execution(representation)
    directory = Path(directory)
    root = DATA_DIR.resolve()
    if not os.environ.get("MISLOCUS_DATA_ROOT") or root == (REPO_ROOT / "data").resolve():
        raise ValueError("Downstream stages require an explicit independent MISLOCUS_DATA_ROOT")
    if not directory.resolve().is_relative_to(root) or directory.resolve() == root:
        raise ValueError("Stage output escapes its analysis root")
    directory.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in directory.iterdir()} - set(allowed)
    if unexpected:
        raise FileExistsError(f"Stage artifacts already exist: {directory}: {sorted(unexpected)}")
    inputs = [Path(p) for p in inputs]
    input_hashes = {str(p.resolve()): sha256(p) for p in inputs}
    parent_hashes = {str(Path(p).resolve()): sha256(p) for p in parents}
    for parent in parents:
        parent = Path(parent).resolve()
        receipt = json.loads(parent.read_text())
        if receipt.get("status") != "complete":
            raise ValueError(f"Incomplete parent: {parent}")
        # Reuse the hashes just computed, closing the gap between require_stage()
        # and this transaction without rereading large parent artifacts.
        for name, digest in receipt["outputs"].items():
            candidate = str((parent.parent / name).resolve())
            if candidate in input_hashes and input_hashes[candidate] != digest:
                raise ValueError(f"Changed parent-bound stage input: {candidate}")
    code = code_fingerprint()
    started = datetime.now(UTC).isoformat()
    binding = {
        "schema_version": 1,
        "status": "started",
        "started_at": started,
        "stage_id": uuid.uuid4().hex,
        "stage": str(directory.resolve()),
        "inputs": input_hashes,
        "parents": parent_hashes,
        "parameters": parameters,
        "code_sha256": code,
        "runtime": {
            "python": platform.python_version(),
            "host": platform.node(),
            "xgboost_build": xgboost_identity(),
            "allocated_gpu": parameters.get("gpu_allocation"),
            **{
                p: version(p)
                for p in ("numpy", "polars", "pandas", "pyarrow", "scipy", "pycytominer", "xgboost", "copairs")
            },
        },
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "MISLOCUS_CLASSIFIER_BACKEND",
                "MISLOCUS_DATA_ROOT",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "POLARS_MAX_THREADS",
                "NUMEXPR_NUM_THREADS",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
                "PYTHONNOUSERSITE",
                "MISLOCUS_CAMPAIGN_SLICE",
            )
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_limits": limits,
    }
    # Exclusive create arbitrates concurrent starts, even before expensive work.
    with (directory / "started.json").open("x") as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
    t0 = time.monotonic()
    try:
        capture_source(directory)
        import pyarrow as pa
        from threadpoolctl import threadpool_limits

        pa.set_cpu_count(int(os.environ.get("POLARS_MAX_THREADS", "2")))
        pa.set_io_thread_count(2)
        with threadpool_limits(limits=int(os.environ.get("OPENBLAS_NUM_THREADS", "1")), user_api="blas"):
            from threadpoolctl import threadpool_info

            binding["runtime"]["threadpools_before_task"] = threadpool_info()
            yield binding
        if input_hashes != {str(p.resolve()): sha256(p) for p in inputs} or code_fingerprint() != code:
            raise ValueError("Stage input or source changed during execution")
        if parent_hashes != {str(Path(p).resolve()): sha256(p) for p in parents}:
            raise ValueError("Stage parent changed during execution")
        verify_source(directory, code)
        record(
            [directory],
            input_paths=inputs + [Path(p) for p in parents],
            duration_seconds=time.monotonic() - t0,
            stage_id=binding["stage_id"],
        )
        outputs = {}
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if relative.parts[0] in allowed or not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"Output symlink/escape: {path}")
            outputs[str(relative)] = sha256(path)
        save_json(
            directory / "stage.json",
            {
                **binding,
                "status": "complete",
                "outputs": outputs,
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": time.monotonic() - t0,
                "final_cgroup_limits": require_bounded_execution(representation),
            },
        )
    except BaseException as error:
        for marker in ("completion.json", "calibration.json"):
            (directory / marker).unlink(missing_ok=True)
        save_json(directory / "failed.json", {"status": "failed", "error": repr(error), "started_at": started})
        raise
