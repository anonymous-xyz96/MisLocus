"""Model-independent staging of an immutable MisLocus crop release.

Standard library only. Preserves archive payload bytes; no image transforms,
split assignment, vocabulary fitting, model weights or training dependencies.
"""
from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

from prot_loc_benchmark.config import CELL_CROP_CHANNEL_FILES
from prot_loc_benchmark.provenance import capture_source, invocation, record, save_json, sha256


def release_inventory(root, *, extra_paths=()):
    """Pin all released crop shards/manifests; optional additional consumer inputs."""
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(
            ['git', '-c', f'safe.directory={root}', '-C', str(root), *args], text=True).strip()

    revision = git('rev-parse', 'HEAD')
    remote = git('remote', 'get-url', 'origin')
    if not re.fullmatch(r'[0-9a-f]{40}', revision) or 'huggingface.co/datasets/' not in remote:
        raise ValueError('Expected a pinned local Hugging Face dataset checkout')
    paths = git('ls-tree', '-r', '--name-only', revision).splitlines()
    manifests = {p for p in paths if p.startswith('manifest/')}
    shards = {p for p in paths if p.startswith('single_cell_crops/') and p.endswith('.tar.gz')}
    if not manifests or not shards:
        raise ValueError('Release must contain crop shards and canonical manifests')
    batches = set()
    for path in shards:
        parts = PurePosixPath(path).parts
        if len(parts) != 3 or not re.fullmatch(r'.+_Batch_[0-9]+', parts[1]):
            raise ValueError(f'Unexpected crop shard path: {path}')
        batches.add(parts[1])
    expected = {f'manifest/manifest_Batch_{b.rsplit("_", 1)[1]}.parquet' for b in batches}
    if manifests != expected:
        raise ValueError('Crop batches and release manifests do not agree')
    selected = manifests | shards | set(extra_paths)
    if not selected.issubset(paths):
        raise ValueError('Missing requested release inputs')
    files = {}
    for path in sorted(selected):
        pointer = git('show', f'{revision}:{path}')
        oid = re.search(r'^oid sha256:([0-9a-f]{64})$', pointer, re.M)
        size = re.search(r'^size (\d+)$', pointer, re.M)
        if oid is None or size is None:
            raise ValueError(f'Expected LFS content hash for {path}')
        local = root / path
        if local.stat().st_size != int(size[1]):
            raise ValueError(f'Incomplete/LFS-pointer download: {local}')
        files[path] = {'sha256': oid[1], 'size': int(size[1])}
        if path in manifests and sha256(local) != oid[1]:
            raise ValueError(f'Manifest differs from pinned revision: {local}')
    return {'revision': revision, 'remote': remote, 'files': files}


def extract(root, output):
    """Verify shards and unpack to NEW storage; completion receipt is written last."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Extraction must not write inside the Hugging Face mirror')
    started = time.perf_counter()
    inventory = release_inventory(root)
    output.mkdir(parents=True, exist_ok=False)
    capture_source(output)
    files = {}
    archives = []
    for relative, expected in inventory['files'].items():
        if not relative.endswith('.tar.gz'):
            continue
        archive = root / relative
        archives.append(archive)
        print(f'Verifying and extracting {relative}', flush=True)
        if sha256(archive) != expected['sha256']:
            raise ValueError(f'Shard hash differs from pinned release: {archive}')
        batch = relative.split('/')[1]
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                parts = PurePosixPath(member.name).parts
                if '..' in parts or PurePosixPath(member.name).is_absolute():
                    raise ValueError(f'Unsafe archive path: {member.name}')
                if member.isdir() and len(parts) == 1:
                    continue
                if (not member.isfile() or len(parts) != 2 or
                        parts[1] not in (*CELL_CROP_CHANNEL_FILES, 'metadata.parquet')):
                    raise ValueError(f'Unexpected crop member: {member.name}')
                destination = output / batch / Path(*parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with tar.extractfile(member) as source, destination.open('xb') as target:
                    while chunk := source.read(8 * 1024 * 1024):
                        target.write(chunk)
                        digest.update(chunk)
                stat = destination.stat()
                if stat.st_size != member.size:
                    raise ValueError(f'Truncated extraction: {destination}')
                files[str(destination.relative_to(output))] = {
                    'sha256': digest.hexdigest(), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    # No model eligibility or normalization decisions are made by this receipt.
    save_json(output / 'extraction.json', {'release': inventory, 'files': files,
              'invocation': invocation(), 'runtime': {'python': platform.python_version(), 'platform': platform.platform()},
              'source_archive_sha256': sha256(output / 'source.tar.gz')})
    record([output], input_paths=archives, duration_seconds=time.perf_counter() - started)
