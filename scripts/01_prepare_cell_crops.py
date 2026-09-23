#!/usr/bin/env python3
"""Inspect or extract a pinned HF crop release to separate storage, for any model.

No downloads, resizing, normalization, segmentation or model dependencies.
Existing release manifests remain in the read-only source checkout. Extraction
preserves the released NPY and per-allele metadata bytes exactly.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prot_loc_benchmark.cell_crops import extract, release_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'extract'])
    parser.add_argument('--release', type=Path, required=True, help='Local git/LFS Hugging Face snapshot')
    parser.add_argument('--crops', type=Path, help='NEW destination outside the release checkout')
    args = parser.parse_args()
    if args.action == 'inspect':
        inventory = release_inventory(args.release)
        shards = [p for p in inventory['files'] if p.endswith('.tar.gz')]
        print(json.dumps({'remote': inventory['remote'], 'revision': inventory['revision'],
                          'batches': sorted({p.split('/')[1] for p in shards}), 'shards': len(shards),
                          'compressed_bytes': sum(inventory['files'][p]['size'] for p in shards)}, indent=2))
        print('Inventory only; archive payload hashes are checked during extraction.')
    else:
        if args.crops is None:
            parser.error('extract requires --crops (new separate storage)')
        extract(args.release, args.crops)


if __name__ == '__main__':
    main()
