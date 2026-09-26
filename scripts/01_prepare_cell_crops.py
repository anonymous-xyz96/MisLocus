#!/usr/bin/env python3
"""Inspect, extract or rehash a pinned HF crop release, for any model.

No downloads, resizing, normalization, segmentation or model dependencies.
Existing release manifests remain in the read-only source checkout. Extraction
preserves the released NPY and per-allele metadata bytes exactly.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prot_loc_benchmark.cell_crops import extract, release_inventory, verify_crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'extract', 'verify'])
    parser.add_argument('--release', type=Path, help='Local git/LFS snapshot (required for inspect/extract)')
    parser.add_argument('--crops', type=Path, help='NEW extraction destination, or existing crops to verify')
    args = parser.parse_args()
    if args.action == 'verify':
        if args.crops is None:
            parser.error('verify requires --crops; reads every payload but writes nothing')
        print(json.dumps(verify_crops(args.crops), indent=2))
        return
    if args.release is None:
        parser.error('inspect/extract require --release')
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
