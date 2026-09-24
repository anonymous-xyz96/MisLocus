#!/usr/bin/env python3
"""Gate and run the six v2 fits from a clean committed checkout (Linux/systemd).

Fresh output only. Without --production, stop after release checks. No extraction
of T4 features or downstream analyses. At most two training jobs use GPUs0/1 and
2/3; every job is bounded by native cgroup RAM/CPU/task limits and monitored.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from prot_loc_benchmark.provenance import capture_source, code_fingerprint, save_json, sha256, verify_source

GIB = 1024 ** 3
LIMITS = {'MemoryMax': '256G', 'MemoryHigh': '192G', 'CPUQuota': '1600%', 'TasksMax': '512',
          'OOMPolicy': 'stop', 'KillMode': 'control-group', 'TimeoutStopSec': '120'}


def host_resources(output):
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {'available_memory_bytes': int(memory['MemAvailable'].split()[0]) * 1024,
            'free_disk_bytes': shutil.disk_usage(output).free,
            'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,temperature.gpu,utilization.gpu',
                                            '--format=csv,noheader,nounits'], text=True).strip()}


def require_resources(resources, *, starting=False):
    if resources['available_memory_bytes'] < (640 if starting else 128) * GIB:
        raise RuntimeError('Insufficient host RAM headroom; refusing to risk shared-host memory pressure')
    if resources['free_disk_bytes'] < (100 if starting else 32) * GIB:
        raise RuntimeError('Insufficient disk headroom')
    rows = [row.split(',') for row in resources['gpu'].splitlines()]
    if len(rows) != 4 or any(int(row[4]) >= 85 for row in rows):
        raise RuntimeError('Expected four GPUs below85C')
    if starting and any(int(row[2]) > 1024 for row in rows):
        raise RuntimeError('GPU memory is occupied; refusing to share a device with another job')


def job_status(state):
    if state['ActiveState'] in ('failed', 'inactive') or state['Result'] != 'success':
        return 'failed'
    if state['SubState'] == 'exited':
        return 'success' if state['ExecMainStatus'] == '0' else 'failed'
    return 'running' if state['SubState'] == 'running' else 'starting'


def verify_completed_run(run, config, fingerprint):
    # Same native-score/hash verifier as extraction. mmap keeps full model and
    # optimizer tensors out of the small controller's resident memory.
    import torch
    from prot_loc_benchmark.representations.subcell_run import verify_selection

    run = Path(run)
    identity = json.loads((run / 'run.json').read_text())
    checkpoint_path = run / 'models/best_model_ap.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    selection = verify_selection(checkpoint_path, checkpoint, run / 'selection.json')
    del checkpoint
    completed = [json.loads(p.read_text()) for p in run.glob('attempts/*/completed.json')]
    if (identity['kind'] != 'production' or identity['code_sha256'] != fingerprint
            or identity['config'] != config or selection['identity'] != identity
            or not any(c.get('identity') == identity and c.get('status') == 'fit_completed'
                       and c.get('selection') == selection
                       and c.get('global_step', -1) >= selection['global_step'] for c in completed)):
        raise RuntimeError('Production completion/selection binding mismatch')
    verify_source(run, fingerprint)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--weights-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--production', action='store_true', help='Authorize production ONLY after all checks pass')
    args = parser.parse_args()
    args.output, args.preflight, args.weights_root = [p.resolve() for p in (args.output, args.preflight, args.weights_root)]
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', 'HEAD'], check=True)
    if subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '--others', '--exclude-standard'], text=True).strip():
        raise RuntimeError('Start from a clean committed worktree, not the documentation working tree')
    evidence = json.loads((args.preflight / 'preflight.json').read_text())
    if args.output.is_relative_to(Path(evidence['release_root'])):
        raise ValueError('Campaign output must not be inside the HF mirror')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'configs').mkdir()
    (args.output / 'logs').mkdir()
    capture_source(args.output)
    fingerprint = code_fingerprint()
    tracker = {'started_at': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
               'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
               'code_sha256': fingerprint, 'source_archive_sha256': sha256(args.output / 'source.tar.gz'),
               'preflight_sha256': sha256(args.preflight / 'preflight.json'),
               'release_revision': evidence['release']['revision'], 'limits_per_job': LIMITS,
               'production_authorized': args.production, 'phase': 'preparing', 'jobs': {}, 'configs': {}}

    def update(phase):
        tracker['phase'] = phase
        tracker['updated_at'] = datetime.now(timezone.utc).isoformat()
        save_json(args.output / 'campaign.json', tracker)

    def run_jobs(jobs):
        if code_fingerprint() != fingerprint:
            raise RuntimeError('Source changed during campaign')
        require_resources(host_resources(args.output), starting=True)
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if occupied:
            raise RuntimeError('A GPU compute process already exists; refusing overlapping jobs')
        active = {}
        try:
            for name, (gpus, command) in jobs.items():
                unit = f'subcell-{uuid.uuid4().hex[:12]}-{name}'
                log = args.output / 'logs' / f'{name}.log'
                argv = ['systemd-run', '--user', '--unit=' + unit, '-p', 'Type=exec', '-p', 'RemainAfterExit=yes',
                        '-p', f'WorkingDirectory={ROOT}', '-p', f'StandardOutput=append:{log}',
                        '-p', f'StandardError=append:{log}']
                for key, value in LIMITS.items():
                    argv += ['-p', f'{key}={value}']
                if controller := os.environ.get('SUBCELL_CONTROLLER_UNIT'):
                    argv += ['-p', f'BindsTo={controller}', '-p', f'After={controller}']
                env = {'PATH': os.environ['PATH'], 'CUDA_VISIBLE_DEVICES': gpus, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'OMP_THREAD_LIMIT': '1',
                       'LD_LIBRARY_PATH': '/run/opengl-driver/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
                       'PYTHONPATH': f'{ROOT}/src:{ROOT}/vendor/subcell_embed:{ROOT}/vendor/subcellportable',
                       'PYTHONUNBUFFERED': '1'}
                for key, value in env.items():
                    argv.append(f'--setenv={key}={value}')
                argv += command
                tracker['jobs'][name] = {'unit': unit, 'command': argv, 'log': str(log), 'status': 'starting'}
                active[name] = unit
                update(tracker['phase'])
                subprocess.run(argv, check=True)
            while active:
                resources = host_resources(args.output)
                require_resources(resources)
                for name, unit in list(active.items()):
                    text = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p',
                        'ActiveState,SubState,ExecMainStatus,Result,MemoryCurrent,MemoryPeak,CPUUsageNSec,TasksCurrent,ControlGroup'], text=True)
                    state = dict(line.split('=', 1) for line in text.splitlines())
                    tracker['jobs'][name]['systemd'] = state
                    tracker['jobs'][name]['status'] = job_status(state)
                    cg = Path('/sys/fs/cgroup') / state['ControlGroup'].lstrip('/')
                    if (cg / 'memory.max').exists():
                        applied = {k: (cg / k).read_text().strip() for k in ('memory.max', 'memory.high', 'cpu.max', 'pids.max')}
                        if applied != {'memory.max': str(256 * GIB), 'memory.high': str(192 * GIB),
                                       'cpu.max': '1600000 100000', 'pids.max': '512'}:
                            raise RuntimeError(f'Resource limits were not applied: {applied}')
                        tracker['jobs'][name]['applied_limits'] = applied
                    if tracker['jobs'][name]['status'] == 'success':
                        if 'applied_limits' not in tracker['jobs'][name]:
                            raise RuntimeError(f'No verified resource-limit receipt for {name}')
                        subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
                        del active[name]
                    elif tracker['jobs'][name]['status'] == 'failed':
                        raise RuntimeError(f'{name} failed: {state}')
                with (args.output / 'resources.jsonl').open('a') as stream:
                    stream.write(json.dumps({'time': time.time(), **resources, 'jobs': tracker['jobs']}) + '\n')
                update(tracker['phase'])
                if active:
                    time.sleep(15)
        finally:
            for unit in active.values():
                subprocess.run(['systemctl', '--user', 'stop', unit], check=False)

    def config_for(family, seed, check=False):
        config = yaml.safe_load((ROOT / f'configs/subcell_finetune_{family}_s{seed}.yaml').read_text())
        weights = 'mae_contrast_supcon_model' if family == 'mae' else 'vit_supcon_model'
        name = f'{family}-check' if check else f'{family}-s{seed}'
        config.update(preflight=str(args.preflight), pretrained_weights=str(args.weights_root / weights / 'encoder.pth'),
                      output=str(args.output / ('checks' if check else 'runs') / name), devices=2, workers=2)
        path = args.output / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        tracker['configs'][name] = {'path': str(path), 'sha256': sha256(path), 'resolved': config}
        return path

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        update('unit-checks')
        env = {**os.environ, 'PYTHONPATH': f'{ROOT}/src:{ROOT}/vendor/subcell_embed', 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        with (args.output / 'logs/unit.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], cwd=ROOT,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        update('frozen-parity')
        run_jobs({'parity': ('0', [sys.executable, str(ROOT / 'tests/subcell_frozen_parity.py'),
                                  '--preflight', str(args.preflight), '--weights-root', str(args.weights_root),
                                  '--device', 'cuda:0', '--output', str(args.output / 'parity.json')])})
        update('full-validation-and-resume')
        launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
        checks = {family: config_for(family, 42, check=True) for family in ('mae', 'vit')}
        run_jobs({family + '-check': (gpus, launch + [str(ROOT / 'tests/subcell_release_check.py'), '--config', str(checks[family])])
                  for family, gpus in (('mae', '0,1'), ('vit', '2,3'))})
        for family in ('mae', 'vit'):
            passed = json.loads((args.output / 'checks' / f'{family}-check/passed.json').read_text())
            if passed['identity']['code_sha256'] != fingerprint or not passed['native_resume_passed']:
                raise RuntimeError('Release gate receipt mismatch')
        verify_source(args.output, fingerprint)
        update('gates-passed')
        if args.production:
            for seed in (42, 43, 44):
                configs = {family: config_for(family, seed) for family in ('mae', 'vit')}
                update(f'production-seed-{seed}')
                run_jobs({f'{family}-s{seed}': (gpus, launch + [str(ROOT / 'scripts/08c_train_subcell_finetune.py'),
                                                          '--config', str(configs[family]), '--fit'])
                          for family, gpus in (('mae', '0,1'), ('vit', '2,3'))})
                for family in ('mae', 'vit'):
                    run = args.output / 'runs' / f'{family}-s{seed}'
                    verify_completed_run(run, tracker['configs'][f'{family}-s{seed}']['resolved'], fingerprint)
        update('complete')
    except BaseException as error:
        tracker['error'] = repr(error)
        update('failed')
        raise


if __name__ == '__main__':
    main()
