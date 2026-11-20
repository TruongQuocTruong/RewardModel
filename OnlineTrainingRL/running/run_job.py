#!/usr/bin/env python3
"""Prepare a single-GPU run, own the Qwen3 server, and supervise training."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_ready(settings, process, timeout):
    deadline = time.monotonic() + timeout
    url = f'http://127.0.0.1:{settings["port"]}/ping'
    attempt = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Reward server exited: {process.returncode}; inspect its log')
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.load(response)
            if payload.get('ready') and payload.get('run_id') == settings['run_id']:
                print(f'Reward server ready: {url}', flush=True)
                return
        except (OSError, ValueError):
            pass
        if attempt % 12 == 0:
            print('Waiting for this job\'s frozen Qwen3 reward model...', flush=True)
        time.sleep(5)
        attempt += 1
    raise TimeoutError('Reward server did not become ready; inspect reward log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    bundle, run = args.bundle.resolve(), args.run_dir.resolve()
    original = json.loads((bundle / 'settings.json').read_text(encoding='utf-8'))
    reward_python = Path(original['reward_env']) / 'bin/python'
    if not reward_python.is_file():
        raise FileNotFoundError(f'Reward Python not found: {reward_python}')
    subprocess.run([sys.executable, '-B', str(bundle / 'prepare_run.py'), '--bundle', str(bundle),
                    '--run-dir', str(run), '--write-probe-mib', str(original.get('write_probe_mib', 512))], check=True)
    settings = json.loads((run / 'settings.json').read_text())
    log = bundle / 'logs' / f'reward_{run.name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    reward = policy = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with log.open('w', encoding='utf-8') as stream:
            reward = subprocess.Popen([str(reward_python), '-B', str(run / 'reward_server.py'),
                                       '--settings', str(run / 'settings.json')],
                                      stdout=stream, stderr=subprocess.STDOUT, cwd=run)
            print(f'Reward log: {log}', flush=True)
            wait_ready(settings, reward, int(settings.get('reward_ready_timeout', 3600)))
            print(f'Train config: {run / "train.yml"}', flush=True)
            print(f'Checkpoints: {run / "checkpoints"}', flush=True)
            policy = subprocess.Popen([sys.executable, '-B', str(run / 'train.py'), '--config', str(run / 'train.yml')], cwd=run)
            while policy.poll() is None:
                if reward.poll() is not None:
                    raise RuntimeError('Reward server exited during training; stopping policy')
                time.sleep(5)
            if policy.returncode:
                raise RuntimeError(f'Policy exited with code {policy.returncode}')
            print('Training completed successfully.', flush=True)
    finally:
        stop(policy)
        stop(reward)


if __name__ == '__main__':
    main()
