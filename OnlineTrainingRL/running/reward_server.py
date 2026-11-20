#!/usr/bin/env python3
"""Single-thread local HTTP reward worker; one frozen Qwen3 on the policy GPU."""
import argparse
import base64
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from reward_runtime import load_reward, forward_pair


def decode_image(encoded):
    from PIL import Image
    with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
        return image.convert('RGB').copy()


def selected_rewards(outputs, names, threshold, empty_policy):
    relevance = [sum(out['relevance_probs'][i] for out in outputs) / len(outputs)
                 for i in range(len(names))]
    selected = [i for i, value in enumerate(relevance) if value >= threshold]
    if not selected:
        if empty_policy == 'error':
            raise ValueError('No relevant metric meets threshold; refusing fabricated/zero reward')
        if empty_policy != 'highest_relevance':
            raise ValueError(f'Unknown empty_metric_policy: {empty_policy}')
        selected = [max(range(len(names)), key=lambda i: relevance[i])]
        print(f'WARNING: no selected metrics; using highest relevance: {names[selected[0]]}', flush=True)
    rewards, details = [], []
    for out in outputs:
        for field in ('scores_A', 'scores_B'):
            scores = {names[i]: float(out[field][i]) for i in selected}
            value = sum(scores.values()) / len(scores)
            if not math.isfinite(value):
                raise ValueError('Nonfinite reward')
            rewards.append(value)
            details.append({'metric_scores': scores, 'overall': value})
    return rewards, details


class RewardWorker:
    def __init__(self, settings):
        self.settings = settings
        args = SimpleNamespace(
            project_root=Path(settings['reward_project']),
            custom_model=Path(settings['reward_project']) / 'src/models/custom_qwen3_vl_reranker.py',
            checkpoint=Path(settings['reward_checkpoint']),
            pretrained_dir=Path(settings['reward_pretrained']), device='cuda:0',
            min_pixels=settings['reward_min_pixels'], max_pixels=settings['reward_max_pixels'],
            prompt_file=None,
        )
        self.runtime = load_reward(args)
        self.requests = 0

    def score(self, payload):
        import torch
        start = time.perf_counter()
        source_data, edited_data, metadata = (payload[key] for key in ('input_images', 'output_image', 'meta_datas'))
        if not len(source_data) == len(edited_data) == len(metadata) or not metadata:
            raise ValueError('Reward request length mismatch/empty')
        parsed = [json.loads(meta) if isinstance(meta, str) else meta for meta in metadata]
        groups = defaultdict(list)
        for i, meta in enumerate(parsed):
            if not isinstance(meta.get('instruction'), str) or not meta.get('sample_key'):
                raise ValueError('Expected raw instruction and sample_key in meta_data')
            groups[meta['sample_key']].append(i)
        rewards, details = [None] * len(metadata), [None] * len(metadata)
        model, processor, module, names, prompt = self.runtime
        # Temporary image transport is local scratch, automatically removed after this request.
        with tempfile.TemporaryDirectory(prefix='omnigen_reward_', dir=os.environ.get('SLURM_TMPDIR') or None) as tmp:
            root = Path(tmp)
            for group, indices in groups.items():
                if len(indices) % 2:
                    raise ValueError(f'{group}: group needs an even number of candidates')
                instruction = parsed[indices[0]]['instruction']
                source = decode_image(source_data[indices[0]][0])
                for i in indices:
                    if len(source_data[i]) != 1 or parsed[i]['instruction'] != instruction:
                        raise ValueError(f'{group}: source/instruction mismatch')
                    check = decode_image(source_data[i][0])
                    if check.size != source.size or check.tobytes() != source.tobytes():
                        raise ValueError(f'{group}: inconsistent source images')
                src = root / 'source.png'
                source.save(src)
                outputs = []
                for pair in range(0, len(indices), 2):
                    a, b = root / 'a.png', root / 'b.png'
                    decode_image(edited_data[indices[pair]]).save(a)
                    decode_image(edited_data[indices[pair + 1]]).save(b)
                    outputs.append(forward_pair(model, processor, module, prompt,
                                                str(src), instruction, str(a), str(b), 'cuda:0'))
                values, metrics = selected_rewards(outputs, names, self.settings['relevance_threshold'],
                                                   self.settings['empty_metric_policy'])
                for i, value, metric in zip(indices, values, metrics):
                    rewards[i], details[i] = value, metric
        self.requests += 1
        torch.cuda.empty_cache()
        print(f'Reward request={self.requests} groups={len(groups)} images={len(rewards)} '
              f'mean={sum(rewards)/len(rewards):.5f} min={min(rewards):.5f} max={max(rewards):.5f} '
              f'seconds={time.perf_counter()-start:.2f}', flush=True)
        return {'scores': details, 'rewards': rewards,
                'reasoning': [json.dumps(row) for row in details], 'meta_data': metadata}


def make_server(worker, port):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            encoded = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.respond(200 if self.path == '/ping' else 404,
                         {'ready': self.path == '/ping', 'run_id': worker.settings.get('run_id')})

        def do_POST(self):
            import traceback
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.respond(200, worker.score(payload))
            except Exception as exc:
                traceback.print_exc()
                self.respond(500, {'error': f'{type(exc).__name__}: {exc}'})

    # Binding failure is fatal, so a job cannot accidentally use another job's server.
    return HTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    if settings['empty_metric_policy'] not in ('error', 'highest_relevance'):
        raise ValueError('Invalid empty_metric_policy')
    worker = RewardWorker(settings)
    server = make_server(worker, settings['port'])
    print(f'Reward ready on 127.0.0.1:{settings["port"]}; frozen Qwen3; no /10 scaling', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
