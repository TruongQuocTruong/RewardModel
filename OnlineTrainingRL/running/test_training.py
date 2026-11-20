"""CPU contract tests; does not import/load the real OmniGen2 or Qwen models."""
import ast
import base64
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from PIL import Image
import prepare_run
import reward_runtime
import reward_server

BUNDLE = Path(__file__).resolve().parent
FIXTURES = BUNDLE / '_verification'


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = FIXTURES / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_metadata_is_unique_source_instruction_not_key(self):
        Image.new('RGB', (64, 32)).save(self.root / 'source.png')
        metadata = self.root / 'metadata.jsonl'
        metadata.write_text('\n'.join(json.dumps(row) for row in [
            {'key': 'a', 'instruction': 'Make red', 'source_img': 'source.png'},
            {'key': 'a', 'instruction': 'Make red', 'source_img': 'source.png'},
            {'key': 'a', 'instruction': 'Make blue', 'source_img': 'source.png'},
        ]))
        rows = prepare_run.prepare_rows(metadata, self.root)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(json.loads(rows[0]['meta_data'])['sample_key'],
                            json.loads(rows[1]['meta_data'])['sample_key'])
        self.assertEqual(rows[0]['input_images'], [str((self.root / 'source.png').resolve())])
        self.assertEqual(json.loads(rows[0]['meta_data'])['instruction'], 'Make red')

    def test_missing_source_fails_before_training(self):
        metadata = self.root / 'metadata.jsonl'
        metadata.write_text(json.dumps({'key': 'x', 'instruction': 'x', 'source_img': 'missing.png'}))
        with self.assertRaises(FileNotFoundError):
            prepare_run.prepare_rows(metadata, self.root)

    def test_write_probe_is_removed(self):
        prepare_run.probe_write_capacity(self.root, 1)
        self.assertFalse((self.root / '.checkpoint_write_probe').exists())
        with self.assertRaises(ValueError):
            prepare_run.probe_write_capacity(self.root, -1)

    def test_offline_model_uses_main_revision(self):
        base = self.root / 'models--owner--model'
        (base / 'refs').mkdir(parents=True)
        (base / 'refs/main').write_text('abc')
        snapshot = base / 'snapshots/abc'
        snapshot.mkdir(parents=True)
        (snapshot / 'config.json').write_text('{}')
        self.assertEqual(prepare_run.resolve_model('', 'owner/model', 'config.json', [self.root]), snapshot.resolve())
        with self.assertRaises(FileNotFoundError):
            prepare_run.resolve_model('', 'owner/absent', 'config.json', [self.root])

    def test_best_student_checkpoint_is_selected(self):
        (self.root / 'best.pt').write_bytes(b'best')
        (self.root / 'other.pt').write_bytes(b'other')
        self.assertEqual(reward_runtime.checkpoint_files(self.root), self.root / 'best.pt')

    def test_prepare_freezes_runnable_job_config(self):
        import yaml
        bundle = self.root / 'bundle'
        bundle.mkdir()
        project = self.root / 'qwen_project'
        (project / 'src/models').mkdir(parents=True)
        (project / 'src/models/custom_qwen3_vl_reranker.py').write_text('# fake architecture')
        reward = self.root / 'reward_pretrained'
        reward.mkdir()
        (reward / 'config.json').write_text('{}')
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'best.pt').write_bytes(b'fake')
        text = self.root / 'text'
        text.mkdir()
        (text / 'config.json').write_text('{}')
        vae = self.root / 'flux/vae'
        vae.mkdir(parents=True)
        (vae / 'config.json').write_text('{}')
        transformer = self.root / 'transformer.bin'
        transformer.write_bytes(b'fake')
        Image.new('RGB', (64, 32)).save(self.root / 'source.png')
        metadata = self.root / 'metadata.jsonl'
        metadata.write_text(json.dumps({'key': 'a', 'source_img': 'source.png', 'instruction': 'Make red'}))
        settings = json.loads((BUNDLE / 'settings.json').read_text())
        settings.update(editscore_repo=str((BUNDLE.parent / 'EditScore').resolve()), metadata=str(metadata.resolve()), data_root=str(self.root.resolve()),
                        reward_project=str(project.resolve()), reward_pretrained=str(reward.resolve()),
                        reward_checkpoint=str(checkpoint.resolve()), omnigen_transformer_bin=str(transformer.resolve()),
                        text_encoder=str(text.resolve()), vae_model=str(vae.parent.resolve()))
        (bundle / 'settings.json').write_text(json.dumps(settings))
        for name in ('train.yml', 'train.py', 'compact_checkpoint.py', 'reward_server.py', 'reward_runtime.py'):
            shutil.copy2(BUNDLE / name, bundle / name)
        run = self.root / 'job_1'
        with patch.object(sys, 'argv', ['prepare_run.py', '--bundle', str(bundle), '--run-dir', str(run)]):
            prepare_run.main()
        config = yaml.safe_load((run / 'train.yml').read_text())
        self.assertEqual(config['model']['pretrained_model_path'], str(transformer.resolve()))
        self.assertEqual(config['train']['batch_size'], 12)
        self.assertEqual(config['train']['rl']['batch_size_per_forward'], 1)
        self.assertEqual(config['train']['rl']['num_update_steps_per_sampling'], 1)
        self.assertEqual(config['output_dir'], str((run / 'checkpoints').resolve()))
        data = json.loads((run / 'training_pairs.jsonl').read_text())
        self.assertEqual(data['input_images'], [str((self.root / 'source.png').resolve())])
        self.assertEqual((run / 'train.py').read_bytes(), (BUNDLE / 'train.py').read_bytes())
        self.assertEqual((run / 'repo_rl_path.txt').read_text(),
                         str((BUNDLE.parent / 'EditScore/examples/OmniGen2-RL').resolve()))
        self.assertEqual(config['train']['gradient_accumulation_steps'], 12)
        self.assertEqual(config['train']['max_train_steps'], 200)

    def test_running_resolves_sibling_repo_from_an_unrelated_cwd(self):
        import os
        settings = json.loads((BUNDLE / 'settings.json').read_text())
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            repo, rl = prepare_run.resolve_repo(BUNDLE, settings)
            self.assertEqual(repo, (BUNDLE.parent / 'EditScore').resolve())
            self.assertTrue((rl / 'omnigen2/grpo/utils.py').is_file())
        finally:
            os.chdir(previous)

    def test_trainer_bootstrap_does_not_depend_on_parent_repo_layout(self):
        bootstrap = (BUNDLE / 'train.py').read_text().split('import dotenv', 1)[0]
        namespace = {'__file__': str(BUNDLE / 'train.py')}
        with patch.object(sys, 'path', list(sys.path)):
            exec(compile(bootstrap, '<bootstrap>', 'exec'), namespace)
            self.assertEqual(Path(sys.path[0]), (BUNDLE.parent / 'EditScore/examples/OmniGen2-RL').resolve())

    def test_metric_average_and_no_implicit_division(self):
        out = {'scores_A': [12, 4, 200, 200, 200, 200],
               'scores_B': [-4, 2, 200, 200, 200, 200],
               'relevance_probs': [0.8, 0.5, 0.1, 0.1, 0.1, 0.1]}
        rewards, details = reward_server.selected_rewards([out], list('abcdef'), 0.5, 'error')
        self.assertEqual(rewards, [8, -1])
        self.assertEqual(details[0]['metric_scores'], {'a': 12, 'b': 4})
        out['relevance_probs'] = [0.1] * 6
        with self.assertRaisesRegex(ValueError, 'No relevant metric'):
            reward_server.selected_rewards([out], list('abcdef'), 0.5, 'error')

    def test_live_http_pair_order_and_failure_contract(self):
        def encoded(value):
            stream = BytesIO()
            Image.new('RGB', (32, 32), (value, 0, 0)).save(stream, format='PNG')
            return base64.b64encode(stream.getvalue()).decode()

        worker = reward_server.RewardWorker.__new__(reward_server.RewardWorker)
        worker.settings = {'relevance_threshold': 0.5, 'empty_metric_policy': 'error'}
        worker.runtime = (None, None, None, list('abcdef'), None)
        worker.requests = 0
        calls = []

        def forward(*args):
            a, b = Path(args[6]), Path(args[7])
            with Image.open(a) as image:
                av = image.getpixel((0, 0))[0]
            with Image.open(b) as image:
                bv = image.getpixel((0, 0))[0]
            calls.append((av, bv))
            return {'scores_A': [av] * 6, 'scores_B': [bv] * 6, 'relevance_probs': [1] * 6}

        @contextmanager
        def scratch(**kwargs):
            folder = self.root / 'scratch'
            folder.mkdir()
            try:
                yield str(folder)
            finally:
                shutil.rmtree(folder)

        payload = {'input_images': [[encoded(1)]] * 4,
                   'output_image': [encoded(v) for v in (10, 20, 30, 40)],
                   'meta_datas': [json.dumps({'sample_key': group, 'instruction': group})
                                  for group in ('a', 'b', 'a', 'b')]}
        server = reward_server.make_server(worker, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        try:
            with patch.object(reward_server, 'forward_pair', forward), \
                 patch.object(reward_server.tempfile, 'TemporaryDirectory', scratch):
                with urllib.request.urlopen(url + '/ping') as response:
                    self.assertTrue(json.load(response)['ready'])
                request = urllib.request.Request(url, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                with urllib.request.urlopen(request) as response:
                    result = json.load(response)
                self.assertEqual(calls, [(10, 30), (20, 40)])
                self.assertEqual(result['rewards'], [10, 20, 30, 40])
                self.assertEqual(result['meta_data'], payload['meta_datas'])
                payload['input_images'][2] = [encoded(2)]
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(urllib.request.Request(url, json.dumps(payload).encode()))
                self.assertEqual(caught.exception.code, 500)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_trainer_single_process_and_original_losses(self):
        source = (BUNDLE / 'train.py').read_text()
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        namespace = {}
        small = ast.Module(body=[functions[name] for name in
            ('_single_scatter_object_list', '_single_broadcast_object_list', '_single_all_gather_object')], type_ignores=[])
        exec(compile(small, '<single_process_helpers>', 'exec'), namespace)
        output = [None]
        namespace['_single_scatter_object_list'](output, [[1, 2]])
        self.assertEqual(output, [[1, 2]])
        namespace['_single_all_gather_object'](output, {'a': 3})
        self.assertEqual(output, [{'a': 3}])
        self.assertNotIn('torch.distributed.scatter_object_list(', source)
        self.assertIn('policy_loss * sample_steps * (sigma_ts[idx] / normalize_factor)', source)
        self.assertIn('/ (2 * sigma_t)', source)
        self.assertIn('compute_single_step_ppo_loss(', source)
        main = functions['main']
        # The original inner-loop break alone could continue on the next dataset batch.
        loops = [node for node in ast.walk(main) if isinstance(node, ast.For)
                 and ast.unparse(node.target) == '(step, batch)']
        self.assertEqual(len(loops), 1)
        self.assertIsInstance(loops[0].body[0], ast.If)
        self.assertIn('global_step >= args.train.max_train_steps', ast.unparse(loops[0].body[0]))
        self.assertIn('if not checkpoints or int(checkpoints[-1].split("-")[1]) < global_step:', source)

    def test_checkpoint_exports_peft_prefix_and_keeps_training_state(self):
        import torch
        from safetensors.torch import load_file
        tree = ast.parse((BUNDLE / 'train.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'save_training_checkpoint')
        namespace = {'Path': Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<checkpoint>', 'exec'), namespace)
        fake_peft = ModuleType('peft')
        fake_peft.get_peft_model_state_dict = lambda *args, **kwargs: {'layer.to_q.lora_A.weight': torch.ones(2, 3)}
        class Config:
            def save_pretrained(self, directory):
                (Path(directory) / 'adapter_config.json').write_text('{}')
        raw = SimpleNamespace(peft_config={'default': Config()})
        class Accelerator:
            def unwrap_model(self, model):
                return raw
            def save_state(self, directory):
                Path(directory).mkdir()
                (Path(directory) / 'optimizer.bin').write_bytes(b'full-state-kept')
        output = self.root / 'checkpoint-1'
        with patch.dict(sys.modules, {'peft': fake_peft}):
            namespace['save_training_checkpoint'](Accelerator(), raw, str(output))
        self.assertEqual((output / 'optimizer.bin').read_bytes(), b'full-state-kept')
        state = load_file(str(output / 'lora_adapter/adapter_model.safetensors'))
        self.assertEqual(list(state), ['base_model.model.layer.to_q.lora_A.weight'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
