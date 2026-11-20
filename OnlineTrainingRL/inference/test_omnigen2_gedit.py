"""CPU/mocked contract checks for the GEdit pretrained generator."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

import generate_omnigen2_gedit as generator


class GEditTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_inference_verification' / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def fixture(self):
        images = self.root / 'data/images/background_change'
        images.mkdir(parents=True)
        Image.new('RGB', (101, 63), 'red').save(images / 'source_name.png')
        metadata = self.root / 'metadata.jsonl'
        row = {'key': 'different_key', 'task_type': 'background_change',
               'image': 'images/background_change/source_name.png', 'instruction': 'Make background blue'}
        metadata.write_text(json.dumps(row) + '\n')
        paths = {name: self.root / name for name in ('text_encoder', 'vae_model')}
        for path in (paths['text_encoder'] / 'config.json', paths['vae_model'] / 'vae/config.json'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        weight = self.root / 'transformer/pytorch_model.bin'
        weight.parent.mkdir()
        weight.write_bytes(b'fake weight, never loaded')
        (weight.parent / 'config.json').write_text('{}')
        settings = self.root / 'settings.json'
        settings.write_text(json.dumps({**{key: str(value) for key, value in paths.items()}, 'omnigen_transformer_bin': str(weight)}))
        args = argparse.Namespace(metadata=metadata, settings=settings, data_root=self.root / 'data',
            output_root=self.root / 'edited', seed=20, steps=50, text_guidance=5.0,
            image_guidance=1.5, time_shift_base_res=168, max_long_edge=1024,
            cpu_offload=True, start_index=0, limit=None, overwrite=False)
        args.pretrained = generator.pretrained_paths(settings)
        args.rows = generator.read_pairs(metadata, args.data_root, args.output_root)
        return args, row

    def test_resize_padding_and_transparency(self):
        canvas, content = generator.prepare_source(Image.new('RGB', (1600, 800)), 1024)
        self.assertEqual(content, (1024, 512))
        self.assertEqual(canvas.size, content)
        canvas, content = generator.prepare_source(Image.new('RGBA', (101, 63), (0, 0, 0, 0)), 1024)
        self.assertEqual(canvas.size, (112, 64))
        self.assertEqual(content, (101, 63))
        self.assertEqual(canvas.getpixel((0, 0)), (255, 255, 255))

    def test_mapping_and_collision(self):
        args, row = self.fixture()
        self.assertEqual(args.rows[0]['target'], args.output_root / 'background_change/source_name.png')
        other = {**row, 'key': 'another_key', 'instruction': 'Make background green'}
        args.metadata.write_text(json.dumps(row) + '\n' + json.dumps(other))
        with self.assertRaisesRegex(ValueError, 'same edited image'):
            generator.read_pairs(args.metadata, args.data_root, args.output_root)

    def test_dry_run_does_not_load_model(self):
        args, _ = self.fixture()
        argv = ['generate_omnigen2_gedit.py', '--settings', str(args.settings), '--metadata', str(args.metadata),
                '--data-root', str(args.data_root), '--output-root', str(args.output_root), '--dry-run']
        with patch.object(sys, 'argv', argv), patch.object(generator, 'generate') as generate:
            generator.main()
        generate.assert_not_called()
        self.assertFalse(args.output_root.exists())

    def test_explicit_paths_and_limit_without_settings(self):
        args, row = self.fixture()
        args.settings.unlink()
        image = args.data_root / 'images/background_change/source_two.png'
        Image.new('RGB', (64, 32)).save(image)
        second = {**row, 'key': 'second', 'image': 'images/background_change/source_two.png'}
        args.metadata.write_text(json.dumps(row) + '\n' + json.dumps(second) + '\n')
        argv = ['generate_omnigen2_gedit.py', '--transformer-path', str(args.pretrained['transformer_dir']),
                '--text-encoder', str(args.pretrained['text_encoder']), '--vae-model', str(args.pretrained['vae_model']),
                '--metadata', str(args.metadata), '--data-root', str(args.data_root),
                '--output-root', str(args.output_root)]
        with patch.object(sys, 'argv', argv), patch.object(generator, 'generate') as generate:
            generator.main()
        self.assertEqual(len(generate.call_args.args[0].rows), 2)
        with patch.object(sys, 'argv', argv + ['--limit', '1']), patch.object(generator, 'generate') as generate:
            generator.main()
        self.assertEqual(len(generate.call_args.args[0].rows), 1)
        self.assertFalse(args.settings.exists())
        with self.assertRaises(FileNotFoundError):
            generator.pretrained_paths(transformer=self.root / 'missing',
                text_encoder=args.pretrained['text_encoder'], vae_model=args.pretrained['vae_model'])

    def test_common_pretrained_root(self):
        args, _ = self.fixture()
        root = self.root / 'pretrained'
        shutil.copytree(args.pretrained['text_encoder'], root / 'Qwen2.5-VL-3B-Instruct')
        shutil.copytree(args.pretrained['vae_model'], root / 'FLUX.1-dev')
        shutil.copytree(args.pretrained['transformer_dir'], root / 'OmniGen2/transformer')
        paths = generator.pretrained_paths(pretrained_root=root)
        self.assertEqual(paths['transformer_bin'], (root / 'OmniGen2/transformer/pytorch_model.bin').resolve())
        self.assertEqual(paths['text_encoder'], (root / 'Qwen2.5-VL-3B-Instruct').resolve())

    def test_generation_resume_and_config_guard(self):
        args, _ = self.fixture()
        calls = []
        class Seed:
            def manual_seed(self, value):
                self.value = value
                return self
        def pipe(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[kwargs['input_images'][0][0].copy()])
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: None),
            inference_mode=nullcontext, autocast=lambda *a, **k: nullcontext(),
            bfloat16='bf16', Generator=lambda **kwargs: Seed())
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        with patch.dict(sys.modules, {'torch': fake_torch, 'fcntl': fake_fcntl}), \
             patch.object(generator, 'ROOT', self.root), \
             patch.object(generator, 'load_pipeline', return_value=(pipe, 'negative', 'mask')) as load:
            generator.generate(args)
            target = args.rows[0]['target']
            self.assertTrue(generator.valid_png(target, (101, 63)))
            self.assertEqual(calls[0]['num_inference_steps'], 50)
            self.assertEqual(calls[0]['generator'].value, 20)
            self.assertEqual(calls[0]['size'], [(112, 64)])
            self.assertEqual(calls[0]['cfg_range'], (0.0, 1.0))
            self.assertNotIn('lora', calls[0])
            generator.generate(args)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual([p.relative_to(args.output_root).as_posix() for p in args.output_root.rglob('*') if p.is_file()],
                             ['background_change/source_name.png'])
            args.seed = 25
            with self.assertRaisesRegex(ValueError, 'config differs'):
                generator.generate(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
