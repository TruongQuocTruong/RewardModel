"""Verify one-file configuration and external-source resolution without a GPU."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import generate_omnigen2_gedit as base
import generate_omnigen2_gedit_fireward as checkpoint
import test_omnigen2_gedit as base_tests


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_inference_verification' / self._testMethodName
        self.root.mkdir(parents=True)
        fixture = base_tests.GEditTests()
        fixture.root = self.root
        self.args, _ = fixture.fixture()
        self.settings = json.loads(self.args.settings.read_text())
        self.settings.update(editscore_repo=str((Path(__file__).resolve().parent.parent / 'EditScore').resolve()),
                             metadata=str(self.args.metadata), data_root=str(self.args.data_root),
                             output_base=str(self.root / 'base'), output_checkpoint=str(self.root / 'checkpoint-output'),
                             seed=5, steps=30, cpu_offload=True, limit=1)
    def tearDown(self):
        shutil.rmtree(self.root)

    def test_base_settings_only_does_not_require_checkpoint(self):
        self.settings['checkpoint'] = '/does/not/exist'
        self.args.settings.write_text(json.dumps(self.settings))
        with patch.object(sys, 'argv', ['base.py', '--settings', str(self.args.settings), '--dry-run']), \
             patch.object(base, 'generate') as generate, redirect_stdout(StringIO()) as output:
            base.main()
        generate.assert_not_called()
        result = json.loads(output.getvalue()[output.getvalue().index('{'):])
        self.assertEqual(result['seed'], 5)
        self.assertEqual(result['steps'], 30)
        self.assertEqual(result['output'], str(self.root / 'base'))

    def test_checkpoint_settings_only_and_cli_override(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'preflight fixture; never loaded')
        self.settings['checkpoint'] = str(adapter)
        self.args.settings.write_text(json.dumps(self.settings))
        with patch.object(sys, 'argv', ['checkpoint.py', '--settings', str(self.args.settings), '--seed', '20', '--dry-run']), \
             patch.object(checkpoint, 'generate') as generate, redirect_stdout(StringIO()) as output:
            checkpoint.main()
        generate.assert_not_called()
        result = json.loads(output.getvalue()[output.getvalue().index('{'):])
        self.assertEqual(result['seed'], 20)
        self.assertFalse(result['pretrained_only'])
        self.assertEqual(result['checkpoint']['kind'], 'peft')


if __name__ == '__main__':
    unittest.main()
