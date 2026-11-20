"""Real CPU tensor restoration with mocked PEFT plumbing; no GPU/backbone loaded."""
import ast
import inspect
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import generate_omnigen2_gedit_fireward as fireward
import test_omnigen2_gedit as base_tests


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    @classmethod
    def from_pretrained(cls, directory, **kwargs):
        return cls(**json.loads((Path(directory) / 'adapter_config.json').read_text()))


class ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.zeros(2, 2))
    def add_adapter(self, config):
        self.to_q = torch.nn.Module()
        self.to_q.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(2, config.r, bias=False)})
        self.to_q.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(config.r, 2, bias=False)})
    def forward(self, hidden_states, timestep=None, text_hidden_states=None,
                text_attention_mask=None, freqs_cis=None, ref_image_hidden_states=None, **kwargs):
        if ref_image_hidden_states is None:
            raise TypeError('Source reference latents were dropped')
        self.seen_reference = ref_image_hidden_states
        return ref_image_hidden_states + self.to_q.lora_B['default'](self.to_q.lora_A['default'](ref_image_hidden_states))


class GenericPeftWrapper(torch.nn.Module):
    def get_base_model(self):
        return self.base_model.model
    def forward(self, *args, **kwargs):
        return self.get_base_model()(*args, **kwargs)


def fake_peft():
    module = ModuleType('peft')
    module.LoraConfig = module.PeftConfig = Config
    def wrap(model, config):
        model.add_adapter(config)
        wrapper = GenericPeftWrapper()
        wrapper.base_model = torch.nn.Module()
        wrapper.base_model.model = model
        return wrapper
    def restore(model, state, **kwargs):
        state = {key.replace('.lora_A.', '.lora_A.default.').replace('.lora_B.', '.lora_B.default.'): value
                 for key, value in state.items()}
        return model.load_state_dict(state, strict=False)
    def export(model, **kwargs):
        return {key.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.'): value
                for key, value in model.state_dict().items() if 'lora_' in key}
    module.get_peft_model = wrap
    module.set_peft_model_state_dict = restore
    module.get_peft_model_state_dict = export
    return module


class FirewardTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_inference_verification' / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)
    def tearDown(self):
        shutil.rmtree(self.root)
    def adapter(self):
        directory = self.root / 'checkpoint-120/lora_adapter'
        directory.mkdir(parents=True)
        (directory / 'adapter_config.json').write_text(json.dumps({'r': 1, 'lora_alpha': 2}))
        state = {'base_model.model.to_q.lora_A.weight': torch.full((1, 2), 0.75),
                 'base_model.model.to_q.lora_B.weight': torch.full((2, 1), 0.25)}
        save_file(state, str(directory / 'adapter_model.safetensors'))
        return directory, state

    def test_portable_adapter_restores_and_freezes(self):
        directory, state = self.adapter()
        checkpoint = fireward.resolve_checkpoint(directory.parent)
        self.assertEqual(checkpoint['kind'], 'peft')
        with patch.dict(sys.modules, {'peft': fake_peft()}):
            model = fireward.apply_checkpoint(ToyTransformer(), checkpoint)
        self.assertIsInstance(model, ToyTransformer)
        self.assertTrue(torch.equal(model.to_q.lora_A['default'].weight, state['base_model.model.to_q.lora_A.weight']))
        self.assertTrue(torch.equal(model.backbone, torch.zeros(2, 2)))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.training)

    def test_full_checkpoint_restores_backbone_and_lora(self):
        directory = self.root / 'checkpoint-120'
        directory.mkdir()
        (self.root / 'train.yml').write_text('train:\n  lora_ft: true\n  lora_rank: 1\n  lora_alpha: 2\n')
        original = ToyTransformer()
        original.add_adapter(Config(r=1))
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.fill_(0.5)
            original.backbone.fill_(7)
        save_file(original.state_dict(), str(directory / 'model.safetensors'))
        checkpoint = fireward.resolve_checkpoint(directory)
        self.assertEqual(checkpoint['kind'], 'full')
        with patch.dict(sys.modules, {'peft': fake_peft()}):
            restored = fireward.apply_checkpoint(ToyTransformer(), checkpoint)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(restored.state_dict()[key], value))

    def test_missing_lora_tensor_is_rejected(self):
        directory, state = self.adapter()
        del state['base_model.model.to_q.lora_B.weight']
        save_file(state, str(directory / 'adapter_model.safetensors'))
        with patch.dict(sys.modules, {'peft': fake_peft()}):
            with self.assertRaisesRegex(RuntimeError, 'LoRA keys differ'):
                fireward.apply_checkpoint(ToyTransformer(), fireward.resolve_checkpoint(directory))

    def test_legacy_direct_adapter_prefix(self):
        directory, state = self.adapter()
        state = {key.removeprefix('base_model.model.'): value for key, value in state.items()}
        save_file(state, str(directory / 'adapter_model.safetensors'))
        with patch.dict(sys.modules, {'peft': fake_peft()}):
            result = fireward.apply_checkpoint(ToyTransformer(), fireward.resolve_checkpoint(directory))
        self.assertTrue(torch.equal(result.to_q.lora_A['default'].weight, state['to_q.lora_A.weight']))

    def test_original_pipeline_predict_preserves_source_and_active_lora(self):
        # Execute the actual repo predict() method, including its signature check.
        path = fireward.resolve_repo(Path(__file__).resolve().parent) / 'omnigen2/pipelines/omnigen2/pipeline_omnigen2.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'OmniGen2Pipeline')
        predict = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'predict')
        namespace = {'inspect': inspect}
        exec(compile(ast.Module(body=[predict], type_ignores=[]), '<original_pipeline_predict>', 'exec'), namespace)
        directory, _ = self.adapter()
        fake_accelerate = ModuleType('accelerate')
        utils = ModuleType('accelerate.utils')
        utils.extract_model_from_parallel = lambda model: model
        fake_accelerate.utils = utils
        reference = torch.ones(1, 2)
        with patch.dict(sys.modules, {'peft': fake_peft(), 'accelerate': fake_accelerate, 'accelerate.utils': utils}):
            checkpoint = fireward.resolve_checkpoint(directory)
            config = Config(r=1)
            broken_wrapper = fake_peft().get_peft_model(ToyTransformer(), config)
            broken_pipeline = SimpleNamespace(mixed_precision=False, transformer=broken_wrapper)
            kwargs = dict(t=torch.tensor(0.5), latents=torch.zeros(1, 2), text_hidden_states=None,
                          text_attention_mask=None, ref_image_hidden_states=reference, freqs_cis=None)
            with self.assertRaisesRegex(TypeError, 'dropped'):
                namespace['predict'](broken_pipeline, **kwargs)
            restored = fireward.apply_checkpoint(ToyTransformer(), checkpoint)
            fixed_pipeline = SimpleNamespace(mixed_precision=False, transformer=restored)
            output = namespace['predict'](fixed_pipeline, **kwargs)
        self.assertIs(restored.seen_reference, reference)
        # Nonzero LoRA values from the checkpoint still affect the prediction.
        self.assertTrue(torch.allclose(output, torch.full((1, 2), 1.375)))
        self.assertIn('ref_image_hidden_states', inspect.signature(restored.forward).parameters)

    def test_full_state_requires_original_lora_config(self):
        directory = self.root / 'checkpoint-120'
        directory.mkdir()
        (directory / 'model.safetensors').write_bytes(b'fake')
        with self.assertRaisesRegex(FileNotFoundError, 'original training'):
            fireward.resolve_checkpoint(directory)

    def test_cpu_preflight_limit_and_checkpoint(self):
        fixture = base_tests.GEditTests()
        fixture.root = self.root
        args, _ = fixture.fixture()
        adapter, _ = self.adapter()
        argv = ['generate_omnigen2_gedit_fireward.py', '--settings', str(args.settings),
                '--checkpoint', str(adapter.parent), '--metadata', str(args.metadata),
                '--data-root', str(args.data_root), '--output-root', str(args.output_root), '--limit', '1', '--dry-run']
        with patch.object(sys, 'argv', argv), patch.object(fireward, 'generate') as generate:
            fireward.main()
        generate.assert_not_called()

    def test_checkpoint_identity_prevents_mixed_outputs(self):
        fixture = base_tests.GEditTests()
        fixture.root = self.root
        args, _ = fixture.fixture()
        adapter, _ = self.adapter()
        args.checkpoint_info = fireward.resolve_checkpoint(adapter)
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        with patch.dict(sys.modules, {'fcntl': fake_fcntl}), patch.object(fireward, 'ROOT', self.root):
            with fireward.output_lock(args):
                pass
            with fireward.output_lock(args):
                pass
            args.checkpoint_info = {**args.checkpoint_info, 'path': 'a-different-checkpoint'}
            with self.assertRaisesRegex(ValueError, 'config differs'):
                with fireward.output_lock(args):
                    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
