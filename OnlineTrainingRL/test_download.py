"""Check server installer preserves user settings/output and rejects traversal."""
from io import BytesIO
from pathlib import Path
import shutil
import unittest
import zipfile

from download_on_server import install


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_upload_verification'
        self.root.mkdir()
    def tearDown(self):
        shutil.rmtree(self.root)

    def archive(self, rows):
        payload = BytesIO()
        with zipfile.ZipFile(payload, 'w') as stream:
            for name, value in rows.items():
                stream.writestr(name, value)
        payload.seek(0)
        return zipfile.ZipFile(payload)

    def test_existing_settings_and_training_output_survive(self):
        (self.root / 'running/outputs').mkdir(parents=True)
        (self.root / 'running/settings.json').write_text('custom paths')
        (self.root / 'running/outputs/checkpoint').write_text('keep weights')
        with self.archive({'repo/running/settings.json': 'defaults', 'repo/running/train.py': 'new code'}) as archive:
            install(archive, self.root)
        self.assertEqual((self.root / 'running/settings.json').read_text(), 'custom paths')
        self.assertEqual((self.root / 'running/outputs/checkpoint').read_text(), 'keep weights')
        self.assertEqual((self.root / 'running/train.py').read_text(), 'new code')
        self.assertEqual(list(self.root.rglob('*.upload')), [])

    def test_archive_traversal_is_rejected(self):
        with self.archive({'repo/../escape.py': 'bad'}) as archive:
            with self.assertRaises(ValueError):
                install(archive, self.root)


if __name__ == '__main__':
    unittest.main()
