#!/usr/bin/env python3
"""Download this public source tree into the prepared HPC directory safely."""
import argparse
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile

REPO = 'ntnghia06/OnlineTrainingRL'
DEFAULT_DESTINATION = '/datastore/anhndt/Truongtq/TrainingEditingModel/OnlineTrainingRL'
PRESERVE = {'running/settings.json', 'inference/settings.json'}


def install(archive, destination):
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written = preserved = 0
    roots = {PurePosixPath(info.filename).parts[0] for info in archive.infolist() if info.filename}
    if len(roots) != 1:
        raise ValueError('Expected one repository root in archive')
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        relative = PurePosixPath(*path.parts[1:])
        if path.is_absolute() or '..' in path.parts or not relative.parts or '.git' in relative.parts:
            raise ValueError(f'Unsafe archive path: {info.filename}')
        if ((info.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError(f'Symlink not supported: {info.filename}')
        target = destination.joinpath(*relative.parts)
        if destination not in target.resolve().parents:
            raise ValueError(f'Path escapes destination: {target}')
        if relative.as_posix() in PRESERVE and target.exists():
            preserved += 1
            print(f'Preserved existing config: {target}', flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix='.' + target.name + '.', suffix='.upload', dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(archive.read(info))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        written += 1
    print(f'Done: wrote {written} source files; preserved {preserved} settings files; destination={destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    url = f'https://codeload.github.com/{REPO}/zip/refs/heads/main'
    print(f'Downloading {REPO}...', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        install(archive, args.destination)


if __name__ == '__main__':
    main()
