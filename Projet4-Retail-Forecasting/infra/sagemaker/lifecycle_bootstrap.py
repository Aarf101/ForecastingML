"""Amorce du conteneur Processing, sans embarquer de secret ni de fichier .env."""

import subprocess
import sys
import tarfile
from pathlib import Path


def main():
    root = Path('/tmp/retail-runtime')
    root.mkdir(exist_ok=True)
    with tarfile.open('/opt/ml/processing/input/bundle/runtime.tar.gz') as archive:
        archive.extractall(root, filter='data')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--break-system-packages',
                    '-q', '-e', str(root), 'boto3>=1.36', 'sagemaker>=2.214,<3'], check=True)
    subprocess.run([sys.executable, '-m', 'infra.sagemaker.lifecycle', *sys.argv[1:]],
                   cwd=root, check=True)


if __name__ == '__main__':
    main()
