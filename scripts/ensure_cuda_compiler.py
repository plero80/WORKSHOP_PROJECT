"""Supply nvcc for DeepSpeed's import checks when a CUDA runtime image lacks it.

This is NVIDIA's standalone compiler redistribution, not a driver installer.
The prebuilt PyTorch/FlashAttention wheels still perform all training compute.
"""
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory
import urllib.request

URL = ('https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/'
       'cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz')
SHA256 = '7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35'
ARCHIVE_ROOT = 'cuda_nvcc-linux-x86_64-12.9.86-archive'


def main():
    target = Path(sys.prefix)/'cuda-toolkit'
    candidates = [target/'bin/nvcc', Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))/'bin/nvcc']
    if shutil.which('nvcc') or any(p.is_file() for p in candidates):
        return
    print('Installing the verified NVIDIA compiler package for OpenRLHF dependency checks.', flush=True)
    with TemporaryDirectory(prefix='cuda-compiler-', dir=sys.prefix) as temporary:
        stage = Path(temporary)
        archive = stage/'nvcc.tar.xz'
        with urllib.request.urlopen(URL, timeout=120) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        with archive.open('rb') as stream:
            checksum = hashlib.sha256(stream.read()).hexdigest()
        if checksum != SHA256:
            raise RuntimeError('CUDA compiler archive checksum mismatch; nothing was installed.')
        # Extract only regular files/directories, never links or device entries.
        with tarfile.open(archive, 'r:xz') as bundle:
            for member in bundle.getmembers():
                destination = (stage/member.name).resolve()
                if not destination.is_relative_to(stage.resolve()):
                    raise RuntimeError('Unsafe compiler archive path.')
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as source, destination.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    destination.chmod(member.mode & 0o777)
        (stage/ARCHIVE_ROOT).replace(target)


if __name__ == '__main__':
    main()
