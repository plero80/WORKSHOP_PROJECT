from __future__ import annotations

import importlib.metadata
import os
import time
from pathlib import Path

from .common import ENGINE_ID, atomic_json, digest, file_sha, read_json, status


def source_fingerprint():
    package = Path(__file__).parent
    return digest({"engine": ENGINE_ID,
                   "package": {p.name: file_sha(p)
                               for p in sorted(package.glob("*.py"))}})


def retry_hub(fn):
    for attempt in range(6):
        try:
            return fn()
        except Exception as e:
            response = getattr(e, "response", None)
            code = getattr(response, "status_code", None)
            if attempt == 5 or (code is not None and code not in (408, 429, 500, 502, 503, 504)):
                raise
            delay = min(30, 2**attempt)
            print(f"Download/metadata retry {attempt + 1}/6 after {type(e).__name__}; waiting {delay}s.", flush=True)
            time.sleep(delay)


def check_runtime(config):
    import torch
    import torch.nn.functional as F
    device = config["runtime"]["device"]
    if not device.startswith("cuda"):
        raise ValueError("Real model experiments require a CUDA GPU. CPU is supported only by the offline tests.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the project GPU environment with a compatible PyTorch installation.")
    torch.cuda.set_device(device)
    if config["runtime"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16. Use a supported GPU or explicitly select float32.")
    # Exercise the configured precision and training kernels before downloads.
    dtype = getattr(torch, config['runtime']['dtype'])
    fused = config['runtime'].get('fused_optimizer', False)
    try:
        probe = torch.randn((32, 32), device=device, dtype=dtype, requires_grad=True)
        loss = (probe @ probe).float().square().mean()
        if config['runtime']['attention'] == 'sdpa':
            query = torch.randn((2, 4, 32, 64), device=device, dtype=dtype, requires_grad=True)
            attended = F.scaled_dot_product_attention(query, query, query, is_causal=True)
            loss = loss + attended.float().square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(probe.grad).all():
            raise RuntimeError('Nonfinite runtime probe.')
        if fused:
            parameter = torch.nn.Parameter(torch.ones(8, device=device))
            parameter.grad = torch.ones_like(parameter)
            optimizer = torch.optim.AdamW([parameter], lr=1e-5, fused=True)
            optimizer.step()
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        raise RuntimeError('The configured CUDA training kernels failed. Use a PyTorch CUDA build '
                           'compatible with this GPU and driver. Details: '+str(error)) from error
    free, total = torch.cuda.mem_get_info(device)
    return {"gpu": torch.cuda.get_device_name(device), "free_gib": free / 2**30, "total_gib": total / 2**30,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "dtype": config['runtime']['dtype'], "attention": config['runtime']['attention'],
            "fused_optimizer": fused}


def resolve_assets(config, output):
    from huggingface_hub import HfApi, snapshot_download
    output = Path(output)
    path = output / "resolved_assets.json"
    # No whoami request; every default repository is public.
    api = HfApi(token=False)
    roles = ['policy', 'proxy', 'judge']
    if 'knn_static_30b' in config['arms']:
        roles.append('judge30b')
    if path.exists():
        resolved = read_json(path)
    else:
        resolved = {}
        for role in roles:
            name = config['models'][role]
            resolved[role] = retry_hub(lambda: api.model_info(name, revision=config["revisions"][role]).sha)
        resolved["dataset"] = retry_hub(lambda: api.dataset_info(config["dataset"]["id"], revision=config["revisions"]["dataset"]).sha)
        atomic_json(path, resolved)
    for role in roles:
        name = config['models'][role]
        status(output, "download", role=role, model=name, revision=resolved[role])
        retry_hub(lambda: snapshot_download(name, revision=resolved[role], token=False,
                  allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.model", "merges.txt", "vocab.txt"], max_workers=2))
    return resolved


def bind_experiment(config, output, resolved, split, runtime):
    import torch
    output = Path(output)
    versions = {name: importlib.metadata.version(name) for name in
                ("transformers", "peft", "accelerate", "datasets", "huggingface-hub", "numpy", "scipy", "scikit-learn")}
    identity = {"config": config, "resolved": resolved, "split_fingerprint": split["fingerprint"],
                "source": source_fingerprint(), "versions": versions, "torch_version": torch.__version__.split("+")[0]}
    fingerprint = digest(identity)
    manifest = output / "manifest.json"
    if manifest.exists() and read_json(manifest)["fingerprint"] != fingerprint:
        raise ValueError("Configuration, data, code, models or library versions changed. Use a new output directory.")
    if not manifest.exists():
        atomic_json(manifest, {"fingerprint": fingerprint, "identity": identity, "runtime_at_creation": runtime,
                               "created_at": time.time(), "protocol": ENGINE_ID})
    return fingerprint
