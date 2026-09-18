# Workshop kNN verification

Verified on 2026-09-18 in an isolated Ubuntu/WSL2 environment with Python
3.10.12, OpenRLHF 0.9.0, PyTorch 2.8.0+cu129, Transformers 4.57.0 and an
RTX 3070 Ti. The real installed OpenRLHF training steps and losses were used;
no replacement implementation or mocked loss was used for the PPO checks.

**147 tests passed**, with 14 third-party deprecation warnings, in 27.09 seconds.
Run the suite in the Linux training environment with `python -m pytest -q`.
Tests use random tiny local Qwen models and do not download pretrained models.

The public `python -m workshop backend-check` command also passed separately:
BF16, SDPA, fused AdamW, two optimizer steps, changed actor/value parameters,
and exact checkpoint continuation. Its output reports `ok: true`,
`exact_resume: true`, and `pretrained_models_downloaded: 0`.

Checks include:

- Actor and critic workers inherit the installed OpenRLHF training-step methods
  unchanged. Instrumentation confirms both upstream losses execute and GAE is
  computed once per rollout.
- Hand-computed masked GAE, frozen reference weights, variable response lengths,
  equivalent gradient accumulation across microbatch sizes, and exact resume.
- Real CUDA updates with standard and fused optimizers, including checkpoint
  continuation, and the configured matrix-multiply/attention training kernels.
- One- and three-answer partial rollouts, a fully ungraded attempt followed by a
  partially graded attempt, and preservation of missing-grade review records.
- Rejection of legacy custom-PPO checkpoints/suites and invalid backend choices.
- All-arm tiny-model preparation, pilot, full execution, resumed checkpoints,
  frozen ridge fitting, seed scheduling, reports, metrics and PDF/PNG plots.
- The existing tail-bias, high-reward optimism, bootstrap, data-split, grader,
  cache, predictor-validation and standalone-copy checks.

Both shell profiles passed syntax checks and actual CLI dry runs. They select
seed 42, all five arms and 400 attempts per arm, with fresh output roots
`outputs/openrlhf_b200` and `outputs/openrlhf_b300`. The runtime profiles still
differ only in their three inference batch limits.

The Linux environment was installed from the pinned package requirements,
CUDA wheels and NVIDIA compiler redistribution used by setup. OpenRLHF's pin
requires Transformers 4.57.0, which PyPI marks as withdrawn for packaging
issues; the explicit pin installed successfully and the full suite above ran
against it. The pinned import dependencies include DeepSpeed and vLLM, but
this integration runs a single-device PyTorch strategy without starting Ray
workers, a DeepSpeed engine or vLLM generation engines.

The previous custom trainer's parity checks belong to the earlier Git revision;
they are not evidence of bitwise equivalence after this migration. New manifests
record the new engine, library version and installed upstream source hashes.

No full pretrained GSM8K experiment or B200/B300 throughput benchmark was run.
Tiny-model checks establish execution and mechanics, not task accuracy or the
scientific effectiveness of kNN/ridge. Start fresh runs under this backend and
keep historical results attributed to their original trainer.
