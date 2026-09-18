# Workshop kNN verification

Verified locally on 2026-09-18 with Python 3.12, PyTorch 2.6.0+cu124 and the
Python dependencies pinned in `requirements.txt`. Core tiny-model tests run on
CPU, with additional CUDA checks when available. No pretrained models are
downloaded by the tests.

After the package rename and runtime optimizations, the full suite passed:
**139 tests passed, 2 third-party SWIG
deprecation warnings**. The `python -m workshop run --seeds 42 43 44 --dry-run`
command confirms all five reward arms and the 400-attempt full schedule.

- **139 tests passed.** This includes real tiny Qwen2/Qwen3/Qwen3-MoE model
  forwards and gradients, all-arm pilot-to-full execution, unchanged completed
  checkpoints on rerun, invalid-grade recovery, frozen ridge fitting, reporting,
  seed scheduling and copying the package/config to an isolated directory.
- CUDA tests exercised BF16 matrix multiplication, SDPA forward/backward and
  fused AdamW kernels. Two tiny-model PPO updates matched the standard optimizer
  within the test tolerance; continuing from a fused-optimizer checkpoint matched
  uninterrupted training exactly. CPU tests use the standard optimizer.
- Grading-cache tests verify one transaction per completed batch, preserved
  answer order and embeddings, durable cache hits after reopening the database,
  and no partially cached batch on serialization failure.
- A separate numerical extraction check compared the new PPO trainer with the
  original shared trainer on identical token batches and terminal rewards.
  Every trainable adapter and value-head parameter matched **exactly after each
  of two successive updates**, including AdamW state carried between updates.
  The comparison used the original trainer's code read directly for this check;
  the standalone package and its test suite have no dependency on that code.
- The original PPO source used for that comparison had SHA-256
  `e31b160a3d2cc42040a23e9d594013025d9b7c945dd0c1de3a073ed0cbc43153`.
- PPO tests separately check that old/reference statistics and GAE are computed
  once per rollout, the base reference remains frozen, gradients agree across
  batching choices, and optimizer checkpoints resume exactly.
- Ridge tests check the actual memory targets, separate question-weighted
  validation, frozen coefficients on resume, exact answer/feature identity,
  absence of final-answer tuning and absence of judge calls in ridge training.
- Optimistic Tail Bias tests use hand-computed 1%, 5% and 10% lower tails,
  positive/negative/zero signed errors, inclusive ties, repeated question IDs,
  missing/nonfinite grades, empty tails and JSON-safe unavailable values.
  Integration checks recompute the metrics from exported predictions, verify
  tail-membership flags, preserve saved source metrics during report generation,
  match shared-base diagnostics to each method's own PPO accuracy, and check
  means/counts across seeds.
- Core paper-analysis tests distinguish high-reward tails from low-predicted-gap
  tails, verify signed optimism summaries, inclusive ties, common reward-comparison
  populations, Kendall/constant-series handling, question-cluster bootstrap and
  recomputation of tail cutoffs in every bootstrap draw. Peak-to-final tests reject
  mixing monitoring peaks with the separate test cohort.
- The all-arm tiny-model integration produces the complete paper tables,
  statistics and PDF/PNG figures with generation and grader calls disabled during
  analysis. Every original experiment file is checked unchanged. A separate saved
  two-seed fixture exercises the public `analyze` command, repeated analysis,
  source preservation and cross-seed summaries without refitting ridge.

No full pretrained-model GPU experiment or throughput benchmark was run for
these changes. Larger batches and fused execution still need measurement on
the training environment.
Tiny random models verify execution and mechanics, not GSM8K accuracy or the
scientific effectiveness of ridge.
