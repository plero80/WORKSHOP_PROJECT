# Reproducing the runs

## 1. Install the environment

Use Linux x86_64 (a GPU machine or WSL2), Python 3.10-3.12 with `venv`,
and an NVIDIA GPU with BF16 support and a driver compatible with CUDA 12.9.
The supplied GPU profiles target B200 and B300. The full experiment loads a
30B teacher during preparation, so it requires substantial GPU memory.
Internet access is needed to install dependencies and download models and GSM8K.

```bash
git clone https://github.com/plero80/WORKSHOP_PROJECT.git
cd WORKSHOP_PROJECT

bash scripts/setup.sh
source .venv/bin/activate
```

Run all subsequent commands from this repository directory. Setup installs
PyTorch 2.8.0/CUDA 12.9, OpenRLHF 0.9.0 and the dependencies in
`requirements.txt` into `.venv`. It also runs a tiny CUDA PPO update and
checkpoint-resume check without downloading pretrained models. To select a
specific Python executable, use `WORKSHOP_PYTHON=python3.11 bash scripts/setup.sh`.

Models and data download automatically on the first experiment run; no manual
dataset preparation or notebook execution is needed.

## 2. Run the experiment

The default configuration in `configs/default.yaml` uses seed **42** and
**400 PPO attempts per arm**: `proxy`, `judge`, `knn_static`,
`knn_static_30b` and `ridge`. It evaluates the base policy and each trained
policy on all **1,319 GSM8K test questions**.

Preview the configuration, then run the full experiment in the foreground:

```bash
python -m workshop run --dry-run
python -m workshop run
```

Results are saved under `outputs/openrlhf/`. The dry run prints the plan
without loading models, downloading data or starting training; it does not
test GPU readiness.

For a background run, choose the launcher matching your GPU instead:

| GPU | Command | Output directory |
|---|---|---|
| B200 | `bash scripts/run_b200.sh` | `outputs/openrlhf_b200/` |
| B300 | `bash scripts/run_b300.sh` | `outputs/openrlhf_b300/` |

Both launchers run the same seed-42 experiment with GPU-specific inference
batch sizes. Append `--dry-run` to preview a launcher. A background run
continues after disconnecting while the machine remains running; keep its
outputs and model cache on persistent storage.

To run three seeds sequentially on one GPU:

```bash
python -m workshop run --config configs/b200.yaml --seeds 42 43 44 --output outputs/three_seeds
```

Use `configs/b300.yaml` for the B300 profile. Keep the same configuration
across the seeds being compared.

## 3. Monitor and resume

For the default foreground run, inspect progress from another terminal with
the environment activated:

```bash
tail -F outputs/openrlhf/seed_42/experiment.log
python -m workshop status --output outputs/openrlhf
```

Substitute your chosen output directory for `outputs/openrlhf` in these and
the reporting commands below. Background launchers also write `launcher.log`
inside their output directory.

To resume an interrupted run, rerun the **same command** after the previous
process has stopped. Keep the same output directory, code, configuration,
model revisions, library versions and seed order. Completed training is
retained; work after the latest checkpoint may be replayed.

For an initial 100-attempt pilot followed by the full 400-attempt run:

```bash
python -m workshop run --stage pilot --output outputs/pilot_then_full
python -m workshop run --output outputs/pilot_then_full
```

The pilot performs monitoring evaluation; the full run adds final test
evaluation. `--updates N` sets the total target, not additional attempts.
Once final test evaluation has fixed the target, use a new output directory
to change it.

## 4. Read or regenerate results

Full runs automatically generate reports, analysis tables and figures.

| Path within the output directory | Contents |
|---|---|
| `suite_report.md`, `suite_summary.json` | Results across arms and seeds |
| `seed_42/report.md` | Per-seed evaluation results |
| `seed_42/predictors/report.md` | kNN and ridge prediction metrics |
| `paper_results/report.md` | Analysis report |
| `paper_results/tables/` | CSV result tables |
| `paper_results/figures/` | PDF and PNG figures |

Regenerate these from saved outputs:

```bash
python -m workshop report --output outputs/openrlhf
python -m workshop analyze --output outputs/openrlhf
```

Analysis uses saved answers, features and predictors on CPU, without new
training or model downloads. Preserve the entire output directory.

## 5. Repeat a previous run

Use the original Git revision and environment, the saved configuration in
`configs/seed_42.json` within the output directory, and the model/dataset
commits recorded in `seed_42/resolved_assets.json`. The default 30B teacher
revision is `main`: set `revisions.judge30b` in your reproduction configuration
to that recorded commit before starting in a fresh output directory.

Pass the saved or adjusted configuration with `--config`, the original seed
list with `--seeds`, and a fresh directory with `--output`. The saved
`seed_42/manifest.json` records configuration, source identity, package
versions and GPU runtime details; `seed_42/data/splits.json` records the data
split. Keep the same hardware and batch settings when comparing repeats,
since they can affect floating-point results and sampled answers.
