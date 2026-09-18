#!/usr/bin/env bash
# Shared launcher: one full seed and the same experiment implementation.
set -euo pipefail

profile="${1:-}"
case "$profile" in
  b200|b300) shift ;;
  *) printf 'Usage: bash scripts/run_seed.sh {b200|b300} [--dry-run]\n' >&2; exit 2 ;;
esac
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != '--dry-run' ]]; }; then
  printf 'Only --dry-run is supported; use python -m workshop for custom runs.\n' >&2
  exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
python_command="${WORKSHOP_PYTHON:-python}"
if [[ -z "${WORKSHOP_PYTHON:-}" && -x .venv/bin/python ]]; then
  python_command="$project_dir/.venv/bin/python"
fi
export HF_HOME="${HF_HOME:-$project_dir/.cache/huggingface}"

output="outputs/openrlhf_$profile"
command=("$python_command" -u -m workshop run --config "configs/$profile.yaml"
         --seeds 42 --stage full --output "$output")
if [[ "${1:-}" == '--dry-run' ]]; then
  exec "${command[@]}" --dry-run
fi

# Validate imports/configuration before detaching. Runtime GPU checks follow in the log.
"${command[@]}" --dry-run > /dev/null
mkdir -p -- "$output"
nohup "${command[@]}" >> "$output/launcher.log" 2>&1 < /dev/null &
pid=$!
printf 'Started full seed 42 (PID %s).\n' "$pid"
printf 'Launcher log: %s/%s/launcher.log\n' "$project_dir" "$output"
printf 'Watch training: tail -F "%s/%s/seed_42/experiment.log"\n' "$project_dir" "$output"
printf 'Results and figures: %s/%s/paper_results/\n' "$project_dir" "$output"
