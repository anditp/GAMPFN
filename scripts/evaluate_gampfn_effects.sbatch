set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 CHECKPOINT OUTPUT_DIR [evaluation arguments...]" >&2
    exit 2
fi

checkpoint=$1
output_dir=$2
shift 2

repo_root=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
python_path="$repo_root/.venv/bin/python"

if [[ ! -x "$python_path" ]]; then
    echo "Missing executable: $python_path" >&2
    exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 2
fi

cd "$repo_root"
mkdir -p "$output_dir"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "Checkpoint: $checkpoint"
echo "Output directory: $output_dir"

exec srun "$python_path" "$repo_root/scripts/evaluate_gampfn_effects.py" \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    "$@"
