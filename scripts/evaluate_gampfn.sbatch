#!/usr/bin/env bash
#SBATCH --wckey=P12DT:python
#SBATCH --partition=an
#SBATCH --job-name=gampfn-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch $0 GAMPFN_CKPT TABICLV2_CKPT OUTPUT_CSV [evaluation arguments...]" >&2
    exit 2
fi

gampfn_checkpoint=$1
tabiclv2_checkpoint=$2
output=$3
shift 3

repo_root=${REPO_ROOT:-"$SLURM_SUBMIT_DIR"}
cd "$repo_root"
mkdir -p "$(dirname "$output")"

exec srun "$repo_root/.venv/bin/python" "$repo_root/scripts/evaluate_gampfn.py" \
    --gampfn "$gampfn_checkpoint" \
    --tabiclv2 "$tabiclv2_checkpoint" \
    --output "$output" \
    "$@"
