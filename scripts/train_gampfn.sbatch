set -euo pipefail

repo_root=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
config_path=${CONFIG_PATH:-"$repo_root/configs/gampfn_additive_stage1.yaml"}
checkpoint_root=${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the high-quota checkpoint directory}
checkpoint_dir="$checkpoint_root/gampfn/additive_stage1"
num_gpus=${NUM_GPUS:-4}

cd "$repo_root"

if [[ ! -x "$repo_root/.venv/bin/torchrun" ]]; then
    echo "Missing executable: $repo_root/.venv/bin/torchrun" >&2
    exit 2
fi
if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path" >&2
    exit 2
fi

mkdir -p "$checkpoint_dir"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "Checkpoint directory: $checkpoint_dir"

exec srun "$repo_root/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node="$num_gpus" \
    -m tabicl.train \
    --config "$config_path" \
    --checkpoint_dir "$checkpoint_dir"
