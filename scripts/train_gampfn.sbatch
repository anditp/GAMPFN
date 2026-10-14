set -euo pipefail

repo_root=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
config_path=${CONFIG_PATH:-"$repo_root/configs/gampfn_additive_stage1.yaml"}
checkpoint_root=${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the high-quota checkpoint directory}
stage_name=${GAMPFN_STAGE:-additive_stage1}
checkpoint_dir=${CHECKPOINT_DIR:-"$checkpoint_root/gampfn/$stage_name"}
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

train_args=()
if [[ -n "${REAL_DATA_DIR:-}" ]]; then
    train_args+=(--real_data_dir "$REAL_DATA_DIR")
fi

shopt -s nullglob
stage_checkpoints=("$checkpoint_dir"/step-*.ckpt)
shopt -u nullglob
if (( ${#stage_checkpoints[@]} == 0 )) && [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    if [[ ! -f "$INIT_CHECKPOINT" ]]; then
        echo "Missing initialization checkpoint: $INIT_CHECKPOINT" >&2
        exit 2
    fi
    echo "Initializing model weights from: $INIT_CHECKPOINT"
    train_args+=(--checkpoint_path "$INIT_CHECKPOINT" --only_load_model true)
fi

exec srun "$repo_root/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node="$num_gpus" \
    -m tabicl.train \
    --config "$config_path" \
    --checkpoint_dir "$checkpoint_dir" \
    "${train_args[@]}"
