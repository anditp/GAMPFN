set -euo pipefail

# Edit this path if the selected Stage 1 checkpoint has a different step.
repo_root=${SLURM_SUBMIT_DIR:-$PWD}
config_path="$repo_root/configs/gampfn_probabilistic_stage2.yaml"
checkpoint_root="$repo_root/results"
init_checkpoint="$checkpoint_root/gampfn/probabilistic_stage1/step-100000.ckpt"
checkpoint_dir="$checkpoint_root/gampfn/probabilistic_stage2_gam_tabicl"
num_gpus=4

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
shopt -s nullglob
stage_checkpoints=("$checkpoint_dir"/step-*.ckpt)
shopt -u nullglob
if (( ${#stage_checkpoints[@]} == 0 )); then
    if [[ ! -f "$init_checkpoint" ]]; then
        echo "Missing probabilistic Stage 1 checkpoint: $init_checkpoint" >&2
        exit 2
    fi
    echo "Initializing Stage 2 model from: $init_checkpoint"
    train_args+=(--checkpoint_path "$init_checkpoint" --only_load_model true)
fi

exec srun "$repo_root/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node="$num_gpus" \
    -m tabicl.train \
    --config "$config_path" \
    --checkpoint_dir "$checkpoint_dir" \
    "${train_args[@]}"
