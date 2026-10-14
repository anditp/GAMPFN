set -euo pipefail

# Edit these paths before submitting the job.
repo_root=${SLURM_SUBMIT_DIR:-$PWD}
config_path="$repo_root/configs/gampfn_additive_stage2.yaml"
checkpoint_root="$repo_root/results"
stage_name="additive_stage2"
real_data_dir="$repo_root/data/tabdpt_regression"
init_checkpoint="$checkpoint_root/gampfn/additive_stage1/step-100000.ckpt"
num_gpus=4

checkpoint_dir="$checkpoint_root/gampfn/$stage_name"

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
train_args+=(--real_data_dir "$real_data_dir")

shopt -s nullglob
stage_checkpoints=("$checkpoint_dir"/step-*.ckpt)
shopt -u nullglob
if (( ${#stage_checkpoints[@]} == 0 )); then
    if [[ ! -f "$init_checkpoint" ]]; then
        echo "Missing initialization checkpoint: $init_checkpoint" >&2
        exit 2
    fi
    echo "Initializing model weights from: $init_checkpoint"
    train_args+=(--checkpoint_path "$init_checkpoint" --only_load_model true)
fi

exec srun "$repo_root/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node="$num_gpus" \
    -m tabicl.train \
    --config "$config_path" \
    --checkpoint_dir "$checkpoint_dir" \
    "${train_args[@]}"
