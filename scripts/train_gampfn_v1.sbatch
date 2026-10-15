set -euo pipefail

# Select "scratch" or "v01". For v01, point init_checkpoint at the exact
# additive checkpoint whose weights should initialize the new architecture.
initialization="v01"
repo_root=${SLURM_SUBMIT_DIR:-$PWD}
config_path="$repo_root/configs/gampfn_v1_stage1.yaml"
checkpoint_root="$repo_root/results"
init_checkpoint="$checkpoint_root/gampfn/additive_stage1/step-65000.ckpt"
num_gpus=4

case "$initialization" in
    scratch|v01) ;;
    *)
        echo "initialization must be scratch or v01, got: $initialization" >&2
        exit 2
        ;;
esac

stage_name="v1_stage1_$initialization"
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
echo "Initialization: $initialization"
echo "Checkpoint directory: $checkpoint_dir"

train_args=()
shopt -s nullglob
stage_checkpoints=("$checkpoint_dir"/step-*.ckpt)
shopt -u nullglob
if (( ${#stage_checkpoints[@]} == 0 )) && [[ "$initialization" == "v01" ]]; then
    if [[ ! -f "$init_checkpoint" ]]; then
        echo "Missing v0.1 initialization checkpoint: $init_checkpoint" >&2
        exit 2
    fi
    echo "Initializing shared weights from: $init_checkpoint"
    train_args+=(
        --checkpoint_path "$init_checkpoint"
        --only_load_model true
        --additive_init_from_v01 true
    )
fi

exec srun "$repo_root/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node="$num_gpus" \
    -m tabicl.train \
    --config "$config_path" \
    --checkpoint_dir "$checkpoint_dir" \
    "${train_args[@]}"
