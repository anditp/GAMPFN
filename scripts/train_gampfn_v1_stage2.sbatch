set -euo pipefail

# Edit the initialization path to select the pretrained v0.1 Stage 2 model.
repo_root=${SLURM_SUBMIT_DIR:-$PWD}
config_path="$repo_root/configs/gampfn_v1_stage2.yaml"
checkpoint_root="$repo_root/results"
real_data_dir="$repo_root/data/tabdpt_regression"
init_checkpoint="$checkpoint_root/gampfn/additive_stage2_gam_tabicl_real/step-20000.ckpt"
checkpoint_dir="$checkpoint_root/gampfn/v1_stage2_gam_tabicl_real"
num_gpus_per_node=4

cd "$repo_root"

if [[ ! -x "$repo_root/.venv/bin/torchrun" ]]; then
    echo "Missing executable: $repo_root/.venv/bin/torchrun" >&2
    exit 2
fi
if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path" >&2
    exit 2
fi
if [[ ! -d "$real_data_dir" ]]; then
    echo "Missing TabDPT regression directory: $real_data_dir" >&2
    exit 2
fi

mkdir -p "$checkpoint_dir"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "Checkpoint directory: $checkpoint_dir"

train_args=(--real_data_dir "$real_data_dir")
shopt -s nullglob
stage_checkpoints=("$checkpoint_dir"/step-*.ckpt)
shopt -u nullglob
if (( ${#stage_checkpoints[@]} == 0 )); then
    if [[ ! -f "$init_checkpoint" ]]; then
        echo "Missing v0.1 initialization checkpoint: $init_checkpoint" >&2
        exit 2
    fi
    echo "Initializing v1 shared weights from: $init_checkpoint"
    train_args+=(
        --checkpoint_path "$init_checkpoint"
        --only_load_model true
        --additive_init_from_v01 true
    )
fi

num_nodes=${SLURM_NNODES:-1}
if (( num_nodes == 1 )); then
    exec srun "$repo_root/.venv/bin/torchrun" \
        --standalone \
        --nproc-per-node="$num_gpus_per_node" \
        -m tabicl.train \
        --config "$config_path" \
        --checkpoint_dir "$checkpoint_dir" \
        "${train_args[@]}"
fi

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
master_port=$((20000 + SLURM_JOB_ID % 20000))
echo "DDP rendezvous: $master_addr:$master_port across $num_nodes nodes"

exec srun \
    --nodes="$num_nodes" \
    --ntasks="$num_nodes" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    "$repo_root/.venv/bin/torchrun" \
    --nnodes="$num_nodes" \
    --nproc-per-node="$num_gpus_per_node" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$master_addr:$master_port" \
    --rdzv-id="$SLURM_JOB_ID" \
    -m tabicl.train \
    --config "$config_path" \
    --checkpoint_dir "$checkpoint_dir" \
    "${train_args[@]}"
