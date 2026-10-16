#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base) on a LeRobot export of the teacher demonstrations.
# Runs in the lerobot-gpu conda env on one GPU (DEMO_GPU, default 0). Extra arguments go to lerobot-train.
# Only the newest checkpoint is kept on disk. TRAIN_VISION=1 also trains the vision encoder
# (scripts/lerobot_train_vision.py).
#
#     scripts/train_smolvla.sh <dataset root> <output dir> [--steps=20000 --batch_size=32 ...]
set -eo pipefail
ROOT="$1"; OUT="$2"; shift 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-gpu
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${DEMO_GPU:-0}
# training episodes only: the export writes them first and records the split in meta/info.json
NUM_TRAIN=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['splits']['train'].split(':')[1])" "$ROOT/meta/info.json")
EPISODES="[$(seq -s, 0 $((NUM_TRAIN - 1)))]"
# keep only the newest complete checkpoint: lerobot-train points checkpoints/last at a checkpoint only
# after it is fully written, so every numbered checkpoint older than that one can go
prune_checkpoints() {
    while sleep 60; do
        local last
        last=$(readlink "$OUT/checkpoints/last" 2>/dev/null) || continue
        last=$(basename "$last")
        for dir in "$OUT"/checkpoints/[0-9]*; do
            [[ -d "$dir" && "$(basename "$dir")" < "$last" ]] && rm -rf "$dir" && echo "[train_smolvla] pruned $(basename "$dir")"
        done
    done
}
prune_checkpoints &
trap 'kill %1 2>/dev/null' EXIT
TRAIN=(lerobot-train)
[[ "${TRAIN_VISION:-0}" == 1 ]] && TRAIN=(python "$(dirname "${BASH_SOURCE[0]}")/lerobot_train_vision.py")
"${TRAIN[@]}" \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --dataset.repo_id="local/$(basename "$ROOT")" \
    --dataset.root="$ROOT" \
    --dataset.episodes="$EPISODES" \
    --dataset.video_backend=pyav \
    --rename_map='{"observation.images.front": "observation.images.camera1"}' \
    --output_dir="$OUT" \
    --job_name="$(basename "$OUT")" \
    --wandb.enable=false \
    "$@"
