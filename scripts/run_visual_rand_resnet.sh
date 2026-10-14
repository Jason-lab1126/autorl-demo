#!/usr/bin/env bash
# Retrying the visually randomized student: ImageNet ResNet-18 (fine-tuned) and random-crop
# augmentation, with fixed-scene teacher episodes (original cube and table) added to the 1,251
# randomized ones. The unchanged scene is checked first; the full visual table (seen + held-out
# conditions) only runs with FULL=1.
#
# Attempt 1 (500 fixed-scene episodes, 50 epochs):
#     NAME=student_visual_rand_resnet scripts/run_visual_rand_resnet.sh          # train + unchanged scene
#     FULL=1 NAME=student_visual_rand_resnet scripts/run_visual_rand_resnet.sh   # + all conditions
#
# Attempt 2 (all 1,250 fixed-scene episodes, 30 epochs):
#     NAME=student_visual_rand_resnet2 TRAIN_ARGS="--encoder resnet18 --crop 76 --epochs 30 --lr 1e-3 --encoder_lr 1e-4 --amp" \
#         DATA="datasets/lift_vision_teacher_visrand.h5 datasets/lift_vision_raw.h5 datasets/lift_vision_teacher_extra.h5" \
#         scripts/run_visual_rand_resnet.sh
set -eo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DEMO_ROOT/env.sh"
cd "$DEMO_ROOT"

NAME=${NAME:-student_visual_rand_resnet}
STUDENT_DIR=logs/student/${NAME#student_}
read -ra DATA <<< "${DATA:-datasets/lift_vision_teacher_visrand.h5 datasets/lift_vision_raw.h5}"
TRAIN_ARGS=${TRAIN_ARGS:-"--encoder resnet18 --crop 76 --epochs 50 --lr 1e-3 --encoder_lr 1e-4 --amp"}
OUT=results/visual
MIN_FREE_MIB=6000

gpu_check() {
    local used total util
    IFS=', ' read -r used total util < <(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    echo "[gpu_check] $(date +%H:%M:%S) before '$1': GPU $CUDA_VISIBLE_DEVICES ${used}/${total} MiB used, ${util}% util"
    if (( total - used < MIN_FREE_MIB )); then
        echo "[gpu_check] less than ${MIN_FREE_MIB} MiB free, stopping"; exit 3
    fi
}

if [[ ! -f "$STUDENT_DIR/student.pt" ]]; then
    gpu_check "training $NAME"
    python -u scripts/train_student.py --data "${DATA[@]}" --out "$STUDENT_DIR" $TRAIN_ARGS --device cuda:0 2>&1 | grep -v Warning
fi

seen=(
    "none|"
    "light_x0.5|--light_scale 0.5"
    "light_x2|--light_scale 2.0"
    "cube_red|--cube_color 0.8 0.1 0.1"
    "table_light_wood|--table_color 0.85 0.75 0.6"
    "camera_5cm|--camera_shift 0 0.05 0"
    "camera_10cm|--camera_shift 0 0.10 0"
)
held_out=(
    "light_x3|--light_scale 3.0"
    "cube_purple|--cube_color 0.6 0.1 0.8"
    "table_purple|--table_color 0.5 0.2 0.6"
    "camera_12cm|--camera_shift 0 0.12 0"
)

evaluate() {  # <condition>
    local key="${1%%|*}" extra="${1#*|}"
    [[ -f "$OUT/$key/$NAME/summary.json" ]] && return 0
    gpu_check "$NAME $key"
    python -u scripts/eval_student.py --policy student --student "$STUDENT_DIR/student.pt" --out_dir "$OUT/$key/$NAME" \
        --save_view "$OUT/$key/${NAME}_view.png" $extra --headless --device cuda:0 | grep -E "success_rate\"|final_dist_mean_m"
}

evaluate "${seen[0]}"
if [[ "${FULL:-0}" == 1 ]]; then
    for condition in "${seen[@]:1}" "${held_out[@]}"; do evaluate "$condition"; done
fi
echo "[run_visual_rand_resnet] done"
