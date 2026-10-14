#!/usr/bin/env bash
# Visual randomization for the vision student: collect 1,250 teacher episodes with randomized
# lighting, cube/table colors and camera offset, train a new student on them (same model and
# settings), and evaluate it on the visual table plus held-out conditions outside the training ranges.
set -eo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DEMO_ROOT/env.sh"
cd "$DEMO_ROOT"

TEACHER=logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline/model_1499.pt
OLD_STUDENT=logs/student/bc_more_teacher/student.pt
NEW_STUDENT_DIR=logs/student/bc_visual_rand
DATA=datasets/lift_vision_teacher_visrand.h5
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

if [[ ! -f "$DATA" ]]; then
    gpu_check "randomized collection"
    python -u scripts/collect_lift_vision.py --checkpoint "$TEACHER" --randomize --num_envs 125 --episodes_per_env 10 \
        --seed 3 --out "$DATA" --headless --device cuda:0 | grep "\[collect\]"
fi
if [[ ! -f "$NEW_STUDENT_DIR/student.pt" ]]; then
    gpu_check "training"
    python -u scripts/train_student.py --data "$DATA" --out "$NEW_STUDENT_DIR" --epochs 20 --device cuda:0 | grep -E "frames|best"
fi

# conditions inside the training ranges (the existing visual table) and held out (outside them)
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

evaluate() {  # <condition> <name> <policy args...>
    local condition="$1" name="$2"; shift 2
    local key="${condition%%|*}" extra="${condition#*|}"
    [[ -f "$OUT/$key/$name/summary.json" ]] && return 0
    gpu_check "$name $key"
    python -u scripts/eval_student.py "$@" --out_dir "$OUT/$key/$name" --save_view "$OUT/$key/${name}_view.png" \
        $extra --headless --device cuda:0 | grep -E "success_rate\"|final_dist_mean_m"
}

for condition in "${seen[@]}" "${held_out[@]}"; do
    evaluate "$condition" student_visual_rand --policy student --student "$NEW_STUDENT_DIR/student.pt"
done
for condition in "${held_out[@]}"; do
    evaluate "$condition" student --policy student --student "$OLD_STUDENT"
    evaluate "$condition" teacher --policy teacher --checkpoint "$TEACHER"
done
python scripts/make_visual_table.py
echo "[run_visual_randomization] done"
