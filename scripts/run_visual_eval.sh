#!/usr/bin/env bash
# Visual robustness: the best student and the teacher under changes that only affect what the camera
# sees. 100 episodes per run, same seed and metric as every other eval. No training.
set -eo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DEMO_ROOT/env.sh"
cd "$DEMO_ROOT"

TEACHER=logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline/model_1499.pt
STUDENT=logs/student/bc_more_teacher/student.pt
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

# name and eval_student.py arguments for each condition
conditions=(
    "none|"
    "light_x0.5|--light_scale 0.5"
    "light_x2|--light_scale 2.0"
    "cube_red|--cube_color 0.8 0.1 0.1"
    "table_light_wood|--table_color 0.85 0.75 0.6"
    "camera_5cm|--camera_shift 0 0.05 0"
    "camera_10cm|--camera_shift 0 0.10 0"
)

for condition in "${conditions[@]}"; do
    name="${condition%%|*}"; extra="${condition#*|}"
    gpu_check "student $name"
    python -u scripts/eval_student.py --policy student --student "$STUDENT" --out_dir "$OUT/$name/student" \
        --save_view "$OUT/$name/student_view.png" $extra --headless --device cuda:0 | grep -E "success_rate\"|final_dist_mean_m"
    gpu_check "teacher $name"
    python -u scripts/eval_student.py --policy teacher --checkpoint "$TEACHER" --out_dir "$OUT/$name/teacher" \
        $extra --headless --device cuda:0 | grep -E "success_rate\"|final_dist_mean_m"
done
python scripts/make_visual_table.py
echo "[run_visual_eval] done"
