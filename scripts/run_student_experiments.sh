#!/usr/bin/env bash
# DAgger vs. more teacher data for the vision student, with matched data budgets.
#   A) 3 DAgger rounds: 250 student-driven episodes each, labelled by the teacher.
#   B) plain BC with 750 extra teacher-driven episodes.
# Both end at 1250 episodes. Every student is trained from scratch for 20 epochs and evaluated on
# the same 100 episodes. Expects datasets/lift_vision_raw.h5 and logs/student/bc_500ep/ to exist.
set -eo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DEMO_ROOT/env.sh"
cd "$DEMO_ROOT"

TEACHER=logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline/model_1499.pt
BASE=datasets/lift_vision_raw.h5
MIN_FREE_MIB=6000

# Before every GPU run: log the load on the pinned GPU and stop if there is too little memory left.
gpu_check() {
    local used total util
    IFS=', ' read -r used total util < <(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    echo "[gpu_check] $(date +%H:%M:%S) before '$1': GPU $CUDA_VISIBLE_DEVICES ${used}/${total} MiB used, ${util}% util"
    if (( total - used < MIN_FREE_MIB )); then
        echo "[gpu_check] less than ${MIN_FREE_MIB} MiB free, stopping"; exit 3
    fi
}

evaluate() {  # <student dir> <results dir>
    gpu_check "eval $1"
    python -u scripts/eval_student.py --policy student --student "$1/student.pt" --out_dir "$2" --headless --device cuda:0 \
        | grep -E "success_rate\"|final_dist_mean_m|wrote"
}

# A) DAgger
data=("$BASE")
student=logs/student/bc_500ep
for round in 1 2 3; do
    gpu_check "DAgger round $round collection"
    python -u scripts/collect_lift_vision.py --actor student --student "$student/student.pt" --action_noise 0 \
        --checkpoint "$TEACHER" --num_envs 125 --episodes_per_env 2 --seed $((10 + round)) \
        --out "datasets/lift_vision_dagger_r$round.h5" --headless --device cuda:0 | grep "\[collect\]"
    data+=("datasets/lift_vision_dagger_r$round.h5")
    student=logs/student/dagger_r$round
    gpu_check "DAgger round $round training"
    python -u scripts/train_student.py --data "${data[@]}" --out "$student" --epochs 20 --device cuda:0 | grep -E "frames|best"
    evaluate "$student" "results/student_dagger_r$round"
done

# B) the same number of extra episodes, from the teacher only
gpu_check "extra teacher collection"
python -u scripts/collect_lift_vision.py --checkpoint "$TEACHER" --num_envs 125 --episodes_per_env 6 --seed 2 \
    --out datasets/lift_vision_teacher_extra.h5 --headless --device cuda:0 | grep "\[collect\]"
gpu_check "BC + more teacher data training"
python -u scripts/train_student.py --data "$BASE" datasets/lift_vision_teacher_extra.h5 --out logs/student/bc_more_teacher \
    --epochs 20 --device cuda:0 | grep -E "frames|best"
evaluate logs/student/bc_more_teacher results/student_bc_more

python scripts/make_student_table.py
echo "[run_student_experiments] done"
