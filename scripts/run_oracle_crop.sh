#!/usr/bin/env bash
# Oracle-crop diagnostic for SmolVLA: does it fail because it cannot see the cube?
#
# Re-collects the same 1,250 teacher episodes (same teacher, seeds, noise and environment counts as
# lift_vision_raw.h5 and lift_vision_teacher_extra.h5) with a second camera rendering at 512x512.
# Every frame keeps the normal 84x84 view and adds a 128x128 crop of the 512 render centered on the
# cube's TRUE position projected into the camera (an oracle crop: the simulator's cube position
# stands in for a perfect detector). The raw files go to /dev/shm (RAM), are exported straight to a
# LeRobot dataset with both images as cameras and then removed. SmolVLA is fine-tuned with the
# frozen run's settings (20k steps, batch 64) and evaluated closed-loop with the oracle crop.
#
#     tmux new -s oracle 'scripts/run_oracle_crop.sh'
#
# Stops if GPU 0 has too little free memory, if the disk drops below MIN_FREE_GB, or if the
# training does not finish by TRAIN_DEADLINE.
set -eo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEMO_ROOT"

TEACHER=logs/rsl_rl/franka_lift/2026-10-13_15-19-32_ppo_baseline/model_1499.pt
SHM=/dev/shm/autorl_oracle_crop
ROOT=datasets/lift_cube_franka_lerobot_1250_oracle_crop
RUN=logs/smolvla/smolvla_1250_oracle_crop
RESULTS=results/smolvla_1250_oracle_crop
MIN_FREE_GB=${MIN_FREE_GB:-40}
MIN_FREE_MIB=12000
TRAIN_DEADLINE=${TRAIN_DEADLINE:-"tomorrow 08:00"}
STATUS=logs/console/oracle_crop_status.txt

say() { echo "[oracle_crop] $(date '+%F %T') $*" | tee -a "$STATUS"; }
stop() { say "STOPPED: $*"; exit 3; }

gpu_check() {
    local used total util
    IFS=', ' read -r used total util < <(nvidia-smi -i 0 --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    say "before '$1': GPU 0 ${used}/${total} MiB used, ${util}% util"
    (( total - used >= MIN_FREE_MIB )) || stop "GPU 0 has less than ${MIN_FREE_MIB} MiB free before '$1'"
}

disk_check() {
    local free_gb
    free_gb=$(df -BG --output=avail "$DEMO_ROOT" | tail -1 | tr -dc 0-9)
    say "disk: ${free_gb} GB free ($1)"
    (( free_gb >= MIN_FREE_GB )) || stop "disk below ${MIN_FREE_GB} GB free ($1)"
}

# 1. collection, in the Isaac Lab environment
mkdir -p "$SHM"
collect() {
    local seed=$1 episodes=$2 out=$3
    [[ -f "$out" ]] && { say "$out exists, skipping collection"; return; }
    gpu_check "collect seed $seed"
    ( source env.sh && python -u scripts/collect_lift_vision.py --checkpoint "$TEACHER" --num_envs 125 \
        --episodes_per_env "$episodes" --seed "$seed" --crop_size 128 --crop_render_size 512 --no_compression \
        --out "$out" --headless --device cuda:0 ) 2>&1 | grep -a "\[collect\]" | tee -a "$STATUS" \
        || stop "collection seed $seed failed"
    [[ -f "$out" ]] || stop "collection seed $seed wrote no file"
}
if [[ ! -f "$ROOT/validation_report.json" ]]; then
    disk_check "before collection"
    collect 1 4 "$SHM/raw.h5"
    collect 2 6 "$SHM/extra.h5"
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-gpu
# Isaac Sim's bundled packages (from env.sh) break LeRobot's imports
unset PYTHONPATH

# 2. same episodes as before? compare with the original raw files (actions and cube positions)
if [[ -f "$SHM/raw.h5" ]]; then
    python - "$SHM/raw.h5" datasets/lift_vision_raw.h5 "$SHM/extra.h5" datasets/lift_vision_teacher_extra.h5 <<'EOF' | tee -a "$STATUS"
import sys, h5py, numpy as np
for new, old in zip(sys.argv[1::2], sys.argv[2::2]):
    a, b = h5py.File(new), h5py.File(old)
    same_len = np.array_equal(a["episode_length"][:], b["episode_length"][:])
    n = min(len(a["action"]), len(b["action"]))
    print(f"[oracle_crop] {new} vs {old}: same episode lengths {same_len}, max |action diff| "
          f"{np.abs(a['action'][:n] - b['action'][:n]).max():.2e}, max |cube diff| "
          f"{np.abs(a['cube_position'][:n] - b['cube_position'][:n]).max():.2e} m, same val split "
          f"{np.array_equal(a['episode_is_val'][:], b['episode_is_val'][:])}")
EOF
fi

# 3. export straight to LeRobot (two cameras), then free the RAM copies
if [[ ! -f "$ROOT/validation_report.json" ]]; then
    say "exporting to $ROOT"
    python -u scripts/export_lerobot.py --raw "$SHM/raw.h5" "$SHM/extra.h5" --root "$ROOT" --no_cube_position \
        --repo_id local/lift_cube_franka_oracle_crop > logs/console/export_lerobot_oracle_crop.log 2>&1 \
        || stop "export failed, see logs/console/export_lerobot_oracle_crop.log"
    grep -a -A 40 VALIDATED logs/console/export_lerobot_oracle_crop.log | tee -a "$STATUS"
fi
[[ -f "$ROOT/validation_report.json" ]] || stop "export did not validate"
rm -f "$SHM/raw.h5" "$SHM/extra.h5"
say "dataset size $(du -sh "$ROOT" | cut -f1)"

# 4. fine-tune SmolVLA, frozen run's settings, with the crop as the second camera
if [[ ! -f "$RUN/checkpoints/last/pretrained_model/model.safetensors" ]] || \
   [[ "$(basename "$(readlink "$RUN/checkpoints/last")")" != 020000 ]]; then
    disk_check "before training"
    gpu_check "training"
    # guard: stop the training if the disk drops below the floor or the deadline passes
    deadline=$(date -d "$TRAIN_DEADLINE" +%s)
    ( while sleep 120; do
        free_gb=$(df -BG --output=avail "$DEMO_ROOT" | tail -1 | tr -dc 0-9)
        if (( free_gb < MIN_FREE_GB )); then
            say "STOPPED: disk below ${MIN_FREE_GB} GB during training (${free_gb} GB)"
            pkill -f "lerobot-train.*$RUN"; exit
        fi
        if (( $(date +%s) > deadline )); then
            say "STOPPED: training not done by $TRAIN_DEADLINE"
            pkill -f "lerobot-train.*$RUN"; exit
        fi
      done ) &
    GUARD=$!
    trap 'kill $GUARD 2>/dev/null' EXIT
    say "training, deadline $(date -d "@$deadline" '+%F %T')"
    RENAME_MAP='{"observation.images.front": "observation.images.camera1", "observation.images.crop": "observation.images.camera2"}' \
        ./scripts/train_smolvla.sh "$ROOT" "$RUN" \
        --steps=20000 --batch_size=64 --save_freq=5000 --log_freq=200 --num_workers=8 \
        > logs/console/smolvla_train_oracle_crop.log 2>&1 \
        || stop "training ended early (exit $?), see logs/console/smolvla_train_oracle_crop.log"
    kill $GUARD 2>/dev/null || true
fi
# keep only the newest checkpoint (the pruning loop in train_smolvla.sh stops before the last one)
last=$(basename "$(readlink "$RUN/checkpoints/last")")
[[ "$last" == 020000 ]] || stop "training did not reach step 20000 (last checkpoint $last)"
for dir in "$RUN"/checkpoints/[0-9]*; do
    [[ "$(basename "$dir")" < "$last" ]] && rm -rf "$dir" && say "pruned $(basename "$dir")"
done

# 5. closed-loop eval: same 100 episodes, replan 10, oracle crop from the true cube position
gpu_check "eval"
python -u scripts/smolvla_server.py --checkpoint "$RUN/checkpoints/last/pretrained_model" \
    > logs/console/smolvla_server_oracle_crop.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 60); do grep -q listening logs/console/smolvla_server_oracle_crop.log && break; sleep 5; done
grep -q listening logs/console/smolvla_server_oracle_crop.log || stop "policy server did not start"
( source env.sh && python -u scripts/eval_student.py --policy smolvla --crop_size 128 --crop_render_size 512 \
    --out_dir "$RESULTS" --headless --device cuda:0 ) > logs/console/smolvla_eval_oracle_crop.log 2>&1 \
    || stop "eval failed, see logs/console/smolvla_eval_oracle_crop.log"
kill $SERVER 2>/dev/null || true
cat "$RESULTS/summary.json" | tee -a "$STATUS"
say "DONE"
