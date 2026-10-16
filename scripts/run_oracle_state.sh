#!/usr/bin/env bash
# Oracle-state diagnostic for SmolVLA: is its failure about perception or about the action side?
#
# Exports the same 1,250 teacher episodes (lift_vision_raw.h5 and lift_vision_teacher_extra.h5)
# with the TRUE cube position appended to observation.state (24 numbers instead of 21), keeps the
# normal 84x84 image, fine-tunes SmolVLA with the frozen run's settings (20k steps, batch 64) and
# evaluates it closed-loop with the true cube position in the state at test time too.
# Only the final checkpoint is saved, to stay above the disk floor.
#
#     tmux new -s oracle 'scripts/run_oracle_state.sh'
#
# Stops if GPU 0 has too little free memory or the disk drops below MIN_FREE_GB.
set -eo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEMO_ROOT"

ROOT=datasets/lift_cube_franka_lerobot_1250_84_oracle_state
RUN=logs/smolvla/smolvla_1250_84_oracle_state
RESULTS=results/smolvla_1250_84_oracle_state
MIN_FREE_GB=${MIN_FREE_GB:-40}
MIN_FREE_MIB=12000
STATUS=logs/console/oracle_state_status.txt

say() { echo "[oracle_state] $(date '+%F %T') $*" | tee -a "$STATUS"; }
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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-gpu
# Isaac Sim's bundled packages (from env.sh) break LeRobot's imports
unset PYTHONPATH

# 1. export with the cube position in the state
if [[ ! -f "$ROOT/validation_report.json" ]]; then
    disk_check "before export"
    say "exporting to $ROOT"
    python -u scripts/export_lerobot.py --raw datasets/lift_vision_raw.h5 datasets/lift_vision_teacher_extra.h5 \
        --root "$ROOT" --no_cube_position --cube_in_state --repo_id local/lift_cube_franka_oracle_state \
        > logs/console/export_lerobot_oracle_state.log 2>&1 \
        || stop "export failed, see logs/console/export_lerobot_oracle_state.log"
    grep -a -A 3 '"observation.state"' "$ROOT/validation_report.json" | tee -a "$STATUS"
fi
say "dataset size $(du -sh "$ROOT" | cut -f1)"

# 2. fine-tune SmolVLA, frozen run's settings
if [[ "$(basename "$(readlink "$RUN/checkpoints/last" 2>/dev/null)")" != 020000 ]]; then
    disk_check "before training"
    gpu_check "training"
    ( while sleep 120; do
        free_gb=$(df -BG --output=avail "$DEMO_ROOT" | tail -1 | tr -dc 0-9)
        if (( free_gb < MIN_FREE_GB )); then
            say "STOPPED: disk below ${MIN_FREE_GB} GB during training (${free_gb} GB)"
            pkill -f "lerobot-train.*$RUN"; exit
        fi
      done ) &
    GUARD=$!
    trap 'kill $GUARD 2>/dev/null' EXIT
    say "training"
    ./scripts/train_smolvla.sh "$ROOT" "$RUN" --steps=20000 --batch_size=64 --save_freq=20000 --log_freq=200 --num_workers=8 \
        > logs/console/smolvla_train_oracle_state.log 2>&1 \
        || stop "training ended early (exit $?), see logs/console/smolvla_train_oracle_state.log"
    kill $GUARD 2>/dev/null || true
fi
[[ "$(basename "$(readlink "$RUN/checkpoints/last")")" == 020000 ]] || stop "training did not reach step 20000"
disk_check "after training"

# 3. closed-loop eval: same 100 episodes, replan 10, true cube position in the state
gpu_check "eval"
python -u scripts/smolvla_server.py --checkpoint "$RUN/checkpoints/last/pretrained_model" \
    > logs/console/smolvla_server_oracle_state.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 60); do grep -q listening logs/console/smolvla_server_oracle_state.log && break; sleep 5; done
grep -q listening logs/console/smolvla_server_oracle_state.log || stop "policy server did not start"
( source env.sh && python -u scripts/eval_student.py --policy smolvla --oracle_state \
    --out_dir "$RESULTS" --headless --device cuda:0 ) > logs/console/smolvla_eval_oracle_state.log 2>&1 \
    || stop "eval failed, see logs/console/smolvla_eval_oracle_state.log"
kill $SERVER 2>/dev/null || true
cat "$RESULTS/summary.json" | tee -a "$STATUS"
say "DONE"
