#!/usr/bin/env bash
# PPO (rsl_rl) training for Isaac-Lift-Cube-Franka-v0, headless.
# Runs from ~/autorl-demo so Isaac Lab writes to ~/autorl-demo/logs/rsl_rl/franka_lift/<timestamp>/.
# Extra args are passed through to Isaac Lab's train.py (e.g. --num_envs 2048 --max_iterations 100).
set -eo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DEMO_ROOT/env.sh"
cd "$DEMO_ROOT"

mkdir -p logs/console
CONSOLE_LOG="logs/console/train_lift_$(date +%Y-%m-%d_%H-%M-%S).log"
echo "[train_lift] GPU=$CUDA_VISIBLE_DEVICES console log: $DEMO_ROOT/$CONSOLE_LOG"

python -u "$ISAACLAB_PATH/scripts/reinforcement_learning/rsl_rl/train.py" \
    --task Isaac-Lift-Cube-Franka-v0 \
    --headless \
    --device cuda:0 \
    "$@" 2>&1 | tee "$CONSOLE_LOG"
