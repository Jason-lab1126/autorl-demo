# Source this to get the Isaac Lab environment used by the demo.
#   source ~/autorl-demo/env.sh
# Pins everything to a single physical GPU (override with DEMO_GPU=<nvidia-smi index>).
export ISAACLAB_PATH="$HOME/IsaacLab"
export ISAACSIM_PATH="$HOME/isaac-sim"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${DEMO_GPU:-0}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate WL
export PYTHONPATH=""
source "$ISAACSIM_PATH/setup_conda_env.sh"
