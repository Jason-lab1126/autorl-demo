"""lerobot-train with SmolVLA's vision encoder trained as well.

LeRobot's options either freeze the whole vision-language model (the default, train_expert_only)
or train it all (train_expert_only=false also trains the language layers). This keeps the default
and only unfreezes the vision encoder, so the run differs from the frozen one in that alone.
Takes the same arguments as lerobot-train; train_smolvla.sh uses it with TRAIN_VISION=1.
"""

import sys

from lerobot.policies.smolvla import smolvlm_with_expert
from lerobot.scripts.lerobot_train import main

frozen_set_requires_grad = smolvlm_with_expert.SmolVLMWithExpertModel.set_requires_grad


def set_requires_grad(self):
    frozen_set_requires_grad(self)
    for parameter in self.get_vlm_model().vision_model.parameters():
        parameter.requires_grad = True


smolvlm_with_expert.SmolVLMWithExpertModel.set_requires_grad = set_requires_grad

if __name__ == "__main__":
    sys.exit(main())
