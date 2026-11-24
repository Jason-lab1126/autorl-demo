"""LLM-only attribution baseline: interface only, not wired to a model yet.

The baseline sees what an engineer would put in a bug report — a compact text summary of one episode
(task outcome, deviation from a nominal-sim replay per joint, residual against each candidate) — and
returns a distribution over the four classes plus H_other. attribution_eval.py calls
``LLMAttributor.predict`` for every kept episode once ``available`` is True.

To connect a model, subclass LLMAttributor, implement ``_complete(prompt) -> str`` (the raw model
reply) and set ``available = True``; ``predict`` builds the prompt and parses the JSON answer.
"""

import json

LABELS = ["H_theta-mass", "H_theta-friction", "H_nu", "H_pi", "H_other"]

PROMPT = """You are debugging a simulated Franka robot lifting a cube. One episode was recorded on the "real"
system; its commanded actions were replayed open loop in a nominal simulator and under candidate
modifications. Decide which cause explains the episode:
  H_theta-mass      the cube is heavier or lighter than modelled
  H_theta-friction  cube/finger friction differs from the model
  H_nu              the simulator's numerics / actuator model differ (not a physical parameter)
  H_pi              the physics is nominal; the policy itself is at fault
  H_other           none of the above
Episode summary (JSON):
{summary}
Reply with JSON only: {{"probabilities": {{label: probability for each of {labels}}}, "reason": "<one sentence>"}}"""


def summarize_episode(features: dict) -> dict:
    """Bug-report summary of one episode from attribution_eval.py's per-episode features."""
    return {
        "task_success": not features["failed"],
        "final_cube_to_goal_cm": round(features["final_dist_cm"], 2),
        "replay_residual_per_candidate": {c: round(float(r), 4) for c, r in features["residual"].items()},
        "noise_floor_residual": round(float(features["floor"]), 4),
        "mean_abs_dev_vs_nominal_replay": {
            name: {"torque_sd": round(float(t), 3), "position_sd": round(float(q), 3)}
            for name, t, q in zip(features["joint_names"], features["dev_tau"], features["dev_q"])
        },
    }


class LLMAttributor:
    available = False

    def _complete(self, prompt: str) -> str:
        raise NotImplementedError("Connect a model by subclassing LLMAttributor.")

    def predict(self, features: dict) -> dict[str, float]:
        prompt = PROMPT.format(summary=json.dumps(summarize_episode(features), indent=1), labels=LABELS)
        reply = json.loads(self._complete(prompt))
        probs = {label: float(reply["probabilities"].get(label, 0.0)) for label in LABELS}
        total = sum(probs.values()) or 1.0
        return {label: p / total for label, p in probs.items()}
