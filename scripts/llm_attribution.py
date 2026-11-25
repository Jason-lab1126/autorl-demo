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


# ---------------------------------------------------------------------------------------------------
# Proposer: an LLM suggests new, directly executable do(nu) / do(theta) candidates for logs that no
# existing candidate explains. Each candidate sets every knob below (nominal value = unchanged);
# to_replay_cfg() turns it into the keyword arguments of an attribute.py "replay_cfg" job.

PROPOSER_MODEL = "claude-sonnet-5-5"

# knob -> (kind, nominal, lower, upper or choices, description)
KNOBS = {
    "physics_substeps": ("int", 1, 1, 4, "multiplier on physics steps per control step: physics dt 0.01 s / k, decimation 2 x k (control rate unchanged)"),
    "robot_solver_iterations": ("int", 8, 1, 64, "PhysX solver position iterations of the robot articulation"),
    "cube_solver_iterations": ("int", 16, 1, 64, "PhysX solver position iterations of the cube"),
    "actuator_model": ("enum", "implicit", None, ["implicit", "explicit_all", "explicit_arm"],
                       "implicit = PhysX joint drives; explicit_* = Isaac Lab IdealPD torques (all joints, or arm only)"),
    "cube_mass_scale": ("float", 1.0, 0.05, 20.0, "multiplier on the cube's mass (inertia scaled with it)"),
    "friction_scale": ("float", 1.0, 0.05, 20.0, "multiplier on static and dynamic friction of the cube and the finger pads"),
    "gripper_effort_scale": ("float", 1.0, 0.1, 2.0, "multiplier on the finger drives' force limit (200 N)"),
}
NUMERICS_KNOBS = ("physics_substeps", "robot_solver_iterations", "cube_solver_iterations", "actuator_model")

SIM_CONFIG = {
    "engine": "Isaac Lab 2.0.2 / Isaac Sim 4.5, PhysX GPU pipeline",
    "physics_dt_s": 0.01, "decimation": 2, "control_rate_hz": 50, "episode_length_steps": 250,
    "robot": "Franka Panda, articulation solver position iterations 8, velocity iterations 0, self-collisions on",
    "actuators": ("implicit PhysX PD drives: joints 1-4 kp 80 kd 4 effort 87 N·m; joints 5-7 kp 80 kd 4 effort 12 N·m;"
                  " fingers kp 2000 kd 100 force 200 N; arm actions are joint-position targets (scale 0.5), gripper binary"),
    "cube": "DexCube scaled 0.8 (~5 cm), rigid body solver position iterations 16, velocity iterations 1, max depenetration velocity 5 m/s",
    "contact": "PhysX defaults for contact/rest offsets; bounce threshold velocity 0.01 m/s; friction correlation distance 0.00625 m",
    "materials": "cube and finger pad friction from the USD assets (nominal = multiplier 1)",
}

PROPOSER_SYSTEM = """You help diagnose sim-to-real gaps for a simulated Franka robot that lifts a cube to a goal.
A recorded episode is replayed open loop (same commanded actions) in a simulator; the residual measures how far
the replayed joint positions and torques are from the recording (normalised; the sensor-noise floor is given).
No existing candidate explains this episode well. Propose new candidate simulator changes that could.

Every candidate must be directly executable: set ALL knobs, using the nominal value for knobs you do not change.
Knobs (nominal value, allowed range):
{knobs}
Prefer changing one or two knobs per candidate. Do not repeat the existing candidates listed in the summary.
Mark each candidate as "do_nu" (a numerics / solver / actuator-model change) or "do_theta" (a physical parameter
change). Give 3 to 5 candidates that are meaningfully different from each other."""

PROPOSER_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["do_nu", "do_theta"]},
                    "rationale": {"type": "string"},
                    "physics_substeps": {"type": "integer"},
                    "robot_solver_iterations": {"type": "integer"},
                    "cube_solver_iterations": {"type": "integer"},
                    "actuator_model": {"type": "string", "enum": ["implicit", "explicit_all", "explicit_arm"]},
                    "cube_mass_scale": {"type": "number"},
                    "friction_scale": {"type": "number"},
                    "gripper_effort_scale": {"type": "number"},
                },
                "required": ["kind", "rationale", *KNOBS],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analysis", "candidates"],
    "additionalProperties": False,
}


def _knob_text() -> str:
    lines = []
    for name, (kind, nominal, lo, hi, desc) in KNOBS.items():
        rng = f"one of {hi}" if kind == "enum" else f"[{lo}, {hi}]"
        lines.append(f"  {name}: {desc}; nominal {nominal}; allowed {rng}")
    return "\n".join(lines)


def validate_candidate(c: dict) -> dict | None:
    """Clamps every knob into its range; None if the candidate changes nothing or is malformed."""
    out = {"kind": c.get("kind"), "rationale": c.get("rationale", "")}
    for name, (kind, nominal, lo, hi, _) in KNOBS.items():
        v = c.get(name, nominal)
        try:
            if kind == "enum":
                v = v if v in hi else nominal
            elif kind == "int":
                v = int(min(max(int(round(float(v))), lo), hi))
            else:
                v = float(min(max(float(v), lo), hi))
        except (TypeError, ValueError):
            return None
        out[name] = v
    if all(out[n] == KNOBS[n][1] for n in KNOBS):
        return None
    return out


def candidate_key(c: dict) -> str:
    """Canonical id of the knobs a candidate changes (identical configurations share one replay)."""
    parts = [f"{n}={c[n]:.3g}" if isinstance(c[n], float) else f"{n}={c[n]}" for n in KNOBS if c[n] != KNOBS[n][1]]
    return "llm_" + "__".join(parts).replace(".", "p")


def candidate_class(c: dict) -> str:
    """Hypothesis class implied by the knobs a candidate changes (not by the model's own label)."""
    if any(c[n] != KNOBS[n][1] for n in NUMERICS_KNOBS):
        return "H_nu"
    import math
    dev = {"H_theta-mass": abs(math.log(c["cube_mass_scale"])), "H_theta-friction": abs(math.log(c["friction_scale"])),
           "H_theta-gripper": abs(math.log(c["gripper_effort_scale"]))}
    return max(dev, key=dev.get)


def to_replay_cfg(c: dict) -> dict:
    """attribute.py replay_cfg: LiftSim keyword arguments plus cube_mass_scale / friction_scale."""
    actuators = {"implicit": False, "explicit_all": True, "explicit_arm": ["panda_shoulder", "panda_forearm"]}[c["actuator_model"]]
    cfg = {"cube_mass_scale": c["cube_mass_scale"], "friction_scale": c["friction_scale"]}
    if c["physics_substeps"] != 1:
        cfg["substep_scale"] = c["physics_substeps"]
    if c["robot_solver_iterations"] != KNOBS["robot_solver_iterations"][1]:
        cfg["robot_solver_iters"] = c["robot_solver_iterations"]
    if c["cube_solver_iterations"] != KNOBS["cube_solver_iterations"][1]:
        cfg["cube_solver_iters"] = c["cube_solver_iterations"]
    if actuators:
        cfg["explicit_actuators"] = actuators
    if c["gripper_effort_scale"] != 1.0:
        cfg["gripper_effort_scale"] = c["gripper_effort_scale"]
    return cfg


def propose(summary: dict, n_samples: int, raw_log_path: str, tag: str) -> list[dict]:
    """Samples the proposer n_samples times; appends every raw reply to raw_log_path (JSON lines).
    Returns the validated candidates of all samples (with their sample index)."""
    import anthropic

    client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY
    system = PROPOSER_SYSTEM.format(knobs=_knob_text())
    user = "Episode summary (JSON):\n" + json.dumps(summary, indent=1)
    candidates = []
    for i in range(n_samples):
        record = {"tag": tag, "sample": i, "model": PROPOSER_MODEL}
        try:
            response = client.beta.messages.create(
                model=PROPOSER_MODEL,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": PROPOSER_SCHEMA}},
            )
        except anthropic.RateLimitError as e:
            record["error"] = f"rate limited: {e.message}"
        except anthropic.APIStatusError as e:
            record["error"] = f"API error {e.status_code}: {e.message}"
        except anthropic.APIConnectionError as e:
            record["error"] = f"connection error: {e}"
        else:
            text = next((b.text for b in response.content if b.type == "text"), "")
            record.update({"served_by": response.model, "stop_reason": response.stop_reason, "raw": text,
                           "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}})
            if response.stop_reason == "end_turn" and text:
                try:
                    reply = json.loads(text)
                except json.JSONDecodeError as e:
                    record["parse_error"] = str(e)
                else:
                    for c in reply.get("candidates", [])[:5]:
                        v = validate_candidate(c)
                        if v is not None:
                            candidates.append({**v, "sample": i})
        with open(raw_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    return candidates
