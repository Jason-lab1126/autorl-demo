"""Settings, candidates and posterior rules shared by attribute.py (rollouts) and attribution_eval.py.

Numpy only, no simulator imports.
"""

import numpy as np

CLASSES = ["H_theta-mass", "H_theta-friction", "H_nu", "H_pi"]

# setting -> (perturbation, multiplier, engine, checkpoint iteration, true class[, injected faults])
# engine: "implicit", "explicit" (all actuators IdealPD) or "explicit_arm" (arm IdealPD, gripper implicit).
# Faults are LiftSim's q_bias / action_delay / gripper_effort_scale.
REAL_SETTINGS = {  # test logs: causes off the candidate grid
    "mass_x1.5": ("cube_mass", 1.5, "implicit", 1499, "H_theta-mass"),
    "mass_x2.5": ("cube_mass", 2.5, "implicit", 1499, "H_theta-mass"),
    "mass_x4": ("cube_mass", 4.0, "implicit", 1499, "H_theta-mass"),
    "fric_x0.7": ("friction", 0.7, "implicit", 1499, "H_theta-friction"),
    "fric_x0.4": ("friction", 0.4, "implicit", 1499, "H_theta-friction"),
    "fric_x0.25": ("friction", 0.25, "implicit", 1499, "H_theta-friction"),
    "policy_300": ("none", 1.0, "implicit", 300, "H_pi"),
    "policy_500": ("none", 1.0, "implicit", 500, "H_pi"),
    "engine_explicit": ("none", 1.0, "explicit", 1499, "H_nu"),
}
# training logs for the supervised and threshold baselines and the H_other threshold: same classes,
# intensities seen neither in the test logs nor on the candidate grid, seeds disjoint from the test seeds
TRAIN_SETTINGS = {
    "mass_x1.75": ("cube_mass", 1.75, "implicit", 1499, "H_theta-mass"),
    "mass_x3.5": ("cube_mass", 3.5, "implicit", 1499, "H_theta-mass"),
    "mass_x6": ("cube_mass", 6.0, "implicit", 1499, "H_theta-mass"),
    "fric_x0.55": ("friction", 0.55, "implicit", 1499, "H_theta-friction"),
    "fric_x0.35": ("friction", 0.35, "implicit", 1499, "H_theta-friction"),
    "fric_x0.15": ("friction", 0.15, "implicit", 1499, "H_theta-friction"),
    "policy_100": ("none", 1.0, "implicit", 100, "H_pi"),
    "policy_700": ("none", 1.0, "implicit", 700, "H_pi"),
    "engine_explicit_arm": ("none", 1.0, "explicit_arm", 1499, "H_nu"),
}
# faults outside the hypothesis space
OTHER_SETTINGS = {
    "other_action_delay": ("none", 1.0, "implicit", 1499, "H_other", {"action_delay": [2, 3]}),
    "other_q_bias": ("none", 1.0, "implicit", 1499, "H_other", {"q_bias": 0.02}),
    "other_grip_x0.5": ("none", 1.0, "implicit", 1499, "H_other", {"gripper_effort_scale": 0.5}),
    "other_grip_x0.25": ("none", 1.0, "implicit", 1499, "H_other", {"gripper_effort_scale": 0.25}),
}
ALL_SETTINGS = {**REAL_SETTINGS, **TRAIN_SETTINGS, **OTHER_SETTINGS}
REF_SETTING = ("none", 1.0, "implicit", 1499, None)
ENGINES = {"implicit": False, "explicit": True, "explicit_arm": ["panda_shoulder", "panda_forearm"]}

# candidate -> (perturbation, multiplier, engine, class)
CANDIDATES = {
    "nominal": ("none", 1.0, "implicit", "H_pi"),
    "mass_x2": ("cube_mass", 2.0, "implicit", "H_theta-mass"),
    "mass_x3": ("cube_mass", 3.0, "implicit", "H_theta-mass"),
    "mass_x5": ("cube_mass", 5.0, "implicit", "H_theta-mass"),
    "fric_x0.5": ("friction", 0.5, "implicit", "H_theta-friction"),
    "fric_x0.3": ("friction", 0.3, "implicit", "H_theta-friction"),
    "fric_x0.2": ("friction", 0.2, "implicit", "H_theta-friction"),
    "numerics": ("none", 1.0, "explicit", "H_nu"),
}


def logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return -np.inf if m == -np.inf else m + np.log(np.exp(x - m).sum())


def new_rule(res: dict, failed: bool, sigma2: float, threshold: float) -> dict:
    """Gaussian likelihood exp(-r / (2 sigma^2)), mean within each class, normalised over the 4 classes.
    H_pi takes the nominal candidate's likelihood only for a failed episode whose nominal residual is
    within the noise-floor threshold."""
    log_l = {c: -r / (2.0 * sigma2) for c, r in res.items()}
    class_ll = {}
    for cls in CLASSES:
        if cls == "H_pi":
            class_ll[cls] = log_l["nominal"] if failed and res["nominal"] <= threshold else -np.inf
        else:
            members = np.array([log_l[c] for c in CANDIDATES if CANDIDATES[c][3] == cls])
            class_ll[cls] = logsumexp(members) - np.log(len(members))
    z = logsumexp(np.array(list(class_ll.values())))
    return {cls: float(np.exp(ll - z)) for cls, ll in class_ll.items()}


def old_rule(res: dict, failed: bool) -> dict:
    """v1 rule: exp(-r / T), T = median residual, summed per class; nominal is H_pi if failed, else none."""
    r = np.array(list(res.values()))
    temp = max(np.median(r), 1e-12)
    post = np.exp(-(r - r.min()) / temp)
    post /= post.sum()
    out = {cls: 0.0 for cls in CLASSES + ["none"]}
    for cand, p in zip(res, post):
        cls = CANDIDATES[cand][3]
        out[cls if cls != "H_pi" or failed else "none"] += p
    return out
