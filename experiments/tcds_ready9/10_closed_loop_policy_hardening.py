#!/usr/bin/env python3
r"""Phase 8 — Closed-loop policy hardening (CORE phase).

A simulated embodied affective-control loop over recorded EEG observations. The
agent must drive a noisy, action-determined environment to a target affective
state and perceptually confirm it. This is a closed-loop *simulation* over
recorded observations -- not physical robot embodiment and not live deployment.

Expected-free-energy (EFE) controller -- defined here, in the output metadata,
and in the manuscript:

  * Belief state  b in Delta^2 : categorical posterior over the latent affective
    state, accumulated by sequential Bayesian updates.
  * Likelihood    P(o|s)       : the ARSPI-Net perceptual estimator (an L2
    logistic readout on the E+D+T substrate) evaluated on the drawn observation.
  * Transition    P(s'|a)       = (1 - eps) * onehot(a) + eps * uniform  (action
    determines the next state up to transition noise eps).
  * Preferred     C             : a distribution peaked on the target s* (prior
    preference over outcomes).
  * Risk(a)       = KL( P(s'|a) || C )            (pragmatic / goal term).
  * Ambiguity(a)  = E_{s'~P(s'|a)} [ H_hat(s') ]  (epistemic term; H_hat(s) is
    the expected observation-posterior entropy for true state s, estimated on
    the training fold).
  * EFE(a)        = Risk(a) + Ambiguity(a).
  * Action rule   a_t = argmin_a EFE(a).
  * Stopping      posterior b(s*) >= theta_conf, or max_steps reached.
  * Oracle        perfect perception: acts a=s* and stops when the true state is
    s*; defines the leak-free upper bound.

Policies: passive (belief-blind), random, pragmatic_only (Risk only),
epistemic_only (Ambiguity only -- formally separable), efe (Risk+Ambiguity),
oracle.

Outputs:
    outputs/tcds_ready9/analysis/closed_loop_policy_metrics.csv
    outputs/tcds_ready9/analysis/closed_loop_policy_summary.csv
    outputs/tcds_ready9/analysis/closed_loop_policy_config.json

Run:
    python experiments/tcds_ready9/10_closed_loop_policy_hardening.py
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402

N_STATES = 3
THETA_CONF = 0.80
MAX_STEPS = 15
PREF_DELTA = 0.02            # preferred-distribution leak onto non-target states


def transition(a: int, eps: float) -> np.ndarray:
    p = np.full(N_STATES, eps / N_STATES)
    p[a] += (1.0 - eps)
    return p


def preferred(target: int) -> np.ndarray:
    c = np.full(N_STATES, PREF_DELTA / (N_STATES - 1))
    c[target] = 1.0 - PREF_DELTA
    return c


def _kl(p, q):
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1)
    return float(np.sum(p * np.log(p / q)))


def _entropy(p):
    p = np.clip(p, 1e-12, 1)
    return float(-np.sum(p * np.log(p)))


def select_action(policy, target, eps, H_hat, rng):
    """Return the action chosen by the given policy."""
    if policy == "passive":
        return target                      # belief-blind: drive to target, no verify
    if policy == "random":
        return int(rng.integers(N_STATES))
    if policy == "oracle":
        return target
    C = preferred(target)
    risk = np.array([_kl(transition(a, eps), C) for a in range(N_STATES)])
    amb = np.array([float(transition(a, eps) @ H_hat) for a in range(N_STATES)])
    if policy == "pragmatic_only":
        score = risk
    elif policy == "epistemic_only":
        score = amb
    elif policy == "efe":
        score = risk + amb
    else:
        score = risk
    return int(np.argmin(score))


def run_episode(policy, eps, target, proba_by_class, H_hat, rng):
    """Return (success, steps, final_belief, believed_class, true_majority)."""
    b = np.full(N_STATES, 1.0 / N_STATES)
    true_counts = np.zeros(N_STATES)
    for step in range(1, MAX_STEPS + 1):
        a = select_action(policy, target, eps, H_hat, rng)
        true_state = int(rng.choice(N_STATES, p=transition(a, eps)))
        true_counts[true_state] += 1
        # draw an observation of the realized true state; use its likelihood
        pool = proba_by_class[true_state]
        L = pool[rng.integers(len(pool))]
        if policy == "oracle":
            # perfect perception: belief collapses onto the true state
            b = np.zeros(N_STATES); b[true_state] = 1.0
        else:
            b = b * np.clip(L, 1e-9, 1.0)
            b = b / b.sum()
        if policy == "passive":
            # belief-blind: trust the action after one step (no perceptual verify)
            return int(true_state == target), step, b, int(target), int(np.argmax(true_counts))
        if policy == "oracle":
            if true_state == target:
                return 1, step, b, target, int(np.argmax(true_counts))
            continue
        if b[target] >= THETA_CONF:
            believed = int(np.argmax(b))
            return int(believed == target), step, b, believed, int(np.argmax(true_counts))
    believed = int(np.argmax(b))
    return int(believed == target and policy not in ("random", "epistemic_only")
               and b[target] >= THETA_CONF), MAX_STEPS, b, believed, int(np.argmax(true_counts))


def main() -> int:
    try:
        data = cr.load_ready9()
    except Exception as e:
        (cfg.OUT_DIR / "CLOSED_LOOP_ERROR.md").write_text(
            f"# Closed-loop phase could not run\n\nInput load failed: {e}\n")
        print(f"[loop] FAILED to load inputs: {e}", file=sys.stderr)
        return 1

    blocks, y, subjects = data["blocks"], data["y"], data["subjects"]
    X = np.hstack([blocks["E"], blocks["D"], blocks["T"]])   # A8 substrate
    label_set = sorted(set(y.tolist()))

    policies = ["passive", "random", "pragmatic_only", "epistemic_only", "efe", "oracle"]
    total_eps = cfg.CLOSED_LOOP_EPISODES
    n_fold_units = len(cfg.SEEDS) * cfg.N_FOLDS_AFFECTIVE
    eps_per_unit = max(1, math.ceil(total_eps / n_fold_units))

    metric_rows = []
    # accumulators: (policy, epsilon) -> lists
    succ = defaultdict(list); steps = defaultdict(list)
    fent = defaultdict(list); brier = defaultdict(list)
    act_dist = defaultdict(lambda: np.zeros(N_STATES))
    fail_modes = defaultdict(lambda: defaultdict(int))

    for seed in cfg.SEEDS:
        cv = cr.subject_grouped_cv(cfg.N_FOLDS_AFFECTIVE, seed)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups=subjects)):
            _, proba_tr, _ = cr.fit_logreg_fold(X[tr], y[tr], X[tr], seed=seed)
            pred_te, proba_te, _ = cr.fit_logreg_fold(X[tr], y[tr], X[te], seed=seed)
            # per-state expected observation-posterior entropy (training fold)
            H_hat = np.array([
                np.mean([_entropy(proba_tr[i]) for i in range(len(tr)) if y[tr][i] == s])
                if (y[tr] == s).any() else math.log(N_STATES)
                for s in range(N_STATES)
            ])
            proba_by_class = {s: proba_te[y[te] == s] for s in range(N_STATES)}
            if any(len(proba_by_class[s]) == 0 for s in range(N_STATES)):
                continue
            rng = np.random.default_rng(13 * seed + fold)
            for ep_eps in cfg.CLOSED_LOOP_EPSILON:
                for _ in range(eps_per_unit):
                    target = int(rng.integers(N_STATES))
                    for pol in policies:
                        s_ok, st, b, believed, true_maj = run_episode(
                            pol, ep_eps, target, proba_by_class, H_hat, rng)
                        key = (pol, ep_eps)
                        succ[key].append(s_ok); steps[key].append(st)
                        fent[key].append(_entropy(b))
                        oneh = np.zeros(N_STATES); oneh[target] = 1.0
                        brier[key].append(float(np.sum((b - oneh) ** 2)))
                        act_dist[key][believed] += 1
                        if not s_ok:
                            fail_modes[key][f"target{target}->believed{believed}"] += 1

    if not succ:
        (cfg.OUT_DIR / "CLOSED_LOOP_ERROR.md").write_text(
            "# Closed-loop phase failed\n\nNo episodes were simulated (fold pools "
            "empty).\n")
        print("[loop] FAILED: no episodes", file=sys.stderr)
        return 1

    summary_rows = []
    for (pol, ep_eps), s in sorted(succ.items()):
        n = len(s); ns = int(np.sum(s))
        lo, hi = cr.wilson_ci(ns, n)
        st = steps[(pol, ep_eps)]
        succ_steps = [st[i] for i in range(n) if s[i]]
        ent = fent[(pol, ep_eps)]
        ent_red = (math.log(N_STATES) - float(np.mean(ent))) / max(1.0, float(np.mean(st)))
        ad = act_dist[(pol, ep_eps)]; ad = (ad / ad.sum()).round(3).tolist()
        summary_rows.append({
            "policy": pol, "epsilon": ep_eps, "n_episodes": n,
            "success_rate": round(ns / n, 4),
            "success_ci_lo": round(lo, 4), "success_ci_hi": round(hi, 4),
            "failure_rate": round(1 - ns / n, 4),
            "median_steps_to_success": (float(np.median(succ_steps)) if succ_steps else None),
            "mean_steps": round(float(np.mean(st)), 3),
            "final_entropy_mean": round(float(np.mean(ent)), 4),
            "entropy_reduction_per_step": round(ent_red, 5),
            "brier_mean": round(float(np.mean(brier[(pol, ep_eps)])), 4),
            "believed_class_dist": ad,
        })
        for i in range(n):
            metric_rows.append({
                "policy": pol, "epsilon": ep_eps, "episode": i,
                "success": s[i], "steps": st[i],
                "final_entropy": round(ent[i], 4),
                "brier": round(brier[(pol, ep_eps)][i], 4),
            })

    cr.write_csv(cfg.ANALYSIS_DIR / "closed_loop_policy_metrics.csv", metric_rows)
    cr.write_csv(cfg.ANALYSIS_DIR / "closed_loop_policy_summary.csv", summary_rows)
    cr.write_json(cfg.ANALYSIS_DIR / "closed_loop_policy_config.json", {
        "provenance": cr.base_provenance(),
        "controller": {
            "belief_state": "categorical posterior over 3 affective states, "
                            "sequential Bayesian accumulation",
            "likelihood": "L2 logistic readout on the E+D+T substrate (ARSPI-Net "
                          "perceptual estimator)",
            "transition": "P(s'|a) = (1-eps) onehot(a) + eps uniform",
            "preferred": f"peaked on target (delta={PREF_DELTA})",
            "risk": "KL(P(s'|a) || preferred)",
            "ambiguity": "E_{s'~P(s'|a)}[ H_hat(s') ]",
            "efe": "risk + ambiguity",
            "action_rule": "argmin_a EFE(a)",
            "stopping": f"b(target) >= {THETA_CONF} or {MAX_STEPS} steps",
            "oracle": "perfect perception upper bound (leak-free)",
        },
        "policies": policies,
        "epsilons": cfg.CLOSED_LOOP_EPSILON,
        "episodes_per_policy_per_epsilon_target": total_eps,
        "episodes_per_fold_unit": eps_per_unit,
        "fold_units": n_fold_units,
        "failure_modes": {f"{p}|eps={e}": dict(v) for (p, e), v in fail_modes.items()},
    })
    n_ep = len(succ[(policies[0], cfg.CLOSED_LOOP_EPSILON[0])])
    print(f"[loop] {len(policies)} policies x {len(cfg.CLOSED_LOOP_EPSILON)} eps; "
          f"~{n_ep} episodes/cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
