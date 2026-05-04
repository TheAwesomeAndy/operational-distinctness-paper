#!/usr/bin/env python3
"""TCDS hardening: embodied closed-loop active-inference demonstration.

Brain-inspired grounding (TCDS CFP topic):
    "Predictive coding and active inference frameworks for embodied
    perception and decision-making."

A simulated embodied perceptual agent observes the same subject over a
sequence of T trials and must infer the subject's underlying affective
class distribution theta in {Neg, Neu, Pos}, where theta lives on the
3-simplex. The agent maintains a Dirichlet posterior over theta and
chooses, at each step, which ARSPI-Net feature-stream "spotlight"
(policy) to attend to. Different feature streams correspond to
different operationally distinct cortical sub-systems (E = LIF-spike
embedding, D = dynamical descriptors, T = topological descriptors,
E+D+T = full multi-stream).

Six policies are compared:

    E_only, D_only, T_only, EDT_full   (fixed policies)
    random                              (uniform random over the four)
    greedy_eig                          (active inference: pick the policy
                                         maximizing expected information
                                         gain about theta at each step)

Per-fold protocol (5-fold subject-grouped CV):

  1. Hold out one subject fold.
  2. For each policy, fit a per-policy LogisticRegression on the
     remaining subjects' (observation_i, class_i) pairs using only that
     policy's feature columns.
  3. For each held-out subject, simulate a length-T sequence by drawing
     T trial indices with replacement from the subject's 3 observations.
  4. Step-by-step:
       a. Action select a_t  (fixed/random/greedy).
       b. Read off the policy's predicted class probabilities for x_t.
       c. Bayesian update of Dirichlet posterior over theta with this
          soft observation.
       d. Record posterior predictive entropy, expected information
          gain, action taken.
  5. Final decision: argmax of the posterior predictive distribution.
     Accuracy is measured against the subject's empirical modal class
     across the sequence.

Outputs (in outputs/tcds_hardening/ and figures/tcds_hardening/):
    embodied_loop_results.csv            per-policy x subject_hash x step
    embodied_loop_summary.csv            per-policy aggregate metrics + CIs
    embodied_loop_diagnostics.json       hashes, runtime, parameters
    fig_embodied_loop.pdf                4-panel figure
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.operational_distinctness import config as cfg
from experiments.operational_distinctness.common import (
    build_feature_blocks,
    hash_subject_array,
    load_inputs,
)

OUT_DIR = REPO_ROOT / "outputs" / "tcds_hardening"
FIG_DIR = REPO_ROOT / "figures" / "tcds_hardening"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_NAMES = {0: "Negative", 1: "Neutral", 2: "Pleasant"}
POLICY_COLORS = {
    "E_only":      "#1b9e77",
    "D_only":      "#d95f02",
    "T_only":      "#7570b3",
    "EDT_full":    "#e7298a",
    "random":      "#666666",
    "greedy_eig":  "#000000",
}
FIXED_POLICIES = ("E_only", "D_only", "T_only", "EDT_full")


def _dirichlet_pred_entropy(alpha: np.ndarray) -> float:
    p = alpha / alpha.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _expected_info_gain(alpha: np.ndarray, prob_vec: np.ndarray) -> float:
    """Approximate EIG of a soft observation under current Dirichlet alpha.

    Uses the deterministic alpha + prob_vec update vs. the current alpha,
    measured as the KL divergence of the new predictive distribution from
    the current predictive. (Standard 1-step EIG approximation; closed-form
    under Dirichlet conjugacy with soft pseudo-counts.)
    """
    p_now = alpha / alpha.sum()
    alpha_new = alpha + prob_vec
    p_new = alpha_new / alpha_new.sum()
    eps = 1e-12
    return float(np.sum(p_new * (np.log(p_new + eps) - np.log(p_now + eps))))


def build_policy_features(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "E_only":   blocks["E"],
        "D_only":   blocks["D"],
        "T_only":   blocks["T"],
        "EDT_full": np.hstack([blocks["E"], blocks["D"], blocks["T"]]),
    }


def fit_policy_classifiers(policy_X: dict[str, np.ndarray],
                           y_train: np.ndarray,
                           train_idx: np.ndarray,
                           ) -> dict[str, tuple[StandardScaler, LogisticRegression]]:
    out = {}
    for name, X in policy_X.items():
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[train_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                 solver="lbfgs", random_state=cfg.RANDOM_STATE)
        clf.fit(Xtr, y_train)
        out[name] = (sc, clf)
    return out


def predict_proba_aligned(scaler: StandardScaler,
                          clf: LogisticRegression,
                          x: np.ndarray) -> np.ndarray:
    """Always returns a 3-vector in [0..1] aligned with class ids 0,1,2."""
    Xs = scaler.transform(x.reshape(1, -1))
    proba = clf.predict_proba(Xs)[0]
    p = np.zeros(3)
    for ci, cls in enumerate(clf.classes_):
        p[int(cls)] = proba[ci]
    p = np.clip(p, 1e-6, 1.0)
    return p / p.sum()


def simulate_subject(subject_obs_indices: np.ndarray,
                     true_classes: np.ndarray,
                     policy_X: dict[str, np.ndarray],
                     fitted: dict[str, tuple[StandardScaler, LogisticRegression]],
                     n_steps: int,
                     rng: np.random.Generator,
                     ) -> dict[str, list[dict]]:
    """Run all 6 policies on one held-out subject.

    Returns {policy_name: [step_record, ...]}.
    """
    # Choose the trial sequence ONCE so all policies see the same observations,
    # making the comparison apples-to-apples.
    if subject_obs_indices.size == 0:
        return {}
    seq = rng.choice(subject_obs_indices, size=n_steps, replace=True)
    seq_classes = true_classes[seq]
    modal_class = int(np.bincount(seq_classes, minlength=3).argmax())

    policy_names = list(FIXED_POLICIES) + ["random", "greedy_eig"]
    records: dict[str, list[dict]] = {p: [] for p in policy_names}
    alphas: dict[str, np.ndarray] = {p: np.ones(3) for p in policy_names}

    H_uniform = float(np.log(3))

    for t in range(n_steps):
        obs_idx = int(seq[t])

        # Precompute predicted-prob vectors for all four feature streams.
        prob_per_policy = {}
        for name in FIXED_POLICIES:
            sc, clf = fitted[name]
            prob_per_policy[name] = predict_proba_aligned(sc, clf, policy_X[name][obs_idx])

        # Action selection per meta-policy:
        chosen_for_random = rng.choice(FIXED_POLICIES)
        eigs_under_alpha = {
            name: _expected_info_gain(alphas["greedy_eig"], prob_per_policy[name])
            for name in FIXED_POLICIES
        }
        chosen_for_greedy = max(eigs_under_alpha, key=eigs_under_alpha.get)

        for meta in policy_names:
            if meta in FIXED_POLICIES:
                a_t = meta
            elif meta == "random":
                a_t = chosen_for_random
            else:  # greedy_eig
                a_t = chosen_for_greedy

            p_obs = prob_per_policy[a_t]
            entropy_before = _dirichlet_pred_entropy(alphas[meta])
            alphas[meta] = alphas[meta] + p_obs
            entropy_after = _dirichlet_pred_entropy(alphas[meta])
            pred = alphas[meta] / alphas[meta].sum()
            decision = int(np.argmax(pred))

            records[meta].append({
                "step": int(t + 1),
                "action": a_t,
                "true_class": int(seq_classes[t]),
                "modal_class": modal_class,
                "posterior_entropy": entropy_after,
                "info_gain_step": entropy_before - entropy_after,
                "info_gain_cumulative": H_uniform - entropy_after,
                "decision_argmax": decision,
                "decision_correct": int(decision == modal_class),
                "alpha_neg": float(alphas[meta][0]),
                "alpha_neu": float(alphas[meta][1]),
                "alpha_pos": float(alphas[meta][2]),
            })

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=10,
                        help="Steps per subject (default 10).")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    args = parser.parse_args()

    print("=" * 72)
    print("Tier-1.3: embodied closed-loop active-inference demo")
    print("=" * 72)
    t0 = time.time()

    ch5, ch67, _ = load_inputs()
    y = np.asarray(ch5["y"]).astype(int)
    subjects = np.asarray(ch5["subjects"])
    subj_hash = hash_subject_array(subjects)
    blocks = build_feature_blocks(ch5, ch67)
    policy_X = build_policy_features(blocks)
    print(f"  policies: {list(policy_X.keys())}  "
          f"+ random + greedy_eig  (n_steps={args.n_steps})")
    for name, X in policy_X.items():
        print(f"    {name:>10s}  shape={X.shape}")

    cv = GroupKFold(n_splits=args.n_folds)
    all_records: list[dict] = []

    rng_master = np.random.default_rng(args.seed)
    for fold_idx, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups=subjects)):
        t_fold = time.time()
        fitted = fit_policy_classifiers(policy_X, y[tr], tr)
        held_out_subjects = np.unique(subjects[te])
        for sid in held_out_subjects:
            # Local rng seeded from master for reproducibility.
            sub_seed = int(rng_master.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(sub_seed)
            sub_obs = np.where((subjects == sid) & np.isin(np.arange(len(y)), te))[0]
            if sub_obs.size == 0:
                continue
            recs_per_policy = simulate_subject(
                sub_obs, y, policy_X, fitted,
                n_steps=args.n_steps, rng=sub_rng,
            )
            sh = subj_hash[sub_obs[0]]
            for policy, recs in recs_per_policy.items():
                for r in recs:
                    r2 = dict(r)
                    r2["policy"] = policy
                    r2["subject_hash"] = sh
                    r2["fold"] = int(fold_idx)
                    all_records.append(r2)
        print(f"  fold {fold_idx + 1}/{args.n_folds}: "
              f"{len(held_out_subjects)} subjects  ({time.time() - t_fold:.1f}s)")

    df = pd.DataFrame(all_records)
    cols_order = ["policy", "subject_hash", "fold", "step",
                  "action", "true_class", "modal_class",
                  "posterior_entropy", "info_gain_step", "info_gain_cumulative",
                  "decision_argmax", "decision_correct",
                  "alpha_neg", "alpha_neu", "alpha_pos"]
    df = df[cols_order]
    res_path = OUT_DIR / "embodied_loop_results.csv"
    df.to_csv(res_path, index=False)
    print(f"  Wrote {res_path}  ({len(df)} rows)")

    # ── per-policy aggregate summary ─────────────────────────────────────
    summary_rows = []
    for policy, sub in df.groupby("policy"):
        last_step = sub[sub["step"] == args.n_steps]
        rng_sum = np.random.default_rng(args.seed + 1)
        # bootstrap CI of mean final entropy
        vals = last_step["posterior_entropy"].values
        boot = rng_sum.choice(vals, size=(2000, vals.size), replace=True).mean(axis=1)
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        summary_rows.append({
            "policy": policy,
            "n_subjects": int(last_step["subject_hash"].nunique()),
            "final_entropy_mean": float(last_step["posterior_entropy"].mean()),
            "final_entropy_sd": float(last_step["posterior_entropy"].std(ddof=1)),
            "final_entropy_ci95_lo": ci_lo,
            "final_entropy_ci95_hi": ci_hi,
            "final_decision_accuracy": float(last_step["decision_correct"].mean()),
            "mean_info_gain_per_step": float(sub["info_gain_step"].mean()),
            "mean_cumulative_info_gain_at_T": float(last_step["info_gain_cumulative"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("final_entropy_mean")
    summary_path = OUT_DIR / "embodied_loop_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Wrote {summary_path}")
    for _, r in summary_df.iterrows():
        print(f"    {r['policy']:>11s}  H_final={r['final_entropy_mean']:.4f}  "
              f"acc={r['final_decision_accuracy']:.3f}  "
              f"info_gain_per_step={r['mean_info_gain_per_step']:.4f}")

    # ── figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))

    # Panel A: per-step posterior entropy curves with bootstrap CI shading
    ax = axes[0]
    H_uniform = float(np.log(3))
    for policy, color in POLICY_COLORS.items():
        sub = df[df["policy"] == policy]
        if sub.empty:
            continue
        steps = np.arange(1, args.n_steps + 1)
        means = sub.groupby("step")["posterior_entropy"].mean().reindex(steps).values
        sds = sub.groupby("step")["posterior_entropy"].std(ddof=1).reindex(steps).values
        ax.plot(steps, means, color=color, label=policy, lw=1.6)
        ax.fill_between(steps, means - sds, means + sds, color=color, alpha=0.13)
    ax.axhline(H_uniform, color="black", ls=":", lw=0.8, label="uniform $H=\\log 3$")
    ax.set_xlabel("step")
    ax.set_ylabel("posterior predictive entropy (nats)")
    ax.set_title("(a) entropy reduction over time")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Panel B: cumulative info gain
    ax = axes[1]
    for policy, color in POLICY_COLORS.items():
        sub = df[df["policy"] == policy]
        if sub.empty:
            continue
        steps = np.arange(1, args.n_steps + 1)
        means = sub.groupby("step")["info_gain_cumulative"].mean().reindex(steps).values
        ax.plot(steps, means, color=color, label=policy, lw=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative info gain (nats)")
    ax.set_title("(b) cumulative info gain")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Panel C: final-decision accuracy bar
    ax = axes[2]
    sumdf = summary_df.copy()
    xs = np.arange(len(sumdf))
    bar_colors = [POLICY_COLORS[p] for p in sumdf["policy"]]
    ax.bar(xs, sumdf["final_decision_accuracy"], color=bar_colors,
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(sumdf["policy"], rotation=20)
    ax.axhline(1 / 3, color="red", ls="--", lw=0.8, label="chance (1/3)")
    ax.set_ylabel("final-decision accuracy (modal class)")
    ax.set_ylim(0, 1)
    ax.set_title("(c) final accuracy after $T$ steps")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel D: greedy_eig action-distribution
    ax = axes[3]
    greedy = df[df["policy"] == "greedy_eig"]
    if not greedy.empty:
        action_counts = (greedy.groupby(["step", "action"]).size()
                          .unstack(fill_value=0)
                          .reindex(columns=list(FIXED_POLICIES), fill_value=0))
        steps = action_counts.index.values
        bottom = np.zeros(len(steps))
        for col in FIXED_POLICIES:
            ax.bar(steps, action_counts[col], bottom=bottom,
                   label=col, color=POLICY_COLORS[col],
                   edgecolor="black", linewidth=0.3)
            bottom += action_counts[col].values
        ax.set_xlabel("step")
        ax.set_ylabel("action count under greedy_eig")
        ax.set_title("(d) greedy active-selection mix")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = FIG_DIR / "fig_embodied_loop.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_path}")

    # ── diagnostics ─────────────────────────────────────────────────────
    files_for_hash = [cfg.CH5_FILE, cfg.CH67_FILE]
    h = hashlib.sha256()
    for f in files_for_hash:
        h.update(str(f).encode())
        h.update(str(f.stat().st_size).encode())
    diagnostics = {
        "script": "experiments/tcds_hardening/run_embodied_loop.py",
        "n_subjects_total": int(np.unique(subjects).size),
        "n_observations": int(len(y)),
        "n_folds": int(args.n_folds),
        "n_steps": int(args.n_steps),
        "policies": list(POLICY_COLORS.keys()),
        "policy_feature_dims": {name: int(X.shape[1]) for name, X in policy_X.items()},
        "summary_table": summary_df.to_dict(orient="records"),
        "input_file_size_hash_sha256": h.hexdigest()[:32],
        "runtime_seconds": float(time.time() - t0),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
    }
    diag_path = OUT_DIR / "embodied_loop_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  Wrote {diag_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
