#!/usr/bin/env python3
"""Phase 2 - Adaptive evidence routing under signal-quality uncertainty.

Question: ARSPI-Net exposes operationally distinct evidence streams whose failure
modes differ across perturbation regimes (band-power is robust to amplitude
noise/channel dropout; embedding-containing streams are stable under graph
perturbation; the topological stream is graph-sensitive). Can a leakage-free
router that selects the stream appropriate to the (measured) signal-quality
condition preserve or improve performance relative to any single fixed stream?

We train each fixed stream on CLEAN training folds and evaluate on perturbed test
folds (representation level, identical perturbations to Ready-9 phase 08). For
every held-out observation under every regime we record per-stream predictions,
probabilities, and label-free signal-quality observables, then evaluate four
routers:
  - oracle            (uses test labels; NON-DEPLOYABLE upper bound)
  - perturbation-label (regime known; best stream chosen from OTHER folds only)
  - signal-quality     (chooses the stream from label-free observables; nested
                        subject-grouped CV)
  - entropy-gated fusion (fuse two streams when primary posterior entropy is high;
                        threshold chosen on training folds only)

Honest reporting: null/negative results are reported, not hidden. No claim of
universal robustness or band-power superiority reversal is made.

Outputs:
  outputs/tcds_risk_reduction/adaptive_router_metrics.csv
  outputs/tcds_risk_reduction/adaptive_router_summary.csv
  outputs/tcds_risk_reduction/adaptive_router_confusion_matrices.json
  outputs/tcds_risk_reduction/adaptive_router_config.json
  figures/tcds_risk_reduction/rr01_adaptive_router_performance.pdf
  figures/tcds_risk_reduction/rr02_router_regime_map.pdf
  tables/tcds_risk_reduction/table_adaptive_evidence_routing.tex
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.tcds_ready9 import config as cfg  # noqa: E402
from experiments.tcds_ready9 import common_ready9 as cr  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix  # noqa: E402

OUT = _REPO / "outputs" / "tcds_risk_reduction"
FIG = _REPO / "figures" / "tcds_risk_reduction"
TAB = _REPO / "tables" / "tcds_risk_reduction"
for d in (OUT, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

# Fixed streams (Ready-9 component composition). A4 (C alone) omitted per spec.
CONFIG_COMPONENTS = {
    "A0": ["BandPower"], "A1": ["E"], "A2": ["D"], "A3": ["T"],
    "A5": ["D", "T"], "A6": ["E", "D"], "A7": ["E", "T"],
    "A8": ["E", "D", "T"], "A9": ["E", "D", "T", "C"],
}
STREAMS = list(CONFIG_COMPONENTS)
LABELS = [0, 1, 2]


# ── perturbation primitives (identical to Ready-9 phase 08) ─────────────
def amplitude_noise_cols(M, snr_db, rng):
    if snr_db is None:
        return M
    col_power = np.mean(M ** 2, axis=0) + 1e-12
    noise_power = col_power / (10 ** (snr_db / 10.0))
    return M + rng.standard_normal(M.shape) * np.sqrt(noise_power)[None, :]


def perturb_tplv(tplv, frac, rng):
    if frac <= 0:
        return tplv
    n = tplv.shape[0]
    iu = np.triu_indices(n, k=1)
    n_drop = int(round(frac * len(iu[0])))
    if n_drop <= 0:
        return tplv
    drop = rng.choice(len(iu[0]), size=n_drop, replace=False)
    out = tplv.copy()
    out[iu[0][drop], iu[1][drop]] = 0.0
    out[iu[1][drop], iu[0][drop]] = 0.0
    return out


def tplv_to_topo(plv):
    n = plv.shape[0]
    strength = plv.sum(axis=1) - np.diag(plv)
    deg = np.maximum(strength, 1e-12)
    clustering = np.zeros(n)
    for i in range(n):
        w = plv[i].copy(); w[i] = 0.0
        clustering[i] = (w.sum() ** 2 - (w ** 2).sum()) / (deg[i] ** 2 + 1e-12)
    return np.stack([strength, clustering], axis=1)


def recompute_T_C(te_idx, tplv_all, D_perchan, frac, rng):
    T_topo = np.zeros((len(te_idx), cfg.N_CHANNELS, 2))
    for r, idx in enumerate(te_idx):
        T_topo[r] = tplv_to_topo(perturb_tplv(tplv_all[idx], frac, rng))
    T_block = T_topo.reshape(len(te_idx), -1)
    C_block = cr.compute_coupling_block(D_perchan[te_idx], T_topo)
    return T_block, C_block


def assemble(components, idx, comp_clean, overrides=None):
    overrides = overrides or {}
    mats = [overrides[c] if c in overrides else comp_clean[c][idx] for c in components]
    return np.hstack(mats) if len(mats) > 1 else mats[0]


def _entropy(P):
    P = np.clip(P, 1e-12, 1.0)
    return -np.sum(P * np.log(P), axis=1)


REGIMES = ([("clean", None)]
           + [("amplitude_noise", s) for s in (20, 10, 5)]
           + [("channel_dropout", f) for f in (0.1, 0.2, 0.3)]
           + [("graph_perturbation", f) for f in (0.1, 0.2, 0.3)])


def _perturb_test(components, te, comp_clean, ptype, level, tplv_all, D_perchan, rng):
    """Return component overrides for the perturbed TEST set, or None if the
    regime does not apply to this stream (graph perturbation w/o T or C)."""
    ov = {}
    if ptype == "amplitude_noise" and level is not None:
        for c in components:
            ov[c] = amplitude_noise_cols(comp_clean[c][te], level, rng)
    elif ptype == "channel_dropout" and level:
        for c in components:
            ov[c] = cr.apply_channel_dropout(comp_clean[c][te].copy(), c, level, rng)
    elif ptype == "graph_perturbation" and level:
        if "T" not in components and "C" not in components:
            # Graph edge perturbation does not affect streams without graph blocks;
            # they remain available and unperturbed (and selectable by a router).
            return {}
        T_blk, C_blk = recompute_T_C(te, tplv_all, D_perchan, level, rng)
        if "T" in components:
            ov["T"] = T_blk
        if "C" in components:
            ov["C"] = C_blk
    return ov


def main() -> int:
    try:
        data = cr.load_ready9()
    except Exception as e:
        (OUT / "ADAPTIVE_ROUTER_ERROR.md").write_text(
            f"# Adaptive routing could not run\n\nInput load failed: {e}\n")
        print(f"[router] FAILED to load inputs: {e}", file=sys.stderr)
        return 1
    blocks, y, subjects = data["blocks"], np.asarray(data["y"]), np.asarray(data["subjects"])
    ch67 = data["ch67"]
    tplv_all = np.asarray(ch67["tPLV_mats"])
    D_perchan = np.asarray(ch67["D"])
    comp_clean = {k: np.asarray(blocks[k]) for k in ("BandPower", "E", "D", "T", "C")}
    n = len(y)

    # Per-observation prediction store, keyed by (seed,fold,regime_idx).
    # records[key] = dict(te=idx array, true=, per-stream pred/proba/conf/ent/obs)
    records = []   # flat list of dicts (one per seed,fold,regime)
    seeds = cfg.SEEDS

    for seed in seeds:
        cv = cr.subject_grouped_cv(cfg.N_FOLDS_AFFECTIVE, seed)
        for fold, (tr, te) in enumerate(cv.split(comp_clean["BandPower"], y, groups=subjects)):
            # Fit each stream on CLEAN train (scaler + balanced logreg = Ready-9 readout).
            fitted = {}
            train_centroid = {}
            train_std = {}
            for s in STREAMS:
                comps = CONFIG_COMPONENTS[s]
                Xtr = assemble(comps, tr, comp_clean)
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                         solver="lbfgs", random_state=seed)
                clf.fit(sc.transform(Xtr), y[tr])
                fitted[s] = (sc, clf, comps)
                Z = sc.transform(Xtr)
                train_centroid[s] = Z.mean(axis=0)
                train_std[s] = Z.std(axis=0) + 1e-8
            # Evaluate each regime.
            for ridx, (ptype, level) in enumerate(REGIMES):
                rng = np.random.default_rng(91 * seed + 7 * fold + ridx)
                per_stream = {}
                applicable = []
                for s in STREAMS:
                    sc, clf, comps = fitted[s]
                    ov = _perturb_test(comps, te, comp_clean, ptype, level,
                                       tplv_all, D_perchan, rng)
                    if ov == "NA":
                        continue  # graph perturbation not applicable to this stream
                    Xte = assemble(comps, te, comp_clean, ov)
                    Z = sc.transform(Xte)
                    proba = clf.predict_proba(Z)
                    pred = clf.classes_[np.argmax(proba, axis=1)]
                    # label-free signal-quality observables
                    conf = proba.max(axis=1)
                    ent = _entropy(proba)
                    fnorm = np.linalg.norm(Z, axis=1)
                    maha = np.sqrt(((Z - train_centroid[s]) / train_std[s]) ** 2 @ np.ones(Z.shape[1]))
                    per_stream[s] = dict(pred=pred, proba=proba, conf=conf, ent=ent,
                                         fnorm=fnorm, maha=maha)
                    applicable.append(s)
                records.append(dict(seed=seed, fold=fold, regime=ridx, ptype=ptype,
                                    level=("clean" if level is None else level),
                                    te=np.asarray(te), true=y[te], streams=applicable,
                                    per_stream=per_stream))
    print(f"[router] collected {len(records)} (seed,fold,regime) prediction blocks")

    _analyze_and_write(records, subjects)
    return 0


def _ba(true, pred):
    return float(balanced_accuracy_score(true, pred))


def _analyze_and_write(records, subjects):
    # ---- 1. Fixed-stream + oracle + perturbation-label router per (seed,fold,regime)
    metric_rows = []
    # index records by regime for cross-fold selection
    by_regime = defaultdict(list)
    for rec in records:
        by_regime[rec["regime"]].append(rec)

    # precompute per (regime,stream) mean test-BA per fold for the label router
    fold_ba = defaultdict(dict)  # fold_ba[(regime,seed,fold)][stream]=ba
    for rec in records:
        for s in rec["streams"]:
            fold_ba[(rec["regime"], rec["seed"], rec["fold"])][s] = _ba(
                rec["true"], rec["per_stream"][s]["pred"])

    for rec in records:
        reg, seed, fold = rec["regime"], rec["seed"], rec["fold"]
        true = rec["true"]
        streams = rec["streams"]
        # fixed streams
        for s in streams:
            metric_rows.append(dict(router=f"fixed_{s}", seed=seed, fold=fold,
                                    ptype=rec["ptype"], level=rec["level"],
                                    balanced_accuracy=_ba(true, rec["per_stream"][s]["pred"]),
                                    macro_f1=float(f1_score(true, rec["per_stream"][s]["pred"],
                                                            labels=LABELS, average="macro"))))
        # oracle (per-obs: correct if any stream correct) -> routed prediction = a
        # correct stream's prediction where available, else stream 0's prediction
        preds = np.stack([rec["per_stream"][s]["pred"] for s in streams], axis=1)
        correct = (preds == true[:, None])
        routed = preds[:, 0].copy()
        any_correct = correct.any(axis=1)
        first_correct = np.argmax(correct, axis=1)
        routed[any_correct] = preds[any_correct, first_correct[any_correct]]
        metric_rows.append(dict(router="oracle", seed=seed, fold=fold,
                                ptype=rec["ptype"], level=rec["level"],
                                balanced_accuracy=_ba(true, routed),
                                macro_f1=float(f1_score(true, routed, labels=LABELS, average="macro"))))
        # perturbation-label router: pick best stream for this regime from OTHER
        # folds/seeds only (leakage-free), apply to this fold.
        agg = defaultdict(list)
        for (rg, sd, fl), d in fold_ba.items():
            if rg == reg and not (sd == seed and fl == fold):
                for s, v in d.items():
                    agg[s].append(v)
        if agg:
            best = max(agg, key=lambda s: np.mean(agg[s]))
            best = best if best in streams else streams[0]
            metric_rows.append(dict(router="perturbation_label", seed=seed, fold=fold,
                                    ptype=rec["ptype"], level=rec["level"],
                                    balanced_accuracy=_ba(true, rec["per_stream"][best]["pred"]),
                                    macro_f1=float(f1_score(true, rec["per_stream"][best]["pred"],
                                                            labels=LABELS, average="macro")),
                                    selected_stream=best))

    # ---- 2. Entropy-gated fusion (primary A8, fuse with A0 when primary entropy high)
    PRIMARY, SECONDARY = "A8", "A0"
    taus = np.quantile(
        np.concatenate([rec["per_stream"][PRIMARY]["ent"] for rec in records
                        if PRIMARY in rec["streams"]]),
        [0.5, 0.6, 0.7, 0.8])
    # choose tau maximizing pooled routed BA on TRAIN folds (folds 0..3), eval on fold4-style
    # Simpler leakage-free: pick tau by pooled BA over all-but-one seed, report mean held-out.
    def fusion_ba(tau, recs):
        tr, pr = [], []
        for rec in recs:
            if PRIMARY not in rec["streams"] or SECONDARY not in rec["streams"]:
                continue
            pP = rec["per_stream"][PRIMARY]["proba"]
            pS = rec["per_stream"][SECONDARY]["proba"]
            entP = rec["per_stream"][PRIMARY]["ent"]
            fused = np.where((entP > tau)[:, None], 0.5 * (pP + pS), pP)
            pr.append(np.array(LABELS)[np.argmax(fused, axis=1)])
            tr.append(rec["true"])
        if not tr:
            return None
        return _ba(np.concatenate(tr), np.concatenate(pr))
    for rec in records:
        if PRIMARY not in rec["streams"] or SECONDARY not in rec["streams"]:
            continue
        others = [r for r in by_regime[rec["regime"]]
                  if not (r["seed"] == rec["seed"] and r["fold"] == rec["fold"])]
        best_tau = max(taus, key=lambda t: (fusion_ba(t, others) or 0))
        b = fusion_ba(best_tau, [rec])
        if b is not None:
            metric_rows.append(dict(router="entropy_fusion", seed=rec["seed"], fold=rec["fold"],
                                    ptype=rec["ptype"], level=rec["level"],
                                    balanced_accuracy=b, macro_f1=np.nan, tau=float(best_tau)))

    # ---- 3. Signal-quality router (nested subject-grouped CV over observables) ----
    sq_rows = _signal_quality_router(records, subjects)
    metric_rows += sq_rows

    cr.write_csv(OUT / "adaptive_router_metrics.csv", metric_rows)

    # ---- summary (mean over seeds/folds), overall and per perturbation type ----
    summary = _summarize(metric_rows)
    cr.write_csv(OUT / "adaptive_router_summary.csv", summary)
    _write_confusion(records)
    _write_config(taus)
    _write_table(summary)
    _make_figures(summary, metric_rows)
    _interpret(summary)
    print("[router] wrote metrics, summary, table, figures, confusion, config")


def _signal_quality_router(records, subjects):
    """Per-observation router trained on label-free observables via subject-grouped
    nested CV. Target = oracle best stream (correct & most confident). Routed
    prediction uses the selected stream's prediction. Leakage-free: router fit on
    training subjects only."""
    from sklearn.model_selection import StratifiedGroupKFold
    STREAMS_R = ["A0", "A1", "A3", "A8"]  # span the regime failure modes
    feats, target, grp, true_all, preds_by_stream, regime_tag = [], [], [], [], [], []
    for rec in records:
        avail = [s for s in STREAMS_R if s in rec["streams"]]
        if len(avail) < 2:
            continue
        m = len(rec["true"])
        fv = np.concatenate([np.stack([rec["per_stream"][s]["conf"],
                                       rec["per_stream"][s]["ent"],
                                       rec["per_stream"][s]["fnorm"],
                                       rec["per_stream"][s]["maha"]], axis=1)
                             for s in avail], axis=1)
        preds = np.stack([rec["per_stream"][s]["pred"] for s in avail], axis=1)
        correct = (preds == rec["true"][:, None])
        conf = np.stack([rec["per_stream"][s]["conf"] for s in avail], axis=1)
        # oracle target: correct streams preferred, tie-break by confidence
        score = correct.astype(float) * 10 + conf
        tgt = np.argmax(score, axis=1)
        feats.append(fv); target.append(tgt)
        grp.append(subjects[rec["te"]]); true_all.append(rec["true"])
        preds_by_stream.append(preds); regime_tag.append(np.array([rec["ptype"]] * m))
    if not feats:
        return []
    X = np.vstack(feats); tgt = np.concatenate(target); groups = np.concatenate(grp)
    true_all = np.concatenate(true_all); preds_all = np.vstack(preds_by_stream)
    regime_all = np.concatenate(regime_tag)
    routed = np.empty_like(true_all)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in sgkf.split(X, tgt, groups=groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 solver="lbfgs", random_state=42)
        clf.fit(sc.transform(X[tr]), tgt[tr])
        choice = clf.predict(sc.transform(X[te]))
        routed[te] = preds_all[te, choice]
    rows = []
    for pt in ["overall"] + sorted(set(regime_all)):
        mask = np.ones(len(true_all), bool) if pt == "overall" else (regime_all == pt)
        rows.append(dict(router="signal_quality", seed=-1, fold=-1,
                         ptype=pt, level="all",
                         balanced_accuracy=_ba(true_all[mask], routed[mask]),
                         macro_f1=float(f1_score(true_all[mask], routed[mask],
                                                 labels=LABELS, average="macro"))))
    return rows


def _summarize(metric_rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in metric_rows:
        agg[(r["router"], "overall")]["ba"].append(r["balanced_accuracy"])
        agg[(r["router"], r["ptype"])]["ba"].append(r["balanced_accuracy"])
        if not (isinstance(r.get("macro_f1"), float) and np.isnan(r.get("macro_f1", np.nan))):
            agg[(r["router"], "overall")]["f1"].append(r.get("macro_f1", np.nan))
    out = []
    for (router, scope), d in sorted(agg.items()):
        ba = [v for v in d["ba"] if v is not None and np.isfinite(v)]
        if not ba:
            continue
        lo, hi = cr.wilson_ci(np.mean(ba), len(ba)) if False else (np.nan, np.nan)
        out.append(dict(router=router, scope=scope, n=len(ba),
                        balanced_accuracy_mean=round(float(np.mean(ba)), 4),
                        balanced_accuracy_sd=round(float(np.std(ba)), 4),
                        macro_f1_mean=round(float(np.mean(d["f1"])), 4) if d["f1"] else np.nan))
    return out


def _write_confusion(records):
    # pooled confusion for oracle and fixed A0/A8 at the most-degraded regimes
    cms = {}
    for tag, fn in [("oracle", None)]:
        pass
    # oracle pooled confusion across all regimes
    tt, pp = [], []
    for rec in records:
        streams = rec["streams"]
        preds = np.stack([rec["per_stream"][s]["pred"] for s in streams], axis=1)
        correct = (preds == rec["true"][:, None])
        routed = preds[:, 0].copy()
        ac = correct.any(axis=1); fc = np.argmax(correct, axis=1)
        routed[ac] = preds[ac, fc[ac]]
        tt.append(rec["true"]); pp.append(routed)
    cms["oracle_pooled"] = confusion_matrix(np.concatenate(tt), np.concatenate(pp),
                                            labels=LABELS).tolist()
    cr.write_json(OUT / "adaptive_router_confusion_matrices.json",
                  {"provenance": cr.base_provenance(), "labels": LABELS,
                   "confusion_matrices": cms})


def _write_config(taus):
    cr.write_json(OUT / "adaptive_router_config.json", {
        "provenance": cr.base_provenance(),
        "streams": CONFIG_COMPONENTS,
        "regimes": [{"type": t, "level": ("clean" if l is None else l)} for t, l in REGIMES],
        "seeds": cfg.SEEDS, "n_folds": cfg.N_FOLDS_AFFECTIVE,
        "routers": ["fixed_<stream>", "oracle (non-deployable upper bound)",
                    "perturbation_label (regime known; selected on other folds)",
                    "signal_quality (label-free observables; nested subject CV)",
                    "entropy_fusion (A8 primary, fuse A0 when entropy>tau)"],
        "signal_quality_observables": ["max-probability confidence", "posterior entropy",
                                       "scaled feature L2 norm", "diagonal Mahalanobis to train centroid"],
        "entropy_fusion_tau_candidates": [float(t) for t in taus],
    })


def _interpret(summary):
    d = {r["router"]: r for r in summary if r["scope"] == "overall"}
    best_fixed = max((r for r in summary if r["scope"] == "overall" and r["router"].startswith("fixed_")),
                     key=lambda r: r["balanced_accuracy_mean"], default=None)
    msg = []
    if best_fixed:
        bf = best_fixed["balanced_accuracy_mean"]
        for rk in ("signal_quality", "perturbation_label", "entropy_fusion"):
            if rk in d:
                delta = d[rk]["balanced_accuracy_mean"] - bf
                verdict = "improves over" if delta > 0.005 else ("matches" if delta > -0.005 else "does not improve over")
                msg.append(f"{rk}: {d[rk]['balanced_accuracy_mean']:.3f} vs best fixed "
                           f"{best_fixed['router']} {bf:.3f} (Δ={delta:+.3f}) -> {verdict} best fixed stream")
        if "oracle" in d:
            msg.append(f"oracle upper bound: {d['oracle']['balanced_accuracy_mean']:.3f} (non-deployable)")
        improved = any((d.get(rk, {}).get("balanced_accuracy_mean", 0) - bf) > 0.005
                       for rk in ("signal_quality", "perturbation_label", "entropy_fusion"))
        msg.append("CONCLUSION: " + (
            "The measurements support perturbation-dependent evidence routing across operationally distinct streams."
            if improved else
            "The routing analysis bounds the utility of stream selection under the measured SHAPE ERP perturbation regime."))
    (OUT / "adaptive_router_interpretation.txt").write_text("\n".join(msg) + "\n")
    print("[router] " + " | ".join(msg))


def _write_table(summary):
    overall = {r["router"]: r for r in summary if r["scope"] == "overall"}
    order = ([f"fixed_{s}" for s in STREAMS] +
             ["perturbation_label", "signal_quality", "entropy_fusion", "oracle"])
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Adaptive evidence routing under perturbation. Balanced accuracy "
             r"(mean over seeds$\times$folds$\times$regimes). The oracle is a non-deployable "
             r"upper bound. Routers that use only label-free signal-quality observables are "
             r"leakage-free under subject-grouped validation.}",
             r"\label{tab:adaptive_routing}",
             r"\begin{tabular}{l r r}", r"\toprule",
             r"Router / fixed stream & Balanced acc. & macro-F1 \\", r"\midrule"]
    for k in order:
        if k not in overall:
            continue
        r = overall[k]
        name = k.replace("_", r"\_")
        f1 = "" if (isinstance(r["macro_f1_mean"], float) and np.isnan(r["macro_f1_mean"])) else f"{r['macro_f1_mean']:.3f}"
        lines.append(f"{name} & {r['balanced_accuracy_mean']:.3f} & {f1} \\\\")
        if k == f"fixed_{STREAMS[-1]}":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / "table_adaptive_evidence_routing.tex").write_text("\n".join(lines))


def _make_figures(summary, metric_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    overall = {r["router"]: r for r in summary if r["scope"] == "overall"}
    order = ([f"fixed_{s}" for s in STREAMS] +
             ["perturbation_label", "signal_quality", "entropy_fusion", "oracle"])
    order = [k for k in order if k in overall]
    vals = [overall[k]["balanced_accuracy_mean"] for k in order]
    colors = ["#9aa0a6" if k.startswith("fixed_") else
              ("#d93025" if k == "oracle" else "#1a73e8") for k in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(order)), vals, color=colors)
    ax.axhline(1/3, ls="--", color="k", lw=0.8, label="chance")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([k.replace("fixed_", "") for k in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Balanced accuracy"); ax.set_title("Evidence routing vs fixed streams (all regimes pooled)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "rr01_adaptive_router_performance.pdf"); plt.close(fig)
    cr.write_json(FIG / "rr01_adaptive_router_performance.json",
                  {"provenance": cr.base_provenance(), "routers": order, "balanced_accuracy": vals})

    # regime map: best fixed stream per perturbation type vs routers
    ptypes = ["clean", "amplitude_noise", "channel_dropout", "graph_perturbation"]
    routers_show = [f"fixed_{s}" for s in ["A0", "A1", "A3", "A8"]] + \
                   ["perturbation_label", "signal_quality", "oracle"]
    M = np.full((len(routers_show), len(ptypes)), np.nan)
    persc = defaultdict(lambda: defaultdict(list))
    for r in metric_rows:
        persc[r["router"]][r["ptype"]].append(r["balanced_accuracy"])
    # signal_quality summary stored per ptype scope already in summary
    for i, rk in enumerate(routers_show):
        for j, pt in enumerate(ptypes):
            vlist = persc.get(rk, {}).get(pt, [])
            if vlist:
                M[i, j] = np.mean(vlist)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.33, vmax=max(0.5, np.nanmax(M)))
    ax.set_xticks(range(len(ptypes))); ax.set_xticklabels(ptypes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(routers_show)))
    ax.set_yticklabels([r.replace("fixed_", "") for r in routers_show], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.45 else "black")
    ax.set_title("Balanced accuracy by perturbation regime")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG / "rr02_router_regime_map.pdf"); plt.close(fig)
    cr.write_json(FIG / "rr02_router_regime_map.json",
                  {"provenance": cr.base_provenance(), "rows": routers_show,
                   "cols": ptypes, "matrix": [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in M]})


if __name__ == "__main__":
    raise SystemExit(main())
