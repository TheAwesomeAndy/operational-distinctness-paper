"""Generate novel manuscript figures for the operational-distinctness paper.

The source-data CSVs are the corrected reporting outputs from PR head
bcafc50ca544da35ee215f030849a34bfb395a4c. These figures are intentionally
not copied from the dissertation; they are generated specifically for this
paper from the corrected reporting pipeline outputs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_public"
OUT = ROOT / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)

# Load data
aff = pd.read_csv(DATA / "affective_ablation_corrected.csv")
clin = pd.read_csv(DATA / "clinical_sensitivity_corrected.csv")
red = pd.read_csv(DATA / "layer_redundancy_corrected.csv")

# Fig 3: mathematical operational-distinctness framework
fig, ax = plt.subplots(figsize=(10.8, 3.8))
ax.axis('off')
items = [
    ("Layer\n$L_i$", "$E, D, T, C$", 0.04),
    ("Predictive\nsufficiency", "$\\mathcal{M}(L_i,Y)$", 0.26),
    ("Additive\nutility", "$\\Gamma_i=\\mathcal{M}([L_b\\|L_i],Y)-\\mathcal{M}(L_b,Y)$", 0.48),
    ("Target-specific\nsensitivity", "$\\mathcal{M}(L_i,Z_k)$", 0.70),
    ("Redundancy\nconstraint", "$\\mathrm{CKA}(L_i,L_j)$", 0.86),
]
for title, eq, x in items:
    w = 0.15 if x < 0.80 else 0.12
    h = 0.38
    patch = FancyBboxPatch((x, 0.40), w, h, boxstyle='round,pad=0.018,rounding_size=0.02',
                           linewidth=1.2, facecolor='white', edgecolor='black')
    ax.add_patch(patch)
    ax.text(x+w/2, 0.63, title, ha='center', va='center', fontsize=10)
    ax.text(x+w/2, 0.47, eq, ha='center', va='center', fontsize=9)
for x1, x2 in [(0.19,0.26),(0.41,0.48),(0.63,0.70),(0.85,0.86)]:
    ax.add_patch(FancyArrowPatch((x1,0.59),(x2,0.59),arrowstyle='->',mutation_scale=12,linewidth=1.1))
ax.text(0.50, 0.17, "Operational distinctness is supported when layers differ in target utility or remain non-equivalent under redundancy analysis.", ha='center', fontsize=10)
save(fig, "fig03_operational_distinctness_formalism.pdf")

# Fig 4: Affective ablation, two panel BA and AUC
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
labels = [f"{r.config_id}\n{r.feature_set}" for _, r in aff.iterrows()]
x = np.arange(len(aff))
yerr = np.vstack([aff['balanced_accuracy']-aff['ci_low'], aff['ci_high']-aff['balanced_accuracy']])
axes[0].bar(x, aff['balanced_accuracy'], yerr=yerr, capsize=3, linewidth=0.7, edgecolor='black')
axes[0].axhline(1/3, linestyle='--', linewidth=1.0, color='black')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=7)
axes[0].set_ylabel('Balanced accuracy')
axes[0].set_ylim(0.30, 0.55)
axes[0].set_title('Affective classification: balanced accuracy')
for i,v in enumerate(aff['balanced_accuracy']):
    axes[0].text(i, v+0.012, f'{v:.3f}', ha='center', fontsize=7)
axes[1].bar(x, aff['macro_roc_auc'], linewidth=0.7, edgecolor='black')
axes[1].axhline(0.5, linestyle='--', linewidth=1.0, color='black')
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7)
axes[1].set_ylabel('Macro ROC-AUC')
axes[1].set_ylim(0.48, 0.72)
axes[1].set_title('Affective classification: macro OVR ROC-AUC')
for i,v in enumerate(aff['macro_roc_auc']):
    axes[1].text(i, v+0.008, f'{v:.3f}', ha='center', fontsize=7)
save(fig, "fig04_affective_ablation_dual_metric.pdf")

# Fig 5: incremental and target-dependent utility
base_e = float(aff.loc[aff.config_id=='A1','balanced_accuracy'].iloc[0])
base_bp = float(aff.loc[aff.config_id=='A0','balanced_accuracy'].iloc[0])
fig, ax = plt.subplots(figsize=(8.8, 3.8))
sel = aff[aff['config_id'].isin(['A1','A6','A7','A8','A9'])].copy()
sel['delta_vs_E'] = sel['balanced_accuracy'] - base_e
xx = np.arange(len(sel))
ax.bar(xx, sel['delta_vs_E'], edgecolor='black', linewidth=0.7)
ax.axhline(0, color='black', linewidth=1.0)
ax.set_xticks(xx); ax.set_xticklabels([f"{r.config_id}\n{r.feature_set}" for _,r in sel.iterrows()], fontsize=8)
ax.set_ylabel('Balanced-accuracy gain relative to E')
ax.set_title('Embedding-centered additive utility in the affective task')
for i,v in enumerate(sel['delta_vs_E']):
    ax.text(i, v + (0.002 if v>=0 else -0.004), f'{v:+.3f}', ha='center', va='bottom' if v>=0 else 'top', fontsize=8)
save(fig, "fig05_embedding_additive_utility.pdf")

# Fig 6: clinical heatmap with best-cell markers
fig, ax = plt.subplots(figsize=(8.6, 4.2))
piv = clin.pivot(index='diagnosis', columns='config_id', values='balanced_accuracy').loc[['SUD','MDD','PTSD','GAD','ADHD'], ['C1','C2','C3','C4','C5','C6']]
im = ax.imshow(piv.values, aspect='auto', vmin=0.40, vmax=0.58)
ax.set_xticks(np.arange(piv.shape[1])); ax.set_xticklabels([f"{c}\n" + clin[clin.config_id==c].feature_set.iloc[0] for c in piv.columns], fontsize=8)
ax.set_yticks(np.arange(piv.shape[0])); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    j_best = int(np.nanargmax(piv.values[i]))
    for j in range(piv.shape[1]):
        text = f'{piv.values[i,j]:.3f}'
        if j == j_best:
            text = '★\n' + text
        ax.text(j, i, text, ha='center', va='center', fontsize=8)
ax.set_title('Exploratory clinical-label sensitivity by layer configuration')
fig.colorbar(im, ax=ax, label='Balanced accuracy')
save(fig, "fig06_clinical_sensitivity_heatmap.pdf")

# Fig 7: best clinical layers BA and AUC
best = clin.loc[clin.groupby('diagnosis')['balanced_accuracy'].idxmax()].set_index('diagnosis').loc[['SUD','MDD','PTSD','GAD','ADHD']].reset_index()
fig, ax = plt.subplots(figsize=(8.8, 3.8))
xx = np.arange(len(best)); width=0.36
ax.bar(xx-width/2, best['balanced_accuracy'], width, label='Balanced accuracy', edgecolor='black', linewidth=0.7)
ax.bar(xx+width/2, best['roc_auc'], width, label='ROC-AUC', edgecolor='black', linewidth=0.7)
ax.axhline(0.5, color='black', linestyle='--', linewidth=1.0)
ax.set_xticks(xx); ax.set_xticklabels([f"{d}\n{c}: {f}" for d,c,f in zip(best.diagnosis,best.config_id,best.feature_set)], fontsize=8)
ax.set_ylim(0.48,0.62)
ax.set_ylabel('Metric value')
ax.set_title('Best layer configuration by clinical label')
ax.legend(frameon=False, fontsize=8)
for i,(ba,auc) in enumerate(zip(best['balanced_accuracy'],best['roc_auc'])):
    ax.text(i-width/2, ba+0.004, f'{ba:.3f}', ha='center', fontsize=7)
    ax.text(i+width/2, auc+0.004, f'{auc:.3f}', ha='center', fontsize=7)
save(fig, "fig07_clinical_best_layer_dual_metric.pdf")

# Fig 8: CKA matrix among layers
layers = ['E','D','T','C']
cka = np.eye(4)
for _,r in red.iterrows():
    if r['source'] in layers and r['target'] in layers:
        i=layers.index(r['source']); j=layers.index(r['target']); cka[i,j]=float(r['linear_cka'])
fig, ax = plt.subplots(figsize=(5.2,4.4))
im=ax.imshow(cka, vmin=0, vmax=1)
ax.set_xticks(np.arange(len(layers))); ax.set_xticklabels(layers)
ax.set_yticks(np.arange(len(layers))); ax.set_yticklabels(layers)
for i in range(4):
    for j in range(4):
        ax.text(j,i,f'{cka[i,j]:.3f}',ha='center',va='center',fontsize=8)
ax.set_title('Layer redundancy matrix (linear CKA)')
fig.colorbar(im, ax=ax, label='CKA')
save(fig, "fig08_layer_redundancy_cka.pdf")

print(f"Generated figures in {OUT}")
