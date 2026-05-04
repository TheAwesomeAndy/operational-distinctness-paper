import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'figures' / 'generated'
OUT.mkdir(parents=True, exist_ok=True)

def box(ax, xy, wh, text, fs=9, fc='white', ec='black', lw=1.1):
    x,y=xy; w,h=wh
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.02',fc=fc,ec=ec,lw=lw)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,wrap=True)
    return p

def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.0,color='black'))

# Fig: evaluation protocol
fig, ax = plt.subplots(figsize=(12,4.2))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.text(0.5,0.95,'Subject-level operational-distinctness evaluation protocol',ha='center',va='center',fontsize=13,fontweight='bold')
xs=[0.04,0.22,0.40,0.58,0.76]
y=0.58; w=0.15; h=0.23
labels=[
    'Input EEG/ERP\n211 subjects\n3 conditions\n34 channels',
    'Feature families\nBandPower, E, D, T, C\nall diagnostics passed',
    'Affective task\nA0--A9\n10-fold grouped CV',
    'Clinical-label task\nC1--C6\nsubject-level averages',
    'Reporting artifacts\nmetrics, predictions,\nfigures, tables, manifest'
]
for i,x in enumerate(xs):
    box(ax,(x,y),(w,h),labels[i],fs=9,fc='#f6f6f6')
    if i>0: arrow(ax,(xs[i-1]+w,y+h/2),(x,y+h/2))
# lower row
box(ax,(0.13,0.18),(0.21,0.18),'Leakage control\nsubject/group separation;\nstandardization fit on train folds only',fs=9,fc='#ffffff')
box(ax,(0.40,0.18),(0.21,0.18),'Primary metrics\nbalanced accuracy, macro AUC,\nmacro-F1, CI summaries',fs=9,fc='#ffffff')
box(ax,(0.67,0.18),(0.21,0.18),'Claim boundary\nclinical-label sensitivity only;\nno diagnostic biomarker claim',fs=9,fc='#ffffff')
arrow(ax,(0.295,0.58),(0.24,0.36)); arrow(ax,(0.485,0.58),(0.505,0.36)); arrow(ax,(0.835,0.58),(0.775,0.36))
fig.tight_layout()
fig.savefig(OUT/'fig02_subject_level_reporting_protocol.pdf',bbox_inches='tight')
plt.close(fig)

# Fig: claim-strength ladder
fig, ax = plt.subplots(figsize=(9,4.2))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.text(0.5,0.94,'Evidence ladder used to constrain manuscript claims',ha='center',va='center',fontsize=13,fontweight='bold')
items=[
('Completed','A0--A9 and C1--C6\nsubject-safe reporting pipeline','Supported: operational layer utility'),
('Completed','Feature-block diagnostics and redundancy CKA','Supported: non-degenerate, non-identical layers'),
('Completed','Comorbidity-adjusted exploratory models','Supported: covariate-qualified sensitivity'),
('Not completed','Permutation-FDR inference','Not supported: statistically confirmed biomarkers')]
ys=[0.72,0.52,0.32,0.12]
for i,(status,method,claim) in enumerate(items):
    fc='#e8f4ea' if status=='Completed' else '#fdecea'
    box(ax,(0.04,ys[i]),(0.16,0.12),status,fs=9,fc=fc)
    box(ax,(0.26,ys[i]),(0.34,0.12),method,fs=9,fc='white')
    box(ax,(0.66,ys[i]),(0.30,0.12),claim,fs=9,fc='white')
    arrow(ax,(0.20,ys[i]+0.06),(0.26,ys[i]+0.06)); arrow(ax,(0.60,ys[i]+0.06),(0.66,ys[i]+0.06))
fig.tight_layout()
fig.savefig(OUT/'fig09_claim_strength_ladder.pdf',bbox_inches='tight')
plt.close(fig)

# Fig: paper distinction triangle
fig, ax = plt.subplots(figsize=(8.5,4.8))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.text(0.5,0.95,'Novelty boundary relative to prior ARSPI-Net manuscripts',ha='center',fontsize=13,fontweight='bold')
box(ax,(0.05,0.55),(0.26,0.25),'SPL geometry paper\nQuestion:\nwhere is condition signal buried?\nObject: subject/condition variance',fs=9,fc='#f5f5ff')
box(ax,(0.37,0.55),(0.26,0.25),'Interpretability paper\nQuestion:\nwhat can be inspected?\nObject: four-level taxonomy',fs=9,fc='#fff7e6')
box(ax,(0.69,0.55),(0.26,0.25),'This paper\nQuestion:\nwhat does each layer do?\nObject: E,D,T,C ablation',fs=9,fc='#e9f7ef')
box(ax,(0.22,0.17),(0.56,0.20),'Unique claim in this manuscript:\nclassification utility and measurement utility dissociate;\nstaged layers are evaluated by operational role, not only endpoint accuracy.',fs=10,fc='white')
arrow(ax,(0.18,0.55),(0.39,0.37)); arrow(ax,(0.50,0.55),(0.50,0.37)); arrow(ax,(0.82,0.55),(0.61,0.37))
fig.tight_layout()
fig.savefig(OUT/'fig10_novelty_boundary.pdf',bbox_inches='tight')
plt.close(fig)
