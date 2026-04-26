# Preparing the input pickles

The reporting pipeline (`experiments/operational_distinctness/run_all.py`)
consumes three files. Two of them — the ARSPI-Net feature pickles — are
**not** in this repo because they're large (≈55 MB and ≈400 MB) and
derive from raw clinical EEG that should not be redistributed. This
folder contains the script and runbook for regenerating them on a
machine that has access to the raw SHAPE EEG.

| Input | What's in it | Where it comes from |
| --- | --- | --- |
| `shape_features_211.pkl`     | E (BSC6/PCA-64 embedding) and BandPower (conv_feats) | `dissoAdventureExperiments/chapter5Experiments/run_chapter5_experiments.py` |
| `ch6_ch7_3class_features.pkl`| D (dynamical) and T_topo (topological) per-channel arrays | **`prepare_inputs/extract_ch67_features.py`** (this repo) |
| `clinical_profile.csv`       | binary diagnosis indicators per subject ID | sourced separately; one row per subject |

The reporting pipeline finds these files via `experiments/operational_distinctness/config.py`,
which checks (in order): explicit env vars → local `data/` directory →
sibling `dissoAdventureExperiments/` checkout.

---

## 1. Raw EEG layout

Place the raw 3-class EEG files in any directory; the extraction script
walks it recursively. Filenames must match the pattern

```
SHAPE_Community_<subject_id>_IAPS<Neg|Neu|Pos>_BC.txt
```

Each file is `(1229, 34)` float64 in µV at 1024 Hz, baseline-corrected,
with rows 0–204 being the 200 ms pre-stimulus baseline.

The full 211-subject dataset is 633 files (3 conditions × 211 subjects;
subject 127 is excluded as a recording anomaly).

---

## 2. Generating `ch6_ch7_3class_features.pkl`

```bash
# 1. Install dependencies (requirements.txt is at the repo root):
pip install -r ../requirements.txt

# 2. Point the script at the raw data and the output location:
export ARSPI_RAW_BATCH_DIR=/path/to/batch_data/
export ARSPI_CH67_OUTPUT=/path/to/operational-distinctness-paper/data/ch6_ch7_3class_features.pkl

# 3. Run with unbuffered output so you see progress live:
python -u prepare_inputs/extract_ch67_features.py
```

### What the script does

For each of the 633 observations (subject × condition), it:

1. Loads the raw `(1229, 34)` µV file.
2. Computes a theta-band tPLV matrix on the post-stimulus segment and
   its weighted strength + clustering descriptors (the **T** block).
3. Downsamples the EEG 4× to 256 Hz, drives a fixed LIF reservoir
   (N=256, β=0.05, θ=0.5, seed=42) on each of the 34 channels.
4. Extracts 7 core dynamical metrics + 4 extras per channel from the
   driven trajectory (the **D** block).

Reservoir parameters and metric definitions are pinned to Chapter 4 of
the dissertation; the script does not expose them as flags.

### Runtime expectations

| Hardware                                | Expected wall time |
| --- | --- |
| Modern Linux/macOS workstation, 633 obs | ~10–20 min          |
| Cloud VM (8 vCPU, NumPy + OpenBLAS)     | ~10–15 min          |
| Windows laptop without OpenBLAS / heavy AV | several hours (avoid) |

Progress is printed every 20 observations. The output pickle is
~300–400 MB.

### Subset run for a quick smoke test

If you just want to validate the pipeline end-to-end without paying the
full extraction cost, point `ARSPI_RAW_BATCH_DIR` at a directory with
only a handful of subjects (e.g. 20 × 3 = 60 files). All downstream
scripts work the same way — they just run on a smaller cohort.

---

## 3. Generating `shape_features_211.pkl`

This file is produced by the Chapter 5 pipeline in the **upstream**
[`dissoAdventureExperiments`](https://github.com/TheAwesomeAndy/dissoAdventureExperiments)
repository. It contains:

```
X_ds          : (633, 256, 34)   downsampled EEG
y             : (633,)           condition label (0=Neg, 1=Neu, 2=Pleasant)
subjects      : (633,)           subject ID
conv_feats    : (633, 34, 5)     band-power baseline (BandPower block)
lsm_bsc6_pca  : (633, 34, 64)    BSC6 / PCA-64 embedding (E block)
```

To regenerate:

```bash
git clone https://github.com/TheAwesomeAndy/dissoAdventureExperiments.git
cd dissoAdventureExperiments
python chapter5Experiments/run_chapter5_experiments.py --data_dir /path/to/batch_data/
# Resulting pickle: chapter5Experiments/results/shape_features_211.pkl
# (path may vary; see that script's --help)
```

Then move/copy the pickle into `data/` of this repo, or set
`ARSPI_CH5_FILE=/path/to/shape_features_211.pkl` before running the
reporting pipeline.

---

## 4. Clinical profile CSV

`clinical_profile.csv` is one row per subject with binary indicator
columns for the diagnoses used in the C1–C6 ablation
(`SUD`, `MDD`, `PTSD`, `GAD`, `ADHD`). It is **not** redistributed
through this repo. Place it at one of the locations resolved by
`config.py`, or set `ARSPI_CLINICAL_FILE`.

---

## 5. After all three inputs are in place

```bash
cd /path/to/operational-distinctness-paper
python experiments/operational_distinctness/run_all.py
# or, with optional hardening:
python experiments/operational_distinctness/run_all.py --include-optional
```

This writes paper-ready CSVs, figures, and LaTeX tables under
`outputs/`, `figures/`, and `tables/`, plus `outputs/.../run_manifest.json`
recording the inputs found and the package versions used.

---

## PHI reminder

Subject IDs are SHA-256 hashed (16-char prefix) in every committed
artifact. **Never** commit the raw EEG files, the two feature pickles,
the clinical CSV, or any intermediate that contains raw subject IDs —
all of those paths are covered by the repo's `.gitignore`.
