ScholarOne source package — ARSPI-Net (double-blind review version)

Contents (self-contained; compiles with pdflatex + IEEEtran.cls):
  main_blind.tex                         anonymized manuscript source
  main_blind_reference.pdf               reference compiled output (10 pages)
  arch_fig_pipeline_overview.pdf         Fig. 1  (architecture)
  fig_reservoir_dynamics.pdf             Fig. 2  (observed computational objects)
  ana01_mechanism_ablation_performance.pdf  Fig. 3  (mechanism ablation)
  ana03_robustness_degradation_curves.pdf   Fig. 4  (perturbation robustness)
  ana07_closed_loop_success_by_policy.pdf   Fig. 5  (closed-loop control)
  table_neural_mechanisms.tex            Table (mechanism mapping)
  table1_feature_blocks.tex              Table (feature blocks)
  table_mechanism_ablation.tex           Table (primary ablation)
  table_closed_loop_policy.tex           Table (closed-loop policy)
  table_reproducibility_panel.tex        Table (reproducibility)

Build:
  pdflatex main_blind.tex
  pdflatex main_blind.tex          (twice for cross-references)

Notes:
  * IEEEtran.cls is supplied by the IEEE/ScholarOne system (documentclass[journal]{IEEEtran}).
  * Bibliography is inline (thebibliography); no .bib/.bbl needed.
  * This is the DOUBLE-BLIND review version: author names, affiliations,
    acknowledgments, institution, dataset name, and repository link are withheld.
  * The non-blind camera-ready is built from ../main.tex on acceptance.
