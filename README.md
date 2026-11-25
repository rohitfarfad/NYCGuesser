# Computer Vision — Project 4: NYCGuessr

NYU Tandon, CS-GY6643, Fall2026. Active code, metrics and progress are in **NYCGuessr.ipynb**; operating rules are in **AGENTS.md**.

## Current metric and status — 2026-11-19

The user supplied the staff announcement changing the GPS denominator from10km to5km. Current working formula: `0.10 * borough_macro_F1 + 0.90 * max(0, 1 - mean_haversine_km / 5)`. Average distances before clipping, preserving the handout aggregation; live Kaggle executable scoring was inaccessible. Use `score_current_predictions` / metric version `mean_haversine_5km_v2` for new experiments (notebook cells90–94). The earlier scorer/trainers/results are frozen v1 replay code.

| Fixed model | Representative v2 score | Spatial v2 score | Mean GPS km (rep / spatial) |
|---|---:|---:|---|
| StreetCLIP | .185297 | .075414 | 4.414989 / 5.987724 |
| Candidate ensemble | .315404 | .075414 | 3.692175 / 5.337459 |
| Previous StreetCLIP + DINOv2 fusion | .346004 | .077237 | 3.543432 / 5.530323 |
| **Fusion + hybrid-meta decoded-distance reranker** | **.365795** | **.077237** | **3.433482 / 5.460754** |
| Experimental pooled-expert ranker +trust (**target unmet; not promoted**) | .417836 | .077237 | 3.144363 / 5.212053 |

These are reused development results from fixed saved predictions, not Kaggle scores. The spatial GPS term is zero because mean errors exceed5km; track raw mean distance as a separate guard. Both seeds improve mean distance; second-seed spatial uncertainty remains. Five current-metric tests passed;18 prediction sets were rescored with original checkpoints/histories unchanged.

Primary goal **.80**, stretch **.90** under v2. At Macro-F1 .85 these require mean errors <=1.028km / <=.472km. The selected decoder-aware model improves GPS locally; the next measurement is its Kaggle result. Large geographic mistakes and spatial generalization remain the main limitations. The mean-distance objective is unchanged; do not blindly halve training-target temperatures.

Ready new CSV: `Kaggle Submissions/nycguessr_decoded_hybrid_20261118.csv`,7,908validated rows. It changes7,738GPS predictions relative to the previous fusion CSV. **New public score unknown.** The user reports about0.42 for the previous fusion after the5-km update; that is a rounded public result for a different CSV, not comparable to local validation as a before/after pair. The earlier .60599 StreetCLIP result is historical v1. Source scores and provenance are in notebook cells129–151 and `outputs/decoded_rerank/`.


No external imagery/maps/coordinates are permitted. The user reported course approval for all pretrained image encoders; retain documented weight provenance and use only competition data for training/retrieval. Real-run flags default off; authorized runs are executed and monitored by the agent. Kaggle upload requires explicit permission.

**Spatial target update — 2026-11-19:** below5km (stretch below4km) is **not yet met**. Best matched experimental spatial means are5.212/5.235km, representative3.144/3.244km. Training-only expert arbitration and pooled-candidate ranking improve GPS, but none reaches the new acceptance threshold. Spatial-only supervision regresses representative performance; its primary gate also fails, and the remaining gate repeats were not started after the90s budget-reservation stop. Total42completed fits814.54s/900s;18inline tests and590artifact checks passed. No new production selection, test predictions, CSV or upload. The ready CSV above remains the older validated hybrid model. Code/results in notebook cells152–173 and `outputs/spatial_target/`. Next proposed task: expand unique geographically cross-fitted calibration coverage beyond2048queries using existing caches; not yet run.

## GeoGuessr technique study — 2026-11-19

Implemented and ran expert-style clue narrowing with the existing StreetCLIP/DINO
prior: OCR, gallery-only rarity/geographic specificity, cross-view deduplication,
visual corroboration, candidate reranking and text recovery. Research used a
Rainbolt demonstration transcript and PlonkIt excerpts; outside images/maps/clue
locations were excluded. Code, metrics, visual examples and diagnosis are in
notebook cells174–184; artifacts in `outputs/geoskills/`.

**No material score gain:** experimental spatial means remain5.212/5.235km;
usable text covers11.62%of locations and reliable gallery matches only3.37%/2.52%
of representative/spatial queries. Both paired spatial confidence intervals
include zero. No production change or new CSV/public result. All47,456training
images recognized,3,703accurate refinements,617.68s combined OCR. Six new tests and
173artifact checks/12replays passed; final75-cell targeted check had zero errors.
Next proposed experiment: benchmark dense local DINO patch matching for structures
and signs. No further extraction/training has started.

## Historical setup and development notes (v1 /10km)

The following earlier notes/results/policy statements are preserved as history; the current status, broad pretrained-encoder permission and scoring above supersede them.

NYU Tandon, CS-GY 6643, Fall 2026. Work in this directory; operating instructions and verified requirements are in `AGENTS.md`.

Current retained method: **frozen ImageNet ResNet18 + a lightweight four-view head**, representative score **0.12568** / spatial **0.11031**. Both are local development results; the 0.90 target remains unmet. Complete code, comparisons and progress are in `NYCGuessr.ipynb`.

The user-supplied inputs have been extracted and audited. Extracted filenames and meaningful dataset subdirectories are preserved; the three source ZIPs were deleted at the user's request after a fresh size/SHA-256 verification of every input file:

| Folder | Contents |
| --- | --- |
| `Q1 Data/` | Three 224x224 masked RGB images and three binary masks; each mask removes 98 of 196 patches. |
| `Q2 Data/` | Three 512x512 noisy RGB images: coffee, cat and astronaut. |
| `Q3 Data/q3/` | `eval_001` through `eval_032`, each with a 128x128 reference and an 800x600 scene. |
| `Data/train/` | 47,456 RGB JPEGs, four 640x640 views for each of 11,864 labeled locations. |
| `Data/test/` | 31,632 RGB JPEGs, four 640x640 views for each of 7,908 unlabeled locations. |
| `Data/train.csv` | Sample ID, four directional image filenames, borough, latitude and longitude. |
| `Data/sample_submission.csv` | Exact 7,908-row submission template; repeated values are placeholders, not test labels. |

All 79,161 images fully decoded without errors; all recorded image-file hashes matched. Q1 masks align exactly to the 16x16 patch grid. All Q3 pairs and Kaggle four-view groups are complete. No exact Kaggle image duplicates were found. Q3 reuses scenes/references across different pairs: retain all 32 pairs. A training-only pHash heuristic found no qualifying near-duplicate groups within 250 metres; this does not prove that all visual overlap is absent.

Raw data and audit outputs are ignored by Git and must remain separate from source code. All 79,163 extracted/manual input files remain unchanged. `DATA_SOURCES.json` records provenance, archive hashes, cleanup status, schemas and findings. Detailed manifests/reports, including `outputs/archive_cleanup.json`, are under `outputs/`. Removing the three ZIPs removed 7.26 GB of redundant archive files; audit reproduction uses the extracted inputs and existing manifest.

Reproduce the read-only raw-data audit from this directory with `python3 -B audit_data.py`. It uses Pillow, NumPy and SciPy, decodes at most two images concurrently, and refreshes derived audit reports only. The inspected system environment has Python 3.12.6, Pillow 12.2.0, NumPy 2.4.4 and SciPy 1.17.1; no dependencies were installed for the raw-input audit. It expects the existing extraction manifest at `outputs/input_manifest.jsonl`.

Training IDs and test IDs each restart at zero; identify inputs by split plus sample ID. Training borough counts match the handout. The median nearest other training location is about 99.9 metres. Fixed representative and spatial validation splits are now saved in `outputs/splits.json`.

The complete Kaggle solution and progress tracker is **`NYCGuessr.ipynb`**. All solution code and checks are inline; no external project helper modules are imported. The notebook contains data contracts, the scorer and CSV validator, training-only priors/RGB controls, scratch-CNN baselines, permitted ImageNet transfer learning, frozen-feature and selective fine-tuning comparisons, experiment records and guarded final CSV generation. Model weights, caches, run logs and predictions belong under ignored output paths.

Open the notebook with a Python environment matching `requirements.txt`. The local **NYCGuessr Project 4** kernel is installed inside `.venv`; that environment reuses the inspected system ML packages and adds the notebook kernel without modifying global packages. Training/export are disabled by default. Configure the notebook's settings cell to run a bounded experiment; CUDA is intended for allocated HPC sessions, and MPS can run bounded local development experiments when HPC allocation is unavailable. Smoke runs are explicitly excluded from performance comparisons.

The saved notebook has passed fresh-kernel restart-and-run-all: **22 code cells, 21 inline checks, zero errors**, with embedded baseline, pilot-progress and training-only diagnostic plots. Two matched five-minute local MPS CNN pilots completed two epochs each. The corrected GPS objective improved the second-epoch mean error from 17.30 to 11.14 km, but neither pilot beat the visual controls. Supervised RGB ridge is the best current control: representative **score 0.03687, Macro-F1 0.36868, mean 10.45 km**; spatial **0.03487, 0.34869, 10.53 km**. Those earlier controls have zero GPS score; **the 0.90 target has not been reached**. Final fitting is gated on beating controls with a positive GPS score. No test submission CSV or Kaggle upload has occurred.

The subsequent small-batch local continuation used 2,048 training locations, batch eight, 128-pixel inputs and an eight-minute cap. It completed 11 epochs; its best head gave **10.42 km**, compared with **10.91 km** for matched RGB ridge. Composite scores were **0.03247 vs 0.03521**, both with GPS zero. Frozen-feature calibration did not improve GPS. A separate 40-location training-only memorization check reached **0.84 km / Macro-F1 1.0**; that result is explicitly excluded from validation/leaderboard claims. That notebook version contained **14 passing checks** and separate held-out and training-only curves. See its recorded decisions before extending compute.

The latest more-data stage expanded to **all 9,492 representative training locations**, retaining the same 2,372 validation locations. It initialized from the earlier training-only checkpoint, preserved geographic-cell identities, and used batch 16 at 128 pixels, fresh AdamW at 0.0005 and a 15-minute cap. Seven epochs completed; epoch six was best: **score 0.04624, Macro-F1 0.38185, GPS score 0.00895, mean/median/p90 9.9105/8.3117/18.2430 km**. This improves the previous best local score of 0.03687, but remains far from 0.90. Retrieval and fixed frozen-feature ridge did not improve on this head. Staten Island remains a major failure (F1 zero, validation mean 28.10 km). This is development validation, not a leaderboard or independent spatial result. The notebook checks warm-start training-ID membership and preserves initialization bytes and learning-rate horizon for eventual full refitting; no final refit has been run.

With ImageNet permission confirmed, the retained approach is now **frozen ImageNet ResNet18 + a small ordered four-view head**. Representative **score 0.12568, Macro-F1 0.50344, mean 9.16297 km**; independently fitted spatial head **score 0.11031, mean 9.30378 km**. One-time feature extraction took **168.87 s** and produced a **92.69 MiB** cache. Both folds' cached baselines/head experiments reran in **22.12 s**; the nonlinear head fits took **9.70 s / 6.73 s**. This is substantially cheaper to iterate than raw-image training. A controlled last-residual-block fine-tuning epoch took **154.61 s** plus validation and scored **0.11383**, below the frozen head; it was rejected along with a head-only continuation control (**0.11338**). The notebook preserves all results, safe reloads, a parameter-change audit, fixed-split provenance and a guarded final frozen-head refit/export path. The actual all-training-data refit, test-feature extraction and test CSV remain unexecuted. **The 0.90 target is unmet; these are development-validation results.**

Torch login access has been verified. Scheduling is currently blocked because the scheduler rejects the default `users` account and requires a project account such as `torch_pr_…`. The project portal requires NYU VPN; course staff may need to enroll the user in the appropriate allocation. The remote environment setup did not complete, no CPU/GPU job was submitted, and the partial data transfer was paused. `outputs/hpc_status.json` records that state; copied remote files must be reconciled before use.

Q4 external imagery/maps/coordinates remain prohibited. On 2026-11-14 the user reported TA/instructor approval for **ImageNet-pretrained weights**. The official ImageNet-1K ResNet18 V1 checkpoint is now downloaded and hashed; other external pretraining sources remain unapproved. Kaggle submissions require explicit user instruction. The notebook's saved results and experiment log, rather than the 0.90 target, determine whether a method is retained.
