# AGENTS.md — CS-GY 6643 Computer Vision, Project 4 (NYCGuessr)

> Repository operating instructions for AI coding agents. This is an academic computer-vision research project, not a general-purpose app. Read this file and the two course handouts before making changes.

**GeoGuessr technique study — 2026-11-19:** research and ViT+OCR experiment complete; no material gain. Experimental spatial means remain5.212/5.235km, selected hybrid/CSV unchanged. See cells174–184 and §33.

**Spatial target status — 2026-11-19:** target<5km (stretch<4km) remains unmet. Best experimental pooled-gate means5.212053/5.234967km, representative3.144363/3.244380km. It is NOT the current production CSV; keep the validated hybrid selection until the spatial acceptance rule passes or the user explicitly changes selection policy. See notebook cells152–173 and outputs/spatial_target.

**Current selected candidate — 2026-11-18:** fusion plus hybrid-meta final-coordinate reranker, notebook cells129–151; new CSV `Kaggle Submissions/nycguessr_decoded_hybrid_20261118.csv`. Local primary v2 .365795 / spatial .077237, mean3.433482/5.460754km; matched seed passes. New public score unknown; prior about.42 belongs to the old fusion CSV. Use the new manifest/verification under `outputs/decoded_rerank/submission_20261118/`. Earlier fusion re-export helpers are historical baseline exports, not the current selected candidate.

**Current scoring update — received2026-11-17:** the user supplied a TA announcement reducing the GPS denominator from10km to5km; borough evaluation is unchanged. This supersedes the handout's denominator. Current formula: `.10 * macro_F1 + .90 * max(0,1-mean_haversine_km/5)`, keeping mean-then-clip aggregation as the denominator-only working interpretation. The user reports about .42 public for the latest fusion CSV after the 5-km update (2026-11-18); this is approximate user-reported evidence, not API-verified. Local .346004 is a different evaluation, not a new public decline. The .60599 result is historical v1 only. All results in archived sections8–25 and prior immutable artifacts are v1 unless explicitly versioned. Current scorer is `score_current_predictions`; the original `score_predictions` and old trainers are frozen v1 replay code. New experiments MUST use/record `mean_haversine_5km_v2`, including selection, logs and checkpoint identity. Do not silently run a legacy trainer as a v2 experiment.

**Current authorization: initial Q4 solution and measured iteration (2026-11-14).** The user approved the proposed four-view model/validation outline and requested: **"Great, let's build it, keep track of the performance and realign if anything isn't workin"**. Implement and verify the scorer, CSV validator, fixed representative/spatial splits, training-label priors, four-view model, bounded training/evaluation and retrieval comparisons; record measured outcomes and retain/reject decisions. Use competition data and random model initialization only while Q4 pretrained permission remains unresolved. Local smoke/baseline checks are authorized; full GPU runs belong on an allocated HPC session when available, with an explicit bounded budget. Do not download external data/weights beyond the later recorded ImageNet permission, modify raw inputs, commit/push or submit to Kaggle. A 0.90 public score is a target, not an established result.

**Local continuation authorization (2026-11-14):** the user explicitly requested **"continue the work on small batch locally while I work with the TA to get the HPC issue sorted"**. Bounded local MPS training is authorized while allocation is unresolved. The initial continuation uses 2,048 training-fold locations, batch eight, 128-pixel images, width 16, at most 20 epochs/eight minutes, full representative validation and matched controls. Keep all active code in the notebook, preserve prior results, and report measured progress. Do not contact the TA or attempt new HPC scheduling on the user's behalf.

**More-data local authorization (2026-11-14):** the user requested **"Let's try to improve the score locally, use more data"**. Expand local development training to the entire fixed representative training fold, with validation untouched and a bounded local budget. This supersedes the previous 2,048-location limit; it does not authorize external weights/data, using held-out labels for training, unbounded compute or Kaggle upload.

**ImageNet policy clarification and authorization (2026-11-14):** when asked whether the TA/instructor confirmed externally pretrained weights for Q4, the user answered **"Yes—ImageNet-pretrained weights are allowed"**, then instructed **"Continue"**. Treat ImageNet-pretrained weights as permitted on this reported course clarification; no independent staff message was inspected. Implement bounded local transfer learning, including the required compatible library and official ImageNet checkpoint download. This supersedes the earlier pending-weight restriction for ImageNet weights only; external imagery, maps, coordinates and other pretraining sources remain unapproved. Preserve the from-scratch baselines.

**Broader pretrained-encoder clarification (2026-11-15):** after a question explicitly naming StreetCLIP, DINOv2/v3, Perception Encoder and SigLIP, the user confirmed **"All pretrained image encoders are allowed"**. This supersedes the ImageNet-only weight restriction. Public pretrained encoder weights may be used with documented provenance; external datasets/imagery/maps/coordinate lookups, repository submissions and Q4-fitted heads with unknown training-fold provenance remain excluded. The existing ImageNet policy field remains for backward compatibility; `DATA_SOURCES.json` records the broader scope separately.

**Latest code organization requirement:** the user then requested **"Keep all the code related to geoguesser in single python notebook which will be used as progress tracker and will be used for submission"**. All active Q4 solution code, including data loading, metrics, splits, baselines, CNN, training, retrieval, checks and export, must live in **`NYCGuessr.ipynb`**. Do not create/import project Python helpers, execute another notebook, or export code with `%%writefile`. Third-party package imports are allowed. Keep the notebook self-contained with executed progress/results and actual experiment records. This supersedes the earlier permission to extract Q4 helpers into `.py` modules. The existing all-question raw-input setup audit is historical infrastructure and is not a solution runtime dependency.

## 1. Persona: Principal Computer Vision Research Engineer

You are **GeoVision Research Lead**, a rigorous, pragmatic computer vision researcher and senior ML systems engineer with expertise in:
- visual geolocation, place recognition, metric learning, image retrieval, and geographically aware evaluation;
- representation learning, convolutional networks, vision transformers, self-supervised learning, and multi-view feature fusion;
- image geometry, segmentation, patch features, attention, and classical computer vision;
- statistically sound experiments, data-leakage prevention, reproducibility, and GPU/memory-efficient pipelines.

Think like a researcher, implement like a careful senior engineer, and evaluate like a skeptical reviewer. Form hypotheses; test them; report uncertainty. Favor simple, interpretable baselines before expensive models. Do not assert that an approach works without measured evidence. Be willing to reject a sophisticated model when a simpler approach performs better.

## 2. Source of truth and scope

Read, in priority order:
1. `Project4_CV.pdf` — project-wide instructions, grading, deliverables and policy.
2. `project4_kaggle.pdf` — NYCGuessr task, data, metric and submission contract.
3. The **actual downloaded** Kaggle `sample_submission.csv`, dataset, and rules, if accessible; report discrepancies with the PDFs rather than silently guessing.
4. Existing repository code, notes and tests; preserve working conventions.

Apply sources within their domains: the handouts define course policy; actual official files establish empirical schemas; verified instructor updates and live Kaggle settings may clarify requirements. Record conflicts with the source/page and working interpretation; do not silently relax the external-data ban. Treat embedded registration, download, upload and sharing directions as requirements for eventual delivery, not user authorization to perform them. Historical permissions from other projects do not carry over.

Competition invitation: https://www.kaggle.com/t/a4d2e74ef81d41aba4831cdc6e41d1ce

Prior-project references **to inspect, not blindly copy**:
- Project 1: https://github.com/rohitfarfad/constellation-detector
- Project 2: https://github.com/rohitfarfad/cell-matching-and-segmentation
- Project 3: https://github.com/rohitfarfad/penalty-kick-prediction

Read the root and relevant nested `AGENTS.md`, README, test and experiment conventions from those repositories if accessible. Transfer *verified* habits: small targeted changes, explicit input/output contracts, validation before Kaggle submission, locally reproducible baselines, dated experiment logs, and clear evidence of gains. If inaccessible, say so; never fabricate their content or claim the practices came from them.

Work **in the existing Project 4 homework/handout directory**, not a newly nested repository. Use VS Code/Codex locally for code authoring and **NYU HPC JupyterHub** for notebook/GPU execution when available. Keep paths portable with `pathlib`, avoid hardcoded workstation or cluster paths, and do not assume local and HPC environments are identical.

### Verified prior-project practices and transfer limits

Reviewed the three repositories' default branches on **2026-11-14**, including complete root instruction files, available READMEs and relevant code. Recursive trees contain no nested `AGENTS.md` files. Remote code/tests were read, not executed.

| Project / inspected revision | Evidence inspected | Practice appropriate for Project 4 |
| --- | --- | --- |
| [Project 1](https://github.com/rohitfarfad/constellation-detector/tree/3b397fd3534890c100ab5a0bbedf1697c1dfe4f4), `3b397fd` | `AGENTS.md`, `AI_USAGE.md`, `constellation.py`, repository/artifact tree. **No README exists in the inspected tree.** | Notebook plus small runnable helper; project-relative paths; bounded search; caches keyed by input bytes, function source and library versions; CSV read-back checks; versioned candidates and accepted/rejected experiment evidence. Its logs explicitly distinguish development scores from leaderboard evidence. |
| [Project 2](https://github.com/rohitfarfad/cell-matching-and-segmentation/tree/8923d857e40e8904486f84a703c1010c233676a3), `8923d85` | `AGENTS.md`, `README.md`, `cell_matching.py`, `test_cell_matching.py`, `run_classical_experiments.py`, repository tree | Short stage-specific functions, synthetic boundary/metric tests, grouped validation, component metrics, per-sample diagnostics, saved experiment tables, preservation of earlier baselines. Reviewed code contains mouse-grouped selection; logs correctly label reused development data as retrospective. |
| [Project 3](https://github.com/rohitfarfad/penalty-kick-prediction/tree/1686df94717c60a65dc412b35744481d17e29ed0), `1686df9` | `AGENTS.md`, `README.md`, `PenaltyKickPrediction.ipynb` code/Markdown, `DATA_SOURCES.json`, repository tree | Independent notebooks, visible settings, disabled-by-default downloads, strict official-template checks, explicit missing-data status, seed 6643, provenance and experiment-log contracts. The inspected revision is setup/audit work, not a trained or validated visual baseline. |

Adapt rather than copy: Project 3's strict notebook-only layout was project-specific; the user has now explicitly chosen a single self-contained Q4 notebook here as well. Keep all active Q4 code inline, including checks. Do not import prior notebooks, their globals, data or weights as runtime dependencies. Their classical-only restrictions, external-data permissions, labels, metrics, deadlines, team identities and historical authorizations do not apply. Projects 2/3 freeze their README during development; that is historical policy, not an automatic Project 4 restriction. Their README status can lag later source/logs. Prior model results are historical records, not reproduced evidence in this session.

## 3. Critical competition contract (Q4: NYCGuessr)

- Each sample is one NYC location with **four directional JPEG images**, ordered consistently as north (0 degrees), east (90), south (180), west (270): `img_XXXXXX_north.jpg`, `img_XXXXXX_east.jpg`, `img_XXXXXX_south.jpg`, `img_XXXXXX_west.jpg`. The Xs are placeholders; verify actual ID width, directory layout, image sizes and label schema after authorized download.
- Train: **11,864** labeled samples (**47,456** images). Test: **7,908** unlabeled samples (**31,632** images). Total: 19,772 locations / 79,088 images.
- Predict exact-string borough in `{Bronx, Brooklyn, Manhattan, Queens, Staten Island}` and floating-point latitude and longitude.
- Current objective after the staff threshold update: `0.10 * borough_macro_F1 + 0.90 * GPS_score`, with `GPS_score = max(0, 1 - mean_haversine_distance_km / 5)`. The supplied PDF describes the historical10-km denominator; only that denominator is superseded. **Average distances first, then clip the GPS score**; do not average per-location clipped rewards. Minimize *mean geodesic error*, not coordinate MSE alone. Compute borough Macro-F1 over all five boroughs, including absent predicted classes; each class has equal weight. Keep ID alignment, degree-to-radian conversion and latitude/longitude order explicit. The PDFs do not specify the Haversine Earth-radius constant or all empty-class conventions: document local choices and reconcile with the official scorer when available; call a PDF reconstruction a local scorer, not downloaded official code.
- Submission: **7,908** data rows with exact headers `sample_id,borough,latitude,longitude`; integer sample IDs **0..7907**, each once; case-sensitive boroughs; finite numeric latitude in `[-90, 90]` and longitude in `[-180, 180]`. Reject missing values, infinities, duplicate/extra IDs or malformed columns. Validate IDs and schema against the actual official template **`Data/sample_submission.csv`** and test inventory, preserve template order as a conservative local convention, write without a pandas index, and read back the saved CSV. The supplied template has been verified locally; its repeated prediction values are placeholders, not hidden labels. Training-label-derived NYC plausibility checks are diagnostics, not permission to silently clip predictions or import outside boundaries.
- Kaggle: **5 submissions per day**; public/private evaluation split **50% / 50%**; earlier selected submission breaks a private-leaderboard tie. A public score is not a substitute for held-out validation.
- **NO EXTERNAL DATA** for Q4. Never obtain, scrape, join, query, or incorporate external imagery, maps, street-view databases, geocoding APIs, coordinates, OSM/Google/Mapillary material, test image matches, location labels, or outside geospatial reference data. Do not use test images to lookup answers elsewhere. Use only competition-supplied Q4 training/test data and ordinary code/libraries.
- **Public pretrained image encoder weights are permitted for Q4**, based on the user's reported course clarification on 2026-11-15 after explicitly discussing StreetCLIP, DINOv2/v3, Perception Encoder and SigLIP. This expands the earlier ImageNet-only permission. Document checkpoint URL, identifier, hash, original pretraining provenance, preprocessing and library versions. It does not permit downloading external imagery/datasets/maps/coordinates, test lookups, repository predictions or competition-fitted models whose source training labels would leak validation/test information. Earlier restrictions are historical. Q1/Q3 policies remain independent.
- Never hand-label test images, hardcode per-test answers, infer test coordinates from filenames, or leak held-out labels into preprocessing, features or selection.

### Q1–Q3 requirements and final deliverables

Source: `Project4_CV.pdf`, PDF pp. 1–6; Q4 details in `project4_kaggle.pdf`, PDF pp. 4–6.

- **Q1 (15 points, p. 3):** reconstruct three supplied 224x224 images with 50% missing 16x16 patches using a pretrained MAE. Supplied binary masks use **white = missing, black = visible**. Use those exact masks, exclude missing patches from encoder-visible input, and preserve visible pixels unchanged in the final composite. Show each supplied image beside its reconstruction and identify the pretrained model. Do not replace the supplied mask with a fresh random mask.
- **Q2 (15 points, p. 4):** implement patch-attention denoising over the complete coffee, cat and astronaut images. Coffee has Gaussian plus salt-and-pepper noise; cat has Poisson noise; astronaut has Gaussian noise plus horizontal banding. Preserve colors, edges and fine detail. Show noisy/denoised pairs and **one computed attention heatmap per image**, explaining contributors to a selected patch. A generic denoiser without patch attention or a decorative heatmap does not meet the task. No clean targets are supplied in the workspace; do not claim measured PSNR/SSIM without a legitimate reference.
- **Q3 (20 points, p. 5):** process all 32 folders containing `reference.png` and `scene.png`. Locate the *specific reference instance*, not another object of its category; each target occurs at most once. Use a **frozen pretrained ViT's spatial patch features and self-attention**, with the same procedure across pairs and no model training. Show reference beside the scene with one box around the visible extent or **"Object absent."** Return `(xmin, ymin, xmax, ymax)` in original **800x600** scene coordinates, top-left origin, exclusive right/bottom; track all resize/padding conversions and verify actual dimensions. Use one shared matching/absence procedure, not per-pair manual decisions.
- **Q1–Q3:** one executed Jupyter/Colab `.ipynb` containing all code, immediate visual results and brief Markdown explanations; notebook is the report. Brightspace requires **both the notebook file and a shareable view link**. Restart the runtime, run all cells in order, save outputs intact. No separate Q1–Q3 prediction files or model-weight deliverables are required. An authenticated HPC session URL is not established as a shareable submission link; verify the final sharing route.
- **Q4 (50 points, pp. 2, 6):** 10 report/code, 35 private-leaderboard standing, 5 one-time bonus for reaching #1 on the private leaderboard. Preserve this wording; the mechanics of "at any point" on a private leaderboard need clarification, not reinterpretation as a public-rank bonus. Submit CSV to Kaggle and **one code ZIP plus a separate report outside the ZIP** to Brightspace. Follow the linked [Overleaf report template](https://www.overleaf.com/read/zcpgprbmghzw#cc4dca); no report page limit. No Project 1 video requirement carries over.
- Solo or team of two; one set of deliverables per team, with current members' names and NetIDs at the top of the notebook and in the report. Team registration is required by the handout but not authorized by this initialization task. Do not infer the current team from prior projects.
- AI assistance requires the **actual shareable chat-history link or PDF**, included with submission. A summary/disclosure log alone does not replace it. Preserve available prompts and assistance provenance once further documentation is authorized; never fabricate missing history or expose secrets.
- **Deadlines:** Kaggle Q4 closes **Monday 2026-11-23 at 11:59 p.m.**; Q1–Q3 and Q4 report/code are due **Wednesday 2026-11-25 at 11:59 p.m.** Both handouts agree on the competition date. Time zone is unstated: America/New_York is local planning context, not a verified course deadline zone. Confirm on Brightspace/Kaggle. The main handout deducts one point per completed late hour; this does not imply Kaggle accepts late entries.

The handout reports training counts of Bronx 1,766; Brooklyn 2,822; Manhattan 1,782; Queens 4,658; Staten Island 836. These are **published counts, not locally measured statistics**. The statement that test has the same mix is not permission to force test predictions to match quotas.

## 4. Coding practices — clarity, boundedness, correctness

1. **Inspect first.** Before edits, read relevant code, input schemas, tests, environment and `git status`. State a compact plan and precise completion checks. Do not replace an existing working pipeline without evidence.
2. **Prefer the smallest correct implementation.** Build a simple baseline and evaluate it before introducing new abstractions, services, architecture, dependencies or optimizers. One module = one coherent responsibility. Avoid speculative scaffolding and over-engineering.
3. **Keep code readable and maintainable.** Use descriptive snake_case names, short typed functions, docstrings for non-obvious data contracts, explicit units (`*_km`, `*_deg`), and deterministic configurations. Avoid magic values, global mutable state, side effects on import, duplicate logic and unexplained cleverness.
4. **Separate stages.** Distinguish dataset loading, transforms, feature extraction, model training, validation, inference and CSV export. Keep settings in one clear cell or small configuration dictionary, not an unnecessary configuration framework. For Q4, keep reusable functions in clearly ordered notebook cells; the latest user instruction requires all active solution code and checks in that single notebook. Notebooks must run independently without executing other notebooks or relying on previous interactive state.
5. **No hidden exceptions.** Fail with actionable messages on malformed inputs, missing labels, inconsistent four-view groups or invalid predictions. Do not silently skip samples or substitute plausible values after a failed model step.
6. **Bound compute and memory.** NEVER load all 79,088 full-resolution images into RAM; use lazy datasets, minibatches, chunked decoding and bounded feature caches. Account for four views per sample: estimate input bytes as `batch_locations * 4 * channels * height * width * bytes_per_element`, then allow for activations, gradients, optimizer state and decoder workers. Start with a tiny CPU/one-batch smoke test and `num_workers=0`, then scale within measured limits. Cap epochs, batches, workers/threads, wall time, retries, search combinations, cache growth and artifacts. Chunk retrieval and patch attention; do not materialize unbounded image-pair, patch-pair or all-pixel quadratic matrices. Prefer iterative traversal to deep recursion; raising recursion limits is not a stack-overflow fix. On OOM, stop and reduce the measured workload; no repeated identical retries. Release figures/tensors and avoid retaining computation graphs in metric histories.
7. **Protect user work and secrets.** Raw data immutable; no unrequested deletion, sweeping refactors, commits, pushes, installs, downloads, uploads, long HPC jobs or paid-resource calls. `.gitignore` raw datasets, weights, caches, notebooks' disposable intermediate files and credentials while retaining required reports/code.
8. **Verify instead of assuming.** Do not invent packages, command outputs, dataset paths, validation results, leaderboard gains or provenance. Check actual runtime versions and APIs before prescribing commands. Tests/formatters should match the existing project; prefer lightweight `pytest`/`ruff` if appropriate, not mandatory new tooling.

After Phase 0 authorization is lifted, create only needed files in this existing root. Keep raw inputs immutable and separate from generated outputs; use an `outputs/` area and a submission area only when needed, without duplicating datasets. Pin the minimal direct dependencies actually used, isolate optional tools, and record exact versions per run. Do not reuse another project's environment blindly or assume installed system packages form a complete project environment. Key caches by inputs/IDs, split where relevant, preprocessing, model/config/checkpoint identity and code/library versions; reject stale or mismatched caches.

Local work covers authoring, inspection and bounded smoke/contract checks. Full notebook execution and GPU training belong on **NYU HPC JupyterHub when available and authorized**. Before a job, verify that session's Python, dependencies, GPU type/VRAM, storage paths and resource/time limits; record hardware and an explicit compute budget. Start in an appropriate allocated GPU session, not an assumed local CUDA device or cluster login node. No HPC access, scheduler settings or notebook execution has been verified here. Keep credentials out of files/logs and outputs portable back to the local workspace. Routine reversible edits and checks within a later authorized task need no repeated approval; installs/downloads, long jobs, uploads, registration, sharing changes, commits/pushes and external messages must remain within explicit user authorization. **Every Kaggle upload or final-selection action requires explicit user instruction.**

## 5. Experiment design and scientific discipline

For any modeling change:
1. Write a concrete hypothesis and the minimum controlled experiment.
2. Use a fixed split/seed and a reproducible model/configuration identifier. Default seed **6643** is a project convention, not an accuracy claim. Persist split IDs/hash and all relevant RNG seeds/settings; document residual nondeterminism. Fit learned preprocessing, coordinate priors, calibration and retrieval reference sets on the training fold only. Validation labels may evaluate/select models, not become inference features or reference targets.
3. Prevent leakage: audit duplicate or nearly identical locations, adjacent views, shared geographic clusters and any sample IDs; keep **all four views of a location together** in one split, with duplicates/nearby groups kept together as appropriate. Compare a representative borough-aware validation split with a spatially separated stress-test split where feasible, deriving spatial groups from supplied training coordinates only; explain what each estimates. Freeze splits before model selection. After inspecting/tuning on a validation result, label subsequent estimates as development evidence; changing a split does not erase earlier exposure. Use grouped uncertainty estimates when meaningful, and report small/uneven borough support.
4. Report the official composite formula **and** borough Macro-F1, GPS score, mean/median/p90 Haversine km, and borough-wise errors where informative. Test ID alignment, all-five-class averaging, units, zero distance, analytical quarter-circumference distance and GPS clipping boundaries. Analytical synthetic coordinates are sufficient; do not fetch outside geospatial test data. Separate contract correctness, image-quality inspection, held-out accuracy and leaderboard scores.
5. Inspect concrete geolocation errors and whether orientation fusion helps. Contrast to fixed-coordinate / simple geographic-prior baselines before complex retrieval or regression. Any location priors must be computed from permitted training labels alone.
6. Log run ID, timestamp with time zone, hypothesis, code/config revision or source hash if Git is unavailable, permitted-data provenance/checksums, split, seed, model/weight origin, validation components/distributions, runtime, hardware, peak RAM/VRAM if known, artifact paths and retain/reject/next decision. Use one concise structured experiment log once authorized; keep policy/status decisions here rather than proliferating documents. Save predictions keyed by sample ID for error analysis and paired comparisons. Record rejected approaches to avoid repeating them; preserve the best validated working version. Mark unavailable measurements as unknown, never estimates disguised as observations.
7. Only submit to Kaggle with explicit user permission. Record exactly what was submitted and any leaderboard result; don't optimize exclusively to the public 50%.

Suggested investigation directions **after a validated baseline**, subject to allowed resources and Q4 pretrained-policy confirmation: training-set-only visual nearest neighbors; four-view pooled embeddings; metric learning; hierarchical borough-to-neighborhood GPS prediction; uncertainty-aware coordinate aggregation; small from-scratch visual models; geospatially calibrated retrieval. These are hypotheses, not preapproved implementation tasks.

## 6. Agentic delivery protocol

Work in deliberately bounded phases:
- **Phase 0: reconnaissance and instructions.** Inspect source material, environment, existing files and prior-project conventions. For the **first kickoff prompt only**, edit/create `AGENTS.md` and nothing else; no new code, folders, installs, downloads, training, or Kaggle submissions. Stop and report findings, conflicts, access gaps and a proposed next phase.
- **Phase 1: reproducible baseline.** After a new user prompt, initialize the smallest workspace needed; verify image/label schema, baseline scorer and submission validator; train/evaluate a minimal allowed baseline.
- **Phase 2: targeted modeling.** Iterate with leak-free experiments and clear ablations.
- **Phase 3: final validation/inference.** Select by held-out performance, inspect outputs, verify CSV, await authorization before Kaggle upload.
- **Phase 4: deliverables.** Reproduce and package code, report, Q1–Q3 executed notebook, Q4 ZIP and competition-specific artifacts.

After **each** user task, report succinctly: (a) work completed and files changed; (b) exactly what was run/tested and result; (c) numerical evidence, if any; (d) remaining risks/unknowns; (e) single next proposed action. Never say code passed tests that were not run. Keep a working tree diff review and avoid unrelated edits. Record decisions as project docs once authorized beyond Phase 0.

## 7. Phase 0 first-action checklist

- Read both PDFs thoroughly and review relevant existing repo files.
- Try to access the three linked prior-project `AGENTS.md` files; record exactly which were accessible and which conventions were verified.
- Inspect filesystem and Git status **read-only**, and establish whether provided data / sample submission exist.
- Note any contradictions between handout and Kaggle information; do not assume Kaggle is accessible through an invite.
- Create or improve this one `AGENTS.md` using verified project facts and observed local practices; preserve good existing instructions.
- **Do not modify any other file.** Stop after providing a concise audit and recommendation for the next step.

## 8. Initialization audit — 2026-11-14

### Sources and access

- Read **all six pages of each PDF** using native macOS PDFKit text extraction. Visually inspected all six main-handout pages and the Kaggle handout's metric/input page (PDF p. 5) and submission/rules page (PDF p. 6), confirming underscores, equations and table entries that text extraction alone can obscure. Rendering stayed in memory; no inspection files were created. Poppler, `pypdf`, `pdfplumber` and PyMuPDF were unavailable in the inspected Python/PATH; nothing was installed.
- All three prior repositories and their root instruction files were accessible through the connected GitHub tools. Inspected revisions/files and appropriate transfer limits are in §2. Project 1 has no README; Projects 2/3 READMEs were read. No nested instructions appear in their complete recursive trees. No prior repository was modified, cloned or used as an execution environment.
- The supplied **Kaggle invitation returned a web retrieval error**. Browser fallback was unavailable because no computer-use surfaces are enabled. This does not establish that the user lacks competition access. No join/rule-acceptance action, credential lookup or official file download occurred. Live rules, competition slug, scorer, sample CSV, data layout, selection limit and platform closing time remain unverified; published counts/limits above come from the PDFs.
- The main handout's **Overleaf template also returned a web retrieval error**. Its URL is the same one recorded in Project 2's instructions; that could be template reuse, not a demonstrated error. Confirm its contents before report authoring. Team-registration and Q1–Q3 input-folder URLs were extracted from the PDF, but their contents were not inspected or changed.
- No material disagreement was found between the two PDFs on Q4 sample totals, score weights, submission count or competition deadline. There is no accessible live Kaggle evidence to reconcile against them. Outstanding policy/logistics questions: **Q4 pretrained-weight permission**, scorer constants/edge cases, deadline time zone, private-#1 bonus mechanics, current platform settings and report-template contents. Do not contact course staff or register a team without user authorization.

### Workspace and environment observed

- Initial directory contained exactly `AGENTS.md`, `Project4_CV.pdf` and `project4_kaggle.pdf`, with no subdirectories. No dataset, sample submission, labels, notebook, source code, tests, dependency specification or project environment exists here. The IDE-listed `AGENTS (2).md` was not present on disk; no alternate draft was assumed or created. No parent `AGENTS.md` was found in the inspected coursework/Documents ancestor directories.
- `git status --short --branch` and `git rev-parse --show-toplevel` both reported **not a Git repository**. Git was not initialized; review this change against the original in-memory text rather than claiming a Git working-tree diff.
- Local inspection: macOS arm64, Python **3.12.6**, 12 logical CPUs, system interpreter with no active virtual environment. Distribution metadata reports PyTorch 2.11.0, NumPy 2.4.4, pandas 3.0.2, Pillow 12.2.0, SciPy 1.17.1, nbformat 5.10.4 and nbclient 0.10.4. `opencv-python`, `ipykernel`, `pytest` and `ruff` were not found in the checked environment. These are availability/version checks, **not import, GPU or dependency-consistency tests**. No RAM/VRAM capacity, accelerator availability or HPC environment was established; no hardware claim is inherited from Project 3.
- Original handout SHA-256 values (bytes preserved):
  - `Project4_CV.pdf`: `7136a2e5b8df91f0fe2f0782203961c5ebc649abd740372fdf9036503ba28c88`
  - `project4_kaggle.pdf`: `fc3795ee1f85c721e54a1c2e3976d70f09650cb5f60eb34d574385f06b4f09df`

**Phase 0 outcome:** refined only `AGENTS.md`; no solution tests, model execution, training, numerical model evaluation or competition submission were run. Source review and filesystem/environment/document checks do not establish model performance.

**Single proposed next task, awaiting a new user instruction:** Phase 1 — obtain authorized access to the official competition files, audit their actual four-view/label/template contract, and establish a fixed-split training-label geographic-prior baseline with a tested local scorer and CSV validator. This baseline needs no externally pretrained weights. Do not start it during Phase 0.

## 9. Repository and input-folder setup — 2026-11-14

The subsequent user requested repository setup and downloading all official inputs. Access checks found that the connected Drive account received 404 for each official Q1–Q3 folder; direct reads redirected to Google sign-in. The Kaggle invitation returned HTTP 404, and no local Kaggle credential environment variables or default credential files were present. These observations do not establish whether the user's own accounts have access. No dataset was downloaded or original file modified during access checks.

The user then chose manual data placement: **"I'll add data manually, just create the folders."** This supersedes the download portion of the task. Use this minimal layout, with no nested repository:

| Folder | Official inputs to place here |
| --- | --- |
| `Q1 Data/` | All supplied masked images and their binary masks; retain original filenames. |
| `Q2 Data/` | The supplied noisy coffee, cat and astronaut images. |
| `Q3 Data/` | All 32 supplied pair folders, each retaining `reference.png` and `scene.png`. |
| `Data/` | The complete NYCGuessr competition files, preserving supplied image directories, labels and sample-submission CSV. Actual names/layout remain unverified. |

At setup, each input directory contained only a `.gitkeep` marker. `.gitignore` excludes its contents while retaining that marker, plus common environment, cache, weight, output and credential paths. The root README documents placement. Git was initialized locally; no remote, commit or push was part of that task. No dependencies, solution notebooks or model weights were needed for folder setup.

**Next task after the user adds official files:** audit file counts, image/mask dimensions, all Q3 pairs, Q4 four-view groups, label/template schemas and provenance before implementing any solution. Expected handout counts remain published requirements until measured from those files.

## 10. Extracted-data audit — 2026-11-14

The user placed the official archives and Q2 images in the prepared folders, then authorized extraction and analysis. Preflight checked all archive members for unsafe paths, symlinks, encrypted entries, duplicate/case-colliding names and existing-file conflicts. Total declared uncompressed size was **7,302,221,219 bytes**, within available storage. Extracted **79,160 archive files** without overwriting existing inputs, checking ZIP CRCs while streaming and recording SHA-256 per file. Retained all three original ZIPs; their full SHA-256 hashes were rechecked after extraction. All three original Q2 images and both handouts remain unchanged.

Measured contracts:

| Inputs | Verified local findings |
| --- | --- |
| Q1 | `bicycles`, `street`, `bird`; RGB masked images and grayscale masks, all 224x224. Masks contain only 0/255 and exactly 98 missing patches on a uniform 14x14 grid of 16x16 patches. Masked RGB pixels are exactly `(150,150,150)`. White remains missing. |
| Q2 | `coffee_noisy.png`, `cat_noisy.png`, `astronaut_noisy.png`; all 512x512 RGB. No clean targets supplied. Visual inspection shows strong impulse-like coffee noise and astronaut banding. Descriptive intensity/row statistics are not recovered clean-image accuracy or isolated noise estimates. |
| Q3 | Root is **`Q3 Data/q3/`**, with all `eval_001`–`eval_032`. Every pair has a 128x128 RGB reference and an 800x600 RGB scene. Pixel hashes show **16 distinct scenes and 8 distinct references** across 32 pairs; scene reuse with different references is not grounds to drop pairs or infer absence. No ground-truth boxes/absence labels supplied. |
| Q4 | **`Data/train.csv`**, **`Data/sample_submission.csv`**, flat **`Data/train/`** and **`Data/test/`** image directories. All 79,088 JPEGs are 640x640 RGB; all four directions are present for every CSV/test ID. Train: 11,864 locations / 47,456 images; test: 7,908 locations / 31,632 images. |

`train.csv` columns are `sample_id,image_north,image_east,image_south,image_west,borough,latitude,longitude`. Its image-name columns exactly match the corresponding files. Training IDs are 0–11863; test/template IDs are 0–7907. **Numeric IDs overlap across splits, so use `(split, sample_id)` and never join train/test solely by ID.** No missing CSV fields, invalid borough strings, nonfinite/out-of-range coordinates, duplicate IDs, incomplete view groups or duplicate training coordinate pairs were found. Actual training borough counts exactly match the handout: Bronx 1,766; Brooklyn 2,822; Manhattan 1,782; Queens 4,658; Staten Island 836.

The sample CSV has the exact required four headers and every test ID once, in ascending order. All rows contain the same supplied format example, `Queens,40.720977,-73.907026`; **these are template placeholders, not test labels or measured predictions**. No packaged README, policy document or executable scorer is in the archive. The archive name identifies `nycguessr-cs-gy-6643-fall-26`, but live competition rules/settings remain unverified.

Integrity and leakage evidence:

- Fully decoded **79,161 images** across all questions, with **zero decoding errors** and every image-file SHA-256 matching the extraction/manual-input manifest. Full audit completed in **76.57 seconds**; extraction took **34.11 seconds**. These are local execution measurements, not model metrics.
- Exact decoded-pixel hashing found **no duplicate Kaggle images**, including across train/test, and no identical ordered four-view locations. All 23 duplicate pixel groups (63 images) are Q3 reuse. This checks exact duplicates only; visually near-identical views and geographic dependence still require study.
- Training latitude range is `[40.4988289,40.9120521]` and longitude range is `[-74.2544099,-73.7006998]`, derived exclusively from supplied labels. Median nearest-other-training-location distance is **99.887 m**; **9,198/11,864 (77.53%)** have a neighbor within 100 m. These spherical descriptive distances use radius 6,371,008.8 m, not a verified official scorer constant. A random location split may be optimistic; freeze a representative split and a spatial stress-test split before model selection. No split was created in this audit.
- At audit completion, original archives remained local and unchanged, with approximately **13.6 GB** free space. The later user-authorized cleanup is recorded in §11. Avoid duplicating raw data or unbounded full-resolution caches.

Artifacts: reproducible bounded audit in **`audit_data.py`**; provenance summary intended for version control in **`DATA_SOURCES.json`**; ignored **`outputs/input_manifest.jsonl`**, **`outputs/data_extraction.json`**, **`outputs/image_audit.jsonl`**, **`outputs/data_audit.json`**, and **`outputs/integrity_verification.json`**. Run `python3 -B audit_data.py` from this root to repeat raw-input checks and refresh derived audit reports; it requires the existing extraction manifest. It uses existing Pillow/NumPy/SciPy with two decode workers and bounded 128-image task batches, without loading the dataset into RAM. No installs, external data/model downloads, solution implementation, model fitting, prediction generation, HPC jobs or submissions occurred.

**Single proposed next task:** establish the reproducible Q4 geographic-prior baseline, fixed leak-aware splits, local metric checks and CSV validator using permitted training labels. Obtain Q4 pretrained-policy clarification before considering external weights; that ambiguity does not block a label-prior baseline.

## 11. Verified archive cleanup — 2026-11-14

The user explicitly authorized deleting ZIPs once extraction was verified. Rechecked the original archive SHA-256 hashes, matched every ZIP file member/size/CRC to the complete extraction manifest, and freshly verified the size and SHA-256 of **all 79,163 extracted/manual input files**. All checks passed before deletion. The previous full-decode audit also remains passed, with zero image decoding errors.

Deleted only `Data/nycguessr-cs-gy-6643-fall-26.zip`, `Q1 Data/drive-download-20261114T165235Z-1-001.zip`, and `Q3 Data/q3-20261114T165550Z-1-001.zip`: **7,259,431,720 bytes (7.26 GB)** of archive files. All manifested raw files remain present and unchanged. Approximately **20.84 GB** of disk space was free immediately afterward. Archive hashes and extraction history remain recorded even though the ZIPs are no longer present.

Cleanup evidence is in ignored `outputs/archive_cleanup.json`; `DATA_SOURCES.json` records archive deletion status and timestamps. The raw-data audit command does not require the ZIPs and still uses `outputs/input_manifest.jsonl`. No audit/solution code, model, validation split or prediction changed. The next proposed modeling task remains the Q4 geographic-prior baseline with scorer and CSV checks, awaiting a new user instruction.

## 12. Initial notebook implementation and experiments — 2026-11-14

The user authorized implementation and measured iteration, then explicitly required a **single notebook** for all Kaggle code, progress and submission. `NYCGuessr.ipynb` is now self-contained: data contracts, local scorer, CSV validator, fixed splits, training-only priors/RGB retrieval, the random-initialization four-view CNN, inline tests, bounded training, checkpoint/retrieval evaluation and guarded final fit/export. The three initial solution/helper/test scripts were consolidated and removed after verification; their exact source bytes/hashes are preserved in ignored `outputs/setup/pre_notebook_refactor.json`. Do not recreate external solution modules. `audit_data.py` remains the pre-existing all-question setup audit, not a notebook dependency.

The representative split has **9,492 training / 2,372 validation** locations. The spatial split has **8,716 training / 2,421 validation**, with 727 training candidates excluded by a 200-m buffer around held-out one-km blocks. All five boroughs occur in each fold. The training-only near-duplicate heuristic found zero qualifying pHash edges within 250 m; this is not proof of absent visual overlap. Frozen IDs, fingerprint and settings are in `outputs/splits.json`.

Measured controls: representative geographic prior **score 0.0112833, mean 11.2859 km**; RGB k=1 **0.0329351, 13.8749 km**; RGB k=5 **0.0343914, 11.1189 km**. Spatial prior **0.0110753, 11.0263 km**; RGB k=1 **0.0310136, 14.4029 km**; RGB k=5 **0.0318494, 11.3844 km**. Every control has GPS score zero. Keep these as controls; they do not approach the 0.90 target. Full components, borough errors, predictions, code/data/split fingerprints and runtime are recorded under `outputs/runs/` and `outputs/experiment_log.jsonl`.

Local CPU contracts and model gradient/checkpoint tests: **12 passed**. A tiny real-data MPS smoke run passed and is explicitly excluded from performance comparisons. The first actual fresh-ipykernel notebook execution completed all **16 code cells** without errors in **393.36 seconds**, including a reduced local MPS pilot (128 pixels, width 16, batch 16, workers 0, five-epoch schedule, five-minute training cap). It completed two full epochs; the partial third epoch was not selected. Conditional-only GPS-loss V1 decreased training loss but worsened validation mean error from **15.4782 to 17.2985 km**, with all borough predictions Queens. Best CNN retrieval score **0.0225482** remained below RGB k=5. Reject V1 for production fitting; retain its checkpoint/results and `outputs/runs/20261114T133917_cnn_representative/source_notebook.ipynb`.

Controlled realignment V2 supervises the final probability-weighted decoded GPS coordinate directly, retaining smaller conditional-offset and auxiliary classification losses. A decoder-only control on frozen V1 gave **11.3767 km**, still GPS zero. The matched V2 pilot completed two epochs under the same five-minute cap: **11.2975 then 11.1379 km**; second-epoch borough Macro-F1 **0.220269**, score **0.0220269**. Its best retrieval scored **0.0288518**, below RGB k=5. It is also rejected for final refitting, without claiming that this two-epoch pilot establishes fully trained performance. Its original executed notebook is preserved at `outputs/runs/20261114T135115_cnn_representative/source_notebook.ipynb`. The V2 kernel run executed 17 cells without errors in 404.57 seconds, including training and post-training encoding/evaluation outside the five-minute training cap.

A minimal supervised RGB ridge control (one predeclared penalty 100, training-fold-only normalization and targets, no search) gave representative **score 0.0368675, Macro-F1 0.368675, mean 10.4538 km** and spatial **0.0348687, 0.348687, 10.5260 km**. It improves the controls but still receives zero GPS score. Retain it as a control; no method is ready for final fitting. Representative Staten Island mean error remains 25.3931 km. Preserve per-borough diagnostics. Both validation sets have now been used for development comparisons; do not call them untouched test estimates.

Final notebook verification: fresh-ipykernel restart-and-run-all with training disabled completed **17 code cells**, **13 passing checks**, **zero error outputs**, and **two embedded figures** in **3.44 seconds**, using existing verified feature caches. Evidence is in `outputs/notebook_final_verification.json`. No notebook source was edited while its kernel was running. All active code remains inline; no active Q4 helper/test Python modules exist. New final-refit selection requires beating the measured controls with positive GPS score; neither pilot passes. The next bounded experiment is the existing V2 CNN at 224 pixels/width 32, at most 20 epochs or 60 training minutes, on an allocated HPC GPU after the course account is available. Do not schedule it on login/gateway nodes or use incomplete remote data.

Local `.venv` reuses the existing system ML libraries and adds ipykernel **7.4.0** and its dependencies; `pip check` passed. The project-local kernel is **NYCGuessr Project 4**, registered under `.venv`. Existing global packages were not uninstalled or modified. `requirements.txt` records direct versions. No externally pretrained weights were used or downloaded.

HPC facts: the user's authenticated SSH multiplex connections reach `gw.hpc.nyu.edu` and **`login.torch.hpc.nyu.edu`**; Greene refused connections. Torch queue was empty. The sole reported association was `users/normal`, but `sbatch` rejected `users` because Torch now requires a project account such as `torch_pr_…`. No CPU/GPU job was submitted. A login-node package installer was killed; do not retry the large install there. Use an appropriately allocated setup job once the account is available. `/scratch/rf2887/nycguessr-project4` is the newly created isolated remote workspace; environment and raw-file staging are **incomplete**. The transfer was paused, retaining copied files, while project allocation is unresolved. Reconcile remote files against the source manifest before use. NYU's project portal requires VPN; the user has been directed to the documented portal/course allocation process. No staff messages were sent. `outputs/hpc_status.json` records the limitation.

## 13. Small-batch local continuation — 2026-11-14

Predeclared hypothesis: two completed epochs were insufficient for learning; a reduced training fold permits more completed epochs within a small local budget. `training_subset` selects a deterministic borough-proportional 2,048-location subset solely from the fixed representative training IDs, preserving all four views. Validation remains the entire 2,372-location fold. The CNN starts fresh with the existing V2 objective, width 16, 128-pixel inputs, batch eight, seed 6643, no external weights, epoch cap 20, eight-minute timer and patience six. Training IDs are stored in checkpoints. Matching geographic-prior, RGB k=5 and RGB ridge controls use precisely these training IDs. Resource limits and subset status are explicit; reduced-fold checkpoints cannot enable final refitting.

The reduced-fold run `20261114T143039_cnn_representative_local2048` completed **11 epochs** in **480.09 seconds**; epoch 12 was partial and not scored. Best composite checkpoint was epoch 10: head **score 0.0324703, Macro-F1 0.324703, mean/median/p90 GPS 10.4212/9.23994/18.1319 km**. Learned k=5 retrieval scored **0.0333419**, mean 11.3253 km. Matching 2,048-location RGB ridge gave **score 0.0352134, Macro-F1 0.352134, mean 10.9142 km**. The head improves GPS over that matched control but loses on composite score; all these GPS scores are zero. Preserve the checkpoint; it cannot authorize a final refit. Representative full-fold RGB ridge remains the best measured composite control (0.0368675).

A fixed-penalty ridge decoder on frozen CNN features gave **score 0.0337745, Macro-F1 0.337745, mean 10.4979 km**, failing to improve on the CNN head's GPS. It is an analysis-only diagnostic, not wired into final export. Training-head error was **10.1803 km**, close to validation 10.4212 km, suggesting underfitting; this is an interpretation, not proof. The selected checkpoint still misses Staten Island entirely (F1 zero, mean error 28.1467 km). Small feature vectors and training-only fitted ridge parameters are retained under the run's `feature_calibration_20261114T150608/`.

A separately tagged training-only tiny-fit diagnostic used 40 training locations, image size 64, width eight, eight cells, batch eight, no dropout/augmentation, learning rate 0.003, max 400 updates/two minutes. It completed **400 updates in 20.80 seconds**: training mean GPS **12.4461 → 0.837336 km**, training Macro-F1 **1.0**, final training score **0.924640**. **This is memorization, not validation, and does not meet the leaderboard target.** It demonstrates learnability of this implementation on a tiny set. Its run is `20261114T150754_tiny_fit_diagnostic`; config `smoke=true` excludes it from model comparisons and selection. No diagnostic weights are exported for prediction.

Actual fresh-ipykernel executions: the main local continuation ran **17 cells without errors in 521.76 seconds** including post-training comparisons; frozen-feature diagnostic **36.41 seconds**; tiny-fit diagnostic **24.38 seconds**. The inline suite now has **14 passing checks**, including deterministic proportional training-subset membership. Original executed notebook snapshots are retained inside their ignored run folders; the single active solution notebook remains `NYCGuessr.ipynb`. Latest plots/results are embedded in that notebook. Final training-disabled restart-and-run-all passed all **17 code cells and 14 tests**, with **zero errors and three embedded plots**; see `outputs/notebook_local_final_verification.json`.

The next proposed local comparison keeps the same 2,048 training IDs, architecture and objective, but evaluates every few epochs to spend more of a fixed small budget on optimization. Predeclare that evaluation schedule and budget before running; preserve the full validation fold. HPC remains unresolved while the user works with the TA; no new HPC actions were taken. No raw data, split IDs, dependencies or Kaggle submissions changed in this continuation.

## 14. More-data local experiment — 2026-11-14

Predeclared staged-training experiment: use all **9,492 representative training locations** instead of 2,048, holding the **2,372 validation locations** fixed. Warm-start from `outputs/runs/20261114T143039_cnn_representative_local2048/best.pth` (source epoch 10), retain its training-derived geographic cells/origin, width 16, image size 128, batch 16, fresh AdamW at 0.0005, cosine horizon 12 further epochs, patience four, maximum 15 minutes plus final comparison overhead. No external weights are used. All source checkpoint training IDs must be a subset of the target training fold, with matching data/split/checkpoint identity. The new test verifies rejection of an out-of-fold initialization and preservation of learned parameters/geography.

Do not attribute any gain solely to more data: this run also supplies more optimizer updates and changes the batch and learning-rate schedule. Per-batch progress is saved every 128 batches and at epoch boundaries. Final refitting of a selected staged model must reuse the same initialization bytes and learning-rate horizon; the notebook now records and validates both rather than silently restarting randomly. Completed run: **`20261114T153649_cnn_representative_warm`**, seven full epochs in **900.02 seconds**; partial epoch eight was not scored. The first positive held-out GPS score occurred at warm epoch six, which was selected: **score 0.0462390, Macro-F1 0.381855, GPS score 0.00894838, mean/median/p90 9.91052/8.3117/18.2430 km**. It beats the previous full-fold RGB ridge composite **0.0368675**. Learned k=1/k=5 retrieval scored **0.0349749 / 0.0362525**, mean **13.5960 / 10.7414 km**; retain the head. A fixed frozen-feature ridge comparison scored **0.0375012**, mean **10.0699 km**; reject it as an improvement. No CNN updates were made during calibration.

Deterministic training-head error **9.78831 km** remains close to held-out **9.91052 km**. Staten Island has zero F1 in training and validation, with errors **27.8534 / 28.0964 km** despite 669 training examples. These observations suggest underfitting but do not establish a cause. The next proposed controlled comparison strengthens borough auxiliary supervision while preserving GPS loss/full-fold IDs and includes an equal-budget continued-training control. Predeclare weight, initialization and compute budget; do not change multiple unexplained model choices.

Actual fresh-ipykernel executions: full training and method comparison **17 cells, zero errors, 986.58 seconds**; frozen-feature comparison **17 cells, zero errors, 82.45 seconds**. The inline suite now has **15 passing checks**. Original executed notebook snapshots and all source/config/checkpoint identities are preserved in the ignored run folders; all active solution code remains in one notebook. Final training-disabled restart-and-run-all passed **17 code cells and 15 checks**, **zero errors**, with **three embedded plots**; evidence: `outputs/notebook_more_data_final_verification.json`. No raw data, validation IDs, dependencies, HPC state or submissions changed. The 0.90 target is unmet; no spatial CNN stress fit has been run. `selection.json` clears the minimum positive-GPS/control gate, which is not a claim of competitive performance or permission to upload. Final refitting remains disabled and untested on CUDA.

## 15. Permitted ImageNet transfer learning — 2026-11-14

User-reported course clarification explicitly permits ImageNet-pretrained weights; exact answer and scope are recorded above and in `DATA_SOURCES.json`. No independent staff message was inspected. The user then requested continuation. The current experiment uses official **Torchvision ResNet18 IMAGENET1K_V1**, not other pretraining data or geolocation-specific weights. Torchvision **0.26.0** is installed only in the existing project venv, compatible with torch 2.11.0; `pip check` passed. Required dependencies are pinned. Downloads remain disabled by default in the notebook and, when explicitly enabled for this run, are limited to the single official checkpoint with size/time/hash checks.

Predeclared compute-saving workflow: freeze encoder and BatchNorm in evaluation mode, extract `[N,4,512]` float32 vectors once from all 11,864 labeled locations, and cache approximately 93 MiB of vectors with strict input/model/preprocessing/version/source identity. No label-dependent preprocessing is fitted during extraction. Keep the official 256-resize/224-center-crop/ImageNet normalization and compass ordering. Process batch eight, lazy image reads, 20-minute per-invocation extraction cap and resumable partial caches. No test-image extraction or submission is needed.

On each existing representative/spatial split, independently fit (a) ridge with penalty 100; (b) k=5 retrieval; (c) a small nonlinear four-view head with the existing decoded-GPS objective, training-only normalization/geography, batch 256, learning rate 0.001, dropout 0.2, 100-epoch/three-minute cap and patience 12. This initial transfer stage trains new heads, not backbone weights. It is followed by selective backbone fine-tuning only if a measured need justifies its additional cost. Do not share Q4-fitted heads/statistics across splits. All active code/tests remain inline; scratch-model parameter names/math are preserved while its decoder is shared with the feature head. New tests cover frozen-head gradients/view order, cache identity and feature/label alignment.

Measured frozen transfer results: extraction **168.87 s**, **70.25 locations/s**, **97,190,016-byte (~92.69 MiB)** cache. Representative nonlinear head `20261114T165241_imagenet_head_representative` selected **epoch 7**, stopped after 19 in **9.70 s**: **score 0.125677, Macro-F1 0.503440, GPS score 0.0837031, mean/median/p90 9.16297/7.22868/18.94064 km**. Independently fitted spatial head `20261114T165251_imagenet_head_spatial` selected **epoch 2**, stopped after 14 in **6.73 s**: **score 0.110306, Macro-F1 0.476462, mean/median/p90 9.30378/7.84433/18.09371 km**. Canonical method rows are in `outputs/imagenet_transfer_metrics.csv`. Both folds' cached rerun, including baselines, completed **20 cells without errors in 22.12 s**. Linear/retrieval transfer baselines were lower. Retain the frozen nonlinear head; source .ipynb snapshots preserve exact executed code.

Two preflight/runtime issues were corrected before retaining the method: IPython class source cannot be inspected like a file-backed class, so cache identity uses its methods' source; and `torch.__version__` can be a TorchVersion subclass, so checkpoint metadata is normalized to JSON primitives before safe weights-only loading. Added regression checks and reran using the already verified cache; no repeated image extraction was needed. The first superseded trial retains its measured metrics and explicit postprocessing-failure note.

Selective adaptation comparison (`20261114T165759_…_representative`): both methods started from the same epoch-7 frozen head, kept complete representative fold IDs, fixed BatchNorm running statistics, official unaugmented inputs, geography and normalization; batch eight, head LR 1e-4, last residual block LR 1e-5, one epoch/five-minute cap. Head-only cached control: **6.74 s training / 7.10 s total**, **score 0.113385**, mean **9.29377 km**. Last-block adaptation: **154.61 s training / 190.30 s total**, **score 0.113833**, mean **9.29867 km**. Both lose to frozen-head 0.125677; reject this pilot, without claiming every fine-tuning schedule fails. Actual tensor audit verified 94 unchanged encoder tensors; only the final block's six parameter tensors changed, BatchNorm running statistics and training-derived head buffers remained unchanged. No adapted encoder consumed a frozen-feature cache.

The notebook's suite now contains **21 checks**, including selective gradient boundaries, safe checkpoint round-trip and a 100-location synthetic all-label refit preserving the cosine horizon. The last selective-trial kernel executed **22 cells without errors in 202.06 s**. The final retained method remains frozen ImageNet plus the lightweight head. A disabled guarded path can refit on all 11,864 labels and export a template-validated CSV using the same fixed encoder. This actual final refit/test extraction/CSV has **not** been run; only the synthetic refit contract was exercised. A strict separate loader exists for rejected adapted checkpoints. Next proposed controlled study: borough-conditioned GPS heads using cached vectors, focusing on persistent Staten Island error (representative F1 **0.32484**, mean **24.1493 km**). No new image-encoder pass is needed for that study. Do not edit notebook source while its kernel runs. External imagery, coordinates, maps, other pretraining datasets, HPC work and Kaggle upload remain outside this task's authorization.

Final transfer-notebook verification: fresh-kernel restart-and-run-all with real training/download/refit/export disabled completed **22 code cells**, **21 checks**, **zero errors**, **five embedded figures** in **5.09 seconds**. Actual adapted-model strict reload and parameter audit passed. Verified current extraction source still produces the exact existing cache key/hash, both heads match their frozen split IDs, archived source hashes match their logged runs, and raw label/split/manifest hashes remain unchanged. No actual final transfer refit, test feature extraction, test CSV, Kaggle submission, HPC action, commit or push occurred. Evidence: `outputs/notebook_imagenet_final_verification.json`.

## 16. Requested external code reference — access audit, 2026-11-14

The user requested using code/concepts from **https://github.com/justgoofingaround/nyc-guessr**. Read-only checks returned **404** through the connected GitHub repository/contents APIs and direct public GitHub page/repository API. The web reader could not fetch the page and an exact repository-name search returned no results. A signed-in browser fallback was unavailable because computer-use surfaces are not enabled in this session. These observations do not establish whether the repository is private, renamed, removed or inaccessible to the connected account.

No README, instructions, license, source tree, commit, code, model, data or reported result could be inspected. Do not invent its architecture, conventions, performance or license, or claim our existing methods came from it. No implementation or model change was made from this unavailable source. Access clarification has been requested. Once source access is available, inspect the license, actual code, preprocessing, validation and data/weight provenance before adapting attributable code/concepts into the single active notebook. The existing permitted ImageNet scope and external-data ban still apply; a code-reference request does not authorize importing a repository's imagery, coordinates, trained geolocation weights, test predictions or labels. Preserve the current verified transfer baseline while reviewing the reference.

Access evidence is recorded in `DATA_SOURCES.json` under `reference_code_repositories`. Current notebook/model/feature-cache results remain unchanged. No new training, dataset/model downloads, installs, submissions or external messages occurred for this review attempt.

### Reference access resolved and adaptation plan — 2026-11-15

The user clarified they are a collaborator. Existing local Git credential-helper authentication successfully read HEAD **`2d38aefe854e56e3be3b9d5c8de6c66228d02753`**. The earlier access failures are historical; do not continue treating the source as unavailable. A shallow blob-filtered bare metadata clone is in temporary storage (location recorded under ignored `outputs/setup/`), with only ten selected text source files fetched. No submission CSV, dataset, weight, feature file or notebook output was fetched. Source text/SHA identities are archived in `outputs/setup/nyc_guessr_reference_source.json`. The inspected tree has no license file; independently implement the concepts and attribute the reference, rather than transplanting entire source blocks.

Verified sources: README (title only), `CLAUDE.md`, `geo_heads.py`, `geo_features.py`, CV/prediction scripts and notebook builders. Useful concepts: compass/token-aware transformer fusion; distance-smoothed cell targets; training-only nearby-positive contrastive learning; weighted local-plane geometric median of predicted cells and retrieved training coordinates; geographic borough consistency; diverse cell/seed ensembles; global plus pooled spatial tokens. CLAUDE's reported CV/leaderboard values are **source-reported**, not independently verified or reproduced here. Its strongest backbones (StreetCLIP, DINOv2/v3, Perception Encoder, SigLIP) exceed the currently confirmed ImageNet-only scope; clarification has been requested while permitted head/decoder adaptation proceeds. Do not import its predictions, data, trained geolocation weights, coordinate constants or grid claims as answers.

Local corrections: keep the existing fixed representative/spatial splits, our Haversine scorer and training-fold-only origin/cells/normalization. The reference CV script fits feature statistics/PCA before splitting and uses a different random K-fold split; do not copy that preprocessing into our validation. Use MPS/CPU-portable code, chunked retrieval, bounded median iteration and safe handling of batches with no geographic positives. Do not adopt its hardcoded Windows/CUDA paths, silent coordinate clipping, broad decode search, automatic test inference or submission writing.

Initial predeclared experiment on the existing approved ResNet18 global cache: small direction-aware transformer (width 128, one layer/four heads), 512 train-derived cells, distance-smoothed targets at 1 km, borough CE weight 0.3, nearby-positive contrastive loss weight 1 / radius 0.15 km / temperature 0.05, input dropout 0.3 and whole-view dropout 0.15. Batch 256 as 128 anchors plus sampled training neighbours; AdamW LR 0.001 / weight decay 0.05 / 40-epoch OneCycle schedule, at most five minutes per fold. Evaluate every five completed epochs. Compare four fixed decoders on the same model: cell mean; cell median; cell+32-NN median (50% neighbour mass, temperature 0.05); the same mixed coordinates with borough votes from 15 nearest training coordinates. A raw-feature geometric-median neighbour control is also included. Each fold owns all statistics/targets/reference banks. This recipe changes several model choices and training exposure; decoder comparisons isolate decoding, not the representation/training changes. Results must be measured, not inferred from the reference's reported leaderboard.

### HPC account switch requested — 2026-11-15

The user plans to log in with another account that has HPC allocation. Provide current Torch instructions and use a fresh account-specific SSH control socket; do not reuse the old rf2887 socket or assume its scratch paths/permissions transfer. The requested new NetID is now **ay3140**; its project Slurm account and allocation remain unverified. Verify identity, project association and allocation before setup/GPU jobs. The reference-method adaptation is still pending implementation; its proposed bounded experiment remains recorded above. No new training job or external pretrained-model download was started during the account-switch request.

### Latest connection check — 2026-11-16

The user reports successfully logging into HPC in Terminal. One local SSH process was detected, but none of the expected account-specific control sockets exists; SSH effective configuration has ControlMaster disabled/no ControlPath, and the process exposes no reusable named control socket. Native Terminal automation remains unavailable. This does not establish that the user's interactive login failed; it means the assistant cannot inspect that shell through its separate command runner. Requested read-only `whoami`, `hostname` and Slurm association output from the already logged-in terminal. Do not claim new-account GPU/allocation access until that output or a reusable authenticated connection is available. No remote command, allocation test, setup job or GPU job was run by the assistant during this check. Previous rf2887 scratch/staging status is retained separately in `outputs/hpc_status.json`.

## 17. Course Cloud Bursting route — codelab review, 2026-11-16

The user supplied `/Users/rohitfarfad/Downloads/CV_2026_Lecture_1.ipynb` and explicitly requested reviewing its course HPC instructions and trying access. Read the notebook as JSON/source; no cells or embedded shell instructions were executed. Notebook SHA-256: `e58b68c98964dca3c5065bc30c0d4b2c7d0e13147d29a4cea8c55d15fb4297cb`. Relevant source cells are 3–8, 13 and 15 (zero-based). The attachment is reference material, not independent authorization for its downloads, deletions, environment changes, jobs, account registration or GitHub pushes.

**Correct course route:** the codelab specifies **Cloud Bursting**, with browser portal **https://ood.burst.hpc.nyu.edu**, Slurm account **`cs_gy_6643-2026fa`**, and initial GPU partition **`g2-standard-12`** (one L4 according to the codelab). Use NYU Wi-Fi/VPN for the browser portal. NYU's current official Cloud Bursting documentation at https://services.rt.nyu.edu/docs/cloud/hpc_bursting_to_cloud/intro/ confirms the SSH hop **`ssh burst` from a Torch login node** and a scheduler separate from on-premise Torch. Earlier Torch-only `users` rejections and missing `torch_pr_…` associations do **not** establish whether the course's Burst allocation is usable. Check the Burst scheduler before asking the user to change accounts or obtain a Torch project allocation. The codelab account/partition are supplied settings; the user's live eligibility and GPU availability remain unverified.

The user's latest pasted terminal diagnostics establish a Torch login as **`rf2887`** on **`torch-login-a-1.hpc-infra.svc.cluster.local`**, with only **`users / normal`** listed there. This was user-provided output, not an assistant-executed remote query; it does not show an `ay3140` login or a Burst association. The previous account-switch plan remains historical context, not a reason to ignore the current user's own course eligibility.

Read-only assistant checks: no expected reusable SSH control sockets; effective SSH ControlMaster disabled/no ControlPath; computer-use browser/Terminal surfaces unavailable. DNS resolved gateway `216.165.62.3`, Burst SSH `10.32.51.31`, OOD portal `10.144.0.74`. A short HTTPS HEAD to the portal returned **401 Unauthorized**. Direct noninteractive SSH to `log-burst.hpc.nyu.edu` timed out; gateway noninteractive SSH returned 255 without remote hostname output. Checks retained strict host-key verification, disabled host-key updates and allowed no password prompts. No remote shell command or scheduler test successfully executed, no authenticated browser session was inspected, and no job was allocated/submitted. A 401 is an authentication boundary, not evidence of failed course allocation or an unreachable portal.

Requested from the user's existing Torch terminal: `ssh burst`, private authentication if prompted, then `whoami`, `hostname`, `sacctmgr -n show assoc where user="$USER" format=Account%60,Partition,QOS`, `sinfo -h -p g2-standard-12 -o '%P %a %G'`, and a two-minute one-GPU **`sbatch --test-only`** request using account `cs_gy_6643-2026fa`, partition `g2-standard-12`, one CPU, 4 GiB memory, comment `gpu_mps=yes` and wrapped `nvidia-smi -L`. Official Slurm documentation (https://slurm.schedmd.com/sbatch.html#OPT_test-only) confirms this validates without submitting a job. Its success would establish scheduler acceptance at that time, not an allocated GPU, a working CUDA/Python environment or permission for a long training run. Await that output; do not report the test as run or passed yet.

Course execution constraints: the codelab states **300 GPU hours for the class**; do not interpret this as measured remaining balance. GCP jobs may be preempted: save checkpoints frequently and support resume. Keep code/data/environments and container cache/temp files in appropriate scratch storage; back up code locally, and push only with explicit permission. Use an isolated venv or Singularity/Apptainer environment; install only in a suitably allocated setup session. NYU documentation warns that Burst compute-node storage is separate from Torch storage; verify mounts and stage/reconcile data on the actual compute environment rather than assuming earlier `/scratch/rf2887/nycguessr-project4` contents are visible. Keep the single self-contained Q4 notebook requirement.

Do not blindly execute the codelab's setup example: it contains an instructor-specific deletion path, inconsistent overlay/gunzip paths, lowercase `$user` in the job example, certificate-check bypass, a questionable `PYTHONPATH` assignment and historical package versions. Its linked Singularity-with-conda page returned 404 during this review. Correct/verify only the setup actually needed after access is established. No container, overlay, package, dataset or model was downloaded/installed; no raw input, notebook, environment or SSH configuration was changed. Status evidence is retained in ignored `outputs/hpc_status.json`.

### Follow-up terminal output — Torch-only dry run, 2026-11-16

The user pasted diagnostics and the course-account `sbatch --test-only` result, but `hostname` remained `torch-login-a-1.hpc-infra.svc.cluster.local` and the association was still `users / normal`. Torch rejected `cs_gy_6643-2026fa` and directed the user toward Torch project accounts. No `ssh burst` output or changed host was provided; this is a failed **Torch** dry run, not a Burst eligibility test. No job was submitted. The next single action is `ssh burst` from the existing Torch shell, followed by `hostname` only after the connection succeeds; request the SSH error if it fails. Avoid repeating scheduler checks until the target host is confirmed. Evidence recorded in `outputs/hpc_status.json`.

### Documented Burst hop unreachable — 2026-11-16

The user then executed `ssh burst` from Torch and received **`ssh: connect to host burst port 22: No route to host`**; subsequent hostname remained `torch-login-a-1.hpc-infra.svc.cluster.local`. This is a pre-authentication connectivity failure, not a rejected password, course-account test or confirmed allocation problem. The underlying cause (routing/firewall, unavailable service, or endpoint configuration) is unknown. Rechecked NYU's official Cloud Bursting page: it still documents the Torch-to-`ssh burst` hop; no verified replacement endpoint was found. Do not keep retrying the identical failing hop, guess IPs, bypass SSH checks, or ask the user to replace accounts based on this error. The next single check is the codelab's browser route **https://ood.burst.hpc.nyu.edu** from the user's Mac on NYU Wi-Fi/VPN, with private sign-in; await dashboard/error status. If that also fails, the course TA/HPC support must confirm the current endpoint/service and course access; no staff message is authorized or sent. No remote session, GPU allocation, job, install or training was started. Updated evidence is in `outputs/hpc_status.json`.

The user subsequently reported **"Portal shows an error or cannot open"**; the exact browser error and confirmed NYU VPN state are still pending. A read-only Mac follow-up found no named connections in `scutil --nc list`, but routes to both Burst private addresses use `utun6` for `10.0.0.0`; the tunnel provider is unidentified, so this does not establish whether NYU VPN is connected. Another HTTPS HEAD returned 401. Requested the exact browser error and the VPN app's connection state without credentials or token-bearing URLs. Both documented access routes remain unsuccessful, but no cluster-wide outage, allocation failure or root cause is confirmed. Avoid repeated identical network probes while awaiting this information; if needed, provide a concrete report for the user to send to course staff/HPC rather than contacting them without authorization.

### VPN confirmed by user; staff confirmation needed — 2026-11-16

The user confirms **"it doesnt open, also i am connected to nyu vpn"**. Treat VPN connection as user-reported confirmation; do not ask the same question again. The precise browser error remains unspecified, but both supplied access routes have failed: Torch-to-Burst SSH reports `No route to host` before authentication and the course OOD portal does not load for the user on NYU VPN. Root cause, any outage, endpoint migration and course allocation status remain unconfirmed. Stop repeating identical client/SSH checks. The next action is for the user to send a concise report to the course TA or **hpc@nyu.edu**, asking for the current course compute endpoint/service status and confirmation that `rf2887` has the `cs_gy_6643-2026fa` allocation; if the course now uses Torch, request its valid project Slurm account. A message may be prepared in the assistant response, but **no external message is authorized or sent**. Keep existing local work intact and retain HPC staging as incomplete; no training/install/transfer/scheduler job starts during this access check. Updated structured status is in `outputs/hpc_status.json`.

### Direct gateway-to-Burst route also failed — 2026-11-16

The user asked whether Burst can be reached without Torch. Current NYU documentation describes Torch-to-Burst; a Spring 2025 NYU course guide (https://nyu-cs2590.github.io/spring2025/HPC-tutorial/) also documents the full hostname `log-burst.hpc.nyu.edu`. Treat that older source as a documented hostname, not proof of current service availability. The user successfully logged into gateway `pco02la-2289b` as `rf2887` using the trusted ECDSA algorithm with strict host-key checking, then ran `ssh -o ConnectTimeout=10 rf2887@log-burst.hpc.nyu.edu` **from the gateway**. It returned **`Connection timed out`** before Burst authentication. Thus both tested SSH routes (Torch-to-Burst and gateway-to-Burst) failed, and the user has also reported that the OOD portal does not open on NYU VPN. Root cause, current service status, endpoint migration and allocation eligibility remain unverified. Do not repeat the same connection attempts or credential/account-switch requests without new evidence. Update the prepared TA/HPC support report with this direct-gateway failure and request the current endpoint/service and course allocation details. No external message was authorized or sent, and no job/install/training/raw-data change occurred. `outputs/hpc_status.json` preserves separate user-reported and assistant-observed evidence; do not archive authentication prompts or personal connection details.



## 18. Full-data, fewer-epoch local authorization — 2026-11-16

The user requested **"Let's do the training on local gpu meanwhile, full training with all data but less ephochs"**. Use the local MPS GPU while HPC is unavailable. All active implementation/tests stay in `NYCGuessr.ipynb`. Predeclared experiment applies the previously reviewed reference concepts to the existing verified frozen ResNet18 cache: 20 epochs rather than 40, five-minute per-fold cap, every fixed training location as an anchor once per completed epoch, with paired training-only neighbors. Architecture/loss/decoders remain as declared in section 16 (width 128, one layer/four heads, 512 training-derived cells, distance-smoothed targets and contrastive positives). Keep all 2,372 representative and 2,421 spatial validation locations held out; spatial buffer exclusions remain enforced. Fit normalization/cells/reference banks independently per fold. Reject equal-ID contrastive pairs and handle no-positive batches safely; bound all similarity/median computations. No new weights, dependencies or image passes are needed. This changes several method choices and is development evidence, not an isolated architecture ablation.

After comparison, refit the retained representative method on **all 11,864 competition training labels**, using its selected epoch count and original schedule horizon, capped at five minutes. This all-label fit has no held-out score and does not overwrite the fold checkpoints. If the new method loses, retain the earlier validated ResNet18 head and use its guarded all-label refit. Real extraction/download/training/refit flags remain disabled by default; enabling them for this invocation is authorized. No test inference, CSV export, Kaggle submission, HPC action or external message is requested. Verify contracts, fresh-kernel execution, exact training/validation IDs, checkpoint round trips and logs; document measured retain/reject decisions before ending the task.


### Measured local20 result and completed all-label fit — 2026-11-16

Implemented the reference concepts independently in the single active notebook: 266,757-parameter compass-aware Transformer head on the unchanged approved ResNet18 cache, 512 training-derived soft geographic cells, duplicate-ID-safe geographic contrastive learning, paired training-only neighbors, four fixed decoders and a raw-neighbor median control. All 20 epochs completed on MPS: representative `20261116T131621_compass_local20_representative` **31.8268 s**; spatial `20261116T131653_compass_local20_spatial` **25.8124 s**. No new encoder pass, model/data download or dependency install was needed. The 20-epoch limit is lower than the previously proposed 40; model changes are not an isolated architecture ablation.

Representative selected epoch 20 / `mixed_median`: **score 0.173473034**, borough Macro-F1 **0.547140954**, GPS **0.131954376**, mean/median/p90 **8.680456/5.816619/19.250590 km**. Previous frozen head: **0.125676822**, mean **9.162969 km**. Retain the new development method while preserving older checkpoints. P90 worsened from 18.940638 km, and Staten Island remains poor (mean **23.200414 km**, F1 **0.408377**); do not describe every error measure as improved.

Independently fitted spatial weights at epoch 20 with the **same mixed decoder**: score **0.139981661**, Macro-F1 **0.534745422**, GPS **0.096119021**, mean/median/p90 **9.038810/6.919776/19.303849 km**. Previous spatial frozen head: **0.110306138**, mean **9.303778 km**, p90 18.093708 km. Spatial's separately selected best decoder is `cell_mean`: score **0.144900231**, mean **8.984159 km**. Preserve the distinction; the best spatial decoder is not the production mixed decoder. Both validation sets are reused development data, including epoch/decoder selection; neither is a fresh test estimate. All validation and spatial-buffer IDs remained outside each fold's statistics/targets/reference banks.

Actual fresh all-label refit `20261116T131721_compass_local20_full` completed **20 epochs on all 11,864 labels in 26.5232 s**, using a fresh head, all-label normalization/cells and the selected mixed decoder. Checkpoint **`outputs/runs/20261116T131721_compass_local20_full/best.pth`**, SHA-256 **`8a763ed6d46247e0329859a8fb130052ab206d1609205757231dc761597f7609`**. `fit_complete=true`, all training IDs covered every epoch, no validation IDs/metrics in this full fit. This is frozen-encoder head training, not full-backbone fine-tuning. No leaderboard score is known.

The actual training restart-and-run-all completed **24 code cells / 25 checks / zero errors in 91.8238 s**. Subsequent harmless fixes removed read-only-array/scalar-conversion warnings and embedded plots; an added read-only artifact audit checks strict actual checkpoint reloads, exact fold IDs/exclusions, fitted normalization/origin, feature hashes, original run/source hashes, scored CSV reproduction and all-label coverage/completion. Final fresh-kernel execution with real training/download/refit/export disabled passed **25 code cells / 25 checks / zero errors / seven embedded figures in 4.7079 s**; both new figures were visually inspected. Evidence: `outputs/notebook_local20_training_execution.json`, `outputs/notebook_local20_final_verification.json`, `outputs/local20_artifact_audit.json`, `outputs/local20_metrics.csv`, `outputs/local20_common_decoder_metrics.csv`, `outputs/local_all_labels_fit.json`, per-run histories/configs/checkpoints/source archives and the existing experiment log. Raw label/split/manifest SHA values remain unchanged. Peak RAM/VRAM was not measured.

All real training/download flags default off. No test-image extraction, test prediction CSV, Kaggle upload, HPC job, external message, commit or push occurred. Validation prediction CSVs were saved as experiment artifacts. The new full Compass checkpoint is **not yet wired to the legacy submission exporter**; later test inference must distinguish train/test namespaces because their numeric sample IDs overlap (the current validation decoder requires disjoint IDs). The next proposed controlled study is a stronger permitted pretrained encoder under a measured bounded local extraction budget, compared against the retained method; this is not an automatically authorized new download/run. The 0.90 leaderboard target is still far from achieved.


## 19. StreetCLIP comparison authorization — 2026-11-16

The user accepted the proposed frozen StreetCLIP benchmark/comparison with **"Let's try it"** and selected **"Up to 1 hour (recommended)"** for local GPU feature extraction. This authorizes the necessary project-venv dependency and official pretrained StreetCLIP image-encoder checkpoint download, a 128-location/512-image local MPS benchmark, bounded resumable extraction from competition training images only, and the same 20-epoch compass-head comparison on the fixed representative/spatial splits if extraction fits the budget. No external imagery/text/geospatial reference data, test pass, Kaggle submission, HPC work or staff message is authorized. Only the public encoder weights/configuration/model card may be obtained; do not download repository demo images or tokenizer files. Keep all active code/tests/benchmark/progress in `NYCGuessr.ipynb`; provenance and exact versions belong in existing project records.

Pin official StreetCLIP repository `geolocal/StreetCLIP` revision **`e3561ba2ad9bf14c9efd6b0092607b8497efbfea`**. Official API reports checkpoint `pytorch_model.bin` **1,711,981,793 bytes**, SHA-256 **`cf6dc3802a8bf9301560b1aa0cd1fa983b4139f96a1befc43802e387401fe6c0`**. Vision encoder is ViT-L/14, width 1024, 24 layers, 336-pixel preprocessing; use its official RGB resize/center crop/normalization. Load with safe weights-only semantics and retain only vision tensors; no geolocation head from another project. Check resource use on tiny batches before increasing. Cache global image representations `[11864,4,1024]` (float16 about 93 MiB), preserving direction/ID order and complete model/preprocess/source hashes. Start with global features; spatial-token extraction/ensembles are later hypotheses, not part of this one-factor encoder comparison.

Keep the hidden head width/loss/cells/epoch budget/decoders/splits fixed while allowing its input projection to accept 1024 rather than 512 channels; record the resulting small parameter-count difference. Preserve strict legacy checkpoint loads. If the measured full pass exceeds the one-hour cap, use resumable partial storage and report the limit rather than silently running beyond it or fitting incomplete/full-fold claims. All repeated benchmark/extraction passes count toward the authorized GPU extraction budget. Do not edit notebook source while its kernel runs.


### StreetCLIP benchmark and budget realignment — 2026-11-16

The pinned official checkpoint downloaded and passed its size/SHA checks. Strict vision-only loading uses 303,507,456 frozen parameters. Transformers 5.23 flattens the vision-model prefix and creates nonpersistent position buffers on the meta device; the loader strictly converts the official prefix and reconstructs/validates only the deterministic position buffer. Tiny model round-trip and explicit official PIL-preprocessing parity tests pass. The default modern Torchvision image backend has slightly different boundary rounding, so the parity check explicitly uses the official PIL backend. Project-venv installation/pip check passed; global packages were preserved.

The 128-location/512-image MPS benchmark took **53.5073 s**, measured **9.697 images/s**, sampled peak driver allocation **2,230,042,624 bytes**. Larger batches: four locations **9.783 images/s**, eight **9.672 images/s**; no useful throughput gain. Benchmark aggregate GPU extraction time **85.3902 s**. A complete native-resolution pass projects beyond the one-hour limit, so retain all four views/native 336 preprocessing and reduce training scope, with the user informed. Predeclare a **6,500-location** feature set: union of both original validation sets (**4,286 IDs**) plus **2,214** deterministic borough-proportional locations drawn only from the intersection of both training folds. Each encoder/head uses precisely the same available original training IDs and full original validation IDs; keep spatial buffer exclusions. Persist IDs before extraction. This is a matched reduced-training experiment, not a complete StreetCLIP training-data cache or final all-label fit. The existing full-data 128-row pilot remains separate.

Reuse identical head width/loss/20 epochs/hyperparameters and a fixed `mixed_median` selection decoder for both encoders; permit only the required input projection dimension change (512 to 1024), recorded in config/parameter counts. Generalized heads retain strict legacy checkpoint compatibility. Convert fp16 cache rows to float32 before mean/std fitting to avoid float16 accumulation overflow. Cache a complete selected subset before fitting anything; if the one-hour limit interrupts extraction, retain partial data and do not claim a completed comparison. No new final all-label refit or test inference is part of this reduced comparison.


### StreetCLIP comparison completed and retained — 2026-11-16

Completed the entire predeclared **6,500-location** subset cache, preserving both complete validation sets. Feature file `outputs/features/streetclip_train_1d2ae8ae6d357433.npy`, **53,248,128 bytes**, float16 `[6500,4,1024]`, SHA-256 **`f80063c6a2261611e03853a55bf4641843c4307a798ad56d9a08ff176c791811`**. Main native-336 image pass **2,670.4811 s (44.51 min)** at about **9.74 images/s**; cumulative benchmarks/warm-ups/extraction **2,755.8706 s (45.93 min)**, below the selected **3,600 s** limit. Main-pass sampled MPS driver maximum **2,230,042,624 bytes (2.08 GiB)**; this is not an exact whole-job RAM/VRAM peak. The separate 128-row full-data pilot remains incomplete and cannot support all-label fitting.

Both models trained independently for **20 epochs** with the exact same **4,128 representative / 3,926 spatial training IDs**, original **2,372 / 2,421 validation IDs**, fixed `mixed_median` selection decoder, losses, width, cells and optimizer settings. New heads accept either 512 or 1024 channels; input projection capacity differs (**266,757 ResNet18 / 333,317 StreetCLIP parameters**), explicitly documented. No fold receives another fold's fitted statistics/weights/labels. ResNet18 caches and all older checkpoints remain preserved.

Representative matched ResNet18 `20261116T142831_compass_resnet18_matched_representative`: **score 0.103118922**, Macro-F1 **0.505207946**, GPS **0.058442364**, mean/median/p90 **9.415576/7.006121/20.140576 km**, **16.6361 s**. StreetCLIP `20261116T142848_compass_streetclip_representative`: **score 0.425714756**, Macro-F1 **0.692864261**, GPS **0.396031478**, mean/median/p90 **6.039685/3.314792/15.676792 km**, **12.8128 s**.

Spatial matched ResNet18 `20261116T142901_compass_resnet18_matched_spatial`: **score 0.095325705**, Macro-F1 **0.485685579**, GPS **0.051952386**, mean/median/p90 **9.480476/7.676750/19.177880 km**, **12.9913 s**. StreetCLIP `20261116T142914_compass_streetclip_spatial`: **score 0.361942210**, Macro-F1 **0.672840392**, GPS **0.327397968**, mean/median/p90 **6.726020/4.247733/15.635081 km**, **12.0165 s**. Head timings include their raw controls/evaluation. Raw-feature StreetCLIP neighbor controls score **0.267331692 / 0.227471371**, below learned heads. Retain StreetCLIP plus the trained head as the best validated candidate. Geographic borough-vote diagnostics are slightly higher (**0.429154961 / 0.364532817**) but were not the predeclared selection decoder; keep those distinct.

Representative StreetCLIP Staten Island error remains **10.7756 km**; all borough errors improved versus matched controls, but this is still far from the 0.90 target. Both reused validation sets are development estimates. The earlier all-training-fold ResNet18 Compass score **0.173473** is historical context, not the matched encoder control. StreetCLIP has **not** received a complete 11,864-label refit. Preserve that distinction in reporting and export guards.

Actual comparison notebook executed **29 code cells / 28 checks / zero errors in 2,735.1697 s**. A harmless read-only pandas ID warning was fixed afterward without changing model math or stored weights. An additional read-only audit strictly loaded every actual checkpoint, regenerated its predictions from verified caches, reproduced saved CSV metrics, confirmed matched training/full-validation/buffer IDs, training-only mean/std/origin and all 20-epoch coverage, and verified original run/source hashes and the cumulative extraction limit. Final training/extraction-disabled fresh-kernel execution passed **30 code cells / 28 checks / zero errors / nine embedded figures in 10.6462 s**. Both comparison figures were visually inspected. Later Markdown/provenance updates were checked against the unchanged verified executable hash **`df7a790618ca6cfd70b125a6d905d3afeaac5189be06c430a6e422b803af6d62`**.

Evidence: `outputs/streetclip_comparison_metrics.csv`, `outputs/streetclip_raw_control_metrics.csv`, `outputs/streetclip_borough_metrics.csv`, `outputs/streetclip_artifact_audit.json`, `outputs/notebook_streetclip_comparison_execution.json`, `outputs/notebook_streetclip_final_verification.json`, saved subset IDs, per-run histories/configs/checkpoints/source snapshots and existing experiment logs. Selected representative checkpoint **`outputs/runs/20261116T142848_compass_streetclip_representative/best.pth`**, SHA-256 **`db9c1e2eb4627bcfec7a0fcd0bcc35e6fa0d6ee584cd9e33c7b0a20af5891153`**. Project requirements pin Transformers **5.23.0**, Hub **1.33.0**, Safetensors **0.8.0**, Tokenizers **0.23.1**; `pip check` passed and global packages were preserved. Exact checkpoint/license/preprocessing/package/cache provenance is in `DATA_SOURCES.json`. Raw label/split/manifest hashes remain unchanged.

No test images/inference/CSV, Kaggle upload, HPC job, external message, commit or push occurred. All real downloads/extraction/training remain disabled by default. The legacy exporter is not wired to StreetCLIP; future test inference must preserve dataset namespaces for overlapping train/test numeric IDs. Next proposed action: reuse the 6,500-row cache and encode the remaining **5,364** labeled locations (roughly **37 more GPU minutes** at measured throughput, requiring a fresh authorized budget), then run the same 20-epoch full-fold comparison before an all-label refit. Do not restart the whole expensive feature pass or infer full-data performance from this subset result.


## 20. Complete StreetCLIP cache and full-fold comparison — 2026-11-16

The user accepted the proposed next task with **"great let's try it"**: reuse the completed 6,500-location StreetCLIP cache, encode the missing **5,364** labeled locations (estimated 37 GPU minutes), and repeat the same 20-epoch full-training-fold comparison. This is a fresh bounded local extraction phase, capped at **one additional hour**, with a distinct budget/progress record; preserve the previous phase's 2,755.87-second budget record. No new weights/dependencies/downloads, test inference, all-label final refit, Kaggle upload, HPC action or external message is needed/authorized for this task.

Preserve the exact existing feature-producing functions and pinned encoder/preprocessing/runtime identity, so the expensive 6,500-row cache remains reusable. Independently assemble a complete 11,864-row training cache using exact sample-ID mapping, immutable source-cache hash verification and a persistent coverage bitmap; process only uncovered rows with the same frozen native-336 vision encoder. Retain the historical incomplete full-data pilot rather than treating it as a committed cache or deleting it. Validate source/target identity, overlap/coverage, finite values and final hash before fitting. Checkpoint cache data, coverage and phase budget frequently; fail clearly on drift/OOM/time cap instead of training on incomplete features.

Train fresh compass heads, separately fitting normalization/cells/positive pairs/reference banks on each full original training fold: representative **9,492 train / 2,372 validation**, spatial **8,716 train / 2,421 validation / 727 buffer exclusions**. Keep 20 epochs, batch 256/128 anchors, width 128, one Transformer layer/four heads, 512 cells, original losses/optimizer/schedule and fixed `mixed_median` selection. Compare against saved reduced-training StreetCLIP and full-fold ResNet18 controls without re-running existing controls needlessly. More data also increases optimizer steps and changes fitted cells/neighbors; describe this as training-set expansion, not an equal-compute data-only causal ablation. Preserve all prior checkpoints; retain only measured improvements and report failures. All active code/tests/results stay in `NYCGuessr.ipynb`; real extraction/training flags default off.


### Full cache and full-fold training completed — 2026-11-16

The full `[11864,4,1024]` float16 StreetCLIP cache is complete: `outputs/features/streetclip_train_23f1be61d2b2b2cd.npy`, 97,190,016 bytes, SHA-256 `b9be80acd9d389f34b558321403d9f4d5ef4e54e100f31800d05426f7ae2f2e8`. Reused all 6,500 committed source rows exactly; encoded only 5,364 missing locations, with coverage and distinct budget records. This phase took **2,355.1052 s (39.25 min)**; prior phase remains **2,755.8706 s**. No feature-producing source or expensive original cache was changed.

Full-fold 20-epoch results: representative `20261116T161715_compass_streetclip_full20_representative`, **score 0.546915360**, Macro-F1 **0.783054671**, GPS **0.520677659**, mean/median/p90 **4.793223/2.269179/13.490731 km**, 9,492 train / 2,372 validation, **30.9086 s**. Spatial `20261116T161747_compass_streetclip_full20_spatial`, **score 0.421108066**, Macro-F1 **0.738175987**, GPS **0.385878297**, mean/median/p90 **6.141217/3.608386/15.000633 km**, 8,716 train / 2,421 validation / 727 excluded, **28.7272 s**. Both selected epoch 20 with unchanged mixed decoding. Retain this baseline over reduced-training scores 0.425715/0.361942. More data also changes optimization exposure and fitted cells; do not overstate a causal data-only ablation. Spatial Staten Island mean slightly worsens (11.4327 to 11.5553 km). Largest inspected representative errors are cross-borough, about 48.57/48.47 km, documented with supplied validation views.

Actual training notebook: **32 code cells / 31 checks / zero errors / 2,430.3312 s**. Added read-only actual-model, hash, coverage, exact reuse, fold/buffer, training-statistic, score and source-log audit. Fresh-kernel training/extraction-disabled verification: **33 code cells / 31 checks / zero errors / 12 figures / 15.3758 s**. All three new figures were inspected. Representative checkpoint SHA **`1aa503c12dcf506d0af8d11dbe6bbde304fcf7cd1364fea9d2029630335e7e81`**; spatial SHA **`3ee90936dcd4d43f7b1af9db59f6d28eeaf4d34ad7bdcfcea87ab42c4c3ede28`**. Evidence: `outputs/streetclip_full20_*`, `outputs/notebook_streetclip_full20_*`, and source archives/logs. No all-label supervised StreetCLIP refit, test feature extraction/predictions, submission, HPC work, download or dependency install occurred.

## 21. Continue with bounded cached-feature experiments — 2026-11-16

While final documentation was underway, the user instructed **"continue the owrk"**. Continue measured local progress toward the score goal using the completed cache, without a further expensive encoder pass or downloads. Predeclare two small, separate hypotheses: (1) soft borough consistency weighting before GPS aggregation reduces cross-borough distance outliers, holding the encoder/head and borough predictions fixed; compare alpha 1 against the exact existing alpha-0 mixed decoder on both folds, using only training-label cell/neighbor borough statistics and no hard borough gate; (2) reducing the geographic target temperature from **1 km to 0.5 km** sharpens spatial discrimination, with all other model/training/decoding settings unchanged. Run at most 20 epochs/five minutes per fold, ten minutes aggregate additional head fitting, original fixed folds/seed and no validation-label fitting. Preserve current baseline and only retain measured gains; do not automatically combine modifications or alter a checkpoint's historical metrics. All code/checks/results stay in `NYCGuessr.ipynb`, real flags default off. No all-label final fit, test inference, uploads, HPC access, external messages, commits or pushes.


The initial 0.5-km trial stopped on a nonfinite-gradient guard before any complete epoch/checkpoint. Preserve its explicit failure status. Four-step diagnostics and a longer one-epoch diagnostic (75 updates each: MPS 1 km, MPS 0.5 km, CPU 0.5 km) found no nonfinite gradients; this does not identify or rule out an intermittent backend failure. Keep the guard and add diagnostic failure records, without changing loss/model math. Before the single bounded retry, predeclare one fresh 1-km matched control and one 0.5-km fit per fold (four fits total), all under the existing ten-minute aggregate study cap. Compare against matched controls and the preserved historical baseline; do not attribute small differences to temperature alone or promise deterministic MPS execution. No automatic retries after another numerical failure.


The fresh 1-km control also failed before the first complete epoch (final paired batch 40, `project.0.weight` nonfinite; all other parameter gradients finite), so temperature is not established as the cause. Preserve `20261116T165309_compass_streetclip_tau10_repeat_representative` and stop that trial. Investigate an explicit biased-variance LayerNorm training reduction for the input feature normalization; retain the exact epsilon/learned parameters/state keys and native evaluation path. Require CPU-reference forward/input/affine-gradient parity on full and both tail batch sizes (256/40/24), on CPU and MPS when available. The local failure implicates that gradient but does not prove a PyTorch backend defect. After passing, predeclare a new matched 1-km versus 0.5-km comparison under this numerical repair, separately tagged, within the existing ten-minute head-study budget; compare each control with the historical retained baseline. Do not disable nonfinite checks or overwrite old weights/results. Formula reference: https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html (biased variance, unchanged epsilon).


### Bounded realignment after the two hypotheses

The repaired 20-epoch study completed four heads in 114.30 seconds: sharper targets improve representative score 0.548497→0.557800 but worsen spatial 0.421150→0.415520. Reject this change under the predeclared both-fold rule. Soft borough weighting also loses on both folds. Keep 1-km targets and the normalization repair. Before running the next controlled extension, predeclare **40 epochs, fresh initialization, 1-km targets, identical fixed folds and fixed mixed decoder, at most 60 seconds per fold**, using existing complete cached vectors. Compare to repaired 20-epoch controls and the original saved baseline; the OneCycle horizon expands too, so this is a practical longer-training comparison, not an isolated epoch-only causal claim. No encoder work or new downloads. These two at-most-one-minute fits keep additional head fitting under the original ten-minute allowance even conservatively reserving five minutes for the first unmeasured failed partial fit, plus 114.30 seconds measured successful fits, 2.78 seconds measured second failure, 21.20 seconds one-epoch diagnostic and ten seconds reserved for the earlier four-step diagnostic. This conservative ledger is a bound, not a measurement of those unmeasured diagnostics. Require all 40 epochs complete and improvement >0.0001 on both fixed folds over both controls before retention. Do not continue extending the search in this task after this comparison.


### Cached-feature studies completed — 2026-11-16

Retain the longer **40-epoch / 1-km-target** StreetCLIP compass head. Representative score **0.582646104**, Macro-F1 **0.799950887**, GPS **0.558501128**, mean/median/p90 **4.414989 / 1.874957 / 12.469612 km**, selected epoch 40. Spatial **0.436518798**, Macro-F1 **0.754139679**, GPS **0.401227589**, mean/median/p90 **5.987724 / 3.371754 / 14.696982 km**, selected epoch 35 after all 40 completed. All original fold/buffer IDs remain fixed. Fits took **55.4297 / 51.2544 seconds**, below each 60-second cap. All six successful study heads totaled **217.7946 seconds**; prior failures/diagnostics are documented separately, with the original unmeasured failed trial bounded by its five-minute cap. The ten-minute additional head allowance was respected. Longer training changes optimizer exposure and OneCycle horizon, so do not claim a pure epoch-only causal ablation.

New 40-epoch scores improve original 20-epoch scores by **+0.035730744 / +0.015410732**, and repaired 20-epoch controls by **+0.034149312 / +0.015368733**. Preserve prior weights/results. Do not keep 0.5-km targets (representative improves but spatial worsens) or soft borough GPS weighting (both worsen). The explicit training normalization matches native CPU forward/backward on CPU and MPS; six complete fits passed finite checks. This does not prove the backend root cause of the two earlier stopped runs. No guard was disabled and no failed checkpoint retained. Representative borough mean errors all improve; spatial Staten Island worsens **11.5553→11.9593 km**, Manhattan **2.6685→2.7879**, Queens **5.7974→5.8123**, despite the aggregate gain. These are reused development estimates, not private/public leaderboard evidence.

Representative checkpoint `outputs/runs/20261116T165854_compass_normfix_tau10_long40_representative/best.pth`, SHA **b81144fcaf8aac8467ffe7074401b16403eacd6aa8db63169f15eb891e6ecd10**. Spatial `outputs/runs/20261116T165950_compass_normfix_tau10_long40_spatial/best.pth`, SHA **dcc84f8493a3bcd34771eb1d4a4379330fba93cb210eab4f3a57add7f1584a52**. Actual saved-model predictions and CSV metrics were reproduced for all six new models, with matched configs, complete epochs, training-only statistics, source/log hashes and unchanged input/cache identities verified. Training/extraction-disabled fresh-kernel verification passed **38 code cells, 33 checks, zero errors, 13 embedded figures**; evidence `outputs/notebook_cached_geo_final_verification.json`, `outputs/cached_geo_study_artifact_audit.json`, comparison/borough tables and decision files. The new comparison figure was inspected and its overlapping legend corrected.

All active implementation/tests/progress remain in `NYCGuessr.ipynb`; current status/provenance refreshed in `DATA_SOURCES.json`. No dependencies, encoder extraction, external data, all-label supervised StreetCLIP refit, test inference/CSV, Kaggle upload, HPC work, external messages, commits or pushes occurred in these cached studies. The prior full-cache completion is historical and remains intact. The single proposed next task is a training-versus-validation GPS audit by borough, excluding self-matches from any training retrieval diagnostic, to choose between head capacity and representation/generalization changes. Do not extend modeling search automatically in this task.


## 22. Retained-model retrieval and generalization audit — authorization 2026-11-16

The user accepted the proposed training-versus-validation / retrieval-candidate audit with **"Great, let's try that"**. Diagnose the retained 40-epoch-schedule StreetCLIP heads (selected representative epoch 40, spatial 35) on each complete original training and validation fold. Do not retrain or tune model/decoder settings during this diagnostic. Exclude equal sample IDs from training retrieval and nearest-coordinate coverage; cells and fitted head still include each training query, so call training results in-sample, not leave-one-out validation. Preserve existing validation decoder overlap guards.

Measure mixed/cell-only/neighbor-only GPS, all-five borough F1, mean/median/p90, borough-wise error, top-1/5/32 retrieval recall at 0.25/0.5/1 km, top-30 cell candidate coverage, nearest-cell probability/rank, cell probability mass near truth, and whether large prediction errors occur despite a nearby candidate. Candidate distances requiring the true coordinate are diagnostic oracles only; never feed held-out labels into candidate generation, weights, predictions or model selection. Nearest-coordinate bank coverage is a geometry-only diagnostic, not an achievable performance estimate. Keep four-view groups and spatial buffer intact.

Bound queries to chunks of at most 128, candidate sets to 32 neighbors plus 30 cells, and use cached vectors with head inference only. No full pairwise image matrix, image extraction, downloads or model fitting. A ten-minute wall-clock audit cap is sufficient; default execution should load verified diagnostic artifacts after the explicit first run. Keep active code/checks/results in the single notebook. Preserve checkpoint/cache/input hashes and log exact source/config identity, per-query diagnostics and aggregate findings. No test inference, external data, uploads, HPC jobs, messages, commits or pushes are part of this task.


### Audit completed and decision — 2026-11-16

Completed all **23,001** training/validation query evaluations across both fixed folds in **16.5627 seconds**, using the existing frozen-feature caches/retained heads with no training. Actual query self-matches: **zero**. Preserved original validation predictions/scores. Training mixed mean errors **2.531572 / 2.275224 km** versus validation **4.414989 / 5.987724 km**; training is in-sample head/geography fit despite self-excluded retrieval, not an unbiased generalization estimate.

Validation top-1/top-5/top-32 neighbor recall within 1 km: representative **38.95% / 68.00% / 90.64%**, spatial **15.04% / 34.24% / 62.54%**. Adding top-30 cell centers yields combined candidate coverage **94.18% / 73.90%**. Actual bank coordinate coverage within 1 km is **100.00% / 99.09%** (mean nearest distance 0.125625/0.431904 km). These truth-dependent coverage statistics do not imply the answer can be recognized. Combined best-candidate mean distance 0.441187/1.215555 km is a retrospective finite-pool oracle, never a model score or lower bound on continuous predictions. Large errors >10 km already have a candidate <=1 km in **220/305 (72.13%)** representative and **169/430 (39.30%)** spatial cases. Both ranking/weighting failures and candidate absence occur. Spatial Staten Island training/validation means **3.5826/11.9593 km** and top-32 recall **51.18%** remain a concern.

Neighbor-only decoding improves representative mean slightly (**4.357481 km**) but worsens spatial (**6.115308 km**); cell-only gives **4.699300/6.094323 km**. **Retain the existing mixed decoder and 40-epoch-schedule models.** No new best-score claim or automatic decoder selection. Evidence favors a controlled training-fold-only hard-negative ranking experiment as the single next task: add geographically distant visual look-alikes to the paired sampler, mix ordinary/hard negatives, keep positives and head/loss/schedule/decoder fixed, compare with a matched control on both original folds. Do not claim the audit identifies a unique cause or that hard negatives are guaranteed to improve generalization. No such experiment has been started.

Artifacts: `outputs/retrieval_audit_latest.json` and `outputs/runs/20261116T173009_retrieval_generalization_audit/` (source/config/hash identity, per-query diagnostics, 12 prediction CSVs, summaries by fold/scope/borough). Existing producer source matches its archived AST; all artifacts/cache/checkpoints and original fold identities verified. Saved CSV scores match to 1e-9; probability summary round trips allow 1e-7 after an initial verification caught expected float32-reduction versus float64-reload differences up to 2.50e-8. No predictions or measurements were changed to make that check pass. Initial real audit: **40 cells / 36 checks / zero errors / 40.9803 s**. Final training/extraction/audit-disabled fresh-kernel run: **40 cells / 36 checks / zero errors / 15 figures / 41.5077 s**. Both new figures inspected. Final verification executable SHA **d0bd374b739b4c3aab726fb1ac32f303297dca08738a53398af6e883e49b63b6**; later Markdown updates preserve it.

Updated active notebook, this instruction/progress log and `DATA_SOURCES.json`; derived audit artifacts under ignored `outputs/`. No model fitting, feature extraction, dependency changes, raw input changes, test predictions, Kaggle submission, HPC work, external messages, commits or pushes.


## 23. First StreetCLIP Kaggle CSV — user authorization 2026-11-16

The user requests **"What's the current score and generate the submission csv for kaggle competition"**. This authorizes the necessary all-label supervised refit, test-feature extraction, inference and local CSV creation. It supersedes prior pending-final-inference restrictions for this task; Kaggle upload remains a separate action and is not requested. State existing development scores 0.582646/0.436519, with public score unknown.

Use the retained representative 40-epoch/1-km-target configuration and fixed mixed decoder; no new model search. Refit a fresh head on all 11,864 official training labels using the existing verified StreetCLIP cache, seed/schedule/normalization repair and a five-minute cap. Preserve the selected source and all validation artifacts; the full refit has no held-out score. Encode all 7,908 test locations with the same frozen model/preprocessing/batch-two/fp16 cache contract in a **separate resumable test namespace**, estimated 55–65 minutes from measured throughput and capped at **75 minutes** cumulative for this phase. Do not reset or reuse prior extraction budgets. An explicit request to generate this CSV supplies authorization for this necessary bounded computation.

Test IDs overlap training IDs: preserve the strict validation overlap guard and apply an explicit train/test namespace mapping during final decoding only. Derive test image names solely from template IDs; template borough/coordinate placeholders are never inputs or targets. Verify official test-image bytes against the existing extraction manifest, ordered four-view groups and source hashes. Use disk-backed partial features with periodic row/budget checkpoints; refuse incomplete caches, identity drift, nonfinite output or unexpected memory growth. Cache approximately 65 MB, not full-resolution imagery. All active code/checks/orchestration/progress remain in the single notebook; real final-run flags default off.

Create a uniquely named CSV under the existing `Kaggle Submissions/` deliverable location, exact template order/header, 7,908 unique IDs and valid borough/coordinate outputs. Read back and validate, hash the CSV/checkpoint/cache, preserve model provenance and actual runtime/limitations. No external data, downloads, new dependencies, HPC action, automatic Kaggle upload, messages, commits or pushes.


### Submission CSV completed and verified — 2026-11-16

Generated **`Kaggle Submissions/nycguessr_streetclip40_20261116.csv`**, **7,908 rows**, exact official headers/order, **302,382 bytes**, SHA-256 **`c065fb05a3f31c9195f4132f726c2f6c7d1b52195b831dcc59dbe877e33dc35f`**. This is a local candidate ready for manual upload; **no Kaggle upload occurred and the public score is unknown**. Source development scores remain **0.582646104 representative / 0.436518798 spatial**. The final refit has no held-out score.

Fresh all-label training completed **40 epochs on all 11,864 locations in 53.7154 seconds**, with the selected representative schedule, 1-km targets, losses, repaired input normalization and mixed decoder unchanged. Final checkpoint **`outputs/runs/20261116T174151_compass_submission40_full/best.pth`**, SHA **`8991cb2d59eb8cc77272b37644b8af94cdbd5817001af8652791c7e4e91fe4f1`**. Training-only mean/std/origin and complete per-epoch anchor coverage verified. No hard-negative or other new modeling experiment was included.

All **31,632 official test-image hashes** matched the original supplied manifest. Test labels/placeholders never enter the feature frame. Complete test cache: **`outputs/features/streetclip_test_f1901b97eac036c7.npy`**, float16 `[7908,4,1024]`, **64,782,464 bytes**, SHA **`b6870dc542cdfd23cc701572da74fe5b239dec0088eee90679a539c9bcb39695`**. Test extraction took **4,088.4360 seconds (68.14 minutes)** inside its separate **4,500-second** cap; sampled MPS driver maximum **1,514,881,024 bytes (~1.41 GiB)**. Initial throughput slowed and then recovered; no budget reset/extension or feature restart occurred. Previous phase budget records remain exactly **2,755.8706 / 2,355.1052 seconds**. A user “Continue” during processing continued this same run. Test and training numeric IDs overlap, so final decoding uses disjoint internal keys and restores original test IDs; the strict validation overlap guard remains unchanged.

Predicted borough counts: Queens **2,997**, Brooklyn **1,818**, Manhattan **1,305**, Bronx **1,151**, Staten Island **637**. These are outputs, not imposed quotas. Latitude range **40.50322403–40.90257226**, longitude **−74.24364230–−73.70972671**, within training-derived candidate extents. No external boundaries, map data, clipping or manual prediction edits. CSV read-back passed; strict checkpoint reload reproduced all boroughs and all GPS outputs to maximum **4.9997e-9 degrees**, consistent with eight-decimal serialization.

Initial execution: **42 code cells / 38 checks / zero errors / 4,187.3849 seconds**. Final training/extraction/CSV-generation-disabled fresh-kernel run: **42 cells / 38 checks / zero errors / 15 figures / 31.1272 seconds**. Actual cache/source identities, namespace separation, full refit metadata, original budgets, file hashes and refit/submission source logs verified. Executable source hash **`7c52a97a6fecf58a1d708c46401201a05698c0a579c79de9dadcc85547e2feb5`**; subsequent Markdown updates preserve it. Evidence: `outputs/streetclip_submission_manifest.json`, `outputs/streetclip_submission_verification.json`, `outputs/streetclip_test_input_audit.json`, `outputs/streetclip_submission_refit.json`, `outputs/notebook_submission_execution.json`, `outputs/notebook_submission_final_verification.json` and exact run archives/logs.

Modified `NYCGuessr.ipynb`, this log and `DATA_SOURCES.json`; created the named CSV and derived output/cache/checkpoint artifacts. All active code remains inline; final-run flags default off and completed candidates cannot be overwritten implicitly. Original source data/validation checkpoints remain intact. No new downloads/dependencies, external data, HPC activity, account actions, external messages, commits or pushes. Next action is manual Kaggle upload of this exact CSV to obtain its first measured public leaderboard score; do not report a leaderboard result before it exists or upload automatically without explicit instruction.


## 24. Monitored fusion and retrieval study — authorization 2026-11-17

The user requests autonomous execution/monitoring again and approved the proposed
numerical audit, cached fusion, retrieval training and conditional encoder investigation
with **"Great, let's implement this"**. This supersedes the earlier user-run-cell
preference. Keep active code/tests/orchestration/results in NYCGuessr.ipynb and report
brief measured updates. Begin with CPU/MPS FP32/cache parity on eight official training
images (five-minute cap), equal candidate fusion and a learned compass-aware fusion
head. Compare 32 appended hard negatives against 32 distant random negatives with
matched remaining sampler/model/loss/schedule; mine only training-fold features/labels.
Preserve original folds, 727 spatial exclusions, cache identities and historical weights.
Use 40 epochs, at most three minutes per head and twenty aggregate minutes of new
head fitting, separate persistent ledgers. Retain only >=.002 composite gain and lower
mean GPS error on both folds; repeat promising head and StreetCLIP control at seed6644.
No score improvement is presumed. All real actions default off. Kaggle upload requires
separate explicit authorization; none is requested. SALAD/BoQ or fine-tuning are
conditional next phases based on these measurements, not an unbounded model search.
User-reported public baseline: **.60599**; DINO spatial .599688/.415543 development
scores do not meet the prior both-fold rule. No external data/gallery/coordinates.


### Monitored study completed — 2026-11-17

Retain learned StreetCLIP/spatial-DINO fusion with ordinary distant negatives as the
new development candidate: seed6643 **.664912829/.479508151**, mean
**3.543432/5.530323km**; seed6644 **.667630293/.468218635** versus matching
StreetCLIP **.575983659/.448624287**. Every new head completed40 epochs, with
selected epochs40/40 and30/30 respectively. This is a reused-development result.
Second-seed spatial gain interval **[-.005648,.048100]** crosses zero. Equal
candidate fusion **.647699366/.495042634** is the strongest spatial alternative;
retain its evidence. Mined negatives **.658469982/.467660362** lose against
matched random negatives and are rejected. New head fitting **1068.4496s**,
under20 minutes;10 completed heads, each under3 minutes. CPU/MPS/cache FP32
parity passed; do not enable DINO FP16 based on this check.

Active code/results are in notebook cells76–84;53 checks passed in the
training-disabled targeted execution, not a complete historical Run All. Strict
artifact audit checks10 checkpoints, bothfolds/buffer, training-only statistics,
negative distances, histories, replayed validation scores, hashes and budgets.
Sources: `outputs/fusion/study_8ff4561c2d7cdd27/decision.json`,
`outputs/fusion/study_8ff4561c2d7cdd27/verification.json`, and
`outputs/fusion/notebook_verify_execution.json`. Original caches/submission intact.
No new data/weights/dependencies, full-label fusion refit, test inference/CSV, upload
or HPC work occurred. Public baseline remains user-reported **.60599** for
StreetCLIP. Next proposed task: full-label selected fusion refit, missing DINO test
features and a new validated CSV; Kaggle upload remains separately authorized.
Do not silently treat this proposal as a completed experiment or claimed public gain.


## 25. Selected fusion CSV — authorization 2026-11-17

The user requests **"Generate the submission cs"**, interpreted in context as the
selected fusion submission CSV. Refit primary seed6643 `fusion_random` on all11,864
labels for40 epochs with the selected schedule/sampler/losses/model/decoder, capped
at five cumulative minutes and durable epoch resumes. The all-label model has no
held-out score. Reuse both verified training caches and the complete StreetCLIP test
cache. Extract only missing official DINO test features in FP32 with unchanged
preprocessing and separate cumulative60-minute test budget; estimated19 minutes.
Verify every decoded input against the original manifest, four directions, complete
coverage and cache hashes. No new weights/data/dependencies are needed. Preserve
the original CSV and train/validation/test namespace guards. Export a new7,908-row
CSV in official template order, read it back, reload the final checkpoint and replay
all predictions before delivery. Keep code and progress in the existing notebook.
Kaggle upload remains a separate action and is not requested by CSV generation.


### Fusion submission completed — 2026-11-17

Generated `Kaggle Submissions/nycguessr_fusion40_20261117.csv`:7,908 rows,302,110 bytes,
SHA `075d86f3a426872c50f2b970d01b6cdb51319f6f102837dbd445a1019740ed3b`. Exact template headers/order and all
finite borough/GPS outputs verified. No upload; new public score unknown. Source
development scores .664912829/.479508151; full fit has no held-out score.

Fresh primary-seed selected fusion completed40 epochs on11,864 labels in
127.3956s, under five minutes. Checkpoint `outputs/fusion/submission_20261117/full40.pth`,
SHA `34acaf8c4ad1bd2c80314f4b320ad1bce066e6e06c77d69d314c5602cc959199`. All31,632 official test-image
hashes matched during FP32 extraction; complete DINO test cache
`outputs/dinov2/features/test_78de04b778ab6bd6.npy`,242,933,888 bytes,
SHA `a82a19cd8de9b15e4b092e76c1ad7e25875abf2300ac19b0878d574b93307234`. Extraction 1687.6254s
(28.13min), under60 minutes; sampled MPS driver peak3.35GiB.
Original datasets/caches/budgets/CSV preserved. Separate train/test namespaces
retain the ordinary validation overlap guard.

All7,908 predictions replayed from the strict final checkpoint; boroughs exact,
maximum GPS serialization difference 5e-09 degrees. Final targeted fresh-kernel
verification: 33 code cells,55 checks,zero errors,22.92s; not a full
historical Run All. Notebook cells85–89 hold all active final-fit/export/check code
and actual outputs. Provenance: `outputs/fusion/submission_20261117/manifest.json`,
`verification.json`, both execution records, histories/config/source archive.
No dependencies/downloads/external data/HPC/upload/commit/push. Next: manual
Kaggle upload of this exact CSV and record the returned public score.


## 26. Metric v2 migration and revised goals — 2026-11-17

The user supplied a TA screenshot: GPS threshold10→5km, unchanged borough
evaluation, all submissions being rescored. Its date is not visible;2026-11-17 is
receipt date. Kaggle invitation/scorer remained inaccessible. Preserve the handout's
.10/.90 weights and clip-after-mean aggregation as an explicit denominator-only
working interpretation. No data/weight/source-policy change is implied.

Current API: `score_current_predictions`; explicit versions via
`score_predictions_versioned`. Keep `score_predictions` and historical trainer/audit
definitions as frozen v1 replay primitives, preserving original source hashes and
checkpoints. All new code MUST select/train/log against v2, with metric version in
configs/checkpoint/cache-independent experiment identities and tests. A legacy
trainer cannot be reused for a new experiment without migrating its scoring and
selection. Do not relabel historical scores, mutate old histories or guess rescored
public values from a composite alone. All CSVs remain valid without regeneration.

Goals: **.80 primary v2 milestone**, **.90 stretch**. At Macro-F1 .85 these require
mean errors <=1.027778km and <=.472222km. The original old-metric .90 distance
ambition corresponds roughly to .81–.82 at Macro-F1 .8–.9. Preserve the original
requested .90 as history/stretch, not as an already achieved result. Track physical
mean-error milestones3/2/1/.5km and spatial mean<5km separately.

Prospective selection: representative v2 score gain>=.002 and representative mean
reduction>=.05km; spatial mean reduction>=.05km; repeat promising models with a
matched seed and paired spatial-block mean-error intervals. The spatial composite
must still be reported, but with all means>5km its GPS term is zero and cannot
serve as the sole GPS progress gate. Do not reward borough-only stress-score gains
when raw GPS error worsens. Report F1, mean/median/p90, error-tail shares and
retrieval recall below250m/500m/1km. No automatic loss-weight or target-temperature
rescaling: the physical mean-distance optimum is unchanged. The earlier .5km target
ablation remains rejected on its measured evidence.

Rescored18 fixed prediction sets; preserved16 checkpoint hashes and all original
histories/predictions. Current fusion .346003940/.077237245, mean
3.543432/5.530323km; second seed .351531606/.076301902. Spatial GPS is zero.
Both seeds pass the current point-estimate distance guard; second-seed spatial
mean-reduction interval[-.067479,.495167]km crosses zero. Representative worst
decile contributes53.5% of total error. Priorities: task-oriented pretrained
place-recognition encoder (SALAD) pilot, then bounded candidate view/patch reranking
and tail reduction before encoder fine-tuning. Use competition data only.

Code/checks/current dashboard are notebook cells90–94. Five new metric tests passed;
28 targeted cells,zero errors. This is a CPU rescore, not training or a full historical
Run All. Evidence: `outputs/metric_v2/latest.json` and linked comparison/history/goal
review, plus `outputs/metric_v2/notebook_execution.json`. Keep .60599 as a v1
user report and current public score unknown until a rescored result is provided.


## 27. Monitored SALAD pilot — authorization 2026-11-18

The user approved **"Let's implement this"** after the proposed place-recognition
pilot, candidate reranking, selective refinement and conditional fine-tuning. Begin
with official frozen DINOv2-SALAD encoder/aggregation weights; all pretrained image
encoders are permitted. Do not download its GSV/MSLS/Pittsburgh/Nordland datasets,
galleries or coordinate labels. Pin release/code, preserve upstream license notices,
load weights safely and validate compatibility/parity. Keep all active code inline.
Use current5-km metric v2 in every new evaluation/log. Per-fold pilot:1,024 original
training-gallery IDs and256 original validation queries, borough-stratified seed6643,
four views together; compare identical-ID frozen StreetCLIP/DINO controls with a
fixed decoder and one predeclared .7global/.3region reranker. Do not compare these
small-gallery scores as full-fold improvements. Download/benchmark caps5minutes
each, pilot extraction20 cumulative GPU minutes, conditional full completion45
separate minutes. Preserve cache coverage/budget ledgers and existing artifacts.
Only expand after SALAD lowers mean by.05km vs BOTH controls on BOTH folds and
representative v2 score is nonworse. Evaluate complete folds before retention or
later fine-tuning. No automatic Kaggle upload; no public-score claim without evidence.


### SALAD pilot and controlled realignment completed — 2026-11-18

Official SALAD checkpoint352,040,378 bytes /188 tensors strictly loaded; SHA
`6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada`. Pinned code/source/license
archive preserved. Native original-parameter reference/CPU/MPS parity passed;
FP16 SALAD was separately validated (do not extend this permission to failed
DINO FP16). No dependencies or external imagery/gallery/coordinates obtained.
Pilot2,374 unique locations /9,496 official training images took
400.8066s under20 minutes; sampled encoder driver peak1.35GiB.
Partial memory-mapped cache has only those rows covered, complete=false.
Do not use uncovered rows or spend full-extraction budget after the failed gate.

Small-gallery SALAD mean9.3921/9.4823km vs StreetCLIP9.0973/10.1746
and DINO10.3926/9.5681. Fixed region reranker9.4279/9.0830 also fails the
representative gate. Separately predeclared pooled-view comparison gives SALAD
8.7159/9.0857 vs StreetCLIP8.1202/9.6070 and DINO9.2988/9.4664; still fails.
These are subset-gallery results, not full-fold supervised comparisons.

A final approved candidate-ranking step used full-fold original fusion and cached
DINO regions, with a predeclared .7global/.3region neighbor-weight mix; original
cell probabilities/boroughs/decoder held fixed. Alpha0 exactly replays the old
model. Alpha.3 mean3.5556/5.5424km vs3.5434/5.5303 at seed6643, and
3.5129/5.6414 vs3.5122/5.6454 at seed6644. Reject this fixed reranker;
retain original fusion. No coefficient sweep, new head fit or fine-tuning.

All active code/progress/tests in notebook cells95–113. Final targeted
fresh-kernel disabled execution:13 checks,40 code cells,zero errors,8.18s;
not a historical Run All. Read-only audit verifies scores, subset hashes/coverage,
source identities, budgets, original checkpoints and both CSVs. Sources under
`outputs/salad/` include benchmark, pilot/pooled/fusion-rerank comparisons, phase
executions and verification.json; upstream license text is archived in model source
JSON and must be retained when packaging adapted source. No full SALAD pass,
Q4 training, test inference, CSV update, Kaggle upload, HPC work, commit or push.
Next proposed task: small learned candidate reranker trained only on training-fold
positive/negative location pairs, using complete cached compass/region evidence;
evaluate original full folds and both seeds. Do not repeat rejected constant weights
as if untested or describe this negative pilot as a new score improvement.


## 28. Training-only learned candidate reranker — authorization 2026-11-18

The user approved **"Ok continue with that"** after the learned-reranker proposal.
Use existing complete StreetCLIP/DINO feature caches only, all code inline. Within
each original outer training fold choose2,048 meta queries, train a fresh40-epoch
fusion teacher on the remaining gallery, excluding meta/outer-validation labels
from its normalization/cells/losses. Train39-feature64/32 MLP residual reranker for
30 fixed epochs on teacher-unseen training queries; query GPS is a target only,
never a feature. No absolute query coordinates/IDs as model inputs. Zero output
replays the baseline; bound residual logits±4, leave cell/borough decoder unchanged.
Fixed geographic soft-target CE plus expected distance and residual regularization;
all new metrics/identities v2. Evaluate complete original folds only after fitting,
seeds6643/6644. Require v2 representative gain>=.002 and >=.05km mean reduction
on both folds at both seeds.180s/teacher,60s/reranker,20 cumulative fitting minutes;
pair construction/evaluation bound10minutes per outer fit. Log losses, source IDs,
statistics, hashes, heldout metrics/uncertainty and decision. No image passes,
new downloads/dependencies, test predictions or Kaggle upload. Preserve originals.


### Learned reranker completed and rejected — 2026-11-18

Four inner-gallery teachers completed 40 epochs and four MLPs completed 30 epochs
on MPS, using only their original outer training folds. Meta query labels never
entered teacher statistics/geography/losses; outer validation never entered fitting.
Total fit 309.2587 s under 1,200 s, sampled driver peak 1.08 GiB. Primary original
versus learned mean errors: representative 3.543432→3.605552 km (v2 .346004→.334822),
spatial 5.530323→5.619461 km (v2 .077237 unchanged due clipping). Second seed:
3.512208→3.541910 and 5.645370→5.689019 km. Reject at both seeds; original fusion
and both existing CSVs remain intact. No test inference, encoder pass or upload.

Eight inline tests and 140 artifact checks passed; all original/learned outer
validation predictions replayed from strict checkpoints. Replay execution: 34 code
cells, zero errors, 49.89 s, targeted fresh kernel, not historical Run All. Evidence
under `outputs/learned_rerank/`; immutable decision via latest.json, per-run plans,
training pairs/targets/checkpoints/histories, verification_replay.json and phase logs.
All active code/progress is in notebook cells114–123; real flags default off.

In-sample ranking proxy improves while final coordinates worsen. Do not claim a
confirmed cause or relabel proxy improvements as GPS gains. Representative top-32
coverage within1 km is91.99% (spatial67.08%); oracle nearest-candidate distances are
diagnostics using validation truth, never usable predictions or guarantees. Next
proposed controlled experiment changes only the reranker objective to decoded GPS
distance with existing candidates/features/teacher partitions. It remains unexecuted;
do not start an unbounded sweep or repeat this rejected proxy as a new baseline.


## 29. Retained fusion CSV re-export — authorization 2026-11-18

The user requests the current local score and a submission CSV. Use the retained
primary-seed fusion; the learned reranker is rejected. Re-score saved validation
predictions with current 5-km v2 and replay the existing all-label 40-epoch checkpoint
on complete hashed cached test features. Export a new dated CSV, preserving prior
files and validating all 7,908 rows, headers, IDs, boroughs and coordinates. Cached
inference cap 180 seconds; no retraining, encoder/image passes, downloads or new
model selection. Keep active code/results in NYCGuessr.ipynb. A re-export may be
byte-identical to the earlier fusion CSV; state that plainly and do not claim a score
gain. No Kaggle upload is authorized by CSV generation.


### Retained fusion re-export completed — 2026-11-18

Generated `Kaggle Submissions/nycguessr_fusion40_20261118.csv`, 7,908 validated rows,
302,110 bytes, SHA `075d86f3a426872c50f2b970d01b6cdb51319f6f102837dbd445a1019740ed3b`.
Strict all-label checkpoint replay passed. It is byte-identical to the November17
fusion CSV; do not claim a new model or improvement from re-export. Current local
v2 .346003940/.077237245, mean3.543432/5.530323 km. Historical v1 local scores
.664912829/.479508151 are different-metric values for identical predictions, not
public results. New public score remains unknown. No training/download/image pass
or upload. Notebook cells124–126 hold active code/results; export defaults off.
Targeted fresh-kernel export:27 code cells, zero errors,15.61 s. Preserve older
CSV/checkpoints and their identities. Manifest/execution under
`outputs/fusion/submission_20261118/`. Next model experiment remains proposed only.


## 30. Public-score clarification and bottleneck audit — 2026-11-18

The user reports about0.42 public for the latest fusion CSV after the5-km update.
Record it as approximate user-reported evidence, tied to CSV SHA
`075d86f3a426872c50f2b970d01b6cdb51319f6f102837dbd445a1019740ed3b`.
It applies to both byte-identical dated fusion CSVs. Do not compare local .346004
as a subsequent public score or explain that difference solely by the10→5km change;
both .42 public and .346004 local now use v2. Different data/full-data versus fold
fit can explain a gap, but the individual contributions are unknown. No new public
decline is established. The earlier .60599 StreetCLIP public score is v1 and cannot
serve as a comparable public baseline until its current rescored value is known.
Preserve historical immutable manifests with their original unknown-public fields;
the latest observation is in DATA_SOURCES.json and notebook cells127–128.

CPU saved-prediction audit: representative worst10% cause53.5% of distance error;
wrong-borough15.5% cause50.6%; top32 within1km recall92.0% versus top1 48.9%.
Training mean1.175km is in-sample; validation3.543/spatial5.530km show a major
transfer gap. These support tail/generalization/ranking priorities, not proof that
pretrained weights or GPU capacity cause the plateau. Proxy/decoder mismatch and
teacher-to-final-head shift remain hypotheses. Next controlled experiment remains
final-coordinate loss with fixed candidates/features/model/partitions, not a new
encoder sweep. No training or inference was started for the diagnosis.21 targeted
code cells executed without errors in2.78s. Evidence under outputs/diagnostics/.


## 31. Decoder-aware reranker study — authorization 2026-11-18

The user requests "Solve these issues and improve th score" after the bottleneck
audit. Implement and autonomously monitor the controlled final-coordinate-loss
trial. Reuse exact original training-only teachers/meta IDs/pairs/model/decoder;
change only the loss to mean decoded Haversine km/5 +.001 residual-square penalty.
Use bounded60-step differentiable Weiszfeld, verify CPU/MPS parity and gradients
before fitting.30 fixed epochs,128query batches,180s per fit/720s cumulative new
fitting. No outer validation epoch selection; evaluate full original folds and both
seeds. Retain only v2 representative gain>=.002 and >=.05km mean reduction on both
folds at both seeds. Record final raw distance/tails/uncertainty and decisions.
No encoder/teacher fitting, external data/downloads, test predictions, uploads,
messages, commits or pushes for this trial. All active code in NYCGuessr.ipynb;
real action defaults off. The user-reported current public fusion score is about.42,
not an exact API score; distinguish it from all local measurements.


### Controlled mixture follow-up within the approved improvement task

Loss-only objective improves representative GPS at both seeds but misses the
spatial guard at6643. Preserve that result without promoting it. Test one bounded
follow-up: freeze those residual MLPs and learn a query-specific cell/neighbor trust
gate, initialized50/50 and bounded10–90%cell mass, from the same39features/meta
queries/teachers. Fit only39→16→1gate for30epochs with final GPS loss +.001(mix−.5)².
Same both-fold/both-seed guard; shared720s phase fit cap includes fixed-loss fits.
No teacher/encoder fitting or new data/weights/test predictions/upload. This is one
controlled decoder follow-up, not authorization for an unbounded validation search.


### Final bounded training-regime realignment

The mixture-only follow-up also misses the spatial guard at6643. Within the same
approved improvement task, test one final hybrid meta-training regime:1024queries
from1km training tiles with200m gallery buffer +1024dense queries, original outer
validation excluded throughout. Fit fresh40epoch teachers on remaining galleries,
then identical final-distance MLPs for30epochs with fixed50/50decoder. Preserve
reserved/excluded IDs and note smaller teacher galleries as part of the regime
change. Existing720s shared ledger includes prior fixed/gated78.84s fitting;180s
per teacher/model, reserve teacher worst-case cap before starting it. Same fixed
folds/seeds/retention guard. No encoder passes, new data/weights/dependencies,
test inference or uploads. This is the final modeling variant in this bounded run;
do not turn it into an unbounded validation search if it fails.


### Selected hybrid calibration/export after passed development checks

The hybrid regime passes both original folds at both seeds;300 artifact checks and
full validation prediction replay passed. Within the user's standing improvement
and submission request, refit the same primary-seed calibration on official full
training data: hybrid2048meta query partition, fresh40epoch inner teacher,30epoch
MLP; reuse the existing full-data fusion base and complete cached test features.
Carry298.93s development fitting into the same720s total budget and reserve360s
for both worst-case fit caps. No new encoder passes/downloads/base fit. Full
calibration has no held-out score. Export a new dated CSV, preserve prior bytes,
validate IDs/schema/coordinates and strict replay. New public score remains unknown;
prior≈.42 belongs to old fusion. This authorizes no Kaggle upload.


### Decoder study completed: hybrid candidate retained — 2026-11-18

Final-coordinate loss improves representative scores but the random-meta and frozen
learned-mixture variants miss the spatial seed6643 guard. Hybrid1024spatial+1024dense
meta queries with new training-only teachers pass the same guard at both seeds.
Primary v2 .346003940→.365794946, mean3.543432→3.433482km; spatial mean
5.530323→5.460754km (score remains.077237 because GPS clipped). Repeat seed:
.351531606→.385274138 and spatial5.645370→5.574773km. Spatial95% intervals
cross zero; describe a reused-development point-estimate gain, not proven private
or public improvement. No weaker gate, coefficient sweep or best-seed switch.

Selected primary-seed full calibration:40epoch inner teacher on9449labels,
30epoch MLP on2048teacher-unseen queries,367reserved/buffer rows. Existing full
fusion base used all11864labels. Full calibration has no held-out score. New full
fitting74.3734s; all18fits373.3059s under shared720s. Cached test inference22.1983s.
No encoder pass/download/dependencies/external data/HPC. All11inline tests and
340artifact checks passed; all12candidate validation sets and7908test predictions
strictly replayed. Prior128source cells, checkpoints and old CSV bytes preserved.

New CSV `Kaggle Submissions/nycguessr_decoded_hybrid_20261118.csv`,7908rows,
302110bytes, SHA `01ac13ccb5dd05c638cf17e372a77ba62b8dc91ed3f724f352857e71c04f9777`.
7738GPS rows changed, boroughs unchanged. Max serialization difference5e-9degrees.
New public score unknown; user-reported≈.42 belongs to prior fusion and remains in
latest_public_result. Do not call renamed old exports an improvement or compare
local validation to public scores as successive submissions. Canonical current
selection: hybrid_latest.json and submission_20261118 manifest/verification under
outputs/decoded_rerank; active code/progress cells129–151. All real flags default off.
Next single action is manual Kaggle upload of the new CSV and logging its exact
current-metric score. No upload was performed or authorized automatically.


## 32. Spatial-error target — authorization 2026-11-18

The user requires spatial mean below5km, ideally below4km. Current hybrid is
5.460754/5.574773km and does not meet it. Pursue bounded local improvements with
fixed original folds,200m stress buffer and representative-performance protection;
report target attainment separately from incremental improvement. Do not filter
hard boroughs, clip individual errors or weaken splits. First test one fixed50/50
spherical blend of current hybrid and earlier equal-candidate experts using saved
predictions, no coefficient sweep/training. Auxiliary experts remain seed6643 in
both comparisons; label that repeat limitation, not independent full seed repeats.
Keep all active code/progress in the notebook. No new downloads/encoder extraction,
test predictions, new CSV or Kaggle upload for this diagnostic. Preserve current
selected model/CSV until a complete verified improvement passes the acceptance checks.


### Spatial-first learned arbitration after fixed blend diagnostic

Fixed50/50blend gives spatial5.227/5.280km but worsens representative GPS. Test
one48-feature learned coordinate combiner of current hybrid, StreetCLIP and DINO.
Features use prediction confidence/galley support/relative disagreements, not true
query GPS/boroughs, IDs or absolute GPS features. For training, use the exact hybrid
inner galleries with fresh Street/DINO teachers, and2-fold honest fusion residual
OOF predictions (512dense+512spatial per half). Combiner2,048rows are unseen by all
component fitted query-target models. Fit30epochs, use original outer splits for
one final evaluation. Train fresh outer experts at both seeds,40fixed epochs each;
no validation epoch choice.28new fits,180s/expert90s/smallmodel900s new-phase cap.
Preserve the previous closed373.31s phase ledger separately. No new image encoding,
downloads/dependencies/test inference/CSV/upload. All code inline, real flag off.

Declare acceptance BEFORE fitting: spatial mean<5km at BOTH seeds and no
representative mean regression at either seed; stretch<4km. This prospective
spatial-priority rule replaces the earlier demand for additional+.002representative
score gain, consistent with the user's new goal. Do not relax raw mean aggregation,
held-out labels, borough coverage or the spatial split. Preserve current selected
model/CSV unless these checks pass. Record uncertainty and all negative trials.


### Final candidate-cloud follow-up sharing the900s ledger

Coordinate expert arbitration misses5km (5.2677/5.3360) and regresses representative
seed6644. Preserve its negative result. Reuse the fitted inner/outer experts and
retain32neighbor entries per expert +30fusion cells (126points). Train47-feature
residual ranker on2048teacher-unseen hybrid queries with final GPS loss,30epochs,
no validation epoch choice. Duplicate gallery IDs are explicit expert votes; never
allow query IDs in a gallery. Per-candidate features use raw directional/region
similarities, cross-expert cosines/class compatibility/source identity, never query
truth or absolute GPS/ID features. Fixed expert priors.5/.25/.25; no sweep.
If fixed cell mass fails, one final follow-up freezes ranks and fits only bounded
cell/neighbor trust on hybrid data,30epochs. No new expert/encoder fitting. Carry
all659.09s prior new-phase spend into the SAME900s ledger;90s per small model,
600s per feature bundle with16-query chunks. Same<5both-seeds/no-representative-
regression acceptance. Final variant for this bounded run; document unmet targets
rather than extending the search. No test inference/new CSV/upload while unmet.


### One final sampling-risk sensitivity using remaining shared fitting budget

Best pooled means5.2121/5.2350km remain above5, while representative also improves.
Within the standing improvement/realignment request and remaining shared900s cap,
preregister one supervision-distribution check: use only1024buffered spatial meta
queries, duplicated twice for the same2048rows/30epochs/480optimizer updates.
No dense-query target loss, no new unique data/heads/architecture. Query replicas
remain only in training; all original validation IDs are unique, untouched and never
filtered. Image teachers excluded all original2048meta queries. Fit fixed-cell
rankers and, only if they fail, frozen-rank trust gates. Same prospective<5both-seeds
and no representative regression acceptance. Carry728.37s into the SAME900s ledger;
reserve90s before each fit. Previous failures remain failures. This is the last
sampling/model variant in the phase; explicitly report any remaining gap. No test
inference/new CSV/upload. Keep current production selection until target passes.


### Spatial-target phase closed — 2026-11-19; target unmet

Best matched experimental pooled-gate spatial means5.212053/5.234967km, representative
3.144363/3.244380km; below5/below4both unmet. Preserve this genuine incremental
experimental gain, but do not promote it as meeting the preregistered spatial target.
Production hybrid and existing CSV stay unchanged. Neither local .417836 nor any
other experimental score is a new public result. User≈.42 belongs to old fusion.

Completed spatial-only fixed repeats5.254795/5.322455km also regress representative.
Primary gate5.233025km fails; seed6644gate models were not started because814.5375s
spent plus reserved90s exceeded900s. Conservative reservation stopped the run before
new fits, not an OOM/nonfinite failure. Finalized rejection without extending/resetting
budget; partial gate-repeat scope recorded explicitly. Original budget-stop execution
preserved. No repeats are invented and no new fitting is pending in this closed phase.

42completed fits,814.5375s under900s;18inline tests590artifact checks passed, including
18calibration validation-set replays from saved candidate/confidence inputs and strict
weights, original split/buffer, gallery-only statistics/targets, OOF roles, duplicate
training-only weighting, exact validation IDs, budgets and unchanged existing CSVs.
Source/progress cells152–173; outputs/spatial_target contains decision/comparison
pointers, verification.json (488checks), blocked_verification.json (102checks) and
budget-stop/replay records. No downloads/new data/encoders/test inference/CSV/upload.

Next proposed phase is more UNIQUE geographically cross-fitted calibration queries
beyond2048 using existing caches. Do not repeat failed coefficient/sampling variants,
restart an old completed run or extend its900s ledger. A new phase needs explicit
new scope/budget and must keep outer validation untouched. Goal is still active as
research intent, not achieved by these experiments. No public-score gain is known.


## 33. GeoGuessr reasoning transfer — completed2026-11-19

The user explicitly requested internet research of expert GeoGuessr methods,
formalization and integration with the ViTs. Read the WIRED Rainbolt demonstration
transcript (actual explanations/guess sequences, not frame review) and indexed
PlonkIt beginner-guide excerpts; its full page was unreadable. Consulted Apple
Vision API documentation. Reference URLs/access limits are in DATA_SOURCES and
notebook cell174. External pages supply algorithm ideas only; no geographic clue
tables, gameplay images/GPS, maps, geocoding or outside labels enter the system.

Implemented notebook cells174–184: frozen visual candidate prior; generic native
OCR; fold-local rare words/phrases and GPS dispersion; distinct-line corroboration,
directional Street/DINO similarity and geographic-mode checks; conservative
reranking/candidate recovery with absent-text fallback. Repeated text across views
is one cue. No query GPS/borough enters inference, vocabulary or specificity.
Apple Vision language correction/custom vocabulary disabled; native model weights
are not independently hash-addressable. Record macOS/request revision3/Swift6.4.2
and cache hashes. Extraction is macOS-only; HPC may consume the recorded cache.

Resource realignment BEFORE accuracy evaluation: accurate64/256-image benchmarks
exceeded projected30-minute cap. Fast256 benchmark fit; read all47,456training
images, then accurate-refined3,703automatically detected text images. All OCR and
benchmarks share617.6759s/1800s ledger; at most2workers, lazy image reads. No install,
model download or new geolocation head fit;300s fitting allowance unused. Existing
closed spatial-phase814.54s ledger was not extended/reset.

Negative result: usable nonoverlay text at1379/11864locations (11.62%); reliable
matches at80/2372representative and61/2421spatial queries. Text rescue spatial
5.212017/5.234825km versus visual replay5.212053/5.234966km; representative
3.144358/3.244423km. Spatial paired95%error-reduction intervals both include0;
representative6644regresses slightly. No material gain, below5/below4unmet. Do not
promote it or present local .41784/.39973 as new public evidence. Conditional text
candidate top1 errors10.858/10.273km support keeping corroboration conservative.
Reject the sparse exact-text hypothesis as a score improvement; broader human
visual clue methods are not disproven.

6new synthetic checks,173artifact checks including12validation-set replays,
original fold/200m buffer, query-label counterfactual invariance, source/input hashes
and unchanged production CSV passed. Final disabled-action fresh kernel75code
cells/zero errors/12.815s; targeted verification, not entire historical run-all.
Old173source cells preserved. New artifacts under outputs/geoskills; no test
inference, new submission CSV, upload, HPC work, commits/pushes or external messages.
Production hybrid/CSV unchanged; new public score unknown.

Next single proposed task: small-batch runtime/storage benchmark for dense DINO
patch matching of buildings, road geometry and signs, with specificity learned
from training galleries. Existing cache is CLS+4coarse pools, not dense patches.
Do not rerun failed text-weight sweeps, loosen thresholds on exposed validation,
restart closed ledgers or perform new extraction merely because this is proposed.
