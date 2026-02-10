### Contrastive Pattern Analysis (High vs Mid/Low)

Data preprocessing
- What high-score solutions did: 
  - Applied medical-specific DICOM transforms (RescaleSlope/Intercept or modality/VOI LUT) and MONOCHROME1 inversion: 10/10 (4346, 4355, 4369, 4413, 4583, 4589, 4650, 4700, 4704, 4781).
  - Normalized intensities and standardized geometry via resizing to ~1024 and tracked scale/padding for inverse mapping: ~9/10 resized; explicit inverse-mapping metadata in 3/10 (4413, 4700, 4704).
  - Included negatives (empty labels or targets) and/or stratified splits by has_box: negatives included in at least 6/10; stratification by has_box in 3/10 (4583, 4650, 4700).
- What low-score solutions did:
  - Used basic DICOM handling but introduced pitfalls:
    - Excluded negatives from training: 3/8 (4350, 4374, 4634).
    - No stratified split by has_box: 8/8 (all).
    - Geometry mishandling:
      - Failed to rescale boxes back to original coordinates: 1/8 (4634).
      - Risky conversions or dataset caps limiting coverage: 2/8 (4322 capped to ≤2000 images; 4729 relied on cached PNGs/subsampling).
      - Global class-agnostic NMS and misapplied LR scheduler: 1/8 (4407).
- Concrete difference: High-scorers consistently preserved medical imaging fidelity and geometry (including inverse mapping) and trained with negative examples; several low-scorers trained on positives-only and/or broke coordinate mapping or post-processing.
- Why do these patterns matter? Detectors need background exposure to calibrate confidence; proper modality transforms and geometry inversion ensure accurate localization on original DICOM dimensions; stratified splits stabilize validation and threshold tuning.
- What should future solvers do? 
  - Always apply modality/VOI LUT, MONOCHROME1 inversion, and consistent normalization.
  - Include negative images in training; consider stratifying train/val by has_box.
  - Record scale/padding per image and invert transforms precisely at inference.
  - Avoid arbitrary dataset caps; prefer reducing resolution over reducing coverage.
  - Maintain a small representative validation set to tune thresholds.

Feature engineering
- What high-score solutions did: Minimal feature engineering with medical preprocessing; optional CLAHE used in 2/10 (4583, 4781); GT box deduplication in 1/10 (4583).
- What low-score solutions did: Largely none beyond basic preprocessing; no label de-noising or contrast enhancement.
- Concrete difference: Targeted label noise reduction and gentle contrast normalization (CLAHE) appeared in top-tier but not in low-scorers.
- Why do these patterns matter? Removing duplicate GT boxes reduces conflicting supervision and NMS issues; CLAHE can enhance subtle findings without harmful distribution shifts if applied consistently.
- What should future solvers do? 
  - Add optional CLAHE with conservative settings; validate on a small val set.
  - Deduplicate near-identical GT boxes per class at high IoU (≥0.9–0.95) and log the impact.

Model selection
- What high-score solutions did: 
  - Used strong pretrained detectors:
    - Faster R-CNN ResNet50 FPN (pretrained DEFAULT/v2): 4/10 (4346, 4355, 4369, 4589).
    - Ultralytics YOLOv8 (pretrained yolov8s/n): 5/10 (4413, 4583, 4650, 4700, 4704).
  - Pretrained initialization in 9/10; one succeeded with careful pipeline despite weights=None (4781).
- What low-score solutions did:
  - From-scratch or fragile initialization more common: 2/8 trained FRCNN from scratch (4634, 4729); 4350 could fallback to random init; 4322 risked training YOLO from YAML-only init; YOLOv8n with small imgsz and short training (4571).
- Concrete difference: Stable pretrained backbones in high-scorers versus from-scratch or risk-prone init among low-scorers.
- Why do these patterns matter? Transfer learning dramatically improves convergence and accuracy with short schedules and limited data.
- What should future solvers do? 
  - Prefer pretrained FRCNN v2 DEFAULT or YOLOv8s; use YOLOv8n only when constrained and pair with sufficient imgsz/epochs.
  - Verify weight loading and class mappings before training/inference.

Training methodology
- What high-score solutions did: 
  - FRCNN: SGD with momentum and weight decay (lr ≈ 0.003–0.005), occasional warmup/scheduling, AMP on CUDA; short but effective epochs (2–6) (4346, 4355, 4369, 4589).
  - YOLOv8: 20–25 epochs, imgsz 768–1024, patience and best.pt selection (4413, 4583, 4650, 4700, 4704).
  - Used small validation splits when available to pick checkpoints.
- What low-score solutions did:
  - Non-ideal optimizers/hypers for detection (AdamW with aggressive lr): 3/8 (4350, 4374, 4607).
  - Broken or ineffective scheduling and very short training: 4407 misapplied scheduler; 4322 few epochs with capped data; 4571/4729 too few epochs and weaker models; often no validation metrics (4350, 4374, 4634, 4729).
- Concrete difference: Conventional, stable training setups (SGD+momentum+wd, AMP; standard YOLO training) versus fragile optimizer choices and broken schedulers without validation.
- Why do these patterns matter? Detection heads are sensitive to optimization; SGD+momentum remains robust. Validation enables threshold calibration and checkpoint selection.
- What should future solvers do? 
  - FRCNN: use SGD (lr ~0.003–0.005, momentum=0.9, wd 1e-4–5e-4), add brief warmup or mild StepLR, enable AMP when possible.
  - YOLOv8: train 20–25 epochs at imgsz 768–1024, use patience and select best.pt.
  - Track a small validation set; select checkpoints by val metrics.

Evaluation & submission
- What high-score solutions did: 
  - Correctly mapped predictions back to original DICOM coordinates using stored scale/padding (4413, 4700, 4704); others rescaled appropriately (4346, 4355, 4589).
  - Sensible thresholds (0.05–0.2) and per-image caps (top 50–60); clipped/rounded coordinates; consistent fallback “No finding.”
- What low-score solutions did:
  - Critical mistakes:
    - No rescaling back to original coordinates: 1/8 (4634).
    - Class-agnostic global NMS suppressing cross-class detections: 1/8 (4407).
    - Inference from base COCO weights when trained weights missing (class mismatch risk): 1/8 (4571).
  - Overly low thresholds without calibration (elevated FPs): 3/8 (4571, 4634, 4322).
- Concrete difference: Faithful inverse geometry and class-aware default post-processing in high-scorers versus coordinate mapping bugs, harmful NMS configs, and incorrect checkpoints in low-scorers.
- Why do these patterns matter? Incorrect coordinates or suppressive NMS can devastate metrics regardless of model quality; mismatched label spaces yield invalid predictions.
- What should future solvers do? 
  - Always invert geometry precisely (account for scale and padding).
  - Use library-default class-aware NMS; avoid class-agnostic global NMS.
  - Ensure inference uses the trained checkpoint with correct class names.
  - Calibrate score thresholds on validation (≈0.1–0.2 for FRCNN; 0.05–0.1 for YOLO).

Notable implementation details
- What high-score solutions did: 
  - Stored per-image scale metadata and robustly selected best.pt/last.pt (4413, 4700, 4704).
  - Set seeds, used AMP and pin_memory for speed/stability (4346, 4369, 4589).
  - Stratified splits when used (4583, 4650, 4700).
- What low-score solutions did:
  - Missing validation/metrics; fragile schedulers and caps; positives-only training without compensations (4322, 4407, 4350, 4374, 4634, 4729).
- Concrete difference: High-scorers emphasized reproducibility, geometry fidelity, and safe checkpointing; low-scorers lacked safeguards.
- Why do these patterns matter? Small engineering lapses often determine leaderboard placement under short training regimes.
- What should future solvers do? 
  - Save and verify scale/pad metadata and inverse mapping.
  - Save and load best checkpoints; set seeds; use AMP when safe.
  - Maintain minimal but reliable validation tracking.

### Unique High-Performing Approaches (Diversity Preservation)
- GT box deduplication to reduce label noise
  - Solution IDs: row_index_4583
  - Why effective: Removes near-duplicate GT boxes per class (IoU ≥ 0.95), reducing conflicting supervision and improving NMS behavior.
  - Preservation guidance: Add a pre-pass to merge/deduplicate overlapping GT per class; log counts; ensure minimal over-pruning by using high IoU thresholds.

- Stratified train/val split by image-level positivity (has_box)
  - Solution IDs: row_index_4583, row_index_4650, row_index_4700
  - Why effective: Ensures balanced positives/negatives across splits, stabilizing validation metrics and threshold calibration with short schedules.
  - Preservation guidance: Stratify by a binary has_box flag; maintain sufficient negatives in both train and val; use 5–10% validation.

- Precise inverse letterbox/scale metadata for submission geometry
  - Solution IDs: row_index_4413, row_index_4704
  - Why effective: Prevents coordinate drift between resized training space and original DICOM dimensions, improving localization accuracy.
  - Preservation guidance: Store per-image sx, sy, pad_x, pad_y, and original/resized dims; deterministically invert transforms during inference.

- Linear warmup for FRCNN training
  - Solution IDs: row_index_4369
  - Why effective: Stabilizes early training with AMP and small batches; reduces gradient spikes in detection heads.
  - Preservation guidance: Use a short LinearLR warmup (~0.5–1 epoch) before the main schedule.

- Optional CLAHE for contrast normalization
  - Solution IDs: row_index_4583, row_index_4781
  - Why effective: Enhances local contrast, aiding detection of subtle findings without major distribution shifts when applied consistently.
  - Preservation guidance: Apply conservatively (tuned clip limit); keep consistent across train and inference; validate impact on a small split.

- Prefer torchvision FRCNN v2 DEFAULT weights
  - Solution IDs: row_index_4346, row_index_4355, row_index_4369
  - Why effective: v2 weights improve feature pyramid quality and training stability versus older variants.
  - Preservation guidance: Use v2 DEFAULT configs; verify transforms/class mappings align with the dataset.

Preserve diversity by maintaining both detector families (FRCNN and YOLOv8) in your toolkit, offering CLAHE-on and CLAHE-off variants, and supporting both stratified and non-stratified splits with consistent inclusion of negatives to hedge against distribution shifts.