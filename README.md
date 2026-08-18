# RIF-YOLO-N

Official implementation of **RIF-YOLO-N: Lightweight P3 Residual Identity Fusion with Resolution-Guided Fine-Tuning for Tiny-Object Detection**.

RIF-YOLO-N is a lightweight detector based on YOLOv8n. It strengthens the existing fine-scale P3 pathway through a **Residual Identity Fusion (RIF)** module while preserving the original anchor-free P3/P4/P5 detection hierarchy. A two-stage **Resolution-Guided Fine-Tuning (RGFT)** strategy further adapts the detector to higher-resolution tiny-object representations without introducing an additional detection head.

---

## Highlights

- Lightweight redesign of the YOLOv8n P3 feature pathway.
- Same-scale fusion of backbone and neck P3 features.
- Adaptive gating of backbone spatial information.
- Identity-safe residual enhancement with zero-initialized residual scaling.
- Original P3/P4/P5 anchor-free detection hierarchy is preserved.
- No additional P2 prediction head.
- Resolution-Guided Fine-Tuning for high-resolution tiny-object detection.
- Evaluation on **VisDrone**, **UAVDT**, and **VEDAI**.
- Controlled baseline, ablation, resolution, class-wise, size-wise, runtime, and qualitative analyses.

---

## Method Overview

RIF-YOLO-N receives two P3 representations:

- backbone-level P3 feature \(F_3^b\), which retains fine spatial information;
- neck-level P3 feature \(F_3^n\), which contains stronger semantic information.

The two representations are projected into a common feature space:

\[
P_n = \phi_n(F_3^n),
\]

\[
P_b = \phi_b(F_3^b).
\]

An adaptive gate controls the contribution of the backbone feature:

\[
G = \sigma(\psi([P_n,P_b])),
\]

\[
\widetilde{P}_b = G \odot P_b.
\]

The fused representation is

\[
U_3 = P_n + \widetilde{P}_b,
\]

followed by lightweight residual refinement:

\[
R_3 = \eta(U_3).
\]

The final enhanced P3 representation is

\[
F_3^{rif} = P_n + \gamma R_3,
\]

where the residual scale is initialized as

\[
\gamma_0 = 0.
\]

The enhanced P3 feature is propagated through the original P3/P4/P5 hierarchy without introducing an additional prediction scale.

---

## Resolution-Guided Fine-Tuning

RGFT uses a two-stage optimization procedure.

### Stage 1: Base Training

RIF-YOLO-N is trained at a base input resolution \(s_b\):

\[
\theta_b^* =
\arg\min_{\theta}
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{L}
\left(
f_\theta(I_i^{s_b}),Y_i
\right).
\]

### Stage 2: High-Resolution Fine-Tuning

The best checkpoint from Stage 1 initializes the same architecture at a higher target resolution \(s_h\):

\[
\theta_h^* =
\arg\min_{\theta}
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{L}
\left(
f_\theta(I_i^{s_h}),Y_i
\right),
\qquad
\theta \leftarrow \theta_b^*,
\qquad
s_h > s_b.
\]

RGFT changes the training procedure only and does not introduce additional inference-time detection modules.

---

## Repository Structure

```text
RIF-YOLO-N/
│
├── README.md
│
├── configs/
│   ├── visdrone.yaml
│   ├── uavdt.yaml
│   └── vedai.yaml
│
├── models/
│   ├── rif_yolo_n.yaml
│   └── yolov8n.yaml
│
├── rif_yolo/
│   ├── __init__.py
│   ├── rif_module.py
│   └── model_registration.py
│
├── scripts/
│   ├── train_baseline.py
│   ├── train_rif_yolo.py
│   ├── train_rgft.py
│   ├── validate_models.py
│   ├── benchmark_runtime_1024.py
│   │
│   ├── prepare_visdrone.py
│   ├── prepare_uavdt.py
│   ├── prepare_vedai.py
│   │
│   ├── evaluate_resolution.py
│   ├── evaluate_classwise.py
│   ├── evaluate_sizewise.py
│   ├── run_ablation.py
│   └── generate_diagnostics.py
│
├── datasets/
│   ├── VisDrone/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   │
│   ├── UAVDT/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── VEDAI-yolo/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── weights/
│   ├── visdrone/
│   │   ├── yolov8n_best.pt
│   │   ├── rif_yolo_n_best.pt
│   │   └── rif_yolo_n_rgft_best.pt
│   │
│   ├── uavdt/
│   │   ├── yolov8n_best.pt
│   │   ├── rif_yolo_n_best.pt
│   │   └── rif_yolo_n_rgft_best.pt
│   │
│   └── vedai/
│       ├── yolov8n_best.pt
│       ├── rif_yolo_n_best.pt
│       └── rif_yolo_n_rgft_1024_best.pt
│
├── results/
│   ├── tables/
│   ├── curves/
│   ├── confusion_matrices/
│   ├── qualitative/
│   ├── ablation/
│   └── runtime/
│
├── figures/
│   ├── architecture/
│   ├── rif_module/
│   ├── rgft/
│   ├── qualitative/
│   └── diagnostics/
│
└── requirements.txt
