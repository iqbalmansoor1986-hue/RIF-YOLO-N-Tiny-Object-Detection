# RIF-YOLO-N

Official research code for **RIF-YOLO-N: Lightweight P3 Residual Identity Fusion with Resolution-Guided Fine-Tuning for Tiny-Object Detection**.

RIF-YOLO-N is a lightweight detector based on YOLOv8n. It strengthens the existing fine-scale P3 pathway through **Residual Identity Fusion (RIF)** while preserving the original anchor-free P3/P4/P5 detection hierarchy. A two-stage **Resolution-Guided Fine-Tuning (RGFT)** strategy further adapts the detector to higher-resolution tiny-object representations without introducing an additional detection head.

---

## Highlights

- Lightweight enhancement of the YOLOv8n P3 feature pathway.
- Same-scale fusion of backbone and neck P3 features.
- Adaptive gating of backbone spatial information.
- Identity-safe residual enhancement with zero-initialized residual scaling.
- Original P3/P4/P5 anchor-free detection hierarchy is preserved.
- No additional P2 detection head or prediction scale.
- Resolution-Guided Fine-Tuning for high-resolution tiny-object detection.
- Evaluation on **VisDrone**, **UAVDT**, and **VEDAI**.
- Controlled baseline, ablation, resolution, runtime, diagnostic, and qualitative analyses.

---

## Method Overview

RIF-YOLO-N combines two complementary P3 representations:

- backbone-level feature \(F_3^b\), which retains fine spatial information;
- neck-level feature \(F_3^n\), which contains stronger semantic information.

The two features are projected into a common feature space:

$$
P_n = \phi_n(F_3^n),
$$

$$
P_b = \phi_b(F_3^b).
$$

An adaptive gate controls the contribution of the backbone representation:

$$
G = \sigma\left(\psi([P_n,P_b])\right),
$$

$$
\widetilde{P}_b = G \odot P_b.
$$

The projected features are fused as

$$
U_3 = P_n + \widetilde{P}_b,
$$

and processed by a lightweight residual refinement operator:

$$
R_3 = \eta(U_3).
$$

The final enhanced P3 representation is

$$
F_3^{rif} = P_n + \gamma R_3,
$$

where the trainable residual scale is initialized as

$$
\gamma_0 = 0.
$$

The resulting \(F_3^{rif}\) replaces the original neck-level P3 representation while the existing downstream P3/P4/P5 hierarchy and anchor-free detection head remain unchanged.

---

## Resolution-Guided Fine-Tuning

RGFT uses a two-stage training procedure without modifying the inference architecture.

### Stage 1: Base Training

RIF-YOLO-N is first trained at a base resolution \(s_b\):

$$
\theta_b^*
=
\arg\min_{\theta}
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{L}
\left(
f_\theta(I_i^{s_b}),Y_i
\right).
$$

### Stage 2: High-Resolution Fine-Tuning

The best checkpoint from the base stage initializes the same architecture at a higher target resolution \(s_h\):

$$
\theta_h^*
=
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
$$

RGFT changes only the training procedure and introduces no additional inference-time detection branch.

---

## Repository Structure

```text
RIF-YOLO-N/
│
├── README.md
│
└── scripts/
    │
    ├── dataset_preparation/
    │   ├── README.md
    │   ├── check_vedai_yolo_labels.py
    │   └── convert_vedai_to_yolo.py
    │
    ├── evaluation/
    │   ├── README.md
    │   ├── benchmark_runtime_1024.py
    │   ├── generate_paper_results.py
    │   ├── make_vedai_visuals.py
    │   └── vedai_confusion_matrices.py
    │
    ├── models/
    │   ├── README.md
    │   ├── rif_yolo_n.yaml
    │   ├── rif_yolov8n_ablation_nogate.yaml
    │   └── rif_yolov8n_ablation_noscale.yaml
    │
    └── training/
        ├── README.md
        ├── train_model.py
        └── train_vedai_rgft_1024.py
```

---

## Repository Components

### Model Configurations

The `scripts/models/` directory contains the architecture definitions used in the experiments.

```text
rif_yolo_n.yaml
```

defines the final RIF-YOLO-N architecture.

The following configurations are used for component-level ablation:

```text
rif_yolov8n_ablation_nogate.yaml
rif_yolov8n_ablation_noscale.yaml
```

These variants evaluate the contribution of adaptive gating and identity-safe residual scaling.

---

### Dataset Preparation

The `scripts/dataset_preparation/` directory currently contains the VEDAI preparation utilities.

```text
convert_vedai_to_yolo.py
check_vedai_yolo_labels.py
```

`convert_vedai_to_yolo.py` converts the original VEDAI annotations into YOLO-compatible axis-aligned bounding boxes.

`check_vedai_yolo_labels.py` verifies the converted images and labels before training and evaluation.

The datasets themselves are **not included** in this repository and must be obtained from their original sources.

---

### Training

The `scripts/training/` directory contains the training utilities used for the reported experiments.

```text
train_model.py
train_vedai_rgft_1024.py
```

`train_model.py` provides the main model-training workflow.

`train_vedai_rgft_1024.py` performs the high-resolution RGFT stage for the final VEDAI \(1024 \times 1024\) experiment.

---

### Evaluation

The `scripts/evaluation/` directory contains utilities for runtime analysis, result generation, and detection diagnostics.

```text
benchmark_runtime_1024.py
generate_paper_results.py
make_vedai_visuals.py
vedai_confusion_matrices.py
```

The runtime benchmark uses:

- batch size 1;
- input resolution \(1024 \times 1024\);
- 50 GPU warm-up iterations;
- 200 synchronized forward passes.

Diagnostic utilities generate qualitative detections and normalized confusion matrices for the reported VEDAI experiments.

---

## Datasets

RIF-YOLO-N is evaluated on three datasets:

| Dataset | Classes | Role |
|---|---:|---|
| VisDrone | 10 | Primary benchmark |
| UAVDT | 3 | External validation |
| VEDAI | 9 | External validation |

The datasets are not redistributed with this repository.

For VEDAI, only visible-light aerial images are used. The original oriented annotations are converted to axis-aligned bounding boxes to match the detection formulation used by RIF-YOLO-N.

The prepared VEDAI split contains:

| Split | Images | Objects |
|---|---:|---:|
| Train | 968 | 2954 |
| Validation | 121 | 368 |
| Test | 121 | 365 |

The nine retained VEDAI classes are:

```text
car
truck
tractor
camping-car
van
other
pickup
boat
plane
```

---

## Main Results

All results below are reported at \(1024 \times 1024\).

### RIF-YOLO-N + RGFT

| Dataset | Precision | Recall | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|
| VisDrone | 0.520 | 0.424 | 0.415 | 0.243 |
| UAVDT | 0.832 | 0.838 | 0.862 | 0.543 |
| VEDAI | 0.799 | 0.586 | 0.669 | 0.419 |

### Comparison with YOLOv8n

| Dataset | YOLOv8n mAP50:95 | RIF-YOLO-N + RGFT mAP50:95 |
|---|---:|---:|
| VisDrone | 0.235 | **0.243** |
| UAVDT | 0.519 | **0.543** |
| VEDAI | 0.252 | **0.419** |

---

## Model Complexity

| Model | Parameters | GFLOPs |
|---|---:|---:|
| YOLOv8n | 3.008M | 8.1 |
| RIF-YOLO-N | 3.061M | 8.8 |

RIF-YOLO-N adds only 0.053M parameters while retaining the original three-scale anchor-free detection hierarchy.

---

## Runtime

Runtime was measured at \(1024 \times 1024\) with batch size 1 on an NVIDIA RTX 2000 Ada Generation Laptop GPU.

| Model | Mean Latency | FPS |
|---|---:|---:|
| YOLOv8n | \(18.272 \pm 6.489\) ms | 54.73 |
| RIF-YOLO-N | \(19.035 \pm 5.322\) ms | 52.54 |

The measured mean latency increase of RIF-YOLO-N over YOLOv8n is approximately 0.763 ms per image, corresponding to 4.17%.

---

## Experimental Environment

The reported experiments were conducted using:

```text
Operating System : Windows
Python           : 3.10
Framework        : PyTorch
Detection        : Ultralytics YOLO
GPU              : NVIDIA RTX 2000 Ada Generation Laptop GPU
CUDA             : Enabled
Primary seed     : 42
```

---

## Paper

**RIF-YOLO-N: Lightweight P3 Residual Identity Fusion with Resolution-Guided Fine-Tuning for Tiny-Object Detection**

The study evaluates VisDrone as the primary tiny-object detection benchmark and uses UAVDT and VEDAI for external validation.

Citation information will be added after publication.

---

## Acknowledgment

This work builds on the Ultralytics YOLO framework. Dataset ownership and licensing remain with the corresponding dataset authors and providers.
