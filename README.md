# Attention-Guided TransUNet for Multi-Class Retinal Fluid Segmentation

**Author:** Animesh Kumar | Newcastle University MSc Advanced Computer Science 2025–26  
**Target:** medRxiv preprint  
**Framework:** PyTorch | **Compute:** Google Colab H100

---

## Clinical Problem

Automated segmentation of three retinal fluid types in OCT scans:
- **IRF** (Intraretinal Fluid) — indicates active diabetic macular oedema and wet AMD
- **SRF** (Subretinal Fluid) — guides anti-VEGF injection frequency  
- **PED** (Pigment Epithelial Detachment) — monitors AMD progression

Manual grading takes 20–40 minutes per volume with up to 15% inter-grader variability.

---

## Architecture

Two-model ensemble: **AttentionTransUNet V2S (22M params)** + **AttentionTransUNetL V2L (127M params)**

| Component | V2S | V2L |
|-----------|-----|-----|
| Encoder | EfficientNetV2S | EfficientNetV2L |
| Bottleneck channels | 256 | 640 |
| Transformer d_model | 256 | 512 |
| Attention heads | 16 | 16 |
| Parameters | ~22M | ~127M |

**Novel contributions:**
1. UCUS — Uncertainty-Weighted Clinical Urgency Score
2. Dual uncertainty — MC Dropout + inter-model disagreement
3. Source-Adaptive BatchNorm — per-scanner domain adaptation
4. Multi-source evaluation — DUKE, AROI, UMN, OPTIMA

---

## Results

### Multi-Seed Validation Dice (mean ± std across seeds 42/123/2024)

| Model | IRF | SRF | PED | Mean Fluid |
|-------|-----|-----|-----|-----------|
| V2S | 0.866 ± 0.007 | 0.827 ± 0.005 | 0.518 ± 0.009 | 0.737 ± 0.005 |
| V2L | 0.916 ± 0.003 | 0.856 ± 0.003 | 0.581 ± 0.018 | **0.784 ± 0.006** |

### Test Set Results (503 slices, 4 sources)

| Metric | Value |
|--------|-------|
| dice_IRF | 0.2043 ± 0.3482 |
| dice_SRF | 0.1712 ± 0.2359 |
| dice_PED | 0.4463 ± 0.4611 |
| dice_mean_fluid | 0.2739 ± 0.2161 |

### Per-Source Breakdown

| Source | IRF | SRF | PED | Mean |
|--------|-----|-----|-----|------|
| AROI | 0.054 | 0.299 | 0.144 | 0.166 |
| DUKE | 0.071 | 0.000 | 0.902 | 0.324 |
| UMN | 0.381 | 0.176 | 0.409 | 0.322 |

### Clinical Safety Metrics

| Metric | Value |
|--------|-------|
| Inter-grader human ceiling | 0.9030 |
| Uncertainty ratio (disagreement vs agreement) | 1.34x (p=3.77e-05) |
| SRF volume correlation | r=0.778 (p=6.33e-04) |
| PED volume correlation | r=0.841 (p=8.64e-05) |

### Ablation Study

| Variant | Mean Dice |
|---------|-----------|
| V2S only | 0.338 |
| V2L only | **0.449** |
| V2S + V2L ensemble | 0.407 |
| Full (V2S+V2L+MC+TTA) | 0.293 |

### INT8 Quantisation (Phase 5B)

| Model | FP32 | INT8 | Compression |
|-------|------|------|-------------|
| V2L | 510MB | 132MB | 3.9x |
| V2S | 91MB | 24MB | 3.8x |

---

## Datasets

| Dataset | Volumes | Classes | Disease |
|---------|---------|---------|---------|
| DUKE DME | 10 | IRF | DME |
| AROI | 24 patients | IRF+SRF+PED | AMD |
| UMN AMD | 24 | SRF | AMD |
| UMN DME | 29 | IRF | DME |

---

## Model Weights

Weights hosted on HuggingFace: [animesh-kumar/oct-fluid-segmentation](https://huggingface.co/animesh-kumar/oct-fluid-segmentation)

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download(
    repo_id="animesh-kumar/oct-fluid-segmentation",
    filename="ckpt_phaseB_V2L_s123.pth"
)
ck = torch.load(path, map_location="cpu")
```

---

## Citation

@misc{kumar2026octseg,
title={Attention-Guided TransUNet for Multi-Class Retinal Fluid Segmentation
in OCT with MC Dropout Uncertainty Quantification},
author={Kumar, Animesh A.},
institution={Newcastle University},
year={2026}
}

