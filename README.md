# Attention-Guided TransUNet for Multi-Class Retinal Fluid Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Weights-blue)](https://huggingface.co/animeshakr/oct-fluid-segmentation)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-green)](https://huggingface.co/spaces/animeshakr/oct-fluid-segmentation)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

**Author:** Animesh Kumar | Newcastle University MSc Advanced Computer Science 2025–26  
**Target:** OMIA 2026 Workshop at MICCAI + medRxiv preprint  
**Framework:** PyTorch | **Compute:** Google Colab H100

---

## Clinical Problem

Automated segmentation of three retinal fluid types in OCT scans:

| Fluid | Full Name | Clinical Significance |
|-------|-----------|----------------------|
| **IRF** | Intraretinal Fluid | Active diabetic macular oedema and wet AMD. Requires anti-VEGF within days. |
| **SRF** | Subretinal Fluid | Guides anti-VEGF injection frequency. |
| **PED** | Pigment Epithelial Detachment | Key AMD progression marker. |

Manual grading takes **20–40 minutes per volume** with up to **15% inter-grader variability**.

---

## Architecture

Two-model ensemble: **AttentionTransUNet V2S (22M)** + **AttentionTransUNetL V2L (127M)**

| Component | V2S | V2L |
|-----------|-----|-----|
| Encoder | EfficientNetV2S | EfficientNetV2L |
| Encoder channels | s1=24 s2=48 s3=64 s4=160 bot=256 | s1=32 s2=64 s3=96 s4=192 bot=640 |
| Transformer d_model | 256 | 512 |
| Attention heads | 16 | 16 |
| Parameters | ~22M | ~127M |
| Phase B val Dice | 0.7443 | 0.7913 |

### Novel Contributions

1. **UCUS** — Uncertainty-Weighted Clinical Urgency Score (Monitor / Review / Urgent)
2. **Dual uncertainty** — MC Dropout (20 passes) + inter-model disagreement
3. **Source-Adaptive BatchNorm** — per-scanner domain adaptation (5 sources)
4. **Multi-source evaluation** — DUKE, AROI, UMN-AMD, UMN-DME

---

## Results

### Multi-Seed Validation Dice (seeds 42 / 123 / 2024)

| Model | IRF | SRF | PED | Mean Fluid |
|-------|-----|-----|-----|-----------|
| V2S | 0.866 ± 0.007 | 0.827 ± 0.005 | 0.518 ± 0.009 | 0.737 ± 0.005 |
| **V2L** | **0.916 ± 0.003** | **0.856 ± 0.003** | **0.581 ± 0.018** | **0.784 ± 0.006** |

### Test Set (503 slices, 4 sources)

| Metric | Mean | Std |
|--------|------|-----|
| dice_IRF | 0.2043 | ±0.3482 |
| dice_SRF | 0.1712 | ±0.2359 |
| dice_PED | 0.4463 | ±0.4611 |
| dice_mean_fluid | 0.2739 | ±0.2161 |

### Per-Source Breakdown

| Source | IRF | SRF | PED | Mean |
|--------|-----|-----|-----|------|
| AROI | 0.054 | 0.299 | 0.144 | 0.166 |
| DUKE | 0.071 | 0.000 | **0.902** | 0.324 |
| UMN | 0.381 | 0.176 | 0.409 | 0.322 |

> DUKE SRF=0.000 is expected — DUKE contains only IRF annotations.

### Clinical Safety

| Metric | Value |
|--------|-------|
| Inter-grader human ceiling | 0.9030 |
| Uncertainty ratio | **1.34×** (p=3.77e-05) ✅ |
| SRF volume correlation | **r=0.778** (p=6.33e-04) ✅ |
| PED volume correlation | **r=0.841** (p=8.64e-05) ✅ |

### Ablation Study

| Variant | Mean Dice |
|---------|-----------|
| V2S only | 0.338 |
| **V2L only** | **0.449** |
| V2S + V2L ensemble | 0.407 |
| Full (V2S+V2L+MC+TTA) | 0.293 |

### INT8 Quantisation

| Model | FP32 | INT8 | Compression |
|-------|------|------|-------------|
| V2L | 510 MB | 132 MB | 3.9× |
| V2S | 91 MB | 24 MB | 3.8× |

---

## Datasets

| Dataset | Volumes | Classes | Disease | Scanner |
|---------|---------|---------|---------|---------|
| DUKE DME | 10 | IRF | DME | Spectralis |
| AROI | 24 patients | IRF+SRF+PED | AMD | Zeiss Cirrus |
| UMN AMD | 24 | SRF | AMD | Spectralis |
| UMN DME | 29 | IRF | DME | Spectralis |

**Split:** Train 4983 slices | Val 552 slices | Test 503 slices

---

## Training

### Phase A — Decoder only (5 epochs)
- Encoder frozen, LR=1e-3, Adam, batch=8

### Phase B — Full fine-tuning (25 epochs)
- Encoder blocks 3–5 unfrozen, LR=1e-4, WarmupCosineDecay
- Batch=4, early stopping patience=7
- Loss: Dice + 0.5 × CrossEntropy

---

## Repository Structure

```
oct-fluid-segmentation/
├── app.py                      ← Streamlit dashboard (5 tabs)
├── requirements.txt            ← HuggingFace Space dependencies
├── LICENSE                     ← MIT License
├── .gitignore
└── assets/
    ├── results/
    │   ├── ablation_results.csv
    │   ├── multiseed_results.csv
    │   └── per_source_results.csv
    └── visualisations/
        ├── segmentation_grid.png
        ├── uncertainty_histogram.png
        ├── ablation_bar.png
        ├── multiseed_violin.png
        └── per_source_heatmap.png
```

> Model weights (.pth, .onnx) are hosted on HuggingFace — too large for GitHub (up to 1526MB each).

---

## Usage

### Load Best Model

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download(
    repo_id="animeshakr/oct-fluid-segmentation",
    filename="ckpt_phaseB_V2L_s123.pth"
)
ck = torch.load(path, map_location="cpu")
# ck["val_dice"] = 0.7913
# ck["epoch"]    = 25
```

### Run Dashboard Locally

```bash
pip install streamlit huggingface_hub onnxruntime opencv-python-headless plotly
streamlit run app.py
```

---

## Model Weights

All weights on HuggingFace: [animeshakr/oct-fluid-segmentation](https://huggingface.co/animeshakr/oct-fluid-segmentation)

| File | Size | Description |
|------|------|-------------|
| `ckpt_phaseB_V2L_s123.pth` | 1526 MB | Best V2L — val_dice=0.7913 |
| `ckpt_phaseB_V2L_s2024.pth` | 1526 MB | V2L seed=2024 — val_dice=0.7841 |
| `ckpt_phaseB_V2S_s42.pth` | 271 MB | Best V2S — val_dice=0.7443 |
| `ckpt_phaseB_V2L_s123_int8.pth` | 132 MB | INT8 quantised V2L |
| `ckpt_phaseB_V2S_s42_int8.pth` | 24 MB | INT8 quantised V2S |
| `deployment/slot*.onnx` | — | ONNX for TensorRT/OpenVINO |
| `demo_results.json` | 87 MB | 20 precomputed demo samples |

---

## Citation

```bibtex
@misc{kumar2026octseg,
  title={Attention-Guided TransUNet for Multi-Class Retinal Fluid Segmentation
         in OCT with MC Dropout Uncertainty Quantification},
  author={Kumar, Animesh A.},
  institution={Newcastle University, UK},
  year={2026},
  note={MSc Advanced Computer Science. Targeting OMIA 2026 Workshop at MICCAI.}
}
```

---

## Related Papers

- Bogunovic et al. (2019) — RETOUCH Challenge, IEEE TMI
- Ronneberger et al. (2015) — U-Net
- Schlemper et al. (2019) — Attention U-Net
- Chen et al. (2021) — TransUNet
- Rasti et al. (2022) — RetiFluidNet (current SOTA on RETOUCH)

---

## License

MIT License — see [LICENSE](LICENSE)
