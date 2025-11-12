# MAFM³: Modular Adaptation of Foundation Models for Multi-Modal Medical AI

<!-- [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) -->
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation of **MAFM³** (Modular Adaptation of Foundation Models for Multi-Modal Medical AI), a framework that enables a single foundation model to adapt to diverse domains, tasks, and modalities using lightweight modular components.

## 🎯 Overview

MAFM³ addresses the challenge of data scarcity in medical imaging by allowing a single pre-trained foundation model (CT-CLIP) to be efficiently adapted to multiple tasks (prognosis, segmentation) and modalities (CT, PET) through lightweight modular components, rather than training separate models from scratch.

### Key Features

- **Single Foundation Model**: Uses CT-CLIP as the base model for all adaptations
- **Lightweight Modular Components**: Minimal trainable parameters for task-specific adaptation
- **Multi-Task Support**: Adapts to both survival prediction (prognosis) and segmentation tasks
- **Multi-Modal Capability**: Supports CT scans, PET scans, and radiology text reports
- **Efficient Training**: Foundation model remains frozen; only modular components are trained

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Prognosis Module](#prognosis-module)
  - [Segmentation Module](#segmentation-module)
- [Methodology](#methodology)
- [Results](#results)
<!-- - [Citation](#citation)
- [License](#license) -->

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU
- PyTorch 2.4+

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Areeb2735/CTscan_prognosis_VLM.git
cd CTscan_prognosis_VLM
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install CT-CLIP and transformer-maskgit packages:**
```bash
# Navigate to transformer_maskgit directory
cd CT-CLIP/transformer_maskgit
pip install -e .

# Return to root and navigate to CT_CLIP directory
cd ../CT_CLIP
pip install -e .

# Return to root
cd ../..
```

4. **Download the pre-trained CT-CLIP model:**
   - Place `CT-CLIP_v2.pt` in the `docs/` directory
   - The model can be obtained from the [CT-CLIP repository](https://github.com/ibrahimethemhamamci/CT-CLIP)

## 📁 Project Structure

```
CTscan_prognosis_VLM/
├── CT-CLIP/                    # CT-CLIP foundation model code
│   ├── CT_CLIP/               # CT-CLIP package
│   │   └── ct_clip/           # Core CT-CLIP implementation
│   ├── scripts/               # Training and inference scripts
│   │   ├── main.py            # Prognosis training script
│   │   ├── main_segmentation.py  # Segmentation training script
│   │   ├── embedding_model_seg.py  # Feature extraction for segmentation
│   │   ├── prognosis_model.py    # Prognosis module architecture
│   │   ├── segmentation_model_again.py  # Segmentation module architecture
│   │   ├── data_inference_hector.py  # Dataset classes
│   │   ├── monai_dataset.py   # MONAI dataset utilities
│   │   └── utils.py           # Utility functions
│   └── transformer_maskgit/  # Vision transformer components
├── dataset/                   # Dataset preprocessing scripts
│   └── hector_pre_process.py
├── docs/                      # Documentation and model weights
│   ├── CT-CLIP_v2.pt         # Pre-trained CT-CLIP model (not included)
│   └── TNM_hector_prompts.csv # Sample data file
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Usage

### Prognosis Module

The prognosis module adapts CT-CLIP for survival prediction using multi-task logistic regression (MTLR).

#### Training

```bash
cd CT-CLIP/scripts
python main.py
```

**Key Components:**
- **Foundation Model**: Pre-trained CT-CLIP (frozen)
- **Modular Component**: Two fully connected layers (512 → 1024 → 512) + MTLR head
- **Input**: CT volumes + radiology text reports
- **Output**: Survival probabilities and risk scores

**Configuration:**
- Modify paths in `main.py`:
  - CT-CLIP model path (line 59)
  - Dataset paths (lines 63-64)
  - Training parameters (batch size, epochs, learning rate)

#### Model Architecture

```python
# From prognosis_model.py
class prognosis_model(nn.Module):
    def __init__(self, clip, num_time_bins, device):
        self.clip = clip  # Frozen foundation model
        self.fc = nn.Linear(512, 1024)      # Modular component
        self.fc_2 = nn.Linear(1024, 512)     # Modular component
        self.mtlr = MTLR(512, num_time_bins) # Task-specific head
```

### Segmentation Module

The segmentation module adapts CT-CLIP for 3D medical image segmentation using a UNETR-style decoder.

#### Pre-computing Embeddings

First, extract hidden states from the foundation model:

```bash
cd CT-CLIP/scripts
python embedding_model_seg.py
```

This generates multi-scale features (layers 3, 6, 9, 12) and saves them for efficient training.

#### Training

```bash
python main_segmentation.py --name experiment_name
```

**Key Components:**
- **Foundation Model**: Pre-trained CT-CLIP (frozen, used for feature extraction)
- **Modular Component**: UNETR decoder with multi-scale skip connections
- **Input**: CT volumes + pre-computed hidden states
- **Output**: Segmentation masks (tumor/lymph node classes)

**Model Architecture:**

```python
# From segmentation_model_again.py
class UNETR(nn.Module):
    def forward(self, image, hidden_state):
        z3, z6, z9, z12 = torch.unbind(hidden_state, dim=1)  # Multi-scale features
        # Progressive upsampling with skip connections
        # Final segmentation output
```

## 🔬 Methodology

### MAFM³ Framework

MAFM³ follows a modular adaptation strategy:

1. **Foundation Model**: CT-CLIP (pre-trained on CT-RATE dataset)
   - Image encoder: CTViT (3D Vision Transformer)
   - Text encoder: BiomedVLP-CXR-BERT
   - Trained for image-text contrastive learning

2. **Modular Components**: Task-specific lightweight adapters
   - **Prognosis**: MLP layers + MTLR head (~1M parameters)
   - **Segmentation**: UNETR decoder (~10M parameters)

3. **Training Strategy**:
   - Foundation model: **Frozen** (no gradients)
   - Modular components: **Trainable** (only these are optimized)
   - Efficient: Minimal memory and compute requirements

### Multi-Modal Support

The framework supports:
- **CT scans**: Primary imaging modality
- **PET scans**: Additional modality (5% Dice improvement)
- **Radiology text**: Clinical reports and descriptions

See `main_ctpt.py` for CT+PET multi-modal implementation.

## 📊 Results

### Prognosis Task
- **Metric**: Concordance Index (C-index)
- **Performance**: Improved survival prediction using multi-modal inputs (CT + text)

### Segmentation Task
- **Metric**: Dice Score
- **Performance**: 
  - Baseline: [Report baseline results]
  - With PET: Improvments in Dice score compared to CT-only

<!-- ## 📝 Citation

If you use MAFM³ in your research, please cite:

```bibtex
@article{mafm3_2024,
  title={MAFM³: Modular Adaptation of Foundation Models for Multi-Modal Medical AI},
  author={[Your Name] and [Co-authors]},
  journal={[Journal/Conference]},
  year={2024}
}
``` -->

<!-- ## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

See [LICENSE](LICENSE) for details. -->

## 🙏 Acknowledgments

- [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) for the foundation model
- [CT-RATE Dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) for training data

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [mohammad.qazi@gmail.com](mailto:mohammad.qazi@mbzuai.ac.ae).

## 🔗 Related Work

- [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP): The foundation model used in this work
- [CT-CHAT](https://github.com/ibrahimethemhamamci/CT-CHAT): Visual-language chat model for 3D chest CT volumes
