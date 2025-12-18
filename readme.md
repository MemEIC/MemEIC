<p align="center">
  <img src="logo.png" alt="MemEIC Logo" width="600">
</p>

<h2 align="center">MemEIC: A Step Toward Continual and Compositional Knowledge Editing</h2>

<p align="center">
  <b>NeurIPS 2025</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.25798">
    <img src="https://img.shields.io/badge/arXiv-2510.25798-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/MemEIC/CCKEB">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow" alt="Hugging Face Datasets">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  </a>
</p>

---

## 📢 News

- **[Dec 2025]** Code will be released (~01.2026)
- **[Oct 2025]** Paper accepted at NeurIPS 2025! 🎉

---

## 📖 Overview

**MemEIC** introduces a novel approach for **Continual and Compositional Knowledge Editing** in Vision-Language Models (VLMs). Our method addresses the challenge of editing multiple knowledge types (visual and textual) while maintaining compositional reasoning capabilities.

### Key Features
- **MEM-E(Modality-Aware External Memory)**: Dual external memory for cross-modal evidence retrieval
- **MEM-I(Internal Separated Knowledge Integration)**: Separate Visual/Textual adapters on FFN layers
- **Knowledge Connector (KC)**: Attention-based LoRA that bridges visual and textual knowledge
- **CCKEB Dataset**: Compositional Chain Knowledge Editing Benchmark

---

## 🛠️ Installation

### Requirements
- Python >= 3.11
- PyTorch >= 2.0.1
- CUDA >= 11.7

### Setup

```bash
# Clone the repository
git clone https://github.com/MemEIC/MemEIC.git
cd MemEIC

# Create conda environment
conda create -n memeic python=3.11
conda activate memeic

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset

### CCKEB (Compositional Chain Knowledge Editing Benchmark)

CCKEB is designed to evaluate **compositional knowledge editing** capabilities, where models must reason over both visual and textual edits.

Download from 🤗 Hugging Face: [MemEIC/CCKEB](https://huggingface.co/datasets/MemEIC/CCKEB)

```bash
# Download dataset
mkdir -p datasets
# Place CCKEB_train.json and CCKEB_eval.json in datasets/
```

**Dataset Structure:**
- `CCKEB_train.json`: Training data for Knowledge Connector learning
- `CCKEB_eval.json`: Evaluation data for compositional editing

---

## 🚀 Usage

### Baselines

#### Fine-Tuning (FT)
```bash
# LLaVA
python test_sequential_editing.py test_LLaVA_FT_Composition_0

# MiniGPT4
python test_sequential_editing.py test_MiniGPT4_FT_composition_0
```

#### LoRA (Single Adapter)
```bash
# LLaVA
python test_sequential_editing.py test_LLaVA_CompositionalEdit_one_lora

# MiniGPT4
python test_sequential_editing.py test_MiniGPT4_CompositionalEdit_one_lora
```

### MemEIC (Ours)

#### Stage 1: Mem-E (External Memory) Training

> 🔧 **Coming Soon**: Pre-trained checkpoints will be released.

Stage 1 trains **Mem-E** - a hybrid external-internal editor that combines:
- **Dual External Memory**: Cross-modal evidence retrieval for visual and textual knowledge

```bash
# Placeholder - checkpoints will be provided
# Mem-E: trained on CCKEB training data
```

#### Stage 2: Knowledge Connector Training

The Knowledge Connector is an attention-based LoRA (`q_proj`, `k_proj`) that learns to compose visual and textual knowledge.

```bash
# LLaVA - Knowledge Connector Training (RAG 70% threshold)
python test_sequential_editing.py test_LLaVA_CompositionalEdit_Connector_attention_rag_70

# MiniGPT4 - Knowledge Connector Training (RAG 50% threshold)
python test_sequential_editing.py test_MiniGPT4_CompositionalEdit_Connector_attention_rag_50
```

**Editing/Training Details:**
- Edit -> Visual/Textual LoRA: FFN layers (`down_proj`, `up_proj`)
- Training -> Knowledge Connector: Self-attention layers (`q_proj`, `k_proj`)
---

## 🤖 Supported Models

| Model | Status |
|-------|--------|
| LLaVA-1.5-7B | ✅ Supported |
| MiniGPT4 | ✅ Supported |
| BLIP2 | 🔧 Experimental |

---

## 📁 Project Structure

```
MemEIC/
├── datasets/                  # Dataset files
├── easyeditor/               # Core editing framework
│   ├── trainer/              # Training modules
│   │   ├── algs/            # Editing algorithms (FT, LoRA, etc.)
│   │   ├── llava/           # LLaVA model support
│   │   ├── blip2_models/    # BLIP2/MiniGPT4 model support
│   │   └── MultimodalTrainer.py
│   └── dataset/              # Dataset loaders
├── hparams/                   # Hyperparameter configs
│   ├── FT/                   # Fine-tuning configs
│   ├── LORA/                 # LoRA baseline configs
│   └── OURS/                 # MemEIC configs
│       └── stage2/           # Knowledge Connector configs
├── test_sequential_editing.py # Main evaluation script
└── README.md
```

---

## 📜 License

This code & dataset are released under the **Apache License 2.0**.

The CCKEB dataset is partially derived from the **VLKEB (NeurIPS'24)** dataset, which is licensed under the BSD 3-Clause License. All original copyright notices are preserved.

---

## 🖊️ Citation

If you use this code or dataset, please cite our paper:

```bibtex
@inproceedings{
seong2025memeic,
title={Mem{EIC}: A Step Toward Continual and Compositional Knowledge Editing},
author={Jin Seong and Jiyun Park and Wencke Liermann and Hongseok Choi and Yoonji Nam and Hyun Kim and Soojong Lim and Namhoon Lee},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Qvj8s2rRUs}
}
```

---

## 🔗 Related Works

We encourage citing the foundational works this project builds upon:

- **VLKEB**: [(NeurIPS'24) VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark](https://github.com/VLKEB/VLKEB)
- **EasyEdit**: [An easy-to-use knowledge editing framework for large language models](https://github.com/zjunlp/EasyEdit)

---

## 🙏 Acknowledgements

This project is built upon [EasyEdit](https://github.com/zjunlp/EasyEdit) and [VLKEB](https://github.com/VLKEB/VLKEB). We thank the authors for their excellent work.


