<p align="center">
  <img src="figs/logo.png" alt="MemEIC Logo" width="600">
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
  <a href="https://huggingface.co/MemEIC/MemEIC-checkpoints">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-orange" alt="Hugging Face Checkpoints">
  </a>
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License">
  </a>
  <a href="https://huggingface.co/datasets/MemEIC/CCKEB">
    <img src="https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/datasets/MemEIC/CCKEB&query=$.downloads&label=Downloads&color=green" alt="Downloads">
  </a>
</p>

## 📢 News
- **[Oct 2025]** Paper accepted at NeurIPS 2025! 🎉
- **[Dec 2025]** We release our dataset, CCKEB
- **[Feb 2026]** We release our code for evaluation
- **[Mar 2026]** We release Stage 1 (Mem-E) & Stage 2 (Mem-I) checkpoints


## 📖 Overview

**MemEIC** introduces a novel approach for **Continual and Compositional Knowledge Editing** in Vision-Language Models (VLMs). Our method addresses the challenge of editing multiple knowledge types (visual and textual) while maintaining compositional reasoning capabilities.

### Key Features
- **MEM-E(Modality-Aware External Memory)**: Dual external memory for cross-modal evidence retrieval
- **MEM-I(Internal Separated Knowledge Integration)**: Separate Visual/Textual adapters on FFN layers
- **Knowledge Connector (KC)**: Attention-based LoRA that bridges visual and textual knowledge
- **CCKEB Dataset**: Compositional Chain Knowledge Editing Benchmark


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

### Pre-trained Models

Please download the models relevant to your experiment and place them in the `hugging_cache` directory.

#### 1. Common Requirements
*Required for all experiments.*
- `hugging_cache/bert-base-uncased`
- `hugging_cache/distilbert-base-cased`
- `hugging_cache/all-MiniLM-L6-v2`

#### 2. Model-Specific Requirements
*Download only the models you intend to use.*

| Model         | Required Files / Directories                                                                                                                               |
| :------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLaVA-1.5** | `hugging_cache/llava-v1.5-7b`<br>`hugging_cache/vicuna-7b-v1.5`<br>`openai/clip-vit-large-patch14-336`                                                     |
| **MiniGPT-4** | `hugging_cache/vicuna-7b`<br>`hugging_cache/pretrained_minigpt4_7b.pth`<br>`hugging_cache/blip2_pretrained_flant5xxl.pth`<br>`hugging_cache/eva_vit_g.pth` |
| **BLIP-2**    | `hugging_cache/opt-2.7b`<br>`hugging_cache/blip2_pretrained_opt2.7b.pth`<br>`hugging_cache/eva_vit_g.pth`                                                  |

> **Note**: You can download these models from [Hugging Face](https://huggingface.co/) or use the links provided in the [VLKEB repository](https://github.com/VLKEB/VLKEB).


## 📦 Dataset

### CCKEB (Compositional Chain Knowledge Editing Benchmark)

Download the dataset from 🤗 Hugging Face: [MemEIC/CCKEB](https://huggingface.co/datasets/MemEIC/CCKEB)

### Directory Structure Setup
Please organize your data as follows to ensure compatibility:

```
datasets/
├── CCKEB_train.json
├── CCKEB_eval.json
└── CCKEB_images/          # Image directory
    └── mmkb_images/       # Contains images 
```


## 🚀 Usage (Test)

### Baselines

#### Fine-Tuning (FT)
```bash
# LLaVA
python test_compositional_edit.py test_LLaVA_FT_comp

# MiniGPT4
python test_compositional_edit.py test_MiniGPT4_FT_comp
```

#### LoRA (single adapter in FFN)
```bash
# LLaVA
python test_compositional_edit.py test_LLaVA_one_lora_comp

# MiniGPT4
python test_compositional_edit.py test_MiniGPT4_one_lora_comp
```

#### SERAC (need to train first)
```bash
# LLaVA
python test_compositional_edit.py test_LLaVA_SERAC_comp

# MiniGPT4
python test_compositional_edit.py test_MiniGPT4_SERAC_comp
```

#### WISE
```bash
# LLaVA
python test_compositional_edit.py test_LLaVA_WISE_comp

# MiniGPT4
python test_compositional_edit.py test_MiniGPT4_WISE_comp
```

### MemEIC (Ours)

#### Stage 1: Mem-E (External Memory) Checkpoints

Download the pre-trained Stage 1 checkpoints from HuggingFace: [MemEIC/MemEIC-checkpoints](https://huggingface.co/MemEIC/MemEIC-checkpoints)

| Model | Checkpoint |
|-------|------------|
| LLaVA-1.5-7B | `llava_stage1.pt` |
| MiniGPT-4 | `minigpt4_stage1.pt` |

Place the downloaded checkpoints in the `checkpoints/` directory:

```bash
mkdir -p checkpoints
# Download from HuggingFace and place files here
```

```
MemEIC/
└── checkpoints/
    ├── llava_stage1.pt
    ├── minigpt4_stage1.pt
    ├── llava_stage2/
    │   ├── connector/
    │   ├── textual/
    │   └── visual/
    └── minigpt4_stage2/
        ├── connector/
        ├── textual/
        └── visual/
```

#### Stage 2: Knowledge Connector (KC)

The Knowledge Connector is an attention-based LoRA (`q_proj`, `k_proj`) that bridges visual and textual knowledge.

**Option 1: Download pre-trained adapters**

Download the pre-trained Stage 2 adapters from HuggingFace: [MemEIC/MemEIC-checkpoints](https://huggingface.co/MemEIC/MemEIC-checkpoints)

| Model | Adapter Directory |
|-------|-------------------|
| LLaVA-1.5-7B | `llava_stage2/` |
| MiniGPT-4 | `minigpt4_stage2/` |

Each adapter directory contains `connector/`, `textual/`, and `visual/` sub-folders.

**Option 2: Train from scratch**

```bash
# LLaVA
python test_compositional_edit.py train_LLaVA_OURS_stage2

# MiniGPT4
python test_compositional_edit.py train_MiniGPT4_OURS_stage2
```

#### Stage 3: MemEIC Evaluation

```bash
# LLaVA
python test_compositional_edit.py test_LLaVA_OURS_comp

# MiniGPT4
python test_compositional_edit.py test_MiniGPT4_OURS_comp
```

**Editing/Training Details:**
- Edit -> Visual/Textual LoRA: FFN layers (`down_proj`, `up_proj`)
- Training -> Knowledge Connector: Self-attention layers (`q_proj`, `k_proj`)
---

## 🤖 Supported Models

| Model        | Status         |
| ------------ | -------------- |
| LLaVA-1.5-7B | ✅ Supported    |
| MiniGPT4     | ✅ Supported    |
| BLIP2        | 🔧 Experimental |

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
├── test_compositional_edit.py # Main evaluation script
└── README.md
```

---

## 📜 License

This code & dataset are released under the **BSD-3-Clause**.

The CCKEB dataset is partially derived from the **VLKEB (NeurIPS'24)** dataset, which is licensed under the BSD 3-Clause License. All original copyright notices are preserved.


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

## 📧 Contact

For questions, issues, or contributions:

- **GitHub Issues**: Please open an issue on our [GitHub repository](https://github.com/MemEIC/MemEIC/issues)
- **Email**: Jin Seong (real_castle@etri.re.kr)

## 🔗 Related Works

We encourage citing the foundational works this project builds upon:

- **VLKEB**: [(NeurIPS'24) VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark](https://github.com/VLKEB/VLKEB)
- **EasyEdit**: [An easy-to-use knowledge editing framework for large language models](https://github.com/zjunlp/EasyEdit)

---

## 🙏 Acknowledgements

This project is built upon [EasyEdit](https://github.com/zjunlp/EasyEdit) and [VLKEB](https://github.com/VLKEB/VLKEB). We thank the authors for their excellent work.
