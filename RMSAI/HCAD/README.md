# H²-CAD

A novel framework for Multimodal Sentiment Analysis with Incomplete Data.


## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- transformers
- numpy
- pyyaml

### Setup

```bash
# Create conda environment
conda create -n h2cad python=3.8
conda activate h2cad

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers numpy pyyaml scikit-learn
```

## Project Structure

```
H2CAD/
├── configs/
│   ├── train_sims.yaml      # SIMS dataset config
│   ├── train_mosi.yaml      # MOSI dataset config
│   └── train_mosei.yaml     # MOSEI dataset config
├── core/
│   ├── dataset.py           # Data loading utilities
│   ├── ema.py               # EMA Teacher implementation
│   ├── losses.py            # H2CADLoss composite loss
│   ├── metric.py            # Evaluation metrics
│   ├── scheduler.py         # Learning rate & distillation schedulers
│   └── utils.py             # Utility functions
├── models/
│   ├── h2cad.py             # Main H²-CAD model
│   ├── hyperbolic.py        # Poincaré ball operations
│   ├── basic_layers.py      # Transformer & fusion modules
│   └── bert.py              # BERT text encoder
├── train.py                 # Training script
├── robust_evaluation.py     # Robustness evaluation
└── README.md
```

## Usage

### Training

```bash
# Train on SIMS dataset
python train.py --config_file configs/train_sims.yaml

# Train on MOSI dataset
python train.py --config_file configs/train_mosi.yaml

# Train on MOSEI dataset
python train.py --config_file configs/train_mosei.yaml

# Train with specific seed
python train.py --config_file configs/train_mosi.yaml --seed 1111
```

### Evaluation

```bash
# Evaluate robustness under different missing rates
python robust_evaluation.py --config_file configs/eval_mosi.yaml --key_eval MAE
```

## Configuration

Key hyperparameters in config files:

```yaml
base:
  seed: 1111
  lr: 0.0001
  weight_decay: 0.0001
  batch_size: 64
  n_epochs: 200

  # Loss weights
  alpha: 0.9           # λ₁: Completeness loss weight
  beta_max: 0.1        # β_max: Max hyperbolic distillation weight
  gamma: 0.1           # λ₂: Reconstruction loss weight

  # Hyperbolic space
  curvature: 1.0       # Poincaré ball curvature c
  hyp_warmup_epochs: 10  # T_warmup
  hyp_margin: 0.1      # τ: Relaxation margin

  # EMA Teacher
  ema_decay: 0.999     # EMA decay rate

model:
  feature_extractor:
    bert_pretrained: 'bert-base-uncased'
    input_dims: [768, 35, 74]      # [text, video, audio]
    hidden_dims: [128, 128, 128]

  hyper_params:
    hyp_dim: 64        # Poincaré embedding dimension
    hyper_depth: 3     # Hyper-modality learning depth
```


```

## Datasets


**You can download from:** https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

**This link is provided by MMSA, not by us.**



## License

This project is licensed under the MIT License.

## Acknowledgments

- BERT implementation from HuggingFace Transformers
- Hyperbolic geometry operations inspired by geoopt library

## Why We Provide the Code

Since the original PRMF code repository is currently inaccessible, we have organized and released our previously archived version to facilitate use by relevant researchers.