# Discrete Flow Matching for Text Generation

This repository contains an implementation of discrete flow matching techniques applied to text generation tasks, specifically focusing on the TinyStories dataset. The project explores the development and evaluation of discrete flow matching models, including experiments on hybrid flow matching approaches that combine empirical discrete flow matching with trained models.

## Project Overview

The primary objective of this project is to implement discrete flow matching methods and conduct experiments to assess their effectiveness in text generation. Discrete flow matching extends continuous flow matching paradigms to discrete state spaces, enabling non-autoregressive generation of high-quality text. This work includes:

- Implementation of Campbell et al. discrete flow models with masked and uniform noise schedules
- Empirical discrete flow matching that leverages training data distributions
- Hybrid flow matching that combines empirical and trained models
- General flow framework based on Gat et al. with flexible probability paths and schedulers

## Key Features

### Core Implementations

- **Campbell Flow Models** (`src/flow/campbell_flow.py`): Implementation of masked and uniform flow matching models following Campbell et al. (2024). Includes `MaskedFMModel` and `UniformFMModel` classes that define probability paths from noise distributions to data distributions.

- **Empirical Discrete Flow Matching** (`src/flow/empirical_dfm.py`): A data-driven approach that computes flow velocities directly from the training dataset. This module provides `EmpiricalDFM` class that can use masked or uniform initial distributions.

- **Hybrid Flow Matching** (`src/flow/hybrid_flow.py`): Novel approach combining empirical DFM for early stages of generation with trained models for final stages. The `HybridMaskedSampler` class allows switching between empirical and trained models at a configurable time point `tau`.

- **General Flow Framework** (`src/flow/general_flow.py`): Implementation of the general flow matching framework from Gat et al. (2024), supporting multiple probability paths, weight schedulers, and corrector schedulers for improved generation quality.

- **Transformer Architectures** (`src/flow/transformer.py`): Custom transformer implementations including `IgnorantTransformer`, `TimeAwareTransformer`, and `SmallModel` designed for flow matching tasks with timestep embeddings.

### Evaluation Tools

- **Evaluation Module** (`src/flow/eval.py`): Comprehensive evaluation tools including `SentenceBERTEvaluator` for measuring semantic similarity between generated and training samples, and perplexity evaluation using language models.

## Repository Structure

```
flow/
├── src/flow/              # Core implementation modules
│   ├── campbell_flow.py   # Campbell et al. flow models
│   ├── empirical_dfm.py   # Empirical discrete flow matching
│   ├── hybrid_flow.py     # Hybrid flow matching sampler
│   ├── general_flow.py    # General flow framework (Gat et al.)
│   ├── transformer.py     # Transformer architectures
│   ├── eval.py            # Evaluation utilities
│   └── utils.py           # Configuration and utilities
├── scripts/               # Training and evaluation scripts
│   ├── train_camp_tinystories.py           # Training script for Campbell flow
│   ├── train_gat_tinystories.py            # Training script for Gat flow
│   ├── evaluate_model.py                   # Model evaluation
│   ├── evaluate_hybrid_tau.py             # Hybrid sampler experiments
│   ├── analyze_entropy_evolution.py        # Entropy analysis
│   └── test_hybrid_sampler.py              # Hybrid sampler testing
├── figures/               # Generated visualizations
│   ├── entropy_heatmap.png
│   ├── entropy_vs_time_average.png
│   ├── entropy_vs_time_positions.png
│   ├── entropy_difference.png
│   └── tau_vs_distance.png
├── results/               # Experimental results and data
└── pyproject.toml         # Project dependencies
```

## Requirements and Setup

### Python Version

This project requires Python 3.11 (specifically `>=3.11, <3.12`).

### Dependencies

Install dependencies using pip:

```bash
pip install torch transformers datasets tqdm numpy einops matplotlib wandb sentence-transformers
```

Or install from the project configuration:

```bash
pip install -e .
```

### Key Dependencies

- `torch`: PyTorch for deep learning
- `transformers`: Hugging Face transformers for tokenizers and models
- `datasets`: Dataset loading and processing
- `sentence-transformers`: For semantic similarity evaluation
- `wandb`: Experiment tracking (optional)

## Usage Examples

### Training a Campbell Flow Model

Train a masked flow matching model on TinyStories:

```python
from flow.transformer import IgnorantTransformer
from flow.utils import Config
from flow.campbell_flow import MaskedFMModel
from transformers import AutoTokenizer

# Setup configuration
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
config = Config(
    num_tokens=len(tokenizer),
    embed_dim=512,
    mlp_dim=2048,
    context_len=32,
    num_layers=16,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Create model
model = IgnorantTransformer(config)
flow_model = MaskedFMModel(config, model, mask_token_id=tokenizer.mask_token_id)

# Training loop (see scripts/train_camp_tinystories.py for full example)
```

### Sampling from a Trained Model

Generate text samples using a trained flow model:

```python
# Load trained model
model.load_state_dict(torch.load("checkpoints/model.pt"))
flow_model.eval()

# Generate samples
with torch.inference_mode():
    samples = flow_model.sample(bs=10, dt=0.001, temperature=1.0)
    
# Decode to text
for sample in samples:
    text = tokenizer.decode(sample.cpu().numpy(), skip_special_tokens=True)
    print(text)
```

### Hybrid Flow Matching

Use hybrid sampling that combines empirical and trained models:

```python
from flow.hybrid_flow import HybridMaskedSampler
from flow.empirical_dfm import EmpiricalDFM
from torch.utils.data import DataLoader

# Setup empirical DFM
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
empirical_dfm = EmpiricalDFM(
    config, 
    dataloader, 
    mask_token_id=tokenizer.mask_token_id,
    initial_type="mask"
)

# Create hybrid sampler
hybrid_sampler = HybridMaskedSampler(config, empirical_dfm, trained_model)

# Sample with tau=0.5 (switch from empirical to trained at midpoint)
samples = hybrid_sampler.sample(bs=10, tau=0.5, dt=0.001)
```

### Evaluating Model Quality

Evaluate generated samples using semantic similarity:

```python
from flow.eval import SentenceBERTEvaluator

evaluator = SentenceBERTEvaluator(device=config.device)
distances, closest_indices = evaluator.calculate_distance_batch(generated_texts)
print(f"Mean distance to training set: {distances.mean():.4f}")
```

## Experiments

### Entropy Evolution Analysis

The project includes analysis of entropy evolution throughout the flow matching process, comparing empirical DFM and trained Campbell flow models. Results are visualized in:

- `figures/entropy_heatmap.png`: Entropy heatmap across positions and timesteps
- `figures/entropy_vs_time_average.png`: Average entropy over time
- `figures/entropy_vs_time_positions.png`: Entropy evolution at different positions
- `figures/entropy_difference.png`: Difference between empirical and trained models

Run the analysis:

```bash
python scripts/analyze_entropy_evolution.py
```

### Hybrid Sampler Tau Evaluation

Experiments evaluate the effect of the transition point `tau` in hybrid flow matching, measuring the distance of generated samples to the training dataset. Results show how different `tau` values affect generation quality:

- `figures/tau_vs_distance.png`: Visualization of distance vs tau values
- `results/tau_distance_data.pt`: Raw experimental data

Run the evaluation:

```bash
python scripts/evaluate_hybrid_tau.py
```

## References

This work builds upon the following foundational papers:

1. **Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R. T. Q., Synnaeve, G., Adi, Y., & Lipman, Y. (2024).** Discrete Flow Matching. *arXiv preprint arXiv:2407.15595*.  
   [Paper](https://arxiv.org/pdf/2407.15595)

2. **Campbell, A., et al. (2024).** Generative Flows on Discrete State-Spaces. *arXiv preprint arXiv:2402.04997*.  
   [Paper](https://arxiv.org/pdf/2402.04997)

## Project Status

This project is currently in the proof of concept/experimental phase. The implementation includes core discrete flow matching methods and initial experiments on the TinyStories dataset. Ongoing work includes:

- Refining model architectures and training procedures
- Conducting comprehensive evaluations of hybrid flow matching
- Exploring different probability paths and schedulers
- Analyzing the trade-offs between empirical and trained models

## Acknowledgments

We acknowledge the authors of the referenced papers for their foundational contributions to discrete flow matching and text generation. Their theoretical frameworks and algorithmic insights have been instrumental in guiding this implementation.

