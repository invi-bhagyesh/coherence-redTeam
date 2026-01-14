# Multi-Trajectory Coherence Detector

Detects deceptive/buggy LLM reasoning via cross-question coherence, then analyzes which attention heads drive the coherence signal. No ground-truth labels required at inference time.

## Overview

The core insight: **honest answers to related questions should be more mutually coherent than deceptive ones**. We measure this using a PMI-like metric comparing joint probability vs product of marginals.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (for generation)
export OPENAI_API_KEY="your-key"

# Run minimal test
python -m coherence_detector.main --num-questions 20 --num-clusters 5

# Full run (requires GPU)
python -m coherence_detector.main
```

## Pipeline

1. **Cluster Questions** - Group semantically similar questions
2. **Generate Answers** - Create honest and deceptive answer sets
3. **Score Coherence** - Compute PMI-like coherence metric
4. **Statistical Analysis** - Logistic regression + AUROC
5. **Mech-Interp** - Identify attention heads driving coherence

## Project Structure

```
coherence_detector/
├── config.py      # Hyperparameters and paths
├── data.py        # Question loading and clustering
├── generate.py    # Honest/deceptive answer generation
├── coherence.py   # PMI-based coherence scoring
├── interp.py      # TransformerLens head analysis
├── main.py        # End-to-end pipeline
└── utils.py       # Logging and I/O helpers
```

## Key Metric

```
coherence = log P(A1, A2, A3, A4 | Q1, Q2, Q3, Q4)
          - Σ log P(Ai | Qi)
```

Positive coherence means answers "make more sense" together than apart. Honest answers should show higher coherence than deceptive ones.

## Outputs

- `outputs/clusters.json` - Question clusters with both answer types
- `outputs/scores.csv` - Coherence scores per cluster
- `outputs/regression_results.txt` - Statistical analysis
- `outputs/interp_heads.json` - Top coherence-contributing heads

## Configuration

Modify `coherence_detector/config.py` or use CLI args:

```python
Config(
    generator_model="gpt-4o-mini",  # Answer generator
    judge_model="Qwen/Qwen2-0.5B", # Coherence scorer
    num_questions=200,
    cluster_size=4,
    num_clusters=50,
)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA recommended for interp analysis
- OpenAI API key for generation

## Citation

If building on this work, please cite the relevant mech-interp literature on logit lens and activation patching.
