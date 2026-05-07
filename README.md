# Self-Pruning Neural Network on CIFAR-10

A PyTorch implementation of a learnable weight-pruning network trained on CIFAR-10. The model uses **soft gate scores** to learn which weights to prune during training itself — no separate post-training pruning step required.

---

## How It Works

Each linear layer is replaced by a `PrunableLinear` layer that maintains a `gate_scores` matrix alongside the weights. During the forward pass, sigmoid-activated gates are multiplied element-wise with the weights:

```
pruned_weights = weight * sigmoid(gate_scores)
```

A **sparsity regularisation loss** (weighted by λ) penalises large gate values, pushing the network to zero out unnecessary connections. Three values of λ are compared: `1e-5`, `5e-5`, and `2e-4`.

A **warmup phase** (8 epochs) trains without the sparsity penalty first, letting the network learn useful features before pruning begins.

---

## Architecture

```
Input (32×32×3 → 3072)
    ↓
PrunableLinear(3072 → 512) + ReLU
    ↓
PrunableLinear(512 → 256) + ReLU
    ↓
PrunableLinear(256 → 10)
    ↓
Output (10 classes)
```

---

## Results

| λ (sparsity weight) | Test Accuracy (%) | Sparsity Level (%) |
|---------------------|-------------------|--------------------|
| 1e-5                |       55.46       |       34.43        |
| 5e-5                |       57.73       |       76.44        |
| 2e-4                |       53.04       |       97.07        |


---

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, falls back to CPU)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook Case_Study.ipynb
```


---

## Project Structure

```
.
├── .gitignore
├── Case_Study.ipynb   # Main notebook
├── README.md
├── image.png          # Output: gate value histogram (generated on run)
├── requirements.txt   # Python dependencies                               
└── Report.pdf
```

---

## Key Concepts

| Term | Meaning |
|------|---------|
| **Gate scores** | Learnable parameters controlling per-weight importance |
| **Sparsity loss** | L1-style penalty on gate values to encourage pruning |
| **Warmup** | Initial epochs without sparsity loss for stable feature learning |
| **Sparsity level** | % of gates below threshold (0.01) — effectively pruned weights |

---

## License

This project is for educational/case study purposes.
