# uncertainty-toolkit

Plug-and-play uncertainty estimation for any PyTorch classifier.

Built as the foundational uncertainty module for the **R3AL.AI SDK** — wraps an arbitrary `nn.Module` and extracts two distinct uncertainty scores per prediction, with no changes to model weights or architecture.

---

## Installation

```bash
git clone https://github.com/your-org/uncertainty-toolkit.git
cd uncertainty-toolkit
pip install -r requirements.txt
pip install -e .
```

Python ≥ 3.9, PyTorch ≥ 2.0 required.

---

## Running the Demo

```bash
python demo/run_demo.py [--epochs 15] [--passes 50] [--no-cache]
```

Downloads Fashion-MNIST, trains a small ResNet classifier, runs MC Dropout with T=50 passes, and saves three diagnostic plots to `visualizations/`.

---

## Quickstart

```python
from uncertainty_toolkit import EpistemicEstimator, AleatoricEstimator
import torch
from uncertainty_toolkit.model import FashionMNISTResNet
from demo.run_demo import get_dataloaders
from uncertainty_toolkit.visualizations import generate_all

# Step 1 — define or load your model
model = FashionMNISTResNet(num_classes=10, dropout_p=0.3)
model.load_state_dict(torch.load("checkpoints/fashion_mnist.pt"))
train_loader, test_loader = get_dataloaders(batch_size=512)

# Part 1 — Epistemic
ep = EpistemicEstimator(model, n_passes=30)
ep_result = ep.estimate(test_loader)

# Part 2 — Aleatoric
al = AleatoricEstimator(model, n_augmentations=20)
al_result = al.estimate(test_loader)

# Part 3 — Plot scatter + histograms + per-class bar chart
generate_all(
    ue=ep_result.mutual_information,      
    ua=al_result.conditional_entropy,     
    correct=ep_result.correct,            
    labels=ep_result.labels,              
    output_dir="visualizations/",         
    class_names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
)
```

---

## Project Structure

```
uncertainty_toolkit/
├── uncertainty_toolkit/
│   ├── __init__.py          
│   ├── base.py              
│   ├── epistemic.py         
│   ├── model.py             
│   └── visualizations.py   
├── demo/
│   └── run_demo.py          
├── visualizations/          
├── checkpoints/             
├── data/                    
├── requirements.txt
└── setup.py
```

---

## Part 1 — Epistemic Uncertainty: Design Decisions

### Method: Monte-Carlo Dropout

MC Dropout interprets dropout as approximate Bayesian inference. At inference time, we keep dropout active and run T forward passes, treating each pass as a sample from the model's approximate posterior over weights.

### Uncertainty Metrics

Two complementary metrics are computed:

**Predictive Entropy** — `H[y | x, D]`
```
H = -Σ_c  p̄_c · log(p̄_c)
where p̄_c = (1/T) Σ_t p_c^(t)
```
Total uncertainty of the mean predictive distribution. High for *both* hard inputs and inputs in low-density regions.

**Mutual Information** — `I[y, ω | x, D]`
```
MI = H[y|x,D] − (1/T) Σ_t H[y|x, ω^(t)]
   = predictive_entropy − mean_per_pass_entropy
```

### Dropout pattern

Dropout layers are activated via `_dropout_active()` a context manager that sets only `nn.Dropout` sublayers to `.train()` while leaving BatchNorm in eval mode. This is the correct approach; naively calling `model.train()` would corrupt BatchNorm behaviour and give misleading results.

---

## Part 2 — Aleatoric Uncertainty: Design Decisions

### Method: Augmentation Disagreement

The `AleatoricEstimator` applies K stochastic augmentations to each input at inference time, runs one deterministic forward pass per view, and measures two uncertainty signals from softmax outputs across views:

```
Ua(x) = mean_c  Var_k [ p_c(x̃_k) ]
```
```
Conditional Entropy — the recommended signal:
Ua(x) = (1/K) Σ_k  H[ p(x̃_k) ]
       = (1/K) Σ_k  ( -Σ_c  p_c(x̃_k) · log p_c(x̃_k) )
```
Computes the entropy of each individual augmented view separately, then averages over K views. This is the expected entropy under input perturbations — how uncertain the model is on average across all augmented views.

---

### Why augmentation disagreement over a learned variance head?
Both approaches are valid. The learned variance head requires fine-tuning (even "a small amount") which means touching the model's weights, that breaks the plug-and-play of this SDK. Augmentation disagreement works on any frozen model, needs no retraining. For example:

if a model's prediction flips under small change of the input, the
input is inherently ambiguous (high aleatoric uncertainty). If predictions stay stable, the input is unambiguous regardless of how the model is.

Trade-offs vs learned variance head
Augmentation disagreement
• Zero changes to model weights : truly plug-and-play
• Immediately applicable to any pretrained model
• Augmentation pipeline is adjustable
• Does not produce a calibrated probability — relative scores only


Learned variance head
• Principled probabilistic interpretation
• Per-class uncertainty (heteroscedastic)
• Requires fine-tuning : not zero-weight-change
• Needs labeled data and a training loop

---

## Part 3 — Visualizations

Three plots are produced and saved to `visualizations/`:

| File | Description |
|------|-------------|
| `scatter_ua_ue.png` | Scatter of Ua vs Ue, coloured by correct / incorrect |
| `uncertainty_histograms.png` | Density distributions per uncertainty type |
| `per_class_breakdown.png` | Grouped bar chart of mean Ua & Ue per class |

## References

- Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML.
- Kendall, A., & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS.

---
