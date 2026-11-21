# Phase E Technical Report: Temporal Disease Progression & Generative Modeling
**Requirements Addressed: REQ-09**  
**Project:** Tomato Leaf Disease and Pest Surveillance System  
**Status:** Completed & Verified (111/111 Tests Passing across Repository)  
**Date:** November 2026  

---

## 1. Executive Summary & Academic Integrity Matrix

Phase E implements recurrent sequence modeling (`src/temporal/`) and generative deep learning (`src/generative/`) to address **REQ-09**.

### Explicit Capability Rigor Verification Table

| Capability | Implemented | Unit Tested | Experimentally Trained | Real Data Evaluated | Scope & Qualification |
|---|:---:|:---:|:---:|:---:|---|
| **Vanilla RNN (Scratch NumPy & PyTorch)** | **YES** | **YES** | **YES** (40 epochs) | **NO** (Synthetic only) | Parity verified against `torch.nn.RNN` within $10^{-5}$. |
| **LSTM (Scratch NumPy & PyTorch)** | **YES** | **YES** | **YES** (40 epochs) | **NO** (Synthetic only) | Parity verified against `torch.nn.LSTM` within $10^{-5}$. |
| **Synthetic Progression Benchmark** | **YES** | **YES** | N/A | **NO** (Pure Simulation) | Strictly isolated by `plant_id`; 0 cross-split leakage. |
| **Convolutional VAE (`ConvVAE`)** | **YES** | **YES** | **YES** (25 epochs on `train.csv`) | **YES** (`val.csv` only) | Evaluated on 160 unseen validation leaf photos; `test.csv` untouched. |
| **Lightweight DCGAN** | **YES** | **YES** | **YES** (20 epochs on `train.csv`) | Qualitative only | Demonstrates adversarial instability (`D` overpowering `G` on $N=600$). |
| **Latent-Space & Activation Analysis** | **YES** | **YES** | N/A | **YES** (VAE on `val.csv`) | 2D PCA Healthy vs Diseased + smooth morphing + RNN/LSTM state norms. |

```
============================== 111 passed in 8.40s ==============================
- Production Core (Stages 1-7): 26 passed
- Phase A (REQ-01, 02, 03): 27 passed
- Phase B (REQ-04, 05): 15 passed
- Phase C (REQ-06): 10 passed
- Phase D (REQ-08): 21 passed
- Phase E (REQ-09, NEW): 12 passed
Total: 111 automated unit tests across 15 test files (100% Green)
```

---

## 2. Step 1 — Dataset & Temporal Data Feasibility Audit

### 2.1 Why PlantVillage Cannot Support Temporal Modeling
The PlantVillage repository (18,160 tomato leaf images) consists of independent, detached leaves photographed once under uniform studio conditions. It contains **zero timestamps**, **zero persistent `plant_id` markers**, and **zero repeated longitudinal captures** of the same specimen. Sorting random static images by lesion severity would fabricate artificial time-series trajectories mixing incompatible biological individuals, lighting profiles, and orientations.

### 2.2 Public Longitudinal Candidate Datasets Surveyed

| Dataset | Provenance | License | Scale & Granularity | Suitability & Local Feasibility Constraints |
|---|---|---|---|---|
| **Tomato Leaf Damage Progression** | Mendeley Data | CC BY 4.0 | Categorized into 3 discrete severity stages (Initial, Intermediate, Advanced) | Useful for cross-sectional severity staging, but lacks continuous per-plant timestamps ($t_0 \dots t_T$) and requires external download. |
| **TomatoMAP** | arXiv / Phenotyping Repo | Academic Research | Multi-pose growth cycle time-series categorized by BBCH stages | Designed for 3D structural growth phenotyping rather than quantitative lesion spread; multi-gigabyte download infeasible in offline sandbox. |
| **Production Setting Time-Series** | SciELO / Zenodo | CC BY 4.0 | Greenhouse sequential captures paired with sensor telemetry | Requires external archive access restricted in current local environment. |

### 2.3 Synthetic Progression Benchmark Design (`src/temporal/sequence_generator.py`)
Because genuine longitudinal field captures are unavailable locally, an explicitly **synthetic** mathematical disease progression benchmark was constructed:
- **Scale**: $N = 600$ distinct simulated plants, each observed over $T = 10$ historical steps predicting step $t = 10$.
- **Four Progression Regimes**:
  1. *Healthy / Stable*: $s_t = \operatorname{clip}(s_0 + \epsilon_t, 0, 1)$
  2. *Linear Progression*: $s_t = \operatorname{clip}(s_0 + r t + \epsilon_t, 0, 1)$
  3. *Logistic Outbreak (S-Curve)*: $s_t = \operatorname{clip}(0.05 + \frac{0.90}{1 + e^{-k(t - t_{\text{mid}})}} + \epsilon_t, 0, 1)$
  4. *Treatment Recovery*: Spread followed by post-fungicide exponential decay.
- **Multivariate Features ($D=3$)**: Severity index $s_t$, finite-difference rate $\Delta s_t = s_t - s_{t-1}$, and simulated leaf wetness stress $h_t$.
- **Sequence-Level Isolation**: Split strictly by `plant_id` into **420 Train / 90 Validation / 90 Test** simulated plants with **zero overlap** across splits.
- *Disclaimer*: This is a purely educational mathematical simulation, not a calibrated biological disease model.

---

## 3. Temporal Modeling: Vanilla RNN vs LSTM (`src/temporal/`)

### 3.1 Mathematical Formulations & From-Scratch Parity
Both `VanillaRNNNumPy` and `LSTMCellNumPy` were implemented from first principles in NumPy:
- **Vanilla RNN**: $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$.
- **LSTM Gates**:
  $$f_t = \sigma(W_f [x_t, h_{t-1}] + b_f), \quad i_t = \sigma(W_i [x_t, h_{t-1}] + b_i), \quad \tilde{C}_t = \tanh(W_c [x_t, h_{t-1}] + b_c)$$
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \quad o_t = \sigma(W_o [x_t, h_{t-1}] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$
When initialized with identical weights, both pure NumPy modules matched `torch.nn.RNN` and `torch.nn.LSTM` within `< 1e-5` floating-point tolerance.

### 3.2 Empirical Comparison on Synthetic Test Split ($N = 90$ Unseen Plants)
Models were trained identically for 40 epochs ($H=32$, $\text{lr}=0.002$, MSE loss, Adam) with checkpoints selected solely via Validation MSE:

| Metric | Vanilla RNN (`best_rnn.pt`) | LSTM (`best_lstm.pt`) | Finding |
|---|---:|---:|---|
| **Trainable Parameters** | **1,217** | 4,769 | LSTM has 3.92x higher capacity |
| **Training Duration (40 Epochs)** | **0.65 s** | 0.69 s | Nearly identical compute cost |
| **Best Validation MSE** | 0.001386 | **0.001192** | LSTM achieved slightly lower val error |
| **Untouched Test MAE** | **0.03098** | 0.03125 | Equivalent within noise floor ($\sigma=0.025$) |
| **Untouched Test RMSE** | **0.03881** | 0.03901 | Virtual statistical tie |
| **Untouched Test $R^2$** | **0.9881** | 0.9880 | Both capture 98.8% variance |
| **Inference Latency / Sequence** | **0.046 ms** | 0.091 ms | RNN is 2x faster per sequence |

#### Scientific Conclusion (Honest Negative / Null Difference Result):
On a short $T=10$ horizon where vanishing gradients across time steps do not accumulate catastrophically, the **LSTM does not significantly outperform the Vanilla RNN** on the held-out synthetic test split ($R^2 = 0.9880$ vs $0.9881$). Meanwhile, the Vanilla RNN accomplishes this parity with **~4x fewer parameters** and **2x lower latency**. LSTM's architectural advantage via the additive Cell State highway primarily dominates when learning dependencies spanning long sequence horizons ($T \ge 50$).

---

## 4. Generative Modeling: ConvVAE & DCGAN (`src/generative/`)

### 4.1 Dataset Strict Isolation
Training and validation used a balanced $64 \times 64$ RGB subset sampled **exclusively** from `data/splits/train.csv` (600 images across Healthy, Early Blight, Bacterial Spot, and Yellow Leaf Curl) and `data/splits/val.csv` (160 validation images). `data/splits/test.csv` was **strictly forbidden** and blocked programmatically.

### 4.2 Convolutional VAE (`src/generative/vae.py`)
- **Architecture**: 4 strided `Conv2d` downsampling layers $\to$ $\mu, \log\sigma^2 \in \mathbb{R}^{16}$ $\to$ Reparameterization $z = \mu + \exp(0.5 \log\sigma^2) \odot \epsilon$ $\to$ 4 `ConvTranspose2d` decoder layers.
- **Loss Objective**: $\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{MSE}} + \beta \mathcal{L}_{\text{KL}}$ ($\beta = 1.0$, Adam $\text{lr}=10^{-3}$, 25 epochs).
- **Training Duration**: 57.25s on Apple Silicon CPU; 1,582,179 parameters.
- **Validation Performance (`val.csv`, 160 unseen images)**:
  - Best Validation ELBO: **109.79**
  - Mean Pixel MSE: **0.00736**
  - Mean PSNR: **22.17 dB**
  - Mean KL Divergence: **19.37**
  - Per-Image Inference Latency: **1.333 ms**
- **Latent Space Analysis (`latent_analysis.py`)**:
  - 2D PCA over the 16D latent space reveals continuous structural variation separating green healthy lamina from chlorotic/necrotic diseased foliage.
  - Linear latent interpolation ($z(\alpha) = (1 - \alpha) z_{\text{healthy}} + \alpha z_{\text{diseased}}$) produces smooth color/shape transitions.

### 4.3 Lightweight DCGAN (`src/generative/gan.py`)
- **Architecture**: `DCGANGenerator` (952,768 params, mapping $z \in \mathbb{R}^{64} \to [-1, 1]^{3 \times 64 \times 64}$) and `DCGANDiscriminator` (694,656 params).
- **Training Stability & Honest Assessment**:
  - Trained for 20 epochs (78.23s) using Adam (`lr=2e-4`, $\beta_1=0.5$), `BCEWithLogitsLoss`, and one-sided label smoothing (`0.9`).
  - Empirical logs show discriminator dominance: $D(x) \approx 0.777$ while $D(G(z)) \approx 0.034$ ($\text{Loss}_D = 0.6225, \text{Loss}_G = 4.1100$).
  - *Limitation*: On small training subsets ($N=600$), adversarial optimization struggles to reach Nash equilibrium compared to the smooth probabilistic convergence of `ConvVAE`. All generated images are low-resolution synthetic educational outputs—they must never be presented as genuine agricultural samples or used to augment production classifiers.

---

## 5. Visualizations Index (7 New Figures)

| Directory & Figure File | Contents & Data Origin Label |
|---|---|
| `artifacts/figures/temporal/synthetic_disease_progression_patterns.png` | Sample trajectories and cohort mean $\pm$ SD across all 4 synthetic mathematical regimes. |
| `artifacts/figures/temporal/rnn_vs_lstm_loss_and_predictions.png` | Log-scale MSE training convergence and actual vs predicted scatter on untouched synthetic test split ($N=90$). |
| `artifacts/figures/temporal/rnn_vs_lstm_state_dynamics.png` | Hidden state L2 norm comparison ($\|h_t\|_2$), LSTM cell state norm ($\|C_t\|_2$), and Forget/Input/Output gate traces. |
| `artifacts/figures/generative/vae_loss_and_kl_curves.png` | Train vs Validation ELBO, Pixel MSE, and KL divergence curves across 25 epochs (Real PlantVillage subset). |
| `artifacts/figures/generative/vae_reconstructions_and_samples.png` | Original validation images vs ConvVAE reconstructions vs synthetic prior draws ($z \sim \mathcal{N}(0,I)$). |
| `artifacts/figures/generative/vae_latent_space_and_interpolation.png` | 2D PCA latent scatter (Healthy vs Diseased) and smooth 8-step synthetic morphing strip. |
| `artifacts/figures/generative/gan_synthetic_samples_and_loss.png` | DCGAN minimax loss trajectories and $3 \times 3$ synthetic generated grid. |

---

## 6. Deployment Gap: What Real Data Is Required
Transitioning from educational simulation to real-world agronomic disease progression forecasting requires:
1. Greenhouse/field RGB time-series anchored by persistent `plant_id` / `leaf_id` tags tracked across multi-day intervals ($t = 0 \dots T$).
2. Co-registered micro-climate sensor telemetry (leaf wetness duration, humidity, temperature).
3. Agronomist-calibrated pixel lesion severity labels at each sequential capture.

---

## 7. Preservation Verification

All SHA-256 checksums match pre-Phase E records identically:
- `artifacts/checkpoints/best_resnet18_weighted.pt`: `b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa` (**UNCHANGED**)
- `data/splits/train.csv`: `528898f3c17bd7efb486c027694c15b2be7a1d371780ce00110236dbc6b1af30` (**UNCHANGED**)
- `data/splits/val.csv`: `92c4b22ba874a841cd4dc5ee73bbd475ed9d7826e295c30c34facd7e2981fa70` (**UNCHANGED**)
- `data/splits/test.csv`: `7d152f32dafc55899733b22a2077fbe7ad45a3c0779c73810e99eb20aff0844b` (**UNCHANGED**)
- **Streamlit App**: Verified active on `http://localhost:8501`.
- **Git**: Zero commits or pushes performed.
- **Phase Boundary**: Stopped cleanly at completion of Phase E (REQ-09); REQ-10 untouched.

