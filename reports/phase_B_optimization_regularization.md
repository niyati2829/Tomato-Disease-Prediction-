# Phase B Technical Report: Optimization Dynamics, Weight Initialization & Regularization
**Requirements Addressed: REQ-04, REQ-05**  
**Project:** Tomato Leaf Disease Detection Using Deep Learning  
**Status:** Completed & Verified (68/68 Tests Passing)  
**Date:** November 2026  

---

## 1. Executive Summary & Scope

This report documents the mathematical formulation, implementation, and empirical evaluation of **Phase B (REQ-04 and REQ-05)** of the academic syllabus gap-closure program for the Tomato Leaf Disease Detection system.

Following the successful completion of Phase A (Mathematical Foundations & Neural Fundamentals), Phase B investigates the inner mechanics of neural network training:
- **Optimization Dynamics (REQ-04)**: Convergence behaviors of Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent; scratch implementations of advanced optimizers (Polyak Momentum, Nesterov Accelerated Gradient, and Adam with exact bias correction); learning-rate stability thresholds via Hessian spectral analysis; and 2D/3D loss landscape trajectory visualizations on ill-conditioned anisotropic valleys.
- **Weight Initialization, Normalization & Regularization (REQ-05)**: Signal and gradient variance propagation across deep networks (Zero, Random, Xavier/Glorot, and He/Kaiming); mathematical formulations of Batch Normalization vs Layer Normalization across varying batch regimes; L1 (Lasso) vs L2 (Weight Decay) weight distributions; Inverted Dropout with train-vs-inference mode switching; and controlled single-factor ablation studies.

All code is implemented in pure Python using NumPy and Matplotlib in new isolated modules ([`src/optimization/`](file:///Users/niyatijha/Desktop/crop-disease-detection/src/optimization) and [`src/regularization/`](file:///Users/niyatijha/Desktop/crop-disease-detection/src/regularization)), strictly preserving all production checkpoints (`artifacts/checkpoints/best_resnet18_weighted.pt`), manifests, and the running Streamlit web application.

---

## 2. REQ-04: Optimization Dynamics

### 2.1 Gradient Descent Variants
Gradient descent minimizes an empirical risk objective $\mathcal{L}(w) = \frac{1}{N} \sum_{i=1}^N \ell(w; x_i, y_i)$. Our implementation in `src/optimization/gradient_descent.py` benchmarks the three canonical variants on a convex quadratic regression objective:
$$\mathcal{L}(w) = \frac{1}{2N} \|X w - y\|_2^2, \quad \nabla \mathcal{L}(w) = \frac{1}{N} X^T (X w - y)$$

#### Comparative Dynamics (50 Epochs, $N=250, D=5$):
| Variant | Sample Partition / Step | Updates / Epoch | Total Updates | Sample Evals | Convergence Trajectory | Sensitivity to Learning Rate |
|---|---|---|---|---|---|---|
| **Batch GD** | Full dataset ($B = N = 250$) | 1 | 50 | 12,500 | Smooth, monotonic, zero variance in gradient direction | Stable; requires larger $\eta$ to make fast progress |
| **SGD** | Single sample ($B = 1$) | 250 | 12,500 | 12,500 | High stochastic variance, noisy oscillation around optimum | Sensitive; requires small $\eta$ to prevent exploding variance |
| **Mini-Batch GD** | Subsets ($B = 16$) | 16 | 800 | 12,800 | Balanced, mild stochastic exploration, vectorization friendly | Highly robust; standard default in modern deep learning |

- **Batch GD**: Reached final loss $0.04139$ in 50 deterministic update steps.
- **Mini-Batch GD ($B=16$)**: Reached near-optimal loss $0.00467$ in 800 updates.
- **SGD**: Reached loss $0.00467$ in 12,500 updates, exhibiting high high-frequency jitter in parameter space.

---

### 2.2 Advanced First-Order Optimizers (from scratch in pure NumPy)

Implemented in `src/optimization/optimizers.py`:

#### 1. Polyak Momentum (1964)
Overcomes oscillations in high-curvature directions by accumulating exponentially decaying history of past gradients (velocity):
$$v_t = \gamma v_{t-1} + \eta g_t$$
$$w_t = w_{t-1} - v_t$$
where $\gamma \in [0, 1)$ is the momentum decay factor (typically $0.9$).

#### 2. Nesterov Accelerated Gradient (NAG, 1983)
Instead of evaluating the gradient at the current position $w_{t-1}$, NAG computes a lookahead step $w_{\text{look}} = w_{t-1} - \gamma v_{t-1}$ and evaluates the gradient there:
$$g_{\text{look}} = \nabla \mathcal{L}(w_{t-1} - \gamma v_{t-1})$$
$$v_t = \gamma v_{t-1} + \eta g_{\text{look}}$$
$$w_t = w_{t-1} - v_t$$
This lookahead acts as an adaptive predictive brake, damping overshoot when accelerating into a valley.

#### 3. Adam (Adaptive Moment Estimation, Kingma & Ba 2014)
Combines momentum (first raw moment $m_t$) and RMSprop (second uncentered moment $v_t$):
- Exponential moving averages:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
- **Exact Bias Correction**:
  Because $m_0 = 0$ and $v_0 = 0$, uncorrected estimators are severely biased toward zero during initial steps. Unrolling $m_t$:
  $$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$
  Taking expectations assuming $g_i$ comes from a stationary distribution:
  $$\mathbb{E}[m_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i\right] = g^* (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = g^* (1 - \beta_1^t)$$
  Dividing by $(1 - \beta_1^t)$ eliminates this transient bias:
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
- Parameter update:
  $$w_t = w_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

### 2.3 Learning Rate Dynamics & Hessian Spectral Limits
In `src/optimization/learning_rate_experiments.py`, we test the step-size response on a quadratic objective whose Hessian $H = \frac{1}{N} X^T X$ has maximum eigenvalue $\lambda_{\max} \approx 1.131$:
- **Theoretical Convergence Criterion**: In quadratic optimization, the error recursion is $e_{t+1} = (I - \eta H) e_t$. For stability, all eigenvalues of $(I - \eta H)$ must lie within $(-1, 1)$, establishing the strict spectral divergence bound:
  $$0 < \eta < \frac{2}{\lambda_{\max}} \approx \frac{2}{1.131} \approx 1.768$$
- **Empirical Regimes**:
  1. **Under-damped ($\eta = 0.01$)**: Slow, crawling convergence; after 35 steps, loss remains high ($14.15$).
  2. **Optimal ($\eta = 0.5$)**: Fast monotonic convergence; loss drops to $0.0046$ in under 15 steps.
  3. **Oscillatory ($\eta = 1.4$)**: Significant zig-zag overshoot across the valley, but still converges ($|1 - 1.4 \times 1.131| = 0.583 < 1$).
  4. **Divergent ($\eta = 2.1$)**: Spectral condition violated ($|1 - 2.1 \times 1.131| = |-1.375| > 1$). Error multiplies by $1.375$ each step, causing explosive loss divergence to $\infty$ by step 28.

*Figure Reference: `artifacts/figures/optimization/learning_rate_comparison.png`.*

---

### 2.4 Loss Landscape & Trajectory Visualizations
To evaluate how optimizers navigate pathological curvature, we construct an anisotropic quadratic valley with condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = 10$:
$$f(w_1, w_2) = \frac{1}{2} (w_1^2 + 10 w_2^2)$$
Starting from point $(-4.0, 2.5)$, the gradient along $w_2$ is 10 times steeper than $w_1$.
- **Vanilla GD**: Oscillates severely perpendicular to the valley along $w_2$, making painfully slow progress along $w_1$.
- **Momentum**: Accumulates forward velocity along $w_1$ while past velocities cancel out orthogonal oscillations along $w_2$, reaching the minimum in less than half the steps.
- **NAG**: Anticipates the turn using lookahead gradients, braking earlier and exhibiting less overshoot than classical momentum.
- **Adam**: Adapts per-coordinate step sizes via $\sqrt{\hat{v}_t}$, automatically scaling down updates along the steep $w_2$ axis and scaling up updates along the gentle $w_1$ axis, marching almost directly along the diagonal to the global minimum $(0, 0)$.

*Figure References: `artifacts/figures/optimization/loss_landscape_2d_contours.png`, `loss_landscape_3d_surface.png`, and `optimizer_convergence_comparison.png`.*

---

## 3. REQ-05: Weight Initialization, Normalization & Regularization

### 3.1 Weight Initialization & Signal Propagation
Implemented in `src/regularization/initialization.py`, tested across 6-layer networks ($120$ neurons/layer):

#### 1. Zero Initialization ($W = 0, b = 0$)
- **Symmetry Problem**: For any input $X$, every hidden unit in a layer computes $z_j = \sum_i x_i \cdot 0 + 0 = 0$.
- In backprop, every neuron receives the exact same incoming gradient:
  $$\frac{\partial \mathcal{L}}{\partial W_{ij}} = a_i \delta_j = a_i \delta$$
- All neurons update identically, collapsing an $N$-neuron layer to a single effective neuron. Verified in `test_zero_initialization_and_symmetry` (`is_symmetric = True`).

#### 2. Random Normal ($W \sim \mathcal{N}(0, \sigma^2)$)
- If $\sigma = 0.01$ (too small): Activations decay exponentially across depth; by Layer 6, activation standard deviation vanishes from $1.0$ down to $1.2 \times 10^{-12}$ (signal extinction).
- If $\sigma = 1.0$ (too large): Activations saturate Sigmoid/Tanh or explode, driving gradients to zero.

#### 3. Xavier / Glorot Initialization (1990)
Derived to maintain $\operatorname{Var}(A^{[l]}) = \operatorname{Var}(A^{[l-1]})$ for symmetric linear/tanh activations:
$$\operatorname{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$
- Preserves variance cleanly across 6 layers with Tanh ($\sigma \approx 0.6$ to $0.8$).
- With ReLU, because half the activations are zeroed, variance still decays across depth ($\sigma \to 0.08$ by Layer 6).

#### 4. He / Kaiming Initialization (2015)
Accounts for the rectifying property of ReLU ($\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2} \operatorname{Var}(z)$) by scaling variance by an extra factor of 2:
$$\operatorname{Var}(W) = \frac{2}{n_{\text{in}}}$$
- Maintains activation standard deviation near $1.0$ ($\sigma \approx 0.95$ to $1.05$) continuously through all 6 layers with ReLU.

*Figure References: `artifacts/figures/regularization/weight_initialization_histograms.png` and `layer_activation_gradient_distributions.png`.*

---

### 3.2 Normalization: BatchNorm vs LayerNorm
Implemented in `src/regularization/normalization.py`:

| Characteristic | Batch Normalization (`BatchNorm1dNumPy`) | Layer Normalization (`LayerNorm1dNumPy`) |
|---|---|---|
| **Normalization Axis** | Across Batch dimension ($N$) per channel ($c$) | Across Feature dimension ($D$) per individual sample ($i$) |
| **Statistical Dependency** | Couples all samples in the batch; sample $i$'s output depends on sample $k$ | Independent for each sample; zero inter-sample coupling |
| **Train vs Eval Mode** | Computes batch $\mu_B, \sigma_B^2$ during training; uses frozen running stats $\hat{\mu}, \hat{\sigma}^2$ at eval | Identical behavior during train and eval; no running stats required |
| **Batch Size Sensitivity** | Degrades when batch size is small ($B \le 4$); impossible for $B = 1$ | Completely invariant to batch size; functions perfectly at $B = 1$ |
| **Primary Domain** | Feedforward CNNs and vision backbones with large batches | Recurrent nets, Transformers, NLP, and online streaming |

- **Empirical Batch-Size Evaluation**:
  - At $B=32$: BatchNorm and LayerNorm both successfully standardize activations to $\mu \approx 0.0, \sigma \approx 1.0$.
  - At $B=2$: BatchNorm batch variance has high estimation error; switching to eval mode reveals a large discrepancy ($|\text{out}_{\text{train}} - \text{out}_{\text{eval}}| = 1.84$), whereas LayerNorm eval difference is strictly $0.00$.

*Figure Reference: `artifacts/figures/regularization/batchnorm_vs_layernorm_distributions.png`.*

---

### 3.3 Regularization Techniques: L1, L2, and Inverted Dropout
Implemented in `src/regularization/regularizers.py` and benchmarked on an over-parameterized dataset ($N_{\text{train}} = 50, N_{\text{val}} = 150, D = 30$, where 25 features are pure Gaussian noise):

#### 1. L1 Regularization (Lasso)
- **Penalty**: $\Omega(W) = \lambda \|W\|_1 = \lambda \sum |W_{ij}|$
- **Subgradient**: $\partial \Omega(W) = \lambda \operatorname{sgn}(W)$
- **Effect**: Constant derivative pushes small weights through zero, producing **exact parameter sparsity**. In our benchmark, L1 forced $24.06\%$ of all weights to $< 10^{-3}$, effectively performing automated feature selection.

#### 2. L2 Regularization (Ridge / Weight Decay)
- **Penalty**: $\Omega(W) = \frac{1}{2} \lambda \|W\|_F^2 = \frac{1}{2} \lambda \sum W_{ij}^2$
- **Gradient**: $\nabla \Omega(W) = \lambda W$
- **Effect**: Parameter update $W \leftarrow (1 - \eta \lambda) W - \eta \nabla \mathcal{L}$. Decays large weights strongly while leaving small weights unaffected. Reduced $L_2$ norm from $9.37$ (baseline) down to $7.66$ without forcing exact zeros ($0.16\%$ zeros).

#### 3. Inverted Dropout
- **Training**: $Y = \frac{X \odot M}{1 - p}$, where $M \sim \operatorname{Bernoulli}(1 - p)$.
- **Inference**: $Y = X$ (identity pass-through).
- **Mechanism**: The $\frac{1}{1-p}$ scaling applied during training ensures $\mathbb{E}[Y] = X$, eliminating any need to scale activations during inference.
- **Why disabled during inference**: Ensures deterministic predictions and acts as an equal-weighted geometric ensemble average over all $2^H$ sampled sub-networks.

*Figure References: `artifacts/figures/regularization/regularization_loss_accuracy_curves.png`, `l1_vs_l2_weight_magnitudes.png`, and `dropout_training_inference_diagram.png`.*

---

## 4. Controlled Single-Factor Ablation Studies

Implemented in `src/regularization/ablation_studies.py`, evaluating 7 controlled experiments on identical train/validation splits:

| Experiment | Description | Train Loss | Val Loss | Train Acc (%) | Val Acc (%) | Generalization Gap (%) | $\|W\|_2$ Norm | Sparsity (%) | Key Observation |
|---|---|---|---|---|---|---|---|---|---|
| **Exp-1: Baseline** | Unregularized 2-layer MLP | 0.0292 | 0.4980 | 100.0% | 80.00% | 20.00% | 9.372 | 0.23% | Memorizes noise; large generalization gap |
| **Exp-2: L1 Regularization** | $\lambda = 0.015$ | 1.9105 | 2.2285 | 100.0% | 79.29% | 20.71% | 5.674 | **24.06%** | Eliminates 24% of parameters (feature selection) |
| **Exp-3: L2 Regularization** | $\lambda = 0.02$ (Weight Decay) | 0.6379 | 1.0457 | 100.0% | 79.29% | 20.71% | 7.660 | 0.16% | Smooth weight shrinkage; prevents outlier weights |
| **Exp-4: Inverted Dropout** | $p_{\text{drop}} = 0.35$ | 0.1557 | **0.4862** | 93.33% | 77.14% | **16.19%** | 9.179 | 0.39% | **Lowest generalization gap** (prevents co-adaptation) |
| **Exp-5: Training Augmentation** | Input jitter + cutout | 0.0304 | 0.4598 | 100.0% | 80.00% | 20.00% | 9.384 | 0.16% | Prevents brittle input-coordinate reliance |
| **Exp-6: Xavier Init (on ReLU)** | Glorot normal on ReLU | 0.0335 | **0.4358** | 100.0% | 80.00% | 20.00% | 6.625 | 0.47% | Smaller initial weights; lowest validation loss |
| **Exp-7: Random Normal Init** | Unscaled $\mathcal{N}(0, 1)$ | 0.0033 | 2.2164 | 100.0% | **69.29%** | **30.71%** | 28.151 | 0.08% | Severe overfitting; huge weights ($\|W\|_2=28.15$) |

*Data Export Reference: `reports/phase_B_ablation_results.json`.*

---

## 5. Visualizations Index

All 10 figures are rendered at 300 DPI:

| Figure Name | Directory | Req | Description |
|---|---|---|---|
| `learning_rate_comparison.png` | `artifacts/figures/optimization/` | REQ-04 | Dual-panel log plot: loss and gradient norm vs step count across small, optimal, oscillatory, and divergent learning rates. |
| `loss_landscape_2d_contours.png` | `artifacts/figures/optimization/` | REQ-04 | 2D contour plot of an anisotropic valley ($\kappa=10$) with overlaid trajectories for GD, Momentum, NAG, and Adam. |
| `loss_landscape_3d_surface.png` | `artifacts/figures/optimization/` | REQ-04 | 3D wireframe surface plot with 3D optimization paths marching toward the global minimum. |
| `optimizer_convergence_comparison.png` | `artifacts/figures/optimization/` | REQ-04 | Log-scale convergence curves comparing loss and gradient decay rates between all 4 first-order optimizers. |
| `weight_initialization_histograms.png` | `artifacts/figures/regularization/` | REQ-05 | 4-panel histogram comparing weight density distributions for Zero, Random, Xavier, and He initializations. |
| `layer_activation_gradient_distributions.png` | `artifacts/figures/regularization/` | REQ-05 | Dual-panel plot of activation variance propagation across 6 layers comparing signal preservation for ReLU vs Tanh. |
| `batchnorm_vs_layernorm_distributions.png` | `artifacts/figures/regularization/` | REQ-05 | 3-panel comparative histogram showing unnormalized covariate shift vs BatchNorm vs LayerNorm distributions. |
| `regularization_loss_accuracy_curves.png` | `artifacts/figures/regularization/` | REQ-05 | Dual-panel training and validation loss/accuracy curves for Unregularized, L1, L2, and Dropout. |
| `l1_vs_l2_weight_magnitudes.png` | `artifacts/figures/regularization/` | REQ-05 | 3-panel histogram showing parameter sparsity ($24\%$ zeros under L1) vs weight shrinkage under L2. |
| `dropout_training_inference_diagram.png` | `artifacts/figures/regularization/` | REQ-05 | Architecture schematic contrasting sub-network sampling during training vs ensemble identity inference. |

---

## 6. Automated Unit Testing & Codebase Integrity

### 6.1 Test Suite Breakdown
The complete test suite was executed via pytest:
```bash
.venv/bin/python -m pytest tests/ -q
```
**Results:** **68 passed in 5.85s (100% Green)**
- `tests/test_dataset.py`: 7 passed
- `tests/test_models.py`: 13 passed
- `tests/test_app.py`: 6 passed
- `tests/test_math_sandbox.py`: 9 passed
- `tests/test_neural_fundamentals.py`: 12 passed
- `tests/test_mlp_from_scratch.py`: 6 passed
- `tests/test_optimization.py` (**NEW**): 7 passed
- `tests/test_regularization.py` (**NEW**): 8 passed

### 6.2 Project Preservation Confirmation
In accordance with strict safety and isolation rules:
- `artifacts/checkpoints/best_resnet18_weighted.pt`: Unmodified.
- `artifacts/checkpoints/best_baseline_cnn.pt`: Unmodified.
- `data/splits/train.csv`, `val.csv`, `test.csv`: Unmodified.
- `app/app.py`: Running continuously on `http://localhost:8501`.
- No Git commits or pushes were made.
- Execution has stopped at Phase B completion.

