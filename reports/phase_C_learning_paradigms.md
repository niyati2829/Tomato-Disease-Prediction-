# Phase C Technical Report: Classical Learning Paradigms & Energy-Based Neural Networks
**Requirements Addressed: REQ-06**  
**Project:** Tomato Leaf Disease Detection Using Deep Learning  
**Status:** Completed & Verified (78/78 Tests Passing)  
**Date:** November 2026  

---

## 1. Executive Summary & Objective

This report documents the mathematical derivation, scratch implementation, and experimental validation of **Phase C (REQ-06: Classical Learning Paradigms & Energy-Based Neural Networks)** for the Tomato Leaf Disease and Pest Surveillance system.

While modern convolutional networks rely on supervised backpropagation and stochastic gradient descent, the foundational history of neural computation is characterized by diverse learning principles:
1. **Correlation-based learning (Hebbian)**: Synaptic plasticity driven by co-activation.
2. **Normalized Hebbian learning (Oja's Rule)**: Unsupervised principal component extraction.
3. **Competitive learning & Self-Organizing Maps (Kohonen SOM)**: Topographic clustering and manifold discovery without labels.
4. **Error-correction learning (Delta Rule / LMS)**: Target-driven feedback minimizing output discrepancy.
5. **Energy-based associative memory (Hopfield Networks)**: Deterministic Lyapunov energy descent and pattern reconstruction.
6. **Energy-based probabilistic learning (Boltzmann Machines / RBM)**: Stochastic thermal sampling and Contrastive Divergence (CD-1).

All implementations are coded from scratch in pure Python using NumPy and Matplotlib, isolated inside [`src/learning_paradigms/`](file:///Users/niyatijha/Desktop/crop-disease-detection/src/learning_paradigms), strictly preserving the production ResNet18 pipeline (`best_resnet18_weighted.pt`), manifests, and the running Streamlit web application.

---

## 2. Mathematical Explanations & Update Formulations

### 2.1 Basic Hebbian Learning
Originating from Donald Hebb's 1949 neurophysiological postulate (*"neurons that fire together, wire together"*):
$$\Delta w = \eta y x \quad \text{where } y = w^T x$$

#### Mathematical Behavior & Limitations:
- **Unsupervised Nature**: Operates without teacher targets; the weight update is driven strictly by the statistical correlation between the presynaptic input $x$ and postsynaptic output $y$.
- **Catastrophic Unbounded Growth**: Taking expectations assuming stationary data with covariance $C = \mathbb{E}[x x^T]$:
  $$\mathbb{E}[\Delta w] = \eta \mathbb{E}[(w^T x) x] = \eta C w$$
  Since $C$ is positive semi-definite with eigenvalues $\lambda_i \ge 0$, any projection along an eigenvector $e_1$ with $\lambda_1 > 0$ compounds exponentially over iterations:
  $$w_{t} \approx w_0 e^{\eta \lambda_1 t} \implies \|w\|_2 \to \infty$$
  In our empirical simulation, basic Hebbian weights exploded from $\|w\|_2 = 1.0$ to $\|w\|_2 = 1.0099 \times 10^{12}$ within 1,500 iterations.

---

### 2.2 Oja's Modified Learning Rule (1982)
Erkki Oja solved the Hebbian instability by introducing a weight-dependent forgetting / decay term that mathematically enforces unit-norm constraint without requiring explicit offline normalization:
$$\Delta w = \eta y (x - y w) = \eta (y x - y^2 w)$$

#### Mathematical Derivation via Taylor Approximation:
Consider normalized Hebbian learning where weights are normalized at each step:
$$w_{t+1} = \frac{w_t + \eta y_t x_t}{\|w_t + \eta y_t x_t\|_2}$$
Expanding the denominator for small learning rate $\eta \ll 1$:
$$\|w_t + \eta y_t x_t\|_2 = \sqrt{w_t^T w_t + 2\eta y_t (w_t^T x_t) + \eta^2 y_t^2 x_t^T x_t} = \sqrt{1 + 2\eta y_t^2 + O(\eta^2)} \approx 1 + \eta y_t^2 + O(\eta^2)$$
Using the binomial expansion $(1 + z)^{-1} \approx 1 - z$:
$$w_{t+1} \approx (w_t + \eta y_t x_t)(1 - \eta y_t^2) \approx w_t + \eta y_t x_t - \eta y_t^2 w_t + O(\eta^2) = w_t + \eta y_t (x_t - y_t w_t)$$

#### Equivalence to Principal Component Analysis (PCA):
At equilibrium ($\mathbb{E}[\Delta w] = 0$):
$$\mathbb{E}[y x - y^2 w] = C w - (w^T C w) w = 0 \implies C w = (w^T C w) w$$
This is the standard eigenvalue equation $C w = \lambda w$, where $\lambda = w^T C w$. Dynamical systems theory guarantees that the only stable fixed point of Oja's differential equation corresponds to the **maximal eigenvalue** $\lambda_1$. Thus, Oja's rule extracts the **first principal component** of the input data stream online.
- In our benchmark on correlated 2D Gaussian data ($\theta = 35^\circ$), Oja's rule stabilized to $\|w\|_2 = 1.0011$ and achieved an exact cosine similarity of **$1.0000$** with the ground-truth principal eigenvector.

> [!NOTE]
> **Mathematical Qualifications & Convergence Assumptions for Oja's Rule**:
> Oja's rule is an online stochastic approximation algorithm. Asymptotic convergence to the true first principal component ($\|w\|_2 \to 1$ and $w \to \pm v_1$) requires specific conditions:
> 1. **Zero-Mean Stationary Distribution**: Inputs must satisfy $\mathbb{E}[x] = 0$. If data is uncentered, Oja's rule aligns with the first raw statistical moment rather than the central covariance eigenvector.
> 2. **Strict Spectral Gap**: The largest eigenvalue must be unique ($\lambda_1 > \lambda_2$). If eigenvalues are degenerate ($\lambda_1 = \lambda_2$), any vector spanning the subspace is a fixed point.
> 3. **Non-Orthogonal Initialization**: The initial weight vector must not be orthogonal to the principal eigenvector ($w_0^T v_1 \ne 0$). If $w_0 \perp v_1$, the projection along $v_1$ is zero and remains zero in expectation.
> 4. **Learning Rate Dynamics**: Exact asymptotic convergence requires decaying step sizes satisfying Robbins-Monro conditions ($\sum \eta_t = \infty, \sum \eta_t^2 < \infty$). Under a constant learning rate $\eta$, weights do not converge to a static point but oscillate within a stochastic diffusion ball around the attractor whose radius is proportional to $\sqrt{\eta}$.

*Figure References: `artifacts/figures/learning_paradigms/hebbian_vs_oja_weight_growth.png` and `oja_pca_direction_alignment.png`.*

---

### 2.3 Competitive Learning & Kohonen Self-Organizing Maps (SOM)

#### 1. Winner-Takes-All (WTA) Competitive Learning
Given an unlabelled dataset $X \in \mathbb{R}^{N \times D}$ and $K$ prototype neurons $\{w_k\}_{k=1}^K$:
- **Competition (BMU Selection)**: For input $x$, the winning neuron $k^*$ is selected via minimum Euclidean distance:
  $$k^* = \arg\min_{k \in \{1, \dots, K\}} \|x - w_k\|_2$$
- **Adaptation**: Only the winner updates its prototype toward the input:
  $$\Delta w_{k^*} = \eta (x - w_{k^*}), \quad \Delta w_j = 0 \; (\forall j \ne k^*)$$
  This partitions feature space into Voronoi cells, minimizing the quantization error:
  $$E_{\text{quant}} = \frac{1}{N} \sum_{i=1}^N \|x_i - w_{k^*(i)}\|_2^2$$

#### 2. Kohonen Self-Organizing Map (SOM)
Extends competitive learning by arranging neurons in a low-dimensional topological lattice (e.g., a $5 \times 5$ 2D grid with coordinates $r_i \in \mathbb{R}^2$):
- **Cooperative Neighborhood Function**:
  $$h_{c, i}(t) = \exp\left(-\frac{\|r_c - r_i\|_2^2}{2 \sigma(t)^2}\right)$$
  where $c$ is the BMU coordinate, $r_i$ is neuron $i$'s coordinate on the grid, and $\sigma(t)$ is the neighborhood radius.
- **Topological Weight Update**:
  $$\Delta w_i = \eta(t) h_{c, i}(t) (x - w_i)$$
- **Cooling Schedules**:
  $$\eta(t) = \eta_0 \exp\left(-\frac{t}{T}\right), \quad \sigma(t) = \sigma_0 \exp\left(-\frac{t}{T / \ln(\sigma_0)}\right)$$
- **Topological Preservation**: Because adjacent lattice neurons are updated together ($h_{c, i} \approx 1$), the discrete grid bends and unfolds continuously through the input distribution, mapping high-dimensional manifolds to 2D while preserving neighborhood relationships.

> [!WARNING]
> **Topology Preservation Limitations & Failure Modes in SOM**:
> A Self-Organizing Map is an unsupervised, non-convex heuristic. Topology preservation is **not unconditionally guaranteed**:
> 1. **Topological Kinks / Manifold Twists**: If the initial neighborhood radius $\sigma_0$ is too small (e.g., $\sigma_0 < 1.0$) or if learning rates decay too rapidly, the lattice can freeze in self-intersecting, twisted local minima that fail to untangle.
> 2. **Dimensionality & Topology Mismatch**: When embedding an intrinsically high-dimensional manifold or a space with non-trivial topology (e.g. toroidal distributions, ring clusters with central holes) into a planar 2D grid, tears and boundary distortions are mathematically inevitable.
> 3. **Dead Neurons**: In Winner-Take-All competition without neighborhood smoothing ($\sigma \to 0$), neurons initialized far from data clusters may never win an input and remain permanently inert.

*Figure References: `artifacts/figures/learning_paradigms/competitive_learning_clusters.png` and `som_topological_organization.png`.*

---

### 2.4 Error-Correction Learning (Delta Rule / LMS / Adaline)
Formulated by Bernard Widrow and Ted Hoff (1960):
$$y = w^T x + b, \quad e = t - y$$
$$\Delta w = \eta (t - y) x, \quad \Delta b = \eta (t - y)$$

#### Supervised vs Unsupervised Distinction:
| Dimension | Error-Correction (Delta Rule) | Hebbian / Oja Learning |
|---|---|---|
| **Teaching Signal** | External ground truth target $t$ required | No teacher signal ($t$ does not exist) |
| **Driving Force** | Discrepancy error $e = (t - y)$ | Presynaptic-postsynaptic correlation $y \cdot x$ |
| **Objective Function** | Minimizes Mean Squared Error $\mathcal{L} = \frac{1}{2} (t - y)^2$ | Maximizes output variance / mutual information |
| **Equilibrium** | Reaches zero updates when predictions match targets ($y = t$) | Continues adjusting based on data distribution |

- On a 2D linearly separable classification benchmark, the Delta Rule reduced MSE loss from $0.0180$ to $0.0084$, shifting the linear decision boundary to achieve **0 misclassifications out of 200 samples**.

*Figure Reference: `artifacts/figures/learning_paradigms/error_correction_delta_rule_training.png`.*

---

### 2.5 Hopfield Networks (Energy-Based Associative Memory)
Invented by John Hopfield (1982), the network comprises $N$ symmetrically interconnected bipolar neurons $s_i \in \{-1, +1\}$ without self-feedback ($W_{ii} = 0$).

#### 1. Hebbian Storage Matrix:
For $P$ prototype memory patterns $\{\xi^\mu\}_{\mu=1}^P \in \{-1, +1\}^N$:
$$W = \frac{1}{N} \sum_{\mu=1}^P \xi^\mu (\xi^\mu)^T - \frac{P}{N} I$$

#### 2. Lyapunov Energy Function:
$$E(s) = -\frac{1}{2} s^T W s = -\frac{1}{2} \sum_{i=1}^N \sum_{j \ne i} W_{ij} s_i s_j$$

#### 3. Mathematical Proof of Monotonic Energy Descent ($\Delta E \le 0$):
Under asynchronous updates, a single unit $i$ is updated:
$$s_i^{\text{new}} = \operatorname{sgn}\left(\sum_{j \ne i} W_{ij} s_j\right) = \operatorname{sgn}(h_i)$$
The change in energy between state $s^{\text{old}}$ and $s^{\text{new}}$:
$$\Delta E = E(s^{\text{new}}) - E(s^{\text{old}}) = -\frac{1}{2} \sum_{j \ne i} W_{ij} s_i^{\text{new}} s_j - \frac{1}{2} \sum_{j \ne i} W_{ji} s_i^{\text{new}} s_j - \left( -\frac{1}{2} \sum_{j \ne i} W_{ij} s_i^{\text{old}} s_j - \frac{1}{2} \sum_{j \ne i} W_{ji} s_i^{\text{old}} s_j \right)$$
Using matrix symmetry ($W_{ij} = W_{ji}$):
$$\Delta E = -\Delta s_i \left( \sum_{j \ne i} W_{ij} s_j \right) = -\Delta s_i \cdot h_i$$
- If $h_i > 0 \implies s_i^{\text{new}} = +1 \implies \Delta s_i \ge 0 \implies -\Delta s_i \cdot h_i \le 0$.
- If $h_i < 0 \implies s_i^{\text{new}} = -1 \implies \Delta s_i \le 0 \implies -\Delta s_i \cdot h_i \le 0$.
- If $s_i^{\text{new}} = s_i^{\text{old}} \implies \Delta s_i = 0 \implies \Delta E = 0$.
- If $h_i = 0$ (tie handling): In our implementation, `activation >= 0` maps to $+1$. If $s_i^{\text{old}} = -1$, $\Delta s_i = +2$, yielding $\Delta E = - \Delta s_i \cdot 0 = 0$. If $s_i^{\text{old}} = +1$, $\Delta s_i = 0 \implies \Delta E = 0$. Thus, tie handling never increases energy ($\Delta E \le 0$ strictly holds).

Therefore, **$\Delta E \le 0$ at every asynchronous step**. Because the state space is finite ($2^N$ states) and energy is bounded below ($E(s) \ge -\frac{1}{2} \sum_{ij} |W_{ij}|$), the system is guaranteed to converge to a local energy minimum (attractor).

> [!IMPORTANT]
> **Asynchronous vs Synchronous Update Dynamics**:
> The monotonic energy-descent proof strictly applies to **asynchronous single-neuron updates** where $W = W^T$ and $W_{ii} = 0$.
> Under **synchronous updates** (where all neurons update simultaneously, $s(t+1) = \operatorname{sgn}(W s(t))$), the static energy function $E(s) = -\frac{1}{2} s^T W s$ is **not guaranteed to decrease monotonically** and can oscillate in 2-cycle limit cycles ($s(t+2) = s(t) \ne s(t+1)$). For synchronous dynamics, the valid Lyapunov function is the bipartite temporal energy $E_{\text{sync}}(s(t), s(t-1)) = -s(t)^T W s(t-1)$.

#### 4. Associative Memory Experiment ($8 \times 8 = 64$ Neurons):
- Stored 3 canonical patterns: Leaf, Plus, Box.
- Injected $28\%$ random bit-flip noise into the Leaf pattern.
- In 120 asynchronous steps, the network migrated down the energy valley (from $E = -7.09$ to $E = -51.88$), reconstructing the original pattern with **$96.9\%$ retrieval accuracy** ($62/64$ bits recovered).
- Verified that energy differences satisfy $\Delta E \le 10^{-9}$ strictly across all steps.

*Figure References: `artifacts/figures/learning_paradigms/hopfield_pattern_recovery.png` and `hopfield_energy_trajectory.png`.*

---

### 2.6 Boltzmann Machines & Restricted Boltzmann Machines (RBM)
While Hopfield networks are deterministic and can get trapped in sub-optimal local minima (spurious attractors), the Boltzmann Machine introduces stochastic thermal fluctuations allowing escape from energy barriers.

#### 1. RBM Architecture & Bipartite Energy Function:
Visible units $v \in \{0, 1\}^V$ and hidden units $h \in \{0, 1\}^H$ form a bipartite graph (no visible-visible, no hidden-hidden connections):
$$E(v, h) = -v^T W h - a^T v - b^T h$$

#### 2. Conditional Independence & Sigmoid Activations:
Because visible units are conditionally independent given hidden units (and vice versa):
$$P(h_j = 1 \mid v) = \sigma\left(\sum_{i=1}^V v_i W_{ij} + b_j\right)$$
$$P(v_i = 1 \mid h) = \sigma\left(\sum_{j=1}^H W_{ij} h_j + a_i\right)$$

#### 3. Contrastive Divergence (CD-1):
Standard maximum likelihood gradient $\frac{\partial \ln P(v)}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$ requires intractable infinite Gibbs sampling to compute the model expectation $\langle \cdot \rangle_{\text{model}}$. Geoffrey Hinton's Contrastive Divergence (2002) runs only $k=1$ Gibbs step:
1. **Positive Phase**: Clamp data $v_0$; sample $h_0 \sim P(h \mid v_0)$.
2. **Negative Phase**: Sample reconstruction $v_1 \sim P(v \mid h_0)$; sample $h_1 \sim P(h \mid v_1)$.
3. **Weight Update**:
   $$\Delta W = \eta \left( v_0 h_0^T - v_1 h_1^T \right)$$
   $$\Delta a = \eta (v_0 - v_1), \quad \Delta b = \eta (h_0 - h_1)$$

- In our experiment on 300 structured $4 \times 4$ binary patterns (bars with noise), RBM CD-1 training reduced reconstruction error from $0.2155$ to $0.0500$, while average free energy settled to $-11.56$.

> [!NOTE]
> **Distinction Between Reconstruction, Free Energy, and Generative Modeling in RBMs**:
> 1. **Training Reconstruction vs Generalization**: Achieving low reconstruction error on a small toy dataset demonstrates the RBM's capacity to auto-encode familiar patterns, but does not prove generalization to unseen distributions or absence of mode collapse.
> 2. **Free Energy vs True Likelihood**: The true marginal data probability is $P(v) = \frac{e^{-\mathcal{F}(v)}}{Z}$, where $Z = \sum_{v'} e^{-\mathcal{F}(v')}$ is the intractable partition function. A decrease in $\mathcal{F}(v)$ on training data alone does not guarantee increased probability $P(v)$ unless $Z$ is controlled; if free energy decreases uniformly across all states, $P(v)$ remains unchanged. The negative phase of CD-1 explicitly counteracts this by raising free energy on model hallucinations ($\Delta \mathcal{F}(v_1) > 0$).
> 3. **Generative Sampling**: Assessing true generative fidelity requires running equilibrium Markov chains ($k \to \infty$) from random initial states or estimating $\ln Z$ via Annealed Importance Sampling (AIS).

*Figure Reference: `artifacts/figures/learning_paradigms/boltzmann_rbm_sampling_energy.png`.*

---

## 3. Comparison of Learning Paradigms

| Learning Paradigm | Primary Driving Signal | Optimization Mechanism | Target Domain | Biological Plausibility |
|---|---|---|---|---|
| **Supervised Error-Correction** (Delta / Backprop) | Ground-truth error $(t - y)$ | Gradient descent on loss $\mathcal{L}$ | Classification, regression, object detection | Low (requires precise non-local backward credit assignment) |
| **Unsupervised Hebbian / Oja** | Presynaptic-postsynaptic correlation $y \cdot x$ | Correlation matrix eigendecomposition | Dimensionality reduction, feature extraction | High (strictly local synaptic plasticity) |
| **Competitive / SOM** | Euclidean distance to prototype $\|x - w_k\|$ | Voronoi clustering, topographic neighborhood coupling | Unsupervised clustering, manifold visualization | High (lateral inhibition in cortical columns) |
| **Energy-Based Deterministic** (Hopfield) | Local magnetic field $h_i = \sum W_{ij} s_j$ | Lyapunov energy minimization $\Delta E \le 0$ | Associative content-addressable memory | Moderate (symmetric recurrent connectivity) |
| **Energy-Based Probabilistic** (Boltzmann / RBM) | Contrastive divergence $\langle v h^T \rangle_{\text{data}} - \langle v h^T \rangle_{\text{model}}$ | Stochastic Gibbs sampling, free energy minimization | Density estimation, generative modeling, feature pretraining | Moderate (probabilistic spiking neurons) |

---

## 4. Practical Relevance vs Educational Scope

### 4.1 Production Pipeline Integration vs Educational Demonstrations
To maintain scientific transparency:
- **Production Classification**: The operational tomato leaf disease detection engine uses our supervised, fine-tuned **ResNet18** (`artifacts/checkpoints/best_resnet18_weighted.pt`, 99.23% test accuracy) running in `app/app.py`.
- **Educational Scope of Phase C**: The classical algorithms developed in this phase (Hebbian, Oja, SOM, Hopfield, and RBM) serve as foundational theoretical and academic demonstrations fulfilling syllabus requirement REQ-06. They are not used to classify tomato disease images in the production pipeline because:
  1. *Hebbian / Oja*: Limited to linear subspace extraction, incapable of learning complex hierarchical visual representations like fungal necrotic halos.
  2. *Competitive / SOM*: Lacks class supervision, yielding coarse clusters that confuse visually similar disease morphologies (e.g., Early Blight vs Target Spot).
  3. *Hopfield Networks*: Severe memory capacity limit ($P_{\max} \approx 0.14 N \approx 9$ patterns for $64$ neurons), prone to spurious attractors when storing real-world continuous multi-spectral agricultural photos.
  4. *RBM*: Computationally superseded by modern deep autoencoders and diffusion models for high-resolution vision generation.

### 4.2 Potential Auxiliary Agricultural Applications
- **SOM**: Unsupervised exploration of unlabeled field sensor readings (temperature, soil humidity, leaf wetness) to identify micro-climate stress zones prior to visible infection.
- **Hopfield / RBM**: Noise-filtering and binary mask inpainting for corrupted QR codes or binary sensor packets from remote field telemetry stations.

---

## 5. Visualizations Index (8 New Figures Generated)

All figures rendered at 300 DPI, stored under `artifacts/figures/learning_paradigms/`:

| Figure Name | Requirement | Description & Key Insight |
|---|---|---|
| `hebbian_vs_oja_weight_growth.png` | REQ-06 | Dual-panel log plot: basic Hebbian weights explode ($10^{12}$), while Oja's rule stabilizes cleanly at $\|w\|_2 = 1.0011$ and converges to PCA alignment ($1.0000$). |
| `oja_pca_direction_alignment.png` | REQ-06 | 2D scatter plot comparing true principal eigenvector from covariance eigendecomposition with Oja's learned weight vector ($|\cos(\theta)| = 1.0000$). |
| `competitive_learning_clusters.png` | REQ-06 | Dual-panel plot: initial to final prototype migration in 2D space, and monotonic quantization MSE reduction over training epochs. |
| `som_topological_organization.png` | REQ-06 | 2D scatter with overlaid $5 \times 5$ lattice mesh showing how Kohonen SOM neurons unfold to model multi-cluster data topology. |
| `error_correction_delta_rule_training.png` | REQ-06 | Dual-panel plot: MSE loss and misclassification count reduction, and decision boundary shift from untrained to zero-error linear separation. |
| `hopfield_pattern_recovery.png` | REQ-06 | 3-panel image: ground truth $8 \times 8$ leaf pattern, $28\%$ corrupted input, and Hopfield recalled state ($96.9\%$ bit match). |
| `hopfield_energy_trajectory.png` | REQ-06 | Line plot demonstrating strict monotonic Lyapunov energy descent ($\Delta E \le 0$) from $E = -7.09$ down to attractor $E = -51.88$. |
| `boltzmann_rbm_sampling_energy.png` | REQ-06 | 4-panel figure: CD-1 reconstruction MSE decay, average free energy progression, and original vs reconstructed $4 \times 4$ binary patterns. |

---

## 6. Full Automated Test Suite Verification & Audit Reconciliation

### 6.1 Phase C Test Count & Mathematical Property Mapping
PyTorch and NumPy test functions are collected by `pytest` at the function level. `tests/test_learning_paradigms.py` contains **10 test functions**, which together verify **12 distinct mathematical properties**:

| # | Verified Mathematical Property | Implementing Pytest Function | Test File | Pytest Status |
|---|---|---|---|---|
| 1 | Basic Hebbian update equation exactness ($\Delta w = \eta y x$) | `test_hebbian_update_equation` | `test_learning_paradigms.py` | **PASSED** |
| 2 | Oja's rule weight stabilization ($\|w\|_2 \approx 1.0$) | `test_oja_rule_stabilization_and_pca_alignment` | `test_learning_paradigms.py` | **PASSED** |
| 3 | Oja's rule PCA eigenvector alignment ($|\cos\theta| > 0.95$) | `test_oja_rule_stabilization_and_pca_alignment` | `test_learning_paradigms.py` | **PASSED** |
| 4 | Competitive learning BMU minimum Euclidean selection | `test_competitive_bmu_selection` | `test_learning_paradigms.py` | **PASSED** |
| 5 | Kohonen SOM Gaussian neighborhood-distance decay | `test_som_neighborhood_influence` | `test_learning_paradigms.py` | **PASSED** |
| 6 | Delta Rule MSE reduction & linear separation | `test_error_correction_delta_rule_convergence` | `test_learning_paradigms.py` | **PASSED** |
| 7 | Hopfield weight matrix symmetry ($W = W^T$) | `test_hopfield_weight_matrix_symmetry_and_zero_diagonal` | `test_learning_paradigms.py` | **PASSED** |
| 8 | Hopfield zero diagonal self-connections ($W_{ii} = 0$) | `test_hopfield_weight_matrix_symmetry_and_zero_diagonal` | `test_learning_paradigms.py` | **PASSED** |
| 9 | Lyapunov energy monotonic descent ($\Delta E \le 0$) | `test_hopfield_energy_monotonic_descent` | `test_learning_paradigms.py` | **PASSED** |
| 10 | Associative memory pattern recovery from noisy input | `test_hopfield_pattern_recovery` | `test_learning_paradigms.py` | **PASSED** |
| 11 | RBM stochastic sigmoid probabilities bounded in $[0, 1]$ | `test_rbm_probabilities_and_energy_definitions` | `test_learning_paradigms.py` | **PASSED** |
| 12 | RBM joint energy definition and formula consistency | `test_rbm_probabilities_and_energy_definitions` | `test_learning_paradigms.py` | **PASSED** |
| *Aux* | RBM CD-1 reconstruction error reduction across epochs | `test_rbm_training_reconstruction_reduction` | `test_learning_paradigms.py` | **PASSED** |

*Note*: No test in `tests/test_learning_paradigms.py` is parametrized. The 12 mathematical properties are tested across 10 collected functions (with properties 2+3, 7+8, and 11+12 asserted jointly within their respective cohesive functions).

### 6.2 Full Repository Test Audit Table
Independently verified via `.venv/bin/python -m pytest tests/ -v`:

| Test Suite | Files | Collected | Passed | Failed | Skipped |
|---|---|---:|---:|---:|---:|
| **Production Core** | `test_dataset.py`, `test_models.py`, `test_app.py` | 26 | 26 | 0 | 0 |
| **Phase A** (REQ-01, 02, 03) | `test_math_sandbox.py` (10), `test_neural_fundamentals.py` (11), `test_mlp_from_scratch.py` (6) | 27 | 27 | 0 | 0 |
| **Phase B** (REQ-04, 05) | `test_optimization.py` (7), `test_regularization.py` (8) | 15 | 15 | 0 | 0 |
| **Phase C** (REQ-06) | `test_learning_paradigms.py` (10) | 10 | 10 | 0 | 0 |
| **Total Repository** | **9 test files** | **78** | **78** | **0** | **0** |

---

## 7. Confirmation of Strict Isolation

- **Production Checkpoint Unmodified**: `artifacts/checkpoints/best_resnet18_weighted.pt` remains untouched.
- **Dataset Manifests Unmodified**: `data/splits/train.csv`, `val.csv`, and `test.csv` remain intact.
- **Interactive Application Active**: Streamlit is currently running at `http://localhost:8501`.
- **Git Repository Clean**: No commits or pushes made.
- **Phase Boundary**: Execution has stopped at the completion of Phase C. Ready for your review before proceeding to Phase D.

