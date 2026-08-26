# Boundary-Stabilized NG-RC for Quantum Transport in 1D XX and XXZ Chains

A local **Next-Generation Reservoir Computing (NG-RC)** framework for predicting and extrapolating long-time spatio-temporal quantum dynamics in one-dimensional **XX** and anisotropic **XXZ** spin chains.

The framework is trained only on short-time **Time-Evolving Block Decimation (TEBD)** simulation data and performs autonomous long-time rollouts of observables including the magnetization density

$$
\langle \sigma_j^z(t) \rangle.
$$

The main objective is to determine whether a local, data-driven dynamical model can learn the relevant transport behavior from short-time quantum dynamics and reliably extrapolate it to significantly longer times.

---

## Repository Structure

```text
├── xx_simulation_data/             # Ground-truth TEBD datasets for XX model dynamics
├── xxz_simulation_data/            # Ground-truth TEBD datasets for XXZ models across anisotropy Δ
├── ngrc_boundary_data/             # NG-RC predicted autonomous rollouts (.txt)
├── diffusion_results_xxz/          # Extracted D(Δ) tables and transport comparison plots
├── ngrc_boundary_plots/            # Spatio-temporal predictions and error heatmaps
├── tenpy_simulation.py             # MPS/TEBD data generator for XX chains
├── xxz_delta_simulation.py         # TEBD batch driver across anisotropy parameter Δ
├── RC-PDE-updated-boundary.py      # Boundary-stabilized local NG-RC trainer and predictor
├── xxz_diffusion.py                # MSD extraction and diffusion constant analysis
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Workflow

The project consists of three main stages:

1. **Generate ground-truth quantum dynamics** using MPS/TEBD simulations.
2. **Train a local boundary-stabilized NG-RC model** on a short-time portion of the dynamics and perform autonomous long-time predictions.
3. **Analyze transport properties**, particularly diffusion in the XXZ chain, by extracting mean squared displacement and effective diffusion constants.

---

## 1. Ground-Truth Generation

### `tenpy_simulation.py`

Generates reference trajectories for the XX spin chain using TEBD through TeNPy.

### `xxz_delta_simulation.py`

Extends the simulations to the anisotropic XXZ model and generates datasets across different values of the anisotropy parameter

$$
\Delta.
$$

The resulting datasets are stored in:

```text
xx_simulation_data/
xxz_simulation_data/
```

### Data Format

Simulation outputs are stored as space-delimited `.txt` files.

The general structure is:

```text
t  x₁(t)  x₂(t)  x₃(t)  ...  x_L(t)
```

where:

* **Column 0** contains the simulation time \(t\),
* **Columns 1 through \(L\)** contain the corresponding observable values at each spatial site.

---

## 2. Autoregressive NG-RC Forecasting

### `RC-PDE-updated-boundary.py`

This script implements the central machine-learning component of the project: a **local, autoregressive Next-Generation Reservoir Computing model**.

The model is trained using only a short-time segment of the TEBD trajectory,

$$
t \leq t_{\mathrm{short}},
$$

and subsequently performs autonomous rollouts beyond the training window.

Predicted trajectories are stored in:

```text
ngrc_boundary_data/
```

### Local Spatial Architecture

Rather than learning the full system as a single global dynamical map, the NG-RC model uses local spatial information.

For each lattice site, the prediction depends on a finite local neighborhood defined by a spatial stencil. The final architecture uses a stencil radius

$$
R = 2.
$$

This allows the model to learn local transport and interaction patterns while maintaining a scalable representation that does not require a separate global feature space for the entire chain.

### Boundary Stabilization

Open boundary conditions can introduce systematic regression artifacts because sites near the edges have incomplete spatial neighborhoods.

To address this, the model uses:

* zero-padded local spatial stencils,
* explicit boundary-distance encoding,

$$
d_{\mathrm{edge}},
$$

which provides the regression model with information about a site's distance from the nearest boundary.

This allows the NG-RC model to distinguish bulk dynamics from boundary dynamics and substantially reduces artificial edge effects during autonomous rollouts.

### Stability During Long-Time Rollouts

Autoregressive prediction can accumulate small local errors over many time steps. To improve numerical stability without imposing artificial global conservation laws, the model applies:

* smooth `tanh`-based step bounding,
* physically motivated range clipping.

The model enforces the global magnetization conservation.
---

## 3. Transport and Diffusion Analysis

### `xxz_diffusion.py`

The final stage analyzes transport properties of the XXZ dynamics and compares the transport behavior extracted from:

* the ground-truth TEBD simulation,
* the long-time NG-RC prediction,
* a short-time baseline extrapolation.

The analysis focuses on the spatial spreading of excess magnetization.

### Mean Squared Displacement

For a normalized excess magnetization profile \(P_j(t)\), the spatial mean squared displacement is computed as

$$
\mathrm{MSD}(t)
=
\sum_{j=0}^{L-1}
\left(j-\mu(t)\right)^2 P_j(t),
$$

where

$$
\mu(t)
=
\sum_{j=0}^{L-1} jP_j(t)
$$

is the center of the distribution.

The MSD quantifies the spreading of the magnetization profile across the spin chain.

### Signal Smoothing

Raw MSD curves can contain numerical fluctuations caused by finite-size effects, TEBD noise, and accumulated prediction errors.

To stabilize the transport analysis, the MSD signal is smoothed using a **Savitzky-Golay filter** before performing the diffusion fit.

### Diffusion Constant Extraction

Within a diffusive regime, the mean squared displacement is expected to scale approximately as

$$
\mathrm{MSD}(t) = 2Dt + C,
$$

where:

* \(D\) is the diffusion constant,
* \(C\) is an offset.

The script fits this relation over a selected evaluation window,

$$
[t_{\min},\,t_{\max}],
$$

to extract an effective diffusion coefficient.

The resulting diffusion constants are compared as functions of the XXZ anisotropy parameter:

$$
D_{\mathrm{true}}(\Delta),
$$

$$
D_{\mathrm{pred}}(\Delta),
$$

and

$$
D_{\mathrm{base}}(\Delta).
$$

These correspond respectively to:

* the diffusion constant extracted from the full TEBD dynamics,
* the diffusion constant extracted from the NG-RC rollout,
* the diffusion constant obtained from the short-time baseline.

The analysis automatically focuses on the selected diffusive regime,

$$
\Delta > 1,
$$

and evaluates the relative error between the predicted and reference transport coefficients.

Results are stored in:

```text
diffusion_results_xxz/
```

---

# Output and Evaluation

The predicted dynamics are evaluated using both direct spatio-temporal comparisons and derived transport quantities.

### Spatio-Temporal Dynamics

The repository contains heatmaps comparing:

* ground-truth TEBD evolution,
* NG-RC autonomous predictions,
* absolute or relative prediction errors.

These are stored in:

```text
ngrc_boundary_plots/
```

The visualizations make it possible to identify:

* the onset of long-time prediction errors,
* propagation of local disturbances,
* boundary artifacts,
* deviations in transport velocity and spreading behavior.

### Transport-Level Evaluation

In addition to pointwise prediction accuracy, the project evaluates whether the learned dynamics reproduce macroscopic transport properties.

The primary transport observable is the diffusion constant

$$
D(\Delta).
$$

This is particularly important because a model may accumulate local prediction errors while still reproducing the correct large-scale transport behavior.

---

# Alternative Approaches Evaluated

Several alternative data-driven approaches were considered and tested before converging on the final local boundary-stabilized NG-RC architecture.

These included:

* **Sparse Identification of Nonlinear Dynamics (SINDy)**
* **Standard deep neural networks**, including:

  * Long Short-Term Memory networks (LSTMs),
  * Gated Recurrent Units (GRUs),
  * Multi-Layer Perceptrons (MLPs)
* **Physics-Informed Neural Networks (PINNs)**
* **Standard Echo State Networks and random Reservoir Computing**
* **Unstabilized and global NG-RC architectures**

These approaches were ultimately less suitable for the combination of requirements in this problem: local spatio-temporal structure, autonomous long-time extrapolation, stability under repeated iteration, and robustness near open boundaries.

The final approach combines the relatively low computational cost and interpretable feature construction of NG-RC with an explicitly local representation and boundary-aware feature encoding.

---

# Method Overview

The complete computational pipeline can be summarized as:

```text
                ┌─────────────────────────┐
                │   XX / XXZ Spin Chain   │
                │       Hamiltonian       │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │     MPS + TEBD          │
                │  Ground-Truth Dynamics  │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   Short-Time Dataset    │
                │    t ≤ t_short          │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Local Boundary-Stabilized│
                │          NG-RC          │
                │                         │
                │  • Local stencil R = 2  │
                │  • Zero padding         │
                │  • Boundary encoding    │
                │  • Nonlinear features   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Autonomous Long-Time    │
                │        Rollout          │
                └────────────┬────────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
        ┌──────────────────┐   ┌──────────────────┐
        │ Spatio-Temporal  │   │ Transport / MSD  │
        │    Comparison    │   │     Analysis     │
        └──────────────────┘   └────────┬─────────┘
                                        │
                                        ▼
                               ┌──────────────────┐
                               │  Diffusion D(Δ)  │
                               └──────────────────┘
```

---

# Summary

This repository investigates whether **short-time quantum many-body dynamics can be extrapolated into the long-time regime using a local data-driven dynamical model**.

The final method uses a boundary-aware local NG-RC architecture trained on TEBD data from one-dimensional XX and XXZ spin chains. The model is designed to:

* learn local spatio-temporal dynamics from short trajectories,
* perform autonomous long-time prediction,
* reduce open-boundary artifacts through explicit boundary encoding,
* remain numerically stable during repeated autoregressive updates,
* reproduce large-scale transport behavior, including diffusion in the XXZ chain.

The resulting framework provides a computationally inexpensive surrogate model for studying long-time quantum transport while retaining direct comparison with high-precision tensor-network simulations.
