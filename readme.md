# Quantum Lab

## Spectral Simulation of a Quantum Stadium Billiard

QuantumLab is an interactive numerical simulation of a two-dimensional quantum billiard system. It computes bound-state eigenfunctions and performs spectral time evolution inside a stadium-type domain inspired by:

> E. J. Heller (1984). Bound-State Eigenfunctions of Classically Chaotic Hamiltonian Systems: Scars of Periodic Orbits.

## Governing Equation

We solve the time-dependent Schrödinger equation (dimensionless units):

$$
i \frac{\partial \psi(\mathbf{r}, t)}{\partial t} = - \nabla^2 \psi(\mathbf{r}, t)
$$

inside a bounded domain $\Omega$ with Dirichlet boundary conditions:

$$
\psi(\mathbf{r}) = 0 \quad \text{for } \mathbf{r} \in \partial \Omega
$$

The stationary eigenvalue problem is:

$$
-\nabla^2 \psi_n(\mathbf{r}) = E_n \psi_n(\mathbf{r})
$$

## Numerical Discretization

The Laplacian is approximated using second-order finite differences on a grid with spacing $\Delta x, \Delta y$:

$$
\nabla^2 \psi_{i,j} \approx \frac{\psi_{i+1,j} - 2\psi_{i,j} + \psi_{i-1,j}}{\Delta x^2} + \frac{\psi_{i,j+1} - 2\psi_{i,j} + \psi_{i,j-1}}{\Delta y^2}
$$

This yields a sparse matrix eigenvalue problem $L \mathbf{u}_n = -E_n \mathbf{u}_n$. Eigenpairs are computed using the **Shift-Invert Spectral Transformation** to target eigenstates near a reference energy $\sigma$:

$$
(L - \sigma I)^{-1} \mathbf{x} = \nu \mathbf{x}
$$

where $\nu = (E_n - \sigma)^{-1}$. This is solved via the Arnoldi/Lanczos iterative method (`scipy.sparse.linalg.eigsh`).

## Spectral Time Evolution

An initial Gaussian wavepacket is defined as:

$$
\psi_0(\mathbf{r}) = \mathcal{N} \exp\left(-\frac{|\mathbf{r}-\mathbf{r}_0|^2}{2w^2}\right)\exp\left(i \mathbf{p}\cdot\mathbf{r}\right)
$$

It is expanded in the computed eigenbasis $\psi_0 = \sum_{n} c_n \psi_n$ with coefficients $c_n = \langle \psi_n | \psi_0 \rangle$. Time evolution is performed analytically in the spectral domain:

$$
\psi(\mathbf{r}, t) = \sum_{n} c_n e^{-i E_n t} \psi_n(\mathbf{r})
$$

## Phase Space Analysis

The simulation implements quasiprobability distributions to visualize quantum-classical correspondence:

### 1. Husimi Q-Function

A projection of the wavefunction onto coherent states $|\alpha_{x_0, p_0}\rangle$ (minimum uncertainty packets):

$$
Q(x_0, p_0) = \frac{1}{\pi} | \langle \alpha_{x_0, p_0} | \psi \rangle |^2
$$

### 2. Wigner Distribution (Marginal)

Computed via the autocorrelation of the wavefunction projected onto the transverse coordinate:

$$
W(p_y) = \frac{1}{\pi \hbar} \int_{-\infty}^{\infty} \phi^*(y+y') \phi(y-y') e^{2i p_y y' / \hbar} dy'
$$

### 3. Momentum Density

Calculated via Fast Fourier Transform (FFT):

$$
\tilde{\rho}(\mathbf{k}) = |\mathcal{F}[\psi(\mathbf{r})]|^2
$$

## Localization Diagnostics

### Inverse Participation Ratio (IPR)

Measures the spatial localization of an eigenstate. High IPR values generally indicate "scarred" states localized on unstable periodic orbits.

$$
\mathrm{IPR}_n = \int_{\Omega} |\psi_n(\mathbf{r})|^4 \, d\mathbf{r}
$$

### Autocorrelation Function

Peaks correspond to quantum revivals or classical orbit recurrences.

$$
C(t) = \left| \langle \psi_0 | \psi(t) \rangle \right|^2
$$

## Classical–Quantum Context

The classical stadium billiard is chaotic and contains unstable periodic orbits. Certain quantum eigenfunctions exhibit enhanced probability density along these classical trajectories — known as quantum scars.

This project numerically explores:

* High-energy eigenstates

* Wavepacket propagation

* Localization measures

It does not include semiclassical monodromy matrix analysis or Lyapunov exponent scaling.

## Controls

| **Key** | **Action** |
| :--- | :--- |
| **Mouse Drag** | Initialize and launch a Gaussian wavepacket with momentum $(p_x, p_y)$ |
| **Space** | Pause / Resume time evolution |
| **R** | Re-diagonalize the Hamiltonian (triggers after parameter changes) |
| **V** | Toggle View Mode (Density $\vert\psi\vert^2$ $\leftrightarrow$ Phase $\arg(\psi)$ ) |
| **P** | Cycle Phase Space Mode (Off $\to$ Husimi $\to$ Wigner) |
| **C** | Cycle Colormaps |

#### Adjustable parameters:

* Geometry parameter $a$ (Stadium elongation)

* Grid resolution $N$

* Target energy $\sigma$

* Time step $\Delta t$

## Installation & Dependencies

```
pip install -r requirements.txt
python quantum_lab_cleaned.py

```

**Dependencies:**

* `pygame` (Rendering and UI)

* `numpy` (Dense arrays, FFT)

* `scipy` (Sparse matrices, Arpack solver)

* `matplotlib` (Colormaps)
