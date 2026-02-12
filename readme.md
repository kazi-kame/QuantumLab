# Quantum Lab

## Demo
![QuantumLab Demo](media/quantumlab_demo.gif)

## Spectral Simulation of a Quantum Stadium Billiard
QuantumLab is an interactive numerical simulation of a two-dimensional quantum billiard system. It computes bound-state eigenfunctions and performs spectral time evolution inside a stadium-type domain inspired by:

E. J. Heller (1984)
Bound-State Eigenfunctions of Classically Chaotic Hamiltonian Systems: Scars of Periodic Orbits

## Governing Equation

We solve the time-dependent Schrödinger equation (dimensionless units):

  ### $i \frac{\partial \psi(\mathbf{r}, t)}{\partial t} =- \nabla^2 \psi(\mathbf{r}, t)$

inside a bounded domain $\Omega$ with Dirichlet boundary conditions:

  ### $\psi(\mathbf{r}) = 0 \quad \text{for } \mathbf{r} \in \partial \Omega $

The stationary eigenvalue problem is:

  ### $- \nabla^2 \psi_n(\mathbf{r}) = E_n \psi_n(\mathbf{r})$

## Numerical Discretization
The Laplacian is approximated using second-order finite differences:

### $\nabla^2 \psi_{i,j}\approx\frac{\psi_{i+1,j} - 2\psi_{i,j} + \psi_{i-1,j}}{\Delta x^2}+\frac{\psi_{i,j+1} - 2\psi_{i,j} + \psi_{i,j-1}}{\Delta y^2}$

This yields a sparse matrix eigenvalue problem:

### $L \mathbf{u}_n = -E_n \mathbf{u}_n$

Eigenpairs are computed using shift-invert mode targeting energy $\sigma$:

### $(L - \sigma I)^{-1}$

## Spectral Time Evolution
An initial Gaussian wavepacket is defined as:

### $\psi_0(\mathbf{r}) = \exp\left(-\frac{|\mathbf{r}-\mathbf{r}_0|^2}{2\sigma^2}\right)\exp\left(i \mathbf{p}\cdot\mathbf{r}\right)$

It is expanded in the eigenbasis:

### $\psi_0 = \sum_{n} c_n \psi_n$

with coefficients

### $c_n = \langle \psi_n \mid \psi_0 \rangle$

Time evolution is performed spectrally:

### $\psi(\mathbf{r}, t) = \sum_{n}c_n e^{-i E_n t}\psi_n(\mathbf{r})$

## Localization Diagnostics
### Inverse Participation Ratio (IPR)
### $\mathrm{IPR}(t) = \int_{\Omega}|\psi(\mathbf{r}, t)|^4 \, d\mathbf{r}$
Higher IPR indicates stronger localization.

## Autocorrelation Function

### $C(t)=\left|\langle \psi_0 \mid \psi(t) \rangle\right|^2$
Peaks correspond to quantum revivals or orbit recurrences.

## Classical–Quantum Context
The classical stadium billiard is chaotic and contains unstable periodic orbits. Certain quantum eigenfunctions exhibit enhanced probability density along these classical trajectories — known as quantum scars.

This project numerically explores:
- High-energy eigenstates

- Wavepacket propagation

- Localization measures

It does not include semiclassical monodromy matrix analysis or Lyapunov exponent scaling.

## Installation
````bash
pip install -r requirements.txt
python quantum_lab.py
````

## Windows Executable
A standalone Windows executable is available in the Releases section.

## Controls
- Drag inside simulation window $\rightarrow$ Launch Gaussian wavepacket
- Space $\rightarrow$ Pause / Resume
- R $\rightarrow$ Recompute eigenstates

#### Adjustable parameters:
- Geometry parameter $a$
- Grid resolution $N$
- Target energy $\sigma$
- Time step $\Delta t$

## Dependencies

- pygame
- numpy
- scipy
- matplotlib

Install with:
````bash
pip install -r requirements.txt
````
