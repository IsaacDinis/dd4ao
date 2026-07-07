# DD4AO — Data-Driven Control for Adaptive Optics

This repository implements **DD4AO**, a data-driven controller design method for
Adaptive Optics (AO) systems, along with supporting utilities (`dd_utils`) and an
example script showing how to design a controller, check its stability, and
compare its closed-loop performance against a standard integrator.

DD4AO is based on the work of Karimi et al. [1]. It synthesizes an Infinite
Impulse Response (IIR) filter of a specified order directly from
non-parametric frequency response magnitudes, using mixed-sensitivity criteria
combining $\mathcal{H}_2$ and $\mathcal{H}_\infty$ norms. For an AO
implementation, the relevant frequency responses are the system transfer
function, typically simplified as a pure delay, and the disturbance frequency
magnitude, obtained from the periodogram of the pseudo open-loop
reconstruction. The controller design is formulated as a convex optimization
problem and solved using CVXPY [2, 3], with Clarabel [4] as the solver for the
resulting Second-Order Cone Programming (SOCP) problem. Stability constraints,
disturbance rejection, measurement noise mitigation, and actuator stroke
penalties are all incorporated into the optimization. The optimization
objective can be summarized as:

$$
\min_{K(z)}\Big( \left\|\Phi(z)\,\mathcal{S}(z)\right\|_2^2 +
\sigma(z)\left\|\mathcal{T}(z)\right\|_\infty^2\Big)
$$

where $K(z)$ is the controller, $\Phi(z)$ is the disturbance frequency
magnitude, $\mathcal{S}(z)$ is the sensitivity function of the closed-loop
system, $\mathcal{T}(z)$ is the complementary sensitivity function, and
$\sigma(z)$ is a sigmoid weighting function that penalizes large stroke
commands at high frequencies.

## Contents

- **`dd4ao.py`** — Core implementation of the DD4AO controller design method.
  Given a system frequency response and a disturbance frequency response, `DD4AO` computes a
  controller that optimizes rejection performance over subject to stability constraints.
- **`dd_utils.py`** — Supporting utilities used throughout the design and
  evaluation pipeline (PSD estimation, system modeling, stability checks,
  performance evaluation, and plotting).
- **`example.py`** — End-to-end example: loads a disturbance, designs a
  DD4AO controller, checks its stability, and compares its performance against
  a simple integrator.

## Scope of the example

The example script performs a simple LTI (linear time-invariant) simulation
only: a single-input single-output pure-delay system, a recorded disturbance
time series, and a noise-free closed-loop simulation. It is **not** an
end-to-end AO simulation — there is no WFS noise, no spatial/modal decomposition,
and no telescope/atmosphere model. It's meant purely to illustrate the DD4AO
design and evaluation workflow on a minimal, self-contained example.

## Requirements

- Python 3.x
- [`astropy`](https://www.astropy.org/) (FITS I/O)
- [`numpy`](https://numpy.org/)
- [`python-control`](https://python-control.readthedocs.io/) (`control` package)
- [`cvxpy`](https://www.cvxpy.org/)
- [`matplotlib`](https://matplotlib.org/)

Install with:

```bash
pip install astropy numpy control cvxpy matplotlib
```

## Usage

The example script demonstrates the typical workflow:

```bash
python example.py
```

This will:

1. Load a disturbance time series from a FITS file (`disturbance_test.fits`).
2. Estimate its PSD via Welch's method.
3. Build a discrete-time system model (pure delay).
4. Design a DD4AO controller for the given system and disturbance.
5. Check closed-loop stability (max eigenvalue, gain margin, phase margin).
6. Simulate closed-loop performance for both the DD4AO controller and a
   reference integrator.
7. Report residual RMS.
8. Produce summary plots (time-domain disturbance, frequency response, PSD
   comparison).

### Configuration

Key parameters (set at the top of `example_dd4ao.py`):

| Parameter | Description                                    | Default |
|-----------|-------------------------------------------------|---------|
| `fs`      | Loop frequency [Hz]                              | 500     |
| `order`   | DD4AO controller order (optimization horizon)    | 10      |
| `n_fft`   | Number of FFT bins for Welch PSD estimation      | 300     |
| `delay`   | Loop delay [frames], discrete-time minimum is 1   | 1.2     |


## References

1. Karimi, A. and Kammer, C., "A data-driven approach to robust control of multivariable systems by convex optimization," *Automatica*, vol. 85, pp. 227–233, 2017. DOI: [10.1016/j.automatica.2017.07.063](https://doi.org/10.1016/j.automatica.2017.07.063)
2. Diamond, S. and Boyd, S., "CVXPY: A Python-embedded modeling language for convex optimization," *Journal of Machine Learning Research*, vol. 17, no. 83, pp. 1–5, 2016.
3. Agrawal, A., Verschueren, R., Diamond, S., and Boyd, S., "A rewriting system for convex optimization problems," *Journal of Control and Decision*, vol. 5, no. 1, pp. 42–60, 2018.
4. Goulart, P. J. and Chen, Y., "Clarabel: An interior-point solver for conic programs with quadratic objectives," 2024. arXiv:[2405.12762](https://arxiv.org/abs/2405.12762)


## License

MIT License. See `LICENSE` for details.
