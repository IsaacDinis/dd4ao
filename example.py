"""
Compare a Data-Driven AO controller (DD4AO) against a simple integrator
for a given disturbance profile, and report closed-loop performance
(residual RMS, PSD integral) against the theoretical best achievable.

Author : Isaac Dinis
Email  : isaac.dinis@unige.ch
Date   : 2026-03-06

"""

import astropy.io.fits as pfits
import numpy as np
import control as ct
from matplotlib import pyplot as plt

import dd4ao
from dd_utils import *  # compute_fft_mag_welch, G_tf, freqresp, check_K_stability,
                         # evaluate_K_performance, theoretical_best_perf, plot_combined

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
fs = 500          # loop frequency [Hz]
order = 10        # DD4AO controller order (optimization horizon)
n_fft = 300       # number of FFT bins for Welch PSD estimation
delay = 1.2       # loop delay [frames]; discrete-time minimum is 1, can be fractional
bandwidth = 10    # [Hz], used only for the summary plot's zoomed-in view

# -----------------------------------------------------------------------
# Load disturbance and inspect it in time domain
# -----------------------------------------------------------------------
dist = pfits.getdata('disturbance_test.fits')
t = np.arange(0, dist.shape[0] / fs, 1 / fs)

plt.figure()
plt.plot(t, dist)
plt.xlabel('t [s]')
plt.ylabel('amp')
plt.title('time domain disturbance')

# -----------------------------------------------------------------------
# Frequency-domain representation of the disturbance (Welch PSD)
# -----------------------------------------------------------------------
dist_fft, f, _ = compute_fft_mag_welch(dist, n_fft, fs)
w = 2 * np.pi * f

# -----------------------------------------------------------------------
# System model: pure delay (discretized) and its frequency response
# -----------------------------------------------------------------------
G = G_tf(delay, fs)
G_resp = freqresp(G, w)

# -----------------------------------------------------------------------
# Design the DD4AO (data-driven) controller
# -----------------------------------------------------------------------
dd = dd4ao.DD4AO(w, G_resp, dist_fft, order, fs)
dd.compute_controller(verbose=True)
K = dd.K

# --- Stability check: eigenvalue of the closed loop + margins ---
max_eig = check_K_stability(K, G)
print(
    f"maximum eigenvalue = {max_eig:.2f}, stable if < 1 "
    f"- gain margin = {dd.gm:.2f} - phase margin = {dd.pm:.2f}"
)

# -----------------------------------------------------------------------
# Baseline controller: simple integrator, for comparison
# -----------------------------------------------------------------------
K0 = ct.tf(np.array([0.5, 0]), np.array([1, -1]), 1 / fs)

# -----------------------------------------------------------------------
# Closed-loop performance evaluation (time domain)
# -----------------------------------------------------------------------
res_K0, u_K0 = evaluate_K_performance(K0, G, dist, fs)
res_K, u_K = evaluate_K_performance(K, G, dist, fs)

print(
    f"int residual RMS = {np.std(res_K0):.4f}, "
    f"dd residual RMS = {np.std(res_K):.4f}"
)

# -----------------------------------------------------------------------
# Summary plot: frequency response, PSDs, zoomed to bandwidth of interest
# -----------------------------------------------------------------------
plot_combined(G, K, K0, dist_fft, f, res_K0, res_K, n_fft, fs, 0, bandwidth)
plt.show()