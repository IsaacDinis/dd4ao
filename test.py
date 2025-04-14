import astropy.io.fits as pfits
import numpy as np
import control as ct
from matplotlib import pyplot as plt
from dd_utils import *
import dd4ao


fs = 500 #Hz
n_fft = 300
delay = 2 #frames

dist = pfits.getdata('disturbance_test.fits')
t = np.arange(0,dist.shape[0]/fs,1/fs)
plt.figure()
plt.plot(t,dist)
plt.xlabel('t [s]')
plt.ylabel('amp')
plt.title('time domain disturbance')

dist_fft,f, _  = compute_fft_mag_welch(dist,n_fft,fs)
w = 2*np.pi*f

order = 15
bandwidth = 10 #[Hz]

G = G_tf(2, fs)
G_resp = freqresp(G, w)


dd = dd4ao.DD4AO(w, G_resp, dist_fft, order, bandwidth, fs)

dd.compute_controller(verbose = True)
K = dd.K

print('maximum eigenvalue = {:.2f}, stable if < 1 - gain margin = {:.2f} - phase phase margin = {:.2f} '.format(check_K_stability(K,G),dd.gm,dd.pm))
K0 = ct.tf(np.array([0.5,0]),np.array([1,-1]), 1/fs)

res_K0, u_K0 = evaluate_K_performance(K0,G,dist,fs)
res_K, u_K = evaluate_K_performance(K,G,dist,fs)



print('int residual RMS = {:.4f}, dd residual RMS = {:.4f}'.format(np.std(res_K0),np.std(res_K)))

res_K0_psd, _, _ = compute_fft_mag_welch(res_K0,n_fft,fs)
res_K_psd, _, _ = compute_fft_mag_welch(res_K,n_fft,fs)
theoretical_best = theoretical_best_perf(dist_fft/f[1])

print('int PSD integral = {:.2f}, dd PSD integral = {:.2f}, theoretical best = {:.2f}'.format(np.sum(res_K0_psd)/f[1],np.sum(res_K_psd)/f[1],theoretical_best))


plot_combined(G, K, K0, dist_fft, f, res_K0, res_K, n_fft, fs,0, bandwidth)
plt.show()

