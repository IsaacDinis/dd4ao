import astropy.io.fits as pfits
import numpy as np
import control as ct
import cvxpy as cp
from matplotlib import pyplot as plt
from utils import *
import dd4ao
# dist = pfits.getdata('data3/tilt_dist.fits').squeeze()
dist = pfits.getdata('data2/disturbance.fits').squeeze()

plt.figure()
plt.plot(dist)
plt.title('pseudo open loop tilt')
plt.show()

# dist_psd = pfits.getdata('data/dist_psd.fits').squeeze()
fs = 4000
dist_psd,f, _  = compute_psd_welch(dist,300,fs)
# f = pfits.getdata('data/f.fits').T.squeeze()
plt.figure()
plt.semilogx(f, 20*np.log10(dist_psd))
plt.title('pseudo open loop tilt PSD')
plt.xlabel("frequency [Hz]")
plt.ylabel("magnitude [dB]")
plt.grid()
plt.show()

w_log,dist_psd_log = interp_log(f*2*np.pi,dist_psd,300)



# plt.figure()
# plt.semilogy(f*2*np.pi,dist_psd)
# plt.semilogy(w_log,dist_psd_log)
# plt.show()


G = G_from_delay(2,fs)
G_resp = freqresp(G, w_log)

dd = dd4ao.dd4ao(w_log, G_resp, dist_psd_log, 20, 30, fs)
K = dd.compute_controler()

# print(check_K_stability(K,G))
print('maximum eigenvalue = {:.2f}, stable if < 1'.format(check_K_stability(K,G)))
K0 = ct.tf(np.array([0.5,0]),np.array([1,-1]), 1/fs)

res_K0 = evaluate_K_performance(K0,G,dist,fs)
res_K = evaluate_K_performance(K,G,dist,fs)

# print(np.std(res_K0[100:]))
# print(np.std(res_K[100:]))

print('int residual RMS = {:.4f}, dd residual RMS = {:.4f}'.format(np.std(res_K0[100:]),np.std(res_K[100:])))

res_K0_psd, _, _ = compute_psd_welch(res_K0[100:],300,fs)
res_K_psd, _, _ = compute_psd_welch(res_K[100:],300,fs)
theoretical_best = theoretical_best_perf(dist_psd/f[1])

print('int PSD integral = {:.2f}, dd PSD integral = {:.2f}, theoretical best = {:.2f}'.format(np.sum(res_K0_psd)/f[1],np.sum(res_K_psd)/f[1],theoretical_best))
# print(theoretical_best)
# print(np.sum(res_K0_psd)/f[1])
# print(np.sum(res_K_psd)/f[1])

# G = ct.tf(np.array([1]),np.array([1,0,0]), 1/fs)


plot_sensitivity(G,K,K0,dist_psd,f)
plot_comp_sensitivity(G,K,K0,f)
plot_res_psd(res_K0, res_K, 200, fs)
# sys = ct.feedback(1,G*K0)
# sys_ss = ct.tf2ss(sys)
# dummy = np.max(np.abs(np.linalg.eig(sys_ss.A).eigenvalues))
