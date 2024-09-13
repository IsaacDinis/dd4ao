import numpy as np
import control as ct
import cvxpy as cp
from matplotlib import pyplot as plt


def freqresp(sys, w):
    return ct.frequency_response(sys,w.squeeze()).fresp.squeeze()

def rcone(x,y,z):
    # rcone_con = [
    #     cp.SOC(x[i] + y[i], cp.vstack([2 * z[i], x[i] - y[i]])) for i in range(x.shape[0])
    # ]
    rcone_con = cp.SOC((x + y).flatten(), cp.hstack([2 * z, x - y]).T)
    rcone_con =  [rcone_con,x >= 0, y >= 0]
    return rcone_con

def logspace(start,stop,num):
    return np.logspace(np.log10(start),np.log10(stop),num)

def interp_log(w,psd,num):
    if w[0] == 0:
        w_log = logspace(w[1],w[-1],num)
    else:
        w_log = logspace(w[0],w[-1],num)
    psd_log = np.interp(w_log,w,psd)
    return w_log, psd_log

def G_from_delay(delay,fs):
    dummy = np.zeros(delay+1)
    dummy[0] = 1
    return ct.tf(np.array([1]),dummy,1/fs)

def check_K_stability(K,G):
    sys = ct.feedback(1, G * K)
    sys_ss = ct.tf2ss(sys)
    return np.max(np.abs(np.linalg.eig(sys_ss.A).eigenvalues))

def evaluate_K_performance(K,G,dist,fs):
    T = dist.shape[0]/fs
    t = np.arange(0,T,1/fs)
    sys = ct.feedback(1, G * K)
    res = ct.forced_response(sys,t,dist.squeeze())
    return res.outputs

def theoretical_best_perf(dist_psd):
    length = np.max(dist_psd.shape)
    return 10**(np.sum(np.log10(dist_psd))/length)*length

def get_normal_direction(r):
    n = 1j*np.diff(r, axis = 0)
    for i in range(len(n)):
        if n[i] == 0:
            n[i] = r[i]
        elif np.imag(np.conj(n[i])*r[i])*np.imag(np.conj(n[i])*r[i+1]) > 0:
            idx = np.argmin(np.abs(r[i:i+1]))
            n[i] = r[i+idx]
    n = n/np.abs(n)
    n = n*np.sign(np.real(np.conj(n)*r[0:-1]))
    return n


def compute_psd_welch(data, fft_size, fs):
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_modes = data.shape[1]

    if fft_size % 2 == 0:
        fft_size += 1

    window_size = fft_size * 2 - 1

    n_frames = (data.shape[0] // window_size) * 2 - 1
    spectrogram = np.zeros((fft_size, n_frames, n_modes))

    window = np.hamming(window_size) * 2

    for mode in range(n_modes):
        for i in range(n_frames):
            data_w = data[(i * fft_size):(i * fft_size + window_size), mode]
            data_w = data_w * window
            psd_w = np.abs(np.fft.fft(data_w)) / window_size
            psd_w = psd_w[:fft_size]
            psd_w[1:] = 2 * psd_w[1:]
            spectrogram[:, i, mode] = psd_w

    spectrogram = spectrogram.squeeze()
    psd = np.mean(spectrogram, axis=1).squeeze()

    f = fs * np.arange(0, (window_size // 2) + 1) / window_size

    return psd, f, spectrogram

def plot_sensitivity(G, K, K0, dist_psd, f):
    if f[0] == 0:
        f = f[1:]
        dist_psd = dist_psd[1:]
    K0_cl = ct.feedback(1, G*K0)
    K_cl = ct.feedback(1, G*K)
    K0_cl_freqresp = np.abs(freqresp(K0_cl, f*2*np.pi))

    K_cl_freqresp = np.abs(freqresp(K_cl, f*2*np.pi))
    plt.figure()
    plt.semilogx(f,20*np.log10(K0_cl_freqresp))
    plt.semilogx(f, 20 * np.log10(K_cl_freqresp))
    plt.semilogx(f, 20 * np.log10(1/dist_psd))
    plt.legend(('integrator', 'datadriven', 'disturbance^-1'))
    plt.title('sensitivity function')
    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude [dB]")
    plt.grid()
    plt.show()
    return

def plot_comp_sensitivity(G, K, K0,f):
    if f[0] == 0:
        f = f[1:]
    K0_cl = ct.feedback(G*K0, G*K0)
    K_cl = ct.feedback(G*K, G*K)
    K0_cl_freqresp = np.abs(freqresp(K0_cl, f*2*np.pi))
    K_cl_freqresp = np.abs(freqresp(K_cl, f*2*np.pi))
    plt.figure()
    plt.semilogx(f,20*np.log10(K0_cl_freqresp))
    plt.semilogx(f, 20 * np.log10(K_cl_freqresp))

    plt.legend(('integrator', 'datadriven'))
    plt.title('complementary sensitivity function')
    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude [dB]")
    plt.grid()
    plt.show()
    return

def plot_res_psd(res_K0, res_K, fft_size, fs):
    res_K0_psd, f, _ = compute_psd_welch(res_K0, fft_size, fs)
    res_K_psd, _, _ = compute_psd_welch(res_K, fft_size, fs)
    plt.figure()
    plt.semilogx(f,20*np.log10(res_K0_psd))
    plt.semilogx(f, 20 * np.log10(res_K_psd))
    plt.title('residual PSD')
    plt.legend(('integrator','datadriven'))
    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude [dB]")
    plt.grid()
    plt.show()
if __name__ == '__main__':
    fs = 3000
    z = ct.tf('z')
    sys = 1/z**2
    plop = ct.frequency_response(sys,np.array([1/fs,2/fs])).fresp.squeeze()