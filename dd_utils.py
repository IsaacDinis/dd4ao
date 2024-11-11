import numpy as np
import control as ct
import cvxpy as cp
from matplotlib import pyplot as plt


def freqresp(sys, w):
    return ct.frequency_response(sys,w.squeeze()).fresp.squeeze()

def rcone(x,y,z):
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

def G_tf(delay, fs):
    dummy = np.zeros(delay+1)
    dummy[0] = 1
    return ct.tf(np.array([1]),dummy,1/fs)

def G_freq_resp(delay,w,fs):
    phase_shift = -w * delay / fs
    return np.exp(1j * phase_shift)

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


def compute_fft_mag_welch(data, fft_size, fs):
    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_modes = data.shape[1]

    window_size = fft_size

    n_frames = (data.shape[0] - window_size) // (fft_size // 2) + 1
    spectrogram = np.zeros((fft_size // 2 + 1, n_frames, n_modes))

    window = np.hamming(window_size)

    for mode in range(n_modes):
        for i in range(n_frames):

            start_idx = i * (fft_size // 2)  # Overlap by 50%
            data_w = data[start_idx:start_idx + window_size, mode]
            data_w = data_w * window
            fft_result = np.fft.rfft(data_w)
            psd_w = (np.abs(fft_result) / fft_size) ** 2
            psd_w[1:-1] *= 2  # Double non-DC, non-Nyquist components
            spectrogram[:, i, mode] = psd_w

    avg_psd = np.mean(spectrogram, axis=1).squeeze()
    magnitude_spectrum = np.sqrt(avg_psd)
    f = np.linspace(0, fs / 2, fft_size // 2 + 1)

    return magnitude_spectrum, f, spectrogram

def plot_sensitivity(ax,G, K, K0, dist_psd, f, bandwidth):
    if f[0] == 0:
        f = f[1:]
        dist_psd = dist_psd[1:]

    val = np.interp(bandwidth * 2 * np.pi, f.squeeze(), dist_psd.squeeze())
    dist_psd = dist_psd / val
    K0_cl = ct.feedback(1, G*K0)
    K_cl = ct.feedback(1, G*K)
    K0_cl_freqresp = np.abs(freqresp(K0_cl, f*2*np.pi))
    K_cl_freqresp = np.abs(freqresp(K_cl, f*2*np.pi))

    ax.semilogx(f,20*np.log10(K0_cl_freqresp))
    ax.semilogx(f, 20 * np.log10(K_cl_freqresp))
    ax.semilogx(f, 20 * np.log10(1/dist_psd))
    ax.legend(('integrator', 'datadriven', 'disturbance^-1'))
    ax.set_title('sensitivity function')
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("magnitude [dB]")
    ax.grid()
    return

def plot_comp_sensitivity(ax,G, K, K0,f):
    if f[0] == 0:
        f = f[1:]
    K0_cl = ct.feedback(G*K0, 1)
    K_cl = ct.feedback(G*K, 1)
    K0_cl_freqresp = np.abs(freqresp(K0_cl, f*2*np.pi))
    K_cl_freqresp = np.abs(freqresp(K_cl, f*2*np.pi))

    ax.semilogx(f,20*np.log10(K0_cl_freqresp))
    ax.semilogx(f, 20 * np.log10(K_cl_freqresp))
    ax.legend(('integrator', 'datadriven'))
    ax.set_title('complementary sensitivity function')
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("magnitude [dB]")
    ax.grid()
    return

def plot_res_psd(ax,res_K0, res_K, fft_size, fs):
    res_K0_psd, f, _ = compute_fft_mag_welch(res_K0, fft_size, fs)
    res_K_psd, _, _ = compute_fft_mag_welch(res_K, fft_size, fs)

    ax.semilogx(f,20*np.log10(res_K0_psd))
    ax.semilogx(f, 20 * np.log10(res_K_psd))
    ax.set_title('residual PSD')
    ax.legend(('integrator','datadriven'))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("magnitude [dB]")
    ax.grid()


def plot_combined(G, K, K0, dist_psd, f, res_K0, res_K, fft_size, fs, mode_n, bandwidth):
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 subplots stacked vertically

    plot_sensitivity(axs[0], G, K, K0, dist_psd, f, bandwidth)

    plot_comp_sensitivity(axs[1], G, K, K0, f)

    plot_res_psd(axs[2], res_K0, res_K, fft_size, fs)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    std_res_K0 = np.std(res_K0)
    std_res_K = np.std(res_K)
    fig.suptitle(f'Mode number: {mode_n}, int rms: {std_res_K0:.4f}, dd rms: {std_res_K:.4f}', fontsize=16)

    plt.show()

def pol_reconstruct(command, measurement, delay):
    delay_floor = int(np.floor(delay))
    delay_ceil = int(np.ceil(delay))
    delay_frac,_ = np.modf(delay)
    pol = measurement[delay_ceil:,:] + (1 - delay_frac) * command[1:-delay_floor,:] + delay_frac * command[:-delay_ceil,:]
    return pol
