import astropy.io.fits as pfits
import numpy as np
import control as ct
import cvxpy as cp
from matplotlib import pyplot as plt
from dd_utils import *
import dd4ao
import time

class K_dd:

    def __init__(self, order, delay, S2M, M2V, training_set_size, fs, gain = 0.3, n_fft = 300, bandwidth = 120):
        self.f = None
        self.pol_psd = None
        self.order = order
        self.delay = delay
        self.S2M = S2M
        self.M2V = M2V
        self.V2M = np.linalg.pinv(self.M2V)
        self.n_modes = S2M.shape[0]
        self.bootstrap_size = 100
        self.training_set_size = training_set_size  -self.bootstrap_size
        self.iter = 0
        self.res_modes = np.zeros((self.training_set_size, self.n_modes))
        self.u_modes = np.zeros((self.training_set_size, self.n_modes))
        self.training_set = np.zeros((self.training_set_size, self.n_modes))
        self.res_mat = np.zeros((self.training_set_size, self.n_modes))
        self.K_mat = np.zeros((2*self.order+1, self.n_modes))
        self.K_mat[0,:] = gain
        self.K_mat[order+1, :] = 1
        self.K_array  = np.empty(self.n_modes,dtype = dd4ao.DD4AO)
        self.fs = fs
        self.n_fft = n_fft
        self.bandwidth = bandwidth
        self.status = "bootstrapping"
        self.state_mat = np.zeros((2*self.order+1, self.n_modes))
        self.res_evaluated = np.zeros((2,self.n_modes))


    def step(self, slopes):
        voltage = self.compute_voltage(slopes)

        self.res_modes = np.roll(self.res_modes, -1, axis = 0)
        self.u_modes = np.roll(self.u_modes, -1, axis=0)
        self.res_modes[-1, :] = self.S2M @ slopes
        self.u_modes[-1, :] = self.V2M @ voltage
        if self.status == "bootstrapping":
            if self.iter == self.bootstrap_size:
                self.status = "training"
                self.iter = -1
            self.iter += 1
        if self.status == "training" :
            if self.iter == self.training_set_size:
                self.training_set = pol_reconstruct(self.u_modes, self.res_modes, self.delay)
                self.compute_K_mat()
                self.status = "trained" 
                self.iter = -1
            self.iter += 1
        return voltage

    def compute_K_mat(self):
        print('Starting controller optimization')
        t_start = time.time()
        self.pol_psd, self.f, _ = compute_fft_mag_welch(self.training_set, self.n_fft, self.fs)
        self.w = 2*np.pi*self.f
        self.G = G_tf(self.delay, self.fs)
        self.G_resp = freqresp(self.G, self.w)
        print("Started controller optimization")
        for i in range(self.n_modes):
            self.K_array[i] = dd4ao.DD4AO(self.w, self.G_resp, self.pol_psd[:,i], self.order, self.bandwidth, self.fs)
            self.K_array[i].compute_controller()
            self.K_mat[:self.order + 1,i] = self.K_array[i].num.squeeze()
            self.K_mat[self.order + 1:,i] = -self.K_array[i].den.squeeze()[1:]
            self.status = "trained"
        elapsed_time = time.time() - t_start
        print('Controller optimized in = {:.2f} s'.format(elapsed_time))

    def compute_single_mode(self,mode_n):
        t_start = time.time()
        pol_mode = pol_reconstruct(self.u_modes[:,mode_n], self.res_modes[:,mode_n], self.delay)
        self.pol_psd, self.f, _ = compute_fft_mag_welch(pol_mode, self.n_fft, self.fs)
        self.w = 2*np.pi*self.f
        self.G_resp = G_freq_resp(self.delay,self.f,self.fs)
        self.K_array[mode_n] = dd4ao.DD4AO(self.w, self.G_resp, self.pol_psd, self.order, self.bandwidth, self.fs)
        self.K_array[mode_n].compute_controller()
        elapsed_time = time.time() - t_start
        print('time elapsed = {:.2f} s'.format(elapsed_time))

    def compute_voltage(self,slopes):
        modes = self.S2M@slopes
        self.state_mat[1:, :] = self.state_mat[0:-1, :]
        self.state_mat[0, :] = modes
        command_mat = np.multiply(self.state_mat, self.K_mat)
        command_dd = np.sum(command_mat, axis=0)
        self.state_mat[self.order, :] = command_dd
        return -self.M2V @ command_dd

    def evaluate_performance(self, int_gain):
        K0 = ct.tf(np.array([int_gain, 0]), np.array([1, -1]), 1 / self.fs)
        for i in range(self.n_modes):
            self.res_evaluated[0,i] = np.std(evaluate_K_performance(K0,self.G ,self.pol_modes[:,i], self.fs)[100:])
            self.res_evaluated[1,i] = np.std(evaluate_K_performance(self.K_array[i].K, self.G, self.pol_modes[:, i], self.fs)[100:])
        plt.figure()
        plt.plot(self.res_evaluated[0,:])
        plt.plot(self.res_evaluated[1,:])
        plt.legend(('integrator', 'datadriven'))
        plt.xlabel("mode")
        plt.ylabel("rms")
        plt.show()

    def analyse_mode(self, int_gain, mode_n,bandwidth):
        K0 = ct.tf(np.array([int_gain, 0]), np.array([1, -1]), 1 / self.fs)
        res_K0 = evaluate_K_performance(K0, self.G, self.pol_modes[:, mode_n], self.fs)[100:]
        res_K = evaluate_K_performance(self.K_array[mode_n].K, self.G, self.pol_modes[:, mode_n], self.fs)[100:]
        plot_combined(self.G, self.K_array[mode_n].K, K0, self.pol_psd[:,mode_n], self.f, res_K0, res_K, self.n_fft, self.fs, mode_n, bandwidth)