"""
DD4AO  convex code

Author : Isaac Dinis
Email  : isaac.dinis@unige.ch
Date   : 2026-03-06
"""


import numpy as np
import cvxpy as cp
import control as ct
from scipy import signal
from dd_utils import *
import time


class DD4AO:
    def __init__(self, w, G_freq, disturbance, order, fs, K0_num=np.array([0.2, 0]),
                 K0_den=np.array([1, -0.99]), Fx=np.array([1]), Fy=np.array([1, -0.99]),
                 n_iter=10, tol=1e-3, high_freq_u_lim=False):

        self.K = None
        self.S_freq = None
        self.T_freq = None
        self.GK_freq = None
        self.K_freq = None
        self.gm = None
        self.pm = None
        self.wcg = None
        self.wcp = None

        if w.ndim == 1:
            w = w[:, np.newaxis]
        if G_freq.ndim == 1:
            G_freq = G_freq[:, np.newaxis]
        if disturbance.ndim == 1:
            disturbance = disturbance[:, np.newaxis]

        self.w = w
        self.G_freq = G_freq
        self.W_norm = self.w / fs
        self.order = order
        bandwidth = fs/4
        val = np.interp(bandwidth * 2 * np.pi, w.squeeze(), disturbance.squeeze())
        self.disturbance = disturbance/val # normalize
        
        self.fs = fs
        self.Ts = 1 / fs
        self.Fx = Fx
        self.Fy = Fy

        den = np.pad(K0_den, (0, self.order + 1 - K0_den.shape[0]))
        num = np.pad(K0_num, (0, self.order + 1 - K0_num.shape[0]))
        num, _ = signal.deconvolve(num, self.Fx)
        den, _ = signal.deconvolve(den, self.Fy)
        self.num = num[:, np.newaxis]
        self.den = den[:, np.newaxis]

        self.radius = 0.99

        self.n_iter = n_iter
        self.tol = tol
        self.obj_prev = np.inf

        self.high_freq_u_lim = high_freq_u_lim

        self.sigm_weight = 10
        self.sigm_lambda = 0.05
        sigm = sigmoid_array(w.shape[0],w.shape[0]-1, self.sigm_lambda) * self.sigm_weight
        W2 = np.multiply(sigm, G_freq.flatten())

        self.W2 = W2[:, np.newaxis]

    def robust_nyquist(self, P, Pc):
        if np.abs(self.w[0]) < 1e-4:
            Pc_start = np.conj(Pc[1])
            P_start = cp.reshape(cp.conj(P[1]), (1, 1), order='C')
        else:
            Pc_start = np.conj(Pc[0])
            P_start = cp.reshape(cp.conj(P[0]), (1, 1), order='C')
        if np.abs(self.w[-1] - np.pi * self.fs) < 1e-4:
            Pc_end = np.conj(Pc[-2])
            P_end = cp.reshape(cp.conj(P[-2]), (1, 1), order='C')
        else:
            Pc_end = np.conj(Pc[-1])
            P_end = cp.reshape(cp.conj(P[-1]), (1, 1), order='C')
        Pc_ = np.vstack((Pc_start, Pc, Pc_end))
        Cp2 = cp.vstack([P_start, P, P_end])
        n = self.get_normal_direction(Pc_)
        polygonalP1 = 2 * cp.real(cp.multiply(np.conj(n), Cp2[:-1]))
        polygonalP2 = 2 * cp.real(cp.multiply(np.conj(n), Cp2[1:]))
        return [polygonalP1 >= 1e-5, polygonalP2 >= 1e-5]

    def con_radius(self, z_, szy, Y_c, Y_n):
        if z_.ndim == 1:
            z_ = z_[:, np.newaxis]
        z_start = np.conj(z_[0])
        z_end = np.conj(z_[-1])
        z2 = np.vstack((z_start, z_, z_end))
        Zys_ = np.power.outer(z2 * self.radius, np.arange(szy - 1, -1, -1)).squeeze()
        if Zys_.ndim == 1:  # happens if controller order is 1
            Zys_ = Zys_[:, np.newaxis]
        Ycs_ = Zys_ @ Y_c
        n = self.get_normal_direction(Ycs_)
        polygonalY1 = 2 * cp.real(cp.multiply(np.conj(n), Zys_[:-1] @ Y_n))
        polygonalY2 = 2 * cp.real(cp.multiply(np.conj(n), Zys_[1:] @ Y_n))
        return [polygonalY1 >= 1e-5, polygonalY2 >= 1e-5]

    def solve_iter(self, verbose):

        X_c = self.num
        Y_c = self.den

        szx = X_c.shape[0]
        szy = Y_c.shape[0]

        X_n = cp.Variable((szx, 1))
        if szy > 1:
            Y_n = cp.vstack([np.ones((1, 1)), cp.Variable((szy - 1, 1))])
        else:
            Y_n = np.ones((1, 1))
        XY_n = cp.vstack([X_n, Y_n])
        z = ct.tf([1, 0], 1, 1 / self.fs)
        z_ = freqresp(z, self.w)
        Zy = np.power.outer(z_, np.arange(szy - 1, -1, -1))
        Zx = np.power.outer(z_, np.arange(szx - 1, -1, -1))
        Fx = freqresp(ct.tf(self.Fx, 1, self.Ts), self.w)
        Fy = freqresp(ct.tf(self.Fy, 1, self.Ts), self.w)
        Fx = Fx[:, np.newaxis]
        Fy = Fy[:, np.newaxis]

        Ycs = Zy @ Y_c
        Xcs = Zx @ X_c
        Yc = Ycs * Fy
        Xc = Xcs * Fx
        ZFy = Zy * Fy
        ZFx = Zx * Fx
        Yf = ZFy @ Y_n
        Xf = ZFx @ X_n

        obj_2 = cp.Variable(1)
        gam_2 = cp.Variable(((self.w).shape[0], 1))

        if self.high_freq_u_lim:
            obj_inf = cp.Variable((1, 1))
            gam_inf = cp.multiply(obj_inf, np.ones(((self.w).shape[0], 1)))

        integ = 1 / (self.fs * 2 * np.pi) * (
                    np.append(np.diff(self.w.squeeze()), 0) + np.insert(np.diff(self.w.squeeze()), 0, 2 * self.w[0]))
        F_a = cp.multiply(self.disturbance, Yf)
        if self.high_freq_u_lim:
            F_b = cp.multiply(self.W2, Xf)
        Pc = self.G_freq * Xc + Yc
        P = cp.multiply(self.G_freq, Xf) + Yf

        PHI = 2 * cp.real(cp.multiply(cp.hstack([cp.multiply(self.G_freq, ZFx), ZFy]), np.conj(Pc))) @ XY_n - np.abs(
            Pc) ** 2
        CON = self.rcone(PHI, gam_2, F_a)

        if self.high_freq_u_lim:
            CON += self.rcone(PHI, gam_inf, F_b)
            CON += [cp.sum(cp.multiply(integ[:, np.newaxis], gam_2)) <= obj_2, obj_2 >= 0, gam_inf >= 0]
        else:
            CON += [cp.sum(cp.multiply(integ[:, np.newaxis], gam_2)) <= obj_2, obj_2 >= 0]

        CON += self.robust_nyquist(P, Pc)
        CON += self.con_radius(z_, szy, Y_c, Y_n)

        if self.high_freq_u_lim:
            prob = cp.Problem(cp.Minimize(obj_2 + obj_inf), CON)
        else:
            prob = cp.Problem(cp.Minimize(obj_2), CON)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False, tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7)
        except cp.error.SolverError as e:
            print("Solver failed:", e)
            return -1


        if isinstance(Y_n, np.ndarray):  # happens if controller order is 1
            K_temp,_,_ = self.get_K(X_n.value,Y_n)
        else:
            K_temp,_,_ = self.get_K(X_n.value,Y_n.value)

        stable = self.check_nyquist_stability(K_temp)
        if not stable:
            print("Controller unstable stopping iterations")
            return -1
        if verbose:
            print('obj = {:.2f}'.format(obj_2.value[0]))
            if self.high_freq_u_lim:
                print('obj_inf = {:.5f}'.format(obj_inf.value[0][0]))

        self.num = X_n.value
        if isinstance(Y_n, np.ndarray):  # happens if controller order is 1
            self.den = Y_n
        else:
            self.den = Y_n.value
        return obj_2.value[0]

    def compute_controller(self, verbose=False):
        t_start = time.perf_counter()
        for i in range(self.n_iter):
            # print(i)
            obj = self.solve_iter(verbose)
            if np.abs(self.obj_prev - obj) < self.tol or obj == -1:
                break
            if verbose:
                print('iter {} obj = {:.5f} diff = {:.5f}'.format(i, obj, np.abs(self.obj_prev - obj)))
            self.obj_prev = obj


        self.K,self.num,self.den = self.get_K(self.num,self.den)
        self.K_freq = freqresp(self.K, self.w)
        self.K_freq = self.K_freq[:, np.newaxis]
        self.GK_freq = self.K_freq * self.G_freq
        self.S_freq = 1 / (1 + self.GK_freq)
        self.T_freq = self.GK_freq / (1 + self.GK_freq)

        self.gm, self.pm, self.wcg, self.wcp = ct.margin(np.abs(self.GK_freq).squeeze(),np.angle(self.GK_freq, deg=True).squeeze(),self.w.squeeze())
        elapsed_time = time.perf_counter() - t_start

        if verbose:
            print('time elapsed = {:.2f} s'.format(elapsed_time))
        return 1

    def get_K(self, num, den): # convolves num and den with fixed part and returns K transfer function

        if num.squeeze().ndim == 0:
            num = signal.convolve(np.reshape(num, (1)), self.Fx)
        else:
            num = signal.convolve(num.squeeze(), self.Fx)

        if den.squeeze().ndim == 0:
            den = signal.convolve(np.reshape(den, (1)), self.Fy)
        else:
            den = signal.convolve(den.squeeze(), self.Fy)

        return ct.tf(num, den, self.Ts), num, den

    def check_nyquist_stability(self, K):
        
        poles = np.roots(K.num[0][0])
        P = np.sum(np.abs(poles) > 1)
        K_freq = freqresp(K, self.w)
        K_freq = K_freq[:, np.newaxis]
        GK_freq = K_freq * self.G_freq
        S_freq = 1 / (1 + GK_freq)
        S_full = np.concatenate([np.conj(S_freq[::-1]),S_freq])
        T = 1 / S_full

        # Phase tracking for encirclement of origin
        phase = np.angle(T.ravel())
        phase_unwrapped = np.unwrap(phase)
        self.phase = phase
        total_phase_change = phase_unwrapped[-1] - phase_unwrapped[0]
       
        # Clockwise encirclements of origin
        N = -total_phase_change / (2 * np.pi)

        # Closed-loop RHP poles
        Z = N + P
        stable =  Z < 1e-2

        return stable

    def rcone(self,x,y,z):
        # rcone_con = [
        #     cp.SOC(x[i] + y[i], cp.vstack([2 * z[i], x[i] - y[i]])) for i in range(x.shape[0])
        # ]
        rcone_con = cp.SOC((x + y).flatten(order = 'C'), cp.hstack([2 * z, x - y]).T)
        rcone_con =  [rcone_con,x >= 0, y >= 0]
        return rcone_con


    def get_normal_direction(self,r):
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
