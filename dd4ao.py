import numpy as np
import cvxpy as cp
import control as ct
from scipy import signal
from dd_utils import *
import time

class DD4AO:
    def __init__(self, W, plant, disturbance, order, bandwidth, fs, overshoot_weight = 0.001, Fx=np.array([1]), Fy=np.array([1,-0.99]), radius=1, n_iter = 1000, tol = 1e-4):
        self.K = None
        if W.ndim == 1:
            W = W[:, np.newaxis]
        if plant.ndim == 1:
            plant = plant[:, np.newaxis]
        if disturbance.ndim == 1:
            disturbance = disturbance[:, np.newaxis]
        self.W = W
        self.plant = plant
        self.W_norm = self.W/fs
        self.order = order
        self.bandwidth = bandwidth
        val = np.interp(bandwidth*2*np.pi, W.squeeze(), disturbance.squeeze())
        self.disturbance = disturbance/val
        self.fs = fs
        self.Ts = 1/fs
        self.Fx = Fx
        self.Fy = Fy

        K0_num = np.array([0.2,0])
        K0_den = np.array([1,-1])

        den = np.pad(K0_den,(0,self.order+1-K0_den.shape[0]))
        num = np.pad(K0_num, (0, self.order + 1 - K0_num.shape[0]))
        num, _ = signal.deconvolve(num, self.Fx)
        den, _ = signal.deconvolve(den, self.Fy)
        self.num = num[:, np.newaxis]
        self.den = den[:, np.newaxis]
        self.radius = radius
        self.n_iter = n_iter
        self.tol = tol
        self.obj_prev = np.inf
        W2 = np.multiply(freqresp(ct.tf(overshoot_weight,1,1/fs), W),plant.flatten())
        self.W2 = W2[:, np.newaxis]

    def robust_nyquist(self,P,Pc):
        if np.abs(self.W[0]) < 1e-4:
            Pc_start = np.conj(Pc[1])
            P_start = cp.reshape(cp.conj(P[1]),(1,1))
        else:
            Pc_start = np.conj(Pc[0])
            P_start = cp.reshape(cp.conj(P[0]),(1,1))
        if np.abs(self.W[-1]-np.pi*self.fs) < 1e-4:
            Pc_end = np.conj(Pc[-2])
            P_end = cp.reshape(cp.conj(P[-2]),(1,1))
        else:
            Pc_end = np.conj(Pc[-1])
            P_end = cp.reshape(cp.conj(P[-1]),(1,1))
        Pc_ = np.vstack((Pc_start,Pc,Pc_end))
        Cp2 = cp.vstack([P_start,P,P_end])
        n = get_normal_direction(Pc_)
        polygonalP1 = 2*cp.real(cp.multiply(np.conj(n),Cp2[:-1]))
        polygonalP2 = 2*cp.real(cp.multiply(np.conj(n),Cp2[1:]))
        return [polygonalP1 >= 1e-5, polygonalP2 >= 1e-5]

    def con_radius(self, z_, szy, Y_c, Y_n):
        if z_.ndim == 1:
            z_ = z_[:, np.newaxis]
        z_start = np.conj(z_[0])
        z_end = np.conj(z_[-1])
        z2 = np.vstack((z_start,z_,z_end))
        Zys_ = np.power.outer(z2*self.radius, np.arange(szy - 1, -1, -1)).squeeze()
        if Zys_.ndim == 1: # happens if controller order is 1
            Zys_ = Zys_[:, np.newaxis]
        Ycs_ = Zys_@Y_c
        n = get_normal_direction(Ycs_)
        polygonalY1 = 2*cp.real(cp.multiply(np.conj(n),Zys_[:-1]@Y_n))
        polygonalY2 = 2*cp.real(cp.multiply(np.conj(n),Zys_[1:]@Y_n))
        return [polygonalY1 >= 1e-5, polygonalY2 >= 1e-5]

    def solve_iter(self):

        X_c = self.num
        Y_c = self.den

        szx = X_c.shape[0]
        szy = Y_c.shape[0]

        X_n = cp.Variable((szx,1))
        if szy > 1:
            Y_n = cp.vstack([np.ones((1,1)), cp.Variable((szy-1, 1))])
        else:
            Y_n = np.ones((1,1))
        XY_n = cp.vstack([X_n, Y_n])
        z = ct.tf([1,0],1,1/self.fs)
        z_ = freqresp(z, self.W)
        Zy = np.power.outer(z_, np.arange(szy-1, -1, -1))
        Zx = np.power.outer(z_, np.arange(szx-1, -1, -1))
        Fx = freqresp(ct.tf(self.Fx, 1, self.Ts), self.W)
        Fy = freqresp(ct.tf(self.Fy, 1, self.Ts), self.W)
        Fx = Fx[:, np.newaxis]
        Fy = Fy[:, np.newaxis]

        Ycs = Zy@Y_c
        Xcs = Zx@X_c
        Yc = Ycs*Fy
        Xc = Xcs*Fx
        ZFy = Zy*Fy
        ZFx = Zx*Fx
        Yf = ZFy@Y_n
        Xf = ZFx@X_n

        obj_2 = cp.Variable(1)
        gam_2 = cp.Variable(((self.W).shape[0], 1))

        obj_inf = cp.Variable((1,1))
        gam_inf = cp.multiply(obj_inf,np.ones(((self.W).shape[0],1)))

        integ = 1 / (self.fs * 2 * np.pi) * (np.append(np.diff(self.W.squeeze()), 0) + np.insert(np.diff(self.W.squeeze()), 0, 2 * self.W[0]))
        F_a = cp.multiply(self.disturbance,Yf)
        F_b = cp.multiply(self.W2, Xf)
        Pc = self.plant*Xc+Yc
        P = cp.multiply(self.plant,Xf)+Yf

        PHI = 2*cp.real(cp.multiply(cp.hstack([cp.multiply(self.plant, ZFx), ZFy]), np.conj(Pc))) @ XY_n - np.abs(Pc) ** 2
        CON = rcone(PHI, gam_2, F_a)
        CON += rcone(PHI, gam_inf, F_b)
        CON += [cp.sum(cp.multiply(integ[:,np.newaxis],gam_2)) <= obj_2, obj_2 >= 0, gam_inf >= 0]
        CON += self.robust_nyquist(P, Pc)
        CON += self.con_radius(z_, szy, Y_c, Y_n)
        prob = cp.Problem(cp.Minimize(obj_2+obj_inf), CON)
        prob.solve(solver=cp.CLARABEL, verbose = False, tol_gap_abs = 1e-4, tol_gap_rel = 1e-4, tol_feas = 1e-4)
        self.num = X_n.value
        if isinstance(Y_n, np.ndarray): # happens if controller order is 1
            self.den = Y_n
        else:
            self.den = Y_n.value
        return obj_2.value[0]

    def compute_controller(self, verbose = False):
        # t_start = time.time()
        for i in range(self.n_iter):
            obj = self.solve_iter()
            if np.abs(self.obj_prev-obj) < self.tol:
                break
            if verbose:
                print('iter {} obj = {:.5f} diff = {:.5f}'.format(i,obj,np.abs(self.obj_prev - obj)))
            self.obj_prev = obj

        self.num = signal.convolve(self.num.squeeze(), self.Fx)
        if self.den.squeeze().ndim == 0:
            self.den = self.Fy
        else:
            self.den = signal.convolve(self.den.squeeze(), self.Fy)
        self.K = ct.tf(self.num, self.den, self.Ts)

        # elapsed_time = time.time() - t_start
        # print('time elapsed = {:.2f} s'.format(elapsed_time))
        return 1



