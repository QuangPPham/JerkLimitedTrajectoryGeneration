import numpy as np
from numpy.polynomial import Polynomial
from quintic_spline import QuinticSplineInterpolation

class ServoRecon(QuinticSplineInterpolation):
    def __init__(self):
        self.spline = []
        self.total_t = []
        self.last_d1, self.last_d2 = [], []
        
    def add_trajectory(self, p, t, offset_time=False):
        """Connect new trajectory with current
        Assume last point of previous trajectory is
        coincident with first point of new trajectory
        """
        self.p = np.atleast_2d(np.asarray(p))
        self.n, self.t = self.p.shape
        if len(self.total_t) == 0:
            self.ts = np.asarray(t)
            self.total_t = self.ts
        else:
            if offset_time:
                self.ts = np.asarray(t) + self.total_t[-1]
            else:
                self.ts = np.asarray(t)
            self.total_t = np.hstack((self.total_t, self.ts[1:]))

        self.l = self.ts[1:] - self.ts[:-1]         # (t-1,)
        self.l_1 = self.l[:-1] + self.l[1:]         # (t-2,)
        self.l_2 = self.l_1[:-1] + self.l[2:]       # (t-3,)
        self.dp = self.p[:, 1:] - self.p[:, :-1]    # (n, t-1)
        self.dp_1 = self.p[:, 2:] - self.p[:, :-2]  # (n, t-2)
        self.dp_2 = self.p[:, 3:] - self.p[:, :-3]  # (n, t-3)

        # Piecewise quintic splines
        self.d1, self.d2 = self.calc_derivatives()
        if len(self.spline) != 0:
            self.d1[:,0] = self.last_d1[:,-1]
            self.d2[:,0] = self.last_d2[:,-1]
        self.last_d1 = self.d1
        self.last_d2 = self.d2
        self.coeff = self.calc_quintic_coef()
        A, B, C, D, E, F = self.coeff
        if len(self.spline) == 0:
            # spline[n][t]
            self.spline = [
                [Polynomial([F[n,i], E[n,i], D[n,i], C[n,i], B[n,i], A[n,i]])
                    for i in range(self.t-1)]
                for n in range(self.n)]
            self.v_spline = [
                [self.spline[n][i].deriv()
                    for i in range(self.t-1)]
                for n in range(self.n)]
            self.a_spline = [
                [self.v_spline[n][i].deriv()
                    for i in range(self.t-1)]
                for n in range(self.n)]
            self.j_spline = [
                [self.a_spline[n][i].deriv()
                    for i in range(self.t-1)]
                for n in range(self.n)]
        else:
            for n in range(self.n):
                self.spline[n] += [
                    Polynomial([F[n,i], E[n,i], D[n,i], C[n,i], B[n,i], A[n,i]])
                        for i in range(self.t-1)
                    ]
                self.v_spline[n] += [
                    self.spline[n][i].deriv()
                        for i in range(self.t-1)
                ]
                self.a_spline[n] += [
                    self.v_spline[n][i].deriv()
                        for i in range(self.t-1)
                ]
                self.j_spline[n] += [
                    self.a_spline[n][i].deriv()
                        for i in range(self.t-1)
                ]

    def interpolate(self, u, i):
        p_interp = [self.spline[n][i](u) for n in range(self.n)]
        v_interp = [self.v_spline[n][i](u) for n in range(self.n)]
        a_interp = [self.a_spline[n][i](u) for n in range(self.n)]
        j_interp = [self.j_spline[n][i](u) for n in range(self.n)]
        return np.array(p_interp), np.array(v_interp), np.array(a_interp), np.array(j_interp) # (n,)

    def get_sampling_reference(self, Ts):
        t_interp = self.total_t
        T_interp = self.total_t[1:] - self.total_t[:-1]
        t_sampling = np.linspace(t_interp[0], t_interp[-1], int((t_interp[-1] - t_interp[0])/Ts)+1)
        p_sampling = np.zeros((self.n, len(t_sampling)))
        v_sampling = np.zeros((self.n, len(t_sampling)))
        a_sampling = np.zeros((self.n, len(t_sampling)))
        j_sampling = np.zeros((self.n, len(t_sampling)))
        j = 0
        for i, t in enumerate(t_sampling):
            t_eval = t - t_interp[j]
            if t_eval > T_interp[j]:
                if j < len(T_interp)-1:
                    t_eval -= T_interp[j]
                    j+=1
            p_sampling[:, i], v_sampling[:, i], a_sampling[:, i], j_sampling[:, i], = self.interpolate(t_eval, j)

        ds_sampling = np.linalg.norm(p_sampling[:,1:] - p_sampling[:,:-1], axis=0)
        s_sampling = np.hstack((0., np.cumsum(ds_sampling)))

        sv_sampling = np.linalg.norm(v_sampling, axis=0)
        sa_sampling = np.linalg.norm(a_sampling, axis=0)
        sj_sampling = np.linalg.norm(j_sampling, axis=0)

        return t_sampling, p_sampling, s_sampling, v_sampling, sv_sampling, a_sampling, sa_sampling, j_sampling, sj_sampling
