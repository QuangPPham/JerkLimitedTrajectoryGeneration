import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import root_scalar

class QuinticSplineInterpolation():
    def __init__(self, p):
        """ Class for interpolating the path through a set of points p using a quintic
        spline, along with a cubic polynomial for estimating first and second derivatives
        
        Param: p with shape (n, t) where n is the number of dimensions and t is the number
        of points
        """
        self.p = np.atleast_2d(np.asarray(p))
        self.n, self.t = self.p.shape
        # difference between points
        self.dp = self.p[:, 1:] - self.p[:, :-1]    # (n, t-1)
        self.dp_1 = self.p[:, 2:] - self.p[:, :-2]  # (n, t-2)
        self.dp_2 = self.p[:, 3:] - self.p[:, :-3]  # (n, t-3)
        # chord length
        # l = np.sqrt(np.sum(np.square(dp), axis=0)) # (t-1,)
        self.l = np.linalg.norm(self.dp, axis=0)
        # sum of chord lengths
        self.l_1 = self.l[:-1] + self.l[1:]   # (t-2,)
        self.l_2 = self.l_1[:-1] + self.l[2:] # (t-3,)

        # Piecewise quintic splines
        self.d1, self.d2 = self.calc_derivatives()
        self.coeff = self.calc_quintic_coef()
        A, B, C, D, E, F = self.coeff
        # spline[n][t]
        self.spline = [
            [Polynomial([F[n,i], E[n,i], D[n,i], C[n,i], B[n,i], A[n,i]])
                for i in range(self.t-1)]
            for n in range(self.n)]

        # Path length
        self.delta_s = 0.
        self.L = 0.
        # self.L, self.S_l = self.calc_path_length(F_des, Ts)
        # self.F = F_des
        # self.Ts = Ts

    def calc_cubic_coef(self, i):
        """ Calculate coefficients for the cubic polynomial used
        to estimate the first and second derivatives of the path
        i = 1 to self.t-3
        """
        delta_mat = np.array([
            [self.l[i-1]**3, self.l[i-1]**2, self.l[i-1]],
            [self.l_1[i-1]**3, self.l_1[i-1]**2, self.l_1[i-1]],
            [self.l_2[i-1]**3, self.l_2[i-1]**2, self.l_2[i-1]]
        ])

        delta_a_mat = [
            np.array([
                [self.dp[n, i-1],   self.l[i-1]**2,   self.l[i-1]],
                [self.dp_1[n, i-1], self.l_1[i-1]**2, self.l_1[i-1]],
                [self.dp_2[n, i-1], self.l_2[i-1]**2, self.l_2[i-1]]
            ]) for n in range(self.n)
        ]

        delta_b_mat = [
            np.array([
                [self.l[i-1]**3,   self.dp[n, i-1],   self.l[i-1]],
                [self.l_1[i-1]**3, self.dp_1[n, i-1], self.l_1[i-1]],
                [self.l_2[i-1]**3, self.dp_2[n, i-1], self.l_2[i-1]]
            ]) for n in range(self.n)
        ]

        delta_c_mat = [
            np.array([
                [self.l[i-1]**3,   self.l[i-1]**2,   self.dp[n, i-1]],
                [self.l_1[i-1]**3, self.l_1[i-1]**2, self.dp_1[n, i-1]],
                [self.l_2[i-1]**3, self.l_2[i-1]**2, self.dp_2[n, i-1]]
            ]) for n in range(self.n)
        ]

        delta = np.linalg.det(delta_mat)
        delta_a = [np.linalg.det(delta_a_mat[n]) for n in range(self.n)]
        delta_b = [np.linalg.det(delta_b_mat[n]) for n in range(self.n)]
        delta_c = [np.linalg.det(delta_c_mat[n]) for n in range(self.n)]

        a = np.array(delta_a) / delta
        b = np.array(delta_b) / delta
        c = np.array(delta_c) / delta

        return a, b, c # (n,)


    def calc_derivatives(self):
        """Calculate first and second derivative
        of the splines at the knot points"""
        t = []
        n = []
        for i in range(1, self.t - 2):
            u = self.l[i-1]
            a, b, c = self.calc_cubic_coef(i)
            # first knot point
            if i == 1:
                t += [c]
                n += [2*b]
            # in-between knot points
            t += [3*a*u**2 + 2*b*u + c]
            n += [6*a*u + 2*b]
            # last two knot points
            if i == self.t - 3:
                u1 = self.l[self.t-4] + self.l[self.t-3]
                t += [3*a*u1**2 + 2*b*u1 + c]
                n += [6*a*u1 + 2*b]
                u2 = self.l[self.t-4] + self.l[self.t-3] + self.l[self.t-2]
                t += [3*a*u2**2 + 2*b*u2 + c]
                n += [6*a*u2 + 2*b]
        
        return np.array(t).T, np.array(n).T # (n, t)
    
    def calc_quintic_coef(self):
        """Calculate the coefficients of the quintic polynomials"""
        t, n = self.d1, self.d2
        A = (6*self.dp - 3*(t[:,1:]+t[:,:-1])*self.l + 0.5*(n[:,1:]-n[:,:-1])*self.l**2) / self.l**5
        B = (-15*self.dp + (7*t[:,1:]+8*t[:,:-1])*self.l + (1.5*n[:,:-1]-n[:,1:])*self.l**2) / self.l**4
        C = (10*self.dp - (4*t[:,1:]+6*t[:,:-1])*self.l - (1.5*n[:,:-1]-0.5*n[:,1:])*self.l**2) / self.l**3
        D = n[:,:-1] / 2
        E = t[:,:-1]
        F = self.p[:,:-1]

        return A, B, C, D, E, F # (n, t-1)
    

    def interpolate(self, u, i):
        p_interp = [self.spline[n][i](u) for n in range(self.n)]
        return np.array(p_interp) # (n,)


    def calc_interpolator_points(self, Ts, F, fs, fe, N_required=None):
        """ Calculate path length
        """
        M = np.round(self.l / (F*Ts)).astype(np.int32)
        dl = self.l / M
        S_l = np.zeros(self.t-1)
        
        p_last = self.p[:,0]
        for i in range(self.t-1):
            for j in range(1, M[i]+1):
                p_spline = self.interpolate(j*dl[i], i)
                ds = np.linalg.norm(p_spline - p_last)
                S_l[i] += ds
                p_last = p_spline
        
        L = np.sum(S_l)
        self.L = L

        """ Calculate interpolator points
            Such that arc length is constant
        """
        f_max = np.max([fs, F, fe])
        if N_required is None:
            N = np.round(self.L / (f_max*Ts)).astype(np.int32)
        else:
            N = N_required

        delta_s = L / N
        self.delta_s = delta_s
        A, B, C, D, E, F = self.coeff
        
        # Initialize the spline
        p_last = self.p[:,0]
        R_points = [p_last]
        u = 0.
        for i in range(self.t-1):
            alpha = [0]*11
            alpha[0] = np.sum(np.square(A[:,i]))
            alpha[1] = 2*(A[:,i] @ B[:,i])
            alpha[2] = np.sum(np.square(B[:,i])) + 2*(A[:,i] @ C[:,i])  #camhong dethuong
            alpha[3] = 2*(B[:,i] @ C[:,i] + A[:,i] @ D[:,i])
            alpha[4] = np.sum(np.square(C[:,i])) + 2*(A[:,i] @ E[:,i] + B[:,i] @ D[:,i])
            while True:
                F_prime = F[:,i] - p_last
                alpha[5] = 2*(A[:,i] @ F_prime + B[:,i] @ E[:,i] + C[:,i] @ D[:,i])
                alpha[6] = np.sum(np.square(D[:,i])) + 2*(B[:,i] @ F_prime + C[:,i] @ E[:,i])
                alpha[7] = 2*(D[:,i] @ E[:,i] + C[:,i] @ F_prime)
                alpha[8] = np.sum(np.square(E[:,i])) + 2*(D[:,i] @ F_prime)
                alpha[9] = 2*(E[:,i] @ F_prime)
                alpha[10]= np.sum(np.square(F_prime)) - delta_s**2
                alpha_reversed = alpha[::-1]
                # Create 10th-order polynomial
                g = Polynomial(alpha_reversed)
                g_prime = g.deriv()
                # solve root using newton-raphson's method
                u_next_guess = u + dl[i]
                f = lambda u: g(u)
                f_prime = lambda u: g_prime(u)
                sol = root_scalar(f=f, fprime=f_prime, x0=u_next_guess)
                if sol.converged:
                    u_next = sol.root
                else:
                    u_next = u_next_guess
                    print("Finding u_next did not converge")
                # if next guess is larger than segment length
                if u_next > self.l[i]:
                    # if it's the last segment, just use the entire segment length
                    if len(R_points) == N:
                        u_next = self.l[i]
                        p_next = self.interpolate(u_next, i)
                        R_points += [p_next]
                        break
                    # else, carry the remainder over to the next segment
                    else:
                        u = u_next - self.l[i] - dl[i]
                        break
                # append new interpolator point
                p_next = self.interpolate(u_next, i)
                R_points += [p_next]
                p_last = p_next
                u = u_next

        return np.array(R_points).T, L, delta_s
