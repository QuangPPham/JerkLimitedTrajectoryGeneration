import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import root, root_scalar
from quintic_spline import QuinticSplineInterpolation

class JerkLimitedProfile():
    def __init__(self, knot_points, Ts, fs, F, fe, A, D, J):
        assert J >= 0. and A >= 0. and D >= 0., "All parameters should be positive or zero"
        
        # To be computed - quantities for each travel phase
        self.T = np.zeros(7)                 # period
        self.Ns = np.zeros_like(self.T)      # number of interpolation steps
        self.Ls = np.zeros_like(self.T)      # Analytical distances
        self.Fs = np.zeros_like(self.T)      # feedate
        self.As = np.zeros_like(self.T)      # Feedrate
        self.Js = np.zeros_like(self.T)      # Jerk
        self.L_prime = np.zeros_like(self.T) # Numerical distances

        # Initialization
        self.q_spline = QuinticSplineInterpolation(knot_points)
        self.F = F
        self.fs = fs
        self.fe = fe
        self.Ts = Ts
        self.N = None
        self.calc_interpolator_points()
        self.A, self.D, self.J = A, D, J
        self.initialize()

        # Fill in start points
        self.As[1:3] = self.A
        self.As[5:7] = -self.D
        self.Js[0] = self.J1
        self.Js[2] = -self.J1
        self.Js[4] = -self.J2
        self.Js[6] = self.J2

        # Travel length quantization
        self.travel_lengths_quantization()

        # Calculate interpolation periods
        self.T_interp = []
        self.continuously_execute()

        ds_interp = np.linalg.norm(self.p_interp[:,1:] - self.p_interp[:,:-1], axis=0)
        self.s_interp = np.hstack((0., np.cumsum(ds_interp)))
        self.t_interp = np.hstack((0., np.cumsum(self.T_interp)))


    def calc_interpolator_points(self, N=None):
        """Call quintic spline interpolator
        """
        self.p_interp, self.L, self.delta_s = self.q_spline.calc_interpolator_points(self.Ts, self.F, self.fs, self.fe, N)
        self.N = self.p_interp.shape[1] - 1 # number of segments = number of points - 1


    def check_interpolator_points(self):
        """Check if interpolation has enough steps
        """
        if self.A != 0. and self.D != 0.:
            if self.N < 4:
                print("Not enough interpolator points, setting N = 4 ...")
                self.calc_interpolator_points(4)
        elif self.A == 0. and self.D == 0.:
            if self.N < 1:
                print("Not enough interpolator points, setting N = 1 ...")
                self.calc_interpolator_points(1)
        else:
            if self.N < 2:
                print("Not enough interpolator points, setting N = 2 ...")
                self.calc_interpolator_points(2)


    def initialize(self):
        """Initialize kinematic variables and 
        interpolation periods for each phase
        """
        # Acceleration and jerk conditions
        self.acc_and_jerk_conditions()

        # Travel length condition
        self.travel_length_condition()

        # Verify number of interpolator points
        self.check_interpolator_points()

        # Get jerked times
        T13 = self.A / self.J1 if self.J1 != 0 else 0.
        T57 = self.D / self.J2 if self.J2 != 0 else 0.
        self.T[0] = T13
        self.T[2] = T13
        self.T[4] = T57
        self.T[6] = T57


    def acc_and_jerk_conditions(self):
        """Acceleration and jerk conditions
        """
        A = np.sign(self.F - self.fs) * np.abs(self.A) # initial accel/decel
        D = np.sign(self.F - self.fe) * np.abs(self.D) # Final accel/decel

        if A != self.A:
            print(f"Modifying A from {self.A} to {A} ... ")
        if D != self.D:
            print(f"Modifying D from {self.D} to {D} ... ")

        self.A = A
        self.D = D

        # Jerk condition
        if self.A != 0 or self.D != 0:
            if self.A == 0:
                J = np.minimum(np.abs(self.D)/self.Ts, self.J)
            elif self.D == 0:
                J = np.minimum(np.abs(self.A)/self.Ts, self.J)
            else:
                J = np.min([self.J, np.abs(self.A)/self.Ts, np.abs(self.D)/self.Ts])
            
            if J != self.J:
                print(f"Modifying J from {self.J} to {J} ... ")

            self.J = J
        
        self.J1 = np.sign(self.A) * self.J # Initial jerk ( J1 = self.J1 = -J3)
        self.J2 = np.sign(self.D) * self.J # Final jerk   (-J5 = self.J2 =  J7)
        
        # Acceleration condition
        if self.A != 0:
            T2 = (self.F - self.fs) / self.A - self.A / self.J1
            if T2 >= 0:
                self.T[1] = T2
            else:
                self.T[1] = 0.
                print(f"Modifying A from {self.A} to ", end="")
                self.A = np.sign(self.A) * np.sqrt(self.J1 * (self.F - self.fs))
                print(self.A, end = " ... \n")

        # Deceleration condition
        if self.D != 0:
            T6 = (self.F - self.fe) / self.D - self.D / self.J2
            if T6 >= 0:
                self.T[5] = T6
            else:
                self.T[5] = 0.
                print(f"Modifying D from {self.D} to ", end="")
                self.D = np.sign(self.D) * np.sqrt(self.J2 * (self.F - self.fe))
                print(self.D, end = " ... \n")


    def travel_length_condition(self):
        """Travel length condition
        """
        yes_A = np.sign(self.A)**2
        yes_D = np.sign(self.D)**2
        alpha = yes_A / (2*self.A + (yes_A-1)) + yes_D / (2*self.D + (yes_D-1)) # so that if A or D are zero the corresponding term is zero
        beta =  yes_A * self.A / (2*self.J1 + (yes_A-1)) + yes_D * self.D / (2*self.J2 + (yes_D-1))
        gamma = yes_A * (self.A*self.fs / (2*self.J1 + (yes_A-1)) - self.fs**2 / (2*self.A + (yes_A-1))) \
              + yes_D * (self.D*self.fe / (2*self.J2 + (yes_D-1)) - self.fe**2 / (2*self.D + (yes_D-1))) \
              - self.L
        T4 = -(alpha*self.F**2 + beta*self.F + gamma) / self.F
        if T4 >= 0.:
            self.T[3] = T4
        else:
            self.T[3] = 0.
            F = (-beta + np.emath.sqrt(beta**2 - 4*alpha*gamma)) / (2*alpha)
            if not np.iscomplex(F):
                print(f"Modifying F from {self.F} to ", end="")
                self.F = F
                print(self.F, end = " ... \n")
            else:
                print("Setting fs and fe to 0.0 ...")
                self.fs = 0.
                self.fe = 0.
            # Reset the q_spline using new feedrates
            self.calc_interpolator_points()
            # Reinitialize
            self.initialize()


    def calc_feedrate(self):
        """Calculate the feedrate at 
            the START of each phase
        """
        self.Fs[0] = self.fs
        self.Fs[1] = self.Fs[0] + self.J1*self.T[0]**2 / 2
        self.Fs[2] = self.Fs[1] + self.A*self.T[1]
        self.Fs[3] = self.Fs[2] + self.A*self.T[2] - self.J1*self.T[2]**2 / 2
        self.Fs[4] = self.Fs[3]
        self.Fs[5] = self.Fs[4] - self.J2*self.T[4]**2 / 2
        self.Fs[6] = self.Fs[5] - self.D*self.T[5]
        # F_final = self.Fs[6] - self.D*self.T[6] + self.J2*self.T[6]**2 / 2

    
    def calc_travel_distance(self):
        """Calculate the travel distance 
            at the END of each phase
        """
        self.Ls[0] = self.Fs[0]*self.T[0] + self.J1*self.T[0]**3 / 6
        self.Ls[1] = self.Fs[1]*self.T[1] + self.A*self.T[1]**2 / 2
        self.Ls[2] = self.Fs[2]*self.T[2] + self.A*self.T[2]**2 / 2 - self.J1*self.T[2]**3 / 6
        self.Ls[3] = self.Fs[3]*self.T[3]
        self.Ls[4] = self.Fs[4]*self.T[4] - self.J2*self.T[4]**3 / 6
        self.Ls[5] = self.Fs[5]*self.T[5] - self.D*self.T[5]**2 / 2
        self.Ls[6] = self.Fs[6]*self.T[6] - self.D*self.T[6]**2 / 2 + self.J2*self.T[6]**3 / 6
        self.Ss = np.hstack((0.0, np.cumsum(self.Ls)))

    def travel_lengths_quantization(self):
        """Quantize the travel length into N segments
        """
        self.calc_feedrate()
        self.calc_travel_distance()

        # get analytical plot here before quantizing the path
        self.t_ana, self.s_ana, self.v_ana, self.a_ana, self.j_ana = self.get_plot(self.Ts)

        # Interpolation steps
        # Jerked acceleration
        self.Ns[0:7:2] = np.round(self.Ls[0:7:2] / self.delta_s)

        # Set to minimum of one
        zero_N = self.Ns[0:7:2] == 0
        nonzero_l = self.Ls[0:7:2] != 0.
        self.Ns[0:7:2][np.logical_and(zero_N, nonzero_l)] = 1

        N_acc = np.round(np.sum(self.Ls[:3]) / self.delta_s)
        N_dec = np.round(np.sum(self.Ls[4:]) / self.delta_s)
        # Jerkless acceleration
        self.Ns[1] = np.clip(N_acc - (self.Ns[0] + self.Ns[2]), a_min=0., a_max=None)
        self.Ns[5] = np.clip(N_dec - (self.Ns[4] + self.Ns[6]), a_min=0., a_max=None)
        # No acceleration
        self.Ns[3] = np.clip(self.N - (N_acc + N_dec), a_min=0., a_max=None)

        # modified N
        if np.sum(self.Ns) != self.N:
            self.N = np.sum(self.Ns)
            self.calc_interpolator_points(self.N)

        # quantized travel lengths
        self.L_prime = self.Ns * self.delta_s

        # adjust for new travel lengths
        self.quantization_adjustment()

        # Recalculate feedrate
        self.calc_feedrate()


    def quantization_adjustment(self):
        """Adjust feedrate and interpolation 
            time based on quantized lengths
        """
        def _func(x):
            A, T1, T3, D, T5, T7 = x
            fx = np.zeros_like(x, dtype=np.float32)
            # A, T1, T3
            fx[0] = self.fs*T1 + A*T1**2 / 6 - self.L_prime[0]
            if self.T[1] != 0:
                fx[1] = -A*T1**2 / 8 + A*T3**2 / 8 - self.fs*T1/2 - self.F*T3/2 + (self.F**2 - self.fs**2)/(2*A) - self.L_prime[1]
                fx[2] = self.F*T3 - A*T3**2 / 6 - self.L_prime[2]
            else:
                fx[1] = A*T1/2 + A*T3/2 + self.fs - self.F
                fx[2] = A*T3**2 / 3 + A*T1*T3 / 2 + self.fs*T3 - self.L_prime[2]

            # D, T5, T7
            fx[3] = self.F * T5 - D*T5**2 / 6 - self.L_prime[4]
            if self.T[5] != 0:
                fx[4] = D*T5**2 / 8 - D*T7**2 / 8 - self.F*T5/2 - self.fe*T7/2 + (self.F**2 - self.fe**2)/(2*D) - self.L_prime[5]
                fx[5] = self.fe*T7 + D*T7**2 / 6 - self.L_prime[6]
            else:
                fx[4] = D*T5/2 + D*T7/2 + self.fe - self.F
                fx[5] = -D*T7**2 / 3 - D*T5*T7 / 2 + self.F*T7 - self.L_prime[6]

            return fx
        
        def _jac(x):
            A, T1, T3, D, T5, T7 = x
            Jx = np.zeros((6, 6), dtype=np.float32)

            # A, T1, T3
            Jx[0, 0] = T1**2 / 6
            Jx[0, 1] = self.fs + A*T1/3
            if self.T[1] != 0:
                Jx[1, 0] = -T1**2 / 8 + T3**2 / 8 - (self.F**2 - self.fs**2)/(2*A**2)
                Jx[1, 1] = -A*T1/4 - self.fs/2
                Jx[1, 2] = A*T3/4 - self.F/2
                Jx[2, 0] = -T3**2 / 6
                Jx[2, 2] = self.F - A*T3/3
            else:
                Jx[1, 0] = T1/2 + T3/2
                Jx[1, 1] = A/2
                Jx[1, 2] = A/2
                Jx[2, 0] = T3**2 / 3 + T1*T3/2
                Jx[2, 1] = A*T3/2
                Jx[2, 2] = 2*A*T3/3 + A*T1/2 + self.fs

            # D, T5, T7
            Jx[3, 3] = -T5**2 / 6
            Jx[3, 4] = self.F - D*T5/3
            if self.T[5] != 0:
                Jx[4, 3] = T5**2 / 8 - T7**2 / 8 - (self.F**2 - self.fe**2)/(2*D**2)
                Jx[4, 4] = D*T5/4 - self.F/2
                Jx[4, 5] = -D*T7/4 - self.fe/2
                Jx[5, 3] = T7**2 / 6
                Jx[5, 5] = self.fe + D*T7/3
            else:
                Jx[4, 3] = T5/2 + T7/2
                Jx[4, 4] = D/2
                Jx[4, 5] = D/2
                Jx[5, 3] = -T7**2 / 3 - T5*T7/2
                Jx[5, 4] = -D*T7/2
                Jx[5, 5] = -2*D*T7/3 - D*T5/2 + self.F

            return Jx
        
        # Solve using root finding
        x0 = np.array([self.A, self.T[0], self.T[2], self.D, self.T[4], self.T[6]])
        sol = root(fun=_func, jac=_jac, x0=x0)
        x_root = sol.x
        self.A, self.T[0], self.T[2], self.D, self.T[4], self.T[6] = x_root

        # Modify jerks
        self.J1 = self.A / self.T[0] if self.T[0] != 0. else 0.
        self.J2 = self.D / self.T[6] if self.T[6] != 0. else 0.


    def continuously_execute(self):
        current_n = 1
        for k in range(7):
            tau_last = 0.
            j0 = self.Js[k]
            a0 = self.As[k]
            f0 = self.Fs[k]
            # s0 = self.Ss[k]
            # for _ in range(int(self.Ns[k])):
            #     s_poly = Polynomial([s0-current_n*self.delta_s, f0, a0/2, j0/6])
            #     current_n += 1
            for n in range(int(self.Ns[k])):
                s_poly = Polynomial([-(n+1)*self.delta_s, f0, a0/2, j0/6])
                s_deriv = s_poly.deriv()
                # solve newton-raphson
                f  = lambda x: s_poly(x)
                df = lambda x: s_deriv(x)
                tau_guess = tau_last+self.Ts
                sol = root_scalar(f=f, fprime=df, x0=tau_guess)
                if sol.converged:
                    tau = sol.root
                else:
                    tau = tau_guess
                    print("Finding tau_next did not converge")
                # Append
                self.T_interp += [tau - tau_last]
                tau_last = tau

        # print(np.sum(self.Ls), np.sum(self.L_prime))
        self.T_interp = np.array(self.T_interp)


    def interpolate(self, t, i):
        j0 = self.Js[i]
        a0 = self.As[i]
        f0 = self.Fs[i]
        s0 = self.Ss[i]
        s = s0 + f0*t + a0*t**2 / 2 + j0*t**3 / 6
        f = f0 + a0*t + j0*t**2 / 2
        a = a0 + j0*t
        j = j0
        return s, f, a, j


    def get_plot(self, Ts):
        T_interp = self.T
        t_interp = np.hstack((0., np.cumsum(T_interp)))
        t_sampling = np.linspace(0., t_interp[-1], int((t_interp[-1])/Ts)+1)
        s_sampling = np.zeros_like(t_sampling)
        f_sampling = np.zeros_like(t_sampling)
        a_sampling = np.zeros_like(t_sampling)
        j_sampling = np.zeros_like(t_sampling)
        j = 0
        for i, t in enumerate(t_sampling):
            t_eval = t - t_interp[j]
            if t_eval > T_interp[j]:
                if j < len(T_interp)-1:
                    t_eval -= T_interp[j]
                    j+=1
            s_sampling[i], f_sampling[i], a_sampling[i], j_sampling[i] = self.interpolate(t_eval, j)

        return t_sampling, s_sampling, f_sampling, a_sampling, j_sampling
