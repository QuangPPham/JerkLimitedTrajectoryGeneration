from path_profile import JerkLimitedProfile
import numpy as np
import matplotlib.pyplot as plt

des_F = 0.1  # m/s
dTs = 0.0005  # s
fs = 0.0
fe = 0.0
A = 3.
D = 3.
J = 100.

# knots = np.array([
#     (0.,   0.),
#     (0.01, 0.005),
#     (0.02, 0.),
#     (0.03, 0.005),
#     [0.04, 0.01]
# ]).T

x = np.linspace(0., 0.01, 4)
knots = np.vstack((x, np.zeros_like(x)))

path_profile = JerkLimitedProfile(knots, dTs, fs, des_F,
                                  fe, A=A, D=D, J=J)

p_interp = path_profile.p_interp
ds = np.linalg.norm(p_interp[:,1:] - p_interp[:,:-1], axis=0)
s_interp = np.hstack((0., np.cumsum(ds)))

T_interp = path_profile.T_interp
t_interp = np.hstack((0., np.cumsum(T_interp)))

t_ana, s_ana = path_profile.t_ana, path_profile.s_ana

plt.plot(knots[0,:], knots[1,:], 'o-')
plt.plot(p_interp[0,:], p_interp[1,:], 'o-')
plt.axis("equal")

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
axs[0].set_title("X-Axis", fontsize="xx-large")
axs[0].set_ylabel("Distance (m)", fontsize="xx-large")
axs[1].set_title("Y-Axis", fontsize="xx-large")
axs[2].set_title("Resultant", fontsize="xx-large")
axs[0].plot(t_interp, p_interp[0,:], 'o-', c="tab:orange")
axs[1].plot(t_interp, p_interp[1,:], 'o-', c="tab:orange")
axs[2].plot(t_interp, s_interp, 'o-', c="tab:orange")
axs[2].plot(t_ana, s_ana, '-', c="tab:blue")
axs[0].set_xlabel("Time (s)", fontsize="xx-large")
axs[1].set_xlabel("Time (s)", fontsize="xx-large")
axs[2].set_xlabel("Time (s)", fontsize="xx-large")

plt.show()
