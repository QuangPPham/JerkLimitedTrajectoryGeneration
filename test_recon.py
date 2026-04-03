from path_profile import JerkLimitedProfile
from servo_reconstruction import ServoRecon
import numpy as np
import matplotlib.pyplot as plt

"""
Create reference motion
"""
# x1 = np.linspace(0.0, 0.05, 4)
# y1 = np.linspace(0.0, 0.02, 4)
# x2 = np.flip(x1)
# y2 = np.linspace(0.02, 0.04, 4)
# knots1 = np.array([x1, y1])
# knots2 = np.array([x2, y2])
# knots = np.hstack((knots1, knots2))

x = np.linspace(0., 0.01, 4)
knots = np.vstack((x, np.zeros_like(x)))

# x = np.linspace(-0.05, 0.05, 20)
# y = 0.05*np.cos(60*x)
# knots = np.array([x, y])

"""
Parameters
"""
des_F = 0.1  # m/s
dTs = 0.0001  # s
fs = 0.
fe = 0.
A = 3.
D = 3.
J = 150.

"""
Create jerk-limited path profiles
"""
path_profile1 = JerkLimitedProfile(knots, dTs, fs, des_F,
                                   fe, A=A, D=D, J=J)

# path_profile2 = JerkLimitedProfile(knots2, dTs, fs, des_F,
#                                    fe, A=A, D=D, J=J)

"""
Get values
"""
p_interp1 = path_profile1.p_interp # interpolation knot points
s_interp1 = path_profile1.s_interp # interpolation travel distance
t_interp1 = path_profile1.t_interp # interpolation time

# p_interp2 = path_profile2.p_interp # firtst point of second path is last point of first path
# s_interp2 = path_profile2.s_interp # + s_interp1[-1] # do not reset travel distance to 0
# t_interp2 = path_profile2.t_interp # + t_interp1[-1] # do not reset time to 0

# p_interp = np.hstack((p_interp1, p_interp2))
# t_interp = np.hstack((t_interp1, t_interp2)) # append time for plotting
# s_interp = np.hstack((s_interp1, s_interp2)) # append distance for plotting

"""
Reconstruction at servo-loop sampling rate
"""
recon = ServoRecon()
recon.add_trajectory(p_interp1, t_interp1)
# recon.add_trajectory(p_interp2, t_interp2, offset_time=True)

t_sampling, p_sampling, s_sampling, v_sampling, sv_sampling, a_sampling, sa_sampling, j_sampling, sj_sampling = recon.get_sampling_reference(dTs)
t_ana, s_ana, v_ana, a_ana, j_ana = path_profile1.t_ana, path_profile1.s_ana, path_profile1.v_ana, path_profile1.a_ana, path_profile1.j_ana

"""
Plotting trajectories
"""
# plt.figure(1)
# plt.plot(knots1[0,:], knots1[1,:], 'o-', c="b", ms=10)
# # plt.plot(knots2[0,:], knots2[1,:], 'o-', c="r", ms=10)
# plt.legend(["Path 1", "Path 2"], fontsize="xx-large")
# plt.axis("equal")

plt.figure(2)
# plt.plot(p_interp[0,:], p_interp[1,:], 'o-', c="r", ms=7.5)
plt.plot(knots[0,:], knots[1,:], 'o', c="b", ms=5)
plt.plot(p_sampling[0,:], p_sampling[1,:], 'o-', c="g", ms=4)
plt.axis("equal")
plt.legend(["Reference knots", "Sampling points"], fontsize="xx-large")

# distances
fig, axs = plt.subplots(4, 3, figsize=(20, 6))
axs[0,0].set_title("X-Axis", fontsize="x-large")
axs[0,0].set_ylabel("Distance (m)", fontsize="xx-large")
axs[0,1].set_title("Y-Axis", fontsize="xx-large")
axs[0,2].set_title("Resultant", fontsize="xx-large")
axs[0,0].plot(t_interp1, p_interp1[0,:], 'o', c="r", ms=5, label="Interpolation points")
axs[0,0].plot(t_sampling, p_sampling[0,:], c="g", lw=2, label="Sampling points")
# axs[0,1].plot(t_interp1, p_interp1[1,:], 'o-', c="b", ms=7.5, label="Interpolation points")
axs[0,1].plot(t_sampling, p_sampling[1,:], c="g", lw=2, label="Sampling points")
axs[0,2].plot(t_interp1, s_interp1, 'o', c="r", ms=5, label="Interpolation points")
axs[0,2].plot(t_sampling, s_sampling, c="g", lw=2, label="Sampling points")
# axs[0,2].plot(t_ana, s_ana, '-', lw=1, c="black", label="Analytical path")

# velocities
axs[1,0].set_ylabel("Velocity (m/s)", fontsize="x-large")
axs[1,0].plot(t_sampling, v_sampling[0,:], c="g", lw=2, label="Sampling points")
axs[1,1].plot(t_sampling, v_sampling[1,:], c="g", lw=2, label="Sampling points")
axs[1,2].plot(t_sampling, sv_sampling, c="g", lw=2, label="Sampling points")
# axs[1,2].plot(t_ana, v_ana, '-', lw=1, c="black", label="Analytical path")

# accelerations
idx_a = range(len(s_sampling)) # sa_sampling < max(A, D)*1.1 # get rid of fluctuations
axs[2,0].set_ylabel(r"Acceleration (m/$s^2$)", fontsize="x-large")
axs[2,0].plot(t_sampling[idx_a], a_sampling[0,idx_a], c="g", lw=2, label="Sampling points")
axs[2,1].plot(t_sampling[idx_a], a_sampling[1,idx_a], c="g", lw=2, label="Sampling points")
axs[2,2].plot(t_sampling[idx_a], sa_sampling[idx_a], c="g", lw=2, label="Sampling points")
# axs[2,2].plot(t_ana, np.abs(a_ana), '-', lw=1, c="black", label="Analytical path")

# jerks
idx_j = range(len(s_sampling)) # sj_sampling < J*1.1 # 
axs[3,0].set_ylabel(r"Jerk (m/$s^3$)", fontsize="xx-large")
axs[3,0].set_xlabel("Time (s)", fontsize="xx-large")
axs[3,1].set_xlabel("Time (s)", fontsize="xx-large")
axs[3,2].set_xlabel("Time (s)", fontsize="xx-large")
axs[3,0].plot(t_sampling[idx_j], j_sampling[0,idx_j], c="g", lw=2, label="Sampling points")
axs[3,1].plot(t_sampling[idx_j], j_sampling[1,idx_j], c="g", lw=2, label="Sampling points")
axs[3,2].plot(t_sampling[idx_j], sj_sampling[idx_j], c="g", lw=2, label="Sampling points")
# axs[3,2].plot(t_ana, np.abs(j_ana), '-', lw=1, c="black", label="Analytical path")

handles, labels = axs[0,2].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

plt.show()
