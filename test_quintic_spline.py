from quintic_spline import QuinticSplineInterpolation
import numpy as np
import matplotlib.pyplot as plt

des_F = 0.1  # m/s
dTs = 0.005   # s
# knots = np.array([
#     (0.,   0.),
#     (0.01, 0.005),
#     (0.02, 0.),
#     (0.03, 0.005),
#     [0.04, 0.01]
# ]).T

# x = np.linspace(-0.05, 0.05, 20)
# y = 0.05*np.cos(60*x)
# knots = np.array([x, y])

x = np.linspace(0., 0.02, 10)
knots = np.vstack((x, np.zeros_like(x)))

# Test quintic spline
q_spline = QuinticSplineInterpolation(knots)
p_interp, _, _ = q_spline.calc_interpolator_points(dTs, des_F, 0., 0.)

# p_interp = np.zeros((2, 41))
# p_interp[:,0] = q_spline.p[:,0]
# dl = np.linalg.norm([1., 0.5]) / 10
# for i in range(4):
#     for j in range(1,11):
#         p_interp[:,i*10+j] = q_spline.interpolate(j*dl, i)

plt.plot(knots[0,:], knots[1,:], 'o-')
plt.plot(p_interp[0,:], p_interp[1,:], 'o-')
plt.axis("equal")
plt.legend(["Reference points", "Interpolation points"], fontsize="xx-large")

plt.show()
