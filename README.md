To do only the quintic spline interpolation:
```
knots = [
[0.01, 0.02, 0.04], # X-axis
[0.00, 0.02, 0.01]  # Y-axis
                    # Z-axis if needed
] 
q_spline = QuinticSplineInterpolation(knots)
p_interp, _, _ = q_spline.calc_interpolator_points(Ts=0.001, F=0.1, fs=0., fe=0.)
 ```

To contruct the path profile (this automatically calls the quintic spline interpolation):
```
path_profile = JerkLimitedProfile(knots, Ts=0.001, fs=0.0, F=0.1,
                                  fe=0.0, A=3., D=3., J=100.)
p_interp = path_profile.p_interp # interpolation knot points
t_interp = path_profile.t_interp # interpolation time
```

To interpolate at the servo loop frequencyL
```
recon = ServoRecon()
recon.add_trajectory(p_interp, t_interp)

# get time, positions, path distance, velocities, feedrate, accelerations, path acceleration, jerks, path jerks
t_sampling, p_sampling, s_sampling, v_sampling, sv_sampling, a_sampling, sa_sampling, j_sampling, sj_sampling = recon.get_sampling_reference(Ts=0.001)
```

To connect 2 or more trajectories:
```
knots2 = [...] # IMPORTANT: firtst point of second path is last point of first path
path_profile2 = JerkLimitedProfile(knots2, Ts, fs,
                                    F, fe, A, D, J)
p_interp2 = path_profile2.p_interp
t_interp2 = path_profile2.t_interp

recon.add_trajectory(p_interp2, t_interp2, offset_time=True)
t_sampling, p_sampling = recon.get_sampling_reference(Ts)[:2]
```
