import numpy as np
import dms_estimator as dmse

np.set_printoptions(suppress=True, precision=2, floatmode="fixed")

cfg = dmse.ModelConfig(
    name="Pyridine",
    build_integrator=dmse.pyridine_problem,
    true_p=[1.81, 0.89, 29.4, 9.21, 0.06, 2.43, 0.06, 5.5, 0.02, 0.5, 2.2],
    x0=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    p_init=[1.0] * 11,
    state_labels=list("ABCDEFG"),
)

num_meas = [50, 100, 500, 1000]
num_mode = [10, 20, 50, 100, 500, 1000]

for n_m in num_meas:
    for n_n in num_mode:
        if n_n > n_m:
            continue
        print(f"#Measurement: {n_m}, #node: {n_n}")
        dt = 5.0 / n_m
        t_grid = np.arange(0.0, 5.0, dt)

        integrator, ode, states, params = cfg.build_integrator(dt)
        X_true, X_meas = dmse.generate_data(
            integrator, t_grid, cfg.x0, cfg.true_p, noise_std=0.01
        )

        p_hat, _ = dmse.estimate(
            ode,
            states,
            params,
            t_grid,
            X_meas,
            cfg.p_init,
            num_shooting=n_n,
            strategy="gn_fast",
        )
        print(p_hat, end="\n\n")
