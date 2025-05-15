# -*- coding: utf-8 -*-
"""
Lotka-Volterra parameter estimation with **non-uniform sampling** (multiple-shooting + Gauss–Newton)
───────────────────────────────────────────────────────────────────────────────────────────────────
Requires: CasADi >= 3.6, NumPy
"""

import casadi as ca
import numpy as np

# =============================================================================
# 1  SETTINGS
# =============================================================================
N  = 3_000          # number of samples  (includes t = 0)
rng = np.random.default_rng(0)

# irregular sampling: 30–60 ms between samples
dt_samples = rng.uniform(0.03, 0.06, N - 1)   # length N-1
DT         = ca.DM(dt_samples)                # constant vector for CasADi
ts         = np.insert(np.cumsum(dt_samples), 0, 0.0)   # absolute time stamps (for reference)

print(ts)

# true & initial parameter values
param_truth  = ca.DM([0.5, 0.025, 0.8, 0.02])   # [alpha, beta, gamma, delta]
param_guess  = ca.DM([1.0, 0.01,  1.0, 0.03])

# =============================================================================
# 2  LOTKA–VOLTERRA ODE
# =============================================================================
x   = ca.MX.sym("x")
y   = ca.MX.sym("y")
states = ca.vertcat(x, y)

alpha = ca.MX.sym("alpha")
beta  = ca.MX.sym("beta")
gamma = ca.MX.sym("gamma")
delta = ca.MX.sym("delta")
params = ca.vertcat(alpha, beta, gamma, delta)

rhs = ca.vertcat(
    alpha * x - beta  * x * y,    # dx/dt
    delta * x * y - gamma * y     # dy/dt
)

ode = ca.Function("ode", [states, params], [rhs])

# =============================================================================
# 3  VARIABLE-STEP RK4 INTEGRATOR
# =============================================================================
dt = ca.MX.sym("dt")

k1 = ode(states, params)
k2 = ode(states + dt/2 * k1, params)
k3 = ode(states + dt/2 * k2, params)
k4 = ode(states + dt   * k3, params)

states_next = states + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
one_step_dt = ca.Function("one_step_dt", [states, params, dt], [states_next])

# =============================================================================
# 4  SYNTHETIC DATA WITH IRREGULAR Δt
# =============================================================================
x0_true = ca.DM([30, 4])       # initial populations

# -- use mapaccum to integrate quickly ---------------------------------------
all_steps = one_step_dt.mapaccum("all_steps", N - 1)
P_const   = ca.repmat(param_truth, 1, N - 1)     # (4 × N-1)
DT_const  = ca.reshape(DT, 1, N - 1)             # (1 × N-1)

X_meas = ca.hcat([x0_true, all_steps(x0_true, P_const, DT_const)])   # 2 × N
Y_meas = np.asarray(X_meas.T)                                         # (N,2)

# add Gaussian measurement noise
Y_meas[:, 0] += 0.1 * rng.standard_normal(N)
Y_meas[:, 1] += 0.1 * rng.standard_normal(N)

# =============================================================================
# 5  HELPER – GAUSS–NEWTON HESSIAN
# =============================================================================
if ca.Importer.has_plugin("clang"):
    with_jit = True; compiler = "clang"
elif ca.Importer.has_plugin("shell"):
    with_jit = True; compiler = "shell"
else:
    print("WARNING: running without JIT – may be slow")
    with_jit = False; compiler = ""

def gauss_newton(e, nlp, V):
    J = ca.jacobian(e, V)
    H = ca.triu(ca.mtimes(J.T, J))
    sigma = ca.MX.sym("sigma")
    hessLag = ca.Function(
        "nlp_hess_l",
        {"x": V, "lam_f": sigma, "hess_gamma_x_x": sigma * H},
        ["x", "p", "lam_f", "lam_g"],
        ["hess_gamma_x_x"],
        dict(jit=with_jit, compiler=compiler),
    )
    return ca.nlpsol(
        "solver", "ipopt", nlp, dict(hess_lag=hessLag, jit=with_jit, compiler=compiler)
    )

# =============================================================================
# 6  MULTIPLE-SHOOTING NLP (UNEVEN Δt)
# =============================================================================
Xv = ca.MX.sym("X", 2, N)          # states at all samples (decision variables)
Pv = params                        # 4 parameters
V  = ca.veccat(Pv, Xv)

# --- per-segment prediction --------------------------------------------------
f_map = one_step_dt.map(N - 1, "thread", 16)
Pmat  = ca.repmat(Pv, 1, N - 1)    # (4 × N-1)  shared parameters
X_pred = f_map(Xv[:, :-1], Pmat, DT_const)   # (2 × N-1)

# --- continuity constraints --------------------------------------------------
gaps = X_pred - Xv[:, 1:]          # enforce X_{k+1}^{var} = RK4(X_k^{var}, …)

# --- residuals --------------------------------------------------------------
e = ca.vec(ca.DM(Y_meas[1:, :]).T - X_pred)   # exclude k=0 (perfectly known)

nlp = {"x": V,
       "f": 0.5 * ca.dot(e, e),
       "g": ca.vec(gaps)}

# =============================================================================
# 7  INITIAL GUESS
# =============================================================================
# use noisy measurements as state guess (could smooth/filter if desired)
X_guess = ca.DM(Y_meas).T
x0_vec  = ca.veccat(param_guess, X_guess)

# =============================================================================
# 8  SOLVE
# =============================================================================
solver = gauss_newton(e, nlp, V)
sol    = solver(x0=x0_vec, lbg=0, ubg=0)

est_params = np.array(sol["x"][:4]).ravel()
print("Estimated [alpha, beta, gamma, delta] = ", est_params)
