# -*- coding: utf-8 -*-
"""
Lotka–Volterra parameter estimation with non-uniform sampling
Multiple-shooting  +  Gauss–Newton  (CasADi)
──────────────────────────────────────────────────────────────
• 支持测量点 N 很大，但射击节点数 M (=len(idx_nodes)) 远小于 N
• 仍然一次并行积分全部 N-1 步
Requires: CasADi ≥ 3.6, NumPy
"""

import casadi as ca
import numpy as np

# =============================================================================
# 1 SETTINGS
# =============================================================================
N  = 3_000                      # 测量点个数 (含 t = 0)
M  = 600                        # 射击节点个数  (M < N)
rng = np.random.default_rng(0)  # 随机种子

# irregular sampling: 30 – 60 ms between samples ------------------------------
dt_samples = rng.uniform(0.03, 0.06, N - 1)     # 长度 N-1
DT         = ca.DM(dt_samples)                 # CasADi 常量
ts         = np.insert(np.cumsum(dt_samples), 0, 0.0)   # 绝对时间戳 (仅参考)

print("First ten time stamps:", ts[:10])

# true & initial parameter values --------------------------------------------
param_truth  = ca.DM([0.5, 0.025, 0.8, 0.02])   # [alpha, beta, gamma, delta]
param_guess  = ca.DM([1.0, 0.01,  1.0, 0.03])

# =============================================================================
# 2 LOTKA–VOLTERRA ODE
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
    alpha * x - beta  * x * y,        # dx/dt
    delta * x * y - gamma * y         # dy/dt
)

ode = ca.Function("ode", [states, params], [rhs])

# =============================================================================
# 3 VARIABLE-STEP RK4 INTEGRATOR
# =============================================================================
dt = ca.MX.sym("dt")

k1 = ode(states, params)
k2 = ode(states + dt/2 * k1, params)
k3 = ode(states + dt/2 * k2, params)
k4 = ode(states + dt   * k3, params)

states_next = states + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
one_step_dt = ca.Function("one_step_dt", [states, params, dt], [states_next])

# =============================================================================
# 4 SYNTHETIC DATA WITH IRREGULAR Δt
# =============================================================================
x0_true = ca.DM([30, 4])                    # true initial populations

# integrate quickly using mapaccum -------------------------------------------
all_steps = one_step_dt.mapaccum("all_steps", N - 1)
P_const   = ca.repmat(param_truth, 1, N - 1)     # (4 × N-1)
DT_const  = ca.reshape(DT, 1, N - 1)             # (1 × N-1)

X_meas = ca.hcat([x0_true, all_steps(x0_true, P_const, DT_const)])   # 2 × N
Y_meas = np.asarray(X_meas.T)                                         # (N,2)

# add Gaussian measurement noise ---------------------------------------------
Y_meas[:, 0] += 0.1 * rng.standard_normal(N)
Y_meas[:, 1] += 0.1 * rng.standard_normal(N)

# =============================================================================
# 5 HELPER – GAUSS–NEWTON HESSIAN
# =============================================================================
if ca.Importer.has_plugin("clang"):
    with_jit = True; compiler = "clang"
elif ca.Importer.has_plugin("shell"):
    with_jit = True; compiler = "shell"
else:
    print("WARNING: running without JIT – may be slow")
    with_jit = False; compiler = ""

def gauss_newton(e, nlp, V):
    """Return an IPOPT solver with Gauss-Newton Hessian."""
    J = ca.jacobian(e, V)
    H = ca.triu(ca.mtimes(J.T, J))          # Gauss-Newton approximation
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
# 6 MULTIPLE-SHOOTING (M < N)
# =============================================================================
# choose M shooting nodes and build mapping arrays ----------------------------
idx_nodes = np.linspace(0, N-1, M, dtype=int)          # 射击节点在测量序列中的索引 (单调递增)
# 对于每一步 k ∈ [0, N-2]，找到它归属的射击段编号 s ∈ [0, M-1]
idx_steps = np.searchsorted(idx_nodes[1:], np.arange(N - 1), side="right")
IDX_STEPS_DM = ca.DM(idx_steps.tolist())               # (1 × N-1) CasADi 常量

# decision variables ----------------------------------------------------------
Xv = ca.MX.sym("X", 2, M)       # 仅 M 个射击节点上的状态
Pv = params                     # 4 待估参数
V  = ca.veccat(Pv, Xv)

# per-step prediction (fully parallel) ----------------------------------------
f_map   = one_step_dt.map(N - 1, "thread", 16)
Pmat    = ca.repmat(Pv, 1, N - 1)                      # (4 × N-1)
X0_steps = Xv[:, IDX_STEPS_DM]                         # (2 × N-1)

C   = ca.DM.zeros(IDX_STEPS_DM.numel())
cum = 0.0                                    # 当前段内累计

for i in range(IDX_STEPS_DM.numel()):
    cum = (cum + DT[i]) if (i and IDX_STEPS_DM[i]==IDX_STEPS_DM[i-1]) else DT[i]
    C[i] = cum

X_pred   = f_map(X0_steps, Pmat, C)             # (2 × N-1)

# continuity constraints (only at segment ends) -------------------------------
gap_list = []
for s in range(M - 1):
    k_end = int(idx_nodes[s + 1])                      # 下一节点在测量序列中的索引
    gap_list.append(X_pred[:, k_end - 1] - Xv[:, s + 1])
gaps = ca.hcat(gap_list)                               # (2 × (M-1))

# residuals (exclude k = 0) ----------------------------------------------------
e = ca.vec(ca.DM(Y_meas[1:, :]).T - X_pred)            # (2*(N-1),)

nlp = {"x": V,
       "f": 0.5 * ca.dot(e, e),
       "g": ca.vec(gaps)}

# =============================================================================
# 7 INITIAL GUESS
# =============================================================================
# 初始状态猜测：取测量值在射击节点时刻
X_guess = ca.DM(Y_meas[idx_nodes, :]).T     # (2 × M)
x0_vec  = ca.veccat(param_guess, X_guess)

# =============================================================================
# 8 SOLVE
# =============================================================================
solver = gauss_newton(e, nlp, V)
sol    = solver(x0=x0_vec, lbg=0, ubg=0)

est_params = np.array(sol["x"][:4]).ravel()
print("\nEstimated [alpha, beta, gamma, delta] =", est_params)
