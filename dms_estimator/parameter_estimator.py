"""
Parameter estimation via constrained nonlinear least–squares (CNLLS)
using direct multiple-shooting and CasADi.

Example:
>>> ode  = cs.Function('f', [x, p], [rhs])
>>> est  = ParameterEstimator(ode, x, p, t_meas, x_meas)
>>> sol  = est.solve('gn_fast')
"""
from typing import Callable, Dict, Sequence, Any, Optional
import casadi as cs
import numpy as np
# # If do visualization
# import csnlp
# from matplotlib import pyplot as plt

from .utils import silence, printyellow, printgreen, timed


class ParameterEstimator:
    """CNLLS parameter estimation via direct multiple‑shooting.

    Parameters
    ----------
    ode
        Continuous‑time right‑hand side *f(x, p) -> xdot*.
    states, params
        CasADi *MX* placeholders holding the symbols for **all** states / parameters.
    t_meas, x_meas
        Measurement time stamps and measured states (shape = *(N, n_x)*).
    num_shooting
        Number of shooting nodes; defaults to one per measurement.
    p_init
        Initial guess for parameters *(n_p,)*.  If *None*, zeros are used.
    residual
        Mapping from the concatenated error vector to the scalar cost.
        Defaults to ½‖e‖² (i.e. Gauss–Newton).
    options
        Misc. solver options, e.g. *{"max_iter": 30, "tol": 1e‑10}*.
    """

    @staticmethod
    def _select_jit() -> tuple[bool, str]:
        """Return *(with_jit, compiler_name)* depending on what CasADi finds."""
        if cs.Importer.has_plugin("clang"):
            return True, "clang"
        if cs.Importer.has_plugin("shell"):
            return True, "shell"
        printyellow("Running **without** CasADi JIT; may be slow.")
        return False, ""

    @staticmethod
    def _has_hsl() -> bool:
        """Whether HSL sparse linear solvers are available (e.g. *ma27*)."""
        if cs.Linsol.has_plugin("ma27"):
            return True
        printyellow("HSL sparse linear solvers unavailable; falling back to built‑ins.")
        return False

    def __init__(
        self,
        ode: cs.Function,
        states: cs.MX,
        params: cs.MX,
        t_meas: Sequence[float],
        x_meas: np.ndarray,
        num_shooting: Optional[int] = None,
        p_init: Optional[Sequence[float]] = None,
        residual: Callable[[cs.MX], cs.MX] = lambda e: 0.5 * cs.dot(e, e),
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(t_meas) != x_meas.shape[0]:
            raise ValueError("t_meas and x_meas have to be the same size.")

        self.ode = ode
        self.states, self.params = states, params
        self.t_meas = list(t_meas)
        self.x_meas = x_meas
        self.N = len(self.t_meas)  # number of measurement instants


        # Number of shooting nodes
        self.num_shooting = self.N if num_shooting is None else int(num_shooting)
        if self.num_shooting == self.N:
            printgreen(
                f"Using the Default: #node=#meas={self.N};"
            )
        if self.num_shooting > self.N:
            printyellow(
                f"WARNING: #node={self.num_shooting} > #meas={self.N};"
                "falling back to one node per measurement."
            )
            self.num_shooting = self.N
        self.less_node = self.num_shooting < self.N
        if self.less_node:
            printyellow(
                f"WARNING: #node={self.num_shooting} < #meas={self.N}; "
                "estimation accuracy may degrade."
            )
        
        # Residual/cos
        self.residual = residual
        # Options
        self.options = options or {}
        # Model dimensions
        self.n_x = int(states.size1())
        self.n_p = int(params.size1())

        # Parameter initial guess
        self.p_init = np.zeros(self.n_p) if p_init is None else np.array(p_init)

        # CasADi backend features
        self.with_jit, self.compiler = self._select_jit()
        self.schur = self._has_hsl()
        # Placeholders (will be filled by _build_cnlls)
        self.x0 = None
        self.errors = None
        self.variables = None
        self.constrain = None # ≡ 0

        self._build_cnlls()

    def _build_cnlls(self) -> None:
        """Build the CNLLS using direct multiple-shooting.

        Simplifying assumptions:
            • g ≡ 0
            • r_2, r_3 ≡ 0
        """
        # Pre‑compute Δt between measurements
        dt_meas = np.diff(self.t_meas).astype(float)  # (N‑1, )
        DT = cs.DM(dt_meas)

        # One‑step RK4 integrator
        dt = cs.MX.sym("dt")
        k1 = self.ode(self.states, self.params)
        k2 = self.ode(self.states + dt / 2 * k1, self.params)
        k3 = self.ode(self.states + dt / 2 * k2, self.params)
        k4 = self.ode(self.states + dt * k3, self.params)

        states_next = self.states + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        one_step_dt = cs.Function(
            "one_step_dt", [self.states, self.params, dt], [states_next]
        )

        # Shooting nodes
        Xv = cs.MX.sym("X", self.n_x, self.num_shooting)

        # Map the integrator in parallel over all *N‑1* sampling intervals
        f_map = one_step_dt.map(self.N - 1, "thread")
        p_broadcast = cs.repmat(self.params, 1, self.N - 1)  # shared params
        if self.less_node:
            # measurement index -> closest shooting node index
            idx_nodes = np.linspace(0, self.N - 1, self.num_shooting, dtype=int)
            # maps each step (interval) to the *shooting node* it belongs to
            idx_steps = np.searchsorted(
                idx_nodes[1:], np.arange(self.N - 1), side="right"
            )
            IDX_STEPS_DM = cs.DM(idx_steps.tolist())
            # States at the **start** of each integration step
            X0_steps = Xv[:, IDX_STEPS_DM]
            # integration horizons
            C = cs.DM.zeros(IDX_STEPS_DM.numel())
            cum = 0.0
            for i in range(IDX_STEPS_DM.numel()):
                cum = (
                    (cum + DT[i])
                    if (i and IDX_STEPS_DM[i] == IDX_STEPS_DM[i - 1])
                    else DT[i]
                )
                C[i] = cum

            X_pred = f_map(X0_steps, p_broadcast, C) # states at measurements

            # Continuity constraints
            gap_list = []
            for s in range(self.num_shooting - 1):
                k_end = int(idx_nodes[s + 1])  # last interval that ends at node s+1
                gap_list.append(X_pred[:, k_end - 1] - Xv[:, s + 1])
            gaps = cs.hcat(gap_list)
            X_guess = cs.DM(self.x_meas[idx_nodes, :]).T # initial guess for Xv
        else:
            X_pred = f_map(
                Xv[:, :-1], p_broadcast, cs.reshape(DT, 1, self.N - 1)
            )  # (n_p × N-1)
            gaps = X_pred - Xv[:, 1:]
            X_guess = cs.DM(self.x_meas).T

        self.x0 = cs.veccat(self.p_init, X_guess)
        self.errors = cs.vec(cs.DM(self.x_meas[1:, :]).T - X_pred)
        self.variables = cs.veccat(self.params, Xv) # Decision variables
        self.constrain = cs.vec(gaps) # ≡ 0

    def solve(self, strategy: str = "gn_fast") -> Dict[str, Any]:
        """
        strategy: *"gn_fast"* (default), *"gn"* or *"ipopt"*.
        """
        _dispatch = {
            "gn_fast": self._solve_gn_fast,
            "gn": self._solve_gn,
            "ipopt": self._solve_ipopt,
        }
        try:
            return _dispatch[strategy]()
        except KeyError as exc:
            raise ValueError(f"Unknown strategy '{strategy}'.") from exc

    @silence
    def _solve_ipopt(self) -> Dict[str, Any]:
        """Full‑space IPOPT (exact Hessian or L-BFGS?)."""
        options = dict()
        # # Faster without
        # options["jit"] = self.with_jit
        # options["compiler"] = self.compiler
        # if self.schur:
        #     options["ipopt.linear_solver"] = "ma27"
        options["ipopt.print_level"] = 0
        solver = cs.nlpsol(
            "solver", "ipopt", 
            {
                "x": self.variables, 
                "f": self.residual(self.errors), 
                "g": self.constrain,
            }, 
            options,
        )
        sol = solver(x0=self.x0, lbg=0, ubg=0)
        return sol

    @silence
    def _solve_gn_fast(self) -> Dict[str, Any]:
        """IPOPT with user‑supplied Hessian callback"""
        # Jacobian of residuals wrt *all* decision vars
        J = cs.jacobian(self.errors, self.variables)
        # # Plot H
        # csnlp.util.plot.spy(cs.mtimes(J.T, J))
        # plt.savefig("H.svg")
        # Upper‑triangular part of JᵀJ
        H = cs.triu(cs.mtimes(J.T, J))
        sigma = cs.MX.sym("sigma") # IPOPT multiplies Hessian by *lam_f*
        hessLag = cs.Function(
            "nlp_hess_l",
            {"x": self.variables, "lam_f": sigma, "hess_gamma_x_x": sigma * H},
            ["x", "p", "lam_f", "lam_g"],
            ["hess_gamma_x_x"],
            dict(jit=self.with_jit, compiler=self.compiler),
        )
        options = {"hess_lag": hessLag, "jit": self.with_jit, "compiler": self.compiler}
        options["ipopt.print_level"] = 0
        # # Faster if not using Schur, why?
        # if self.schur:
        #     options["ipopt.linear_solver"] = "ma27"
        solver = cs.nlpsol(
            "solver",
            "ipopt",
            {
                "x": self.variables, 
                "f": self.residual(self.errors), 
                "g": self.constrain,
            },
            options,
        )
        return solver(x0=self.x0, lbg=0, ubg=0)

    def _solve_gn(self) -> Dict[str, Any]:
        """Pure Gauss–Newton with in‑house QP and back‑tracking line‑search."""
        w_sym = self.variables
        f_sym = self.residual(self.errors)
        g_sym = self.constrain
        nvar = w_sym.numel()
        ncons = g_sym.numel()

        # Functions for residuals, constraints, Jacobians
        R_sym = self.errors
        R_fun = cs.Function("R_fun", [w_sym], [R_sym])
        g_fun = cs.Function("g_fun", [w_sym], [g_sym])
        f_fun = cs.Function("f_fun", [w_sym], [f_sym])

        JR_sym = cs.jacobian(R_sym, w_sym)
        JG_sym = cs.jacobian(g_sym, w_sym)
        JR_fun = cs.Function("JR_fun", [w_sym], [JR_sym])
        JG_fun = cs.Function("JG_fun", [w_sym], [JG_sym])

        @silence
        def solve_qp(H_, g_, A_, lbA_, ubA_):
            """
            min 0.5 dw^T H_ dw + g_^T dw
            s.t. A_*dw in [lbA_, ubA_],  dw in [lbx_, ubx_].
            """
            n_ = H_.shape[0]
            dw = cs.MX.sym("dw", n_, 1)

            obj = 0.5 * cs.mtimes([dw.T, H_, dw]) + cs.dot(g_, dw)
            lhs = cs.mtimes(A_, dw) if A_.shape[0] else cs.DM.zeros((0, 1))

            qp_dict = {"x": dw, "f": obj, "g": lhs}
            solver = cs.qpsol("tmp_qp", "qpoases", qp_dict, 
                {
                    'printLevel': 'none', 
                    'sparse': True, 
                    'schur': self.schur, 
                    'hessian_type':'posdef', 
                    'numRefinementSteps': 0,
                }
            )
            sol = solver(lbg=lbA_, ubg=ubA_)
            return sol["x"].full().ravel()

        # Gauss–Newton loop
        max_iter = self.options.get("max_iter", 20)
        tol = self.options.get("tol", 1e-12)
        w = cs.DM(self.x0)

        last_norm_R = np.inf
        for it in range(max_iter):
            # Evaluate residuals / constraints
            R_val = np.array(R_fun(w)).ravel()
            G_val = np.array(g_fun(w)).ravel() if ncons > 0 else np.array([])
            f_val = float(f_fun(w))

            norm_R = np.linalg.norm(R_val, 2)
            # norm_G = np.linalg.norm(G_val, np.inf) if G_val.size > 0 else 0.0

            # Convergence test
            if (last_norm_R - norm_R) / norm_R < tol:
                print(f"[gn] Converged at iter={it}, f={f_val:.3e}")
                break

            # Build GN system  H = JᵀJ , g = Jᵀr
            JR = JR_fun(w)
            H = cs.mtimes(cs.transpose(JR), JR)
            g = cs.mtimes(cs.transpose(JR), R_val)

            # Assemble QP matrices
            if ncons > 0:
                A = JG_fun(w)
                lbA_dm = ubA_dm = cs.DM(-G_val.reshape((-1, 1)))
            else:
                A = cs.DM.zeros((0, nvar))
                lbA_dm = ubA_dm = cs.DM.zeros((0, 1))

            # Solve the QP for search direction
            dw = solve_qp(H, g, A, lbA_dm, ubA_dm)

            # Back-tracking line-search (Armijo)
            if it > 0:
                desc = np.dot(np.array(g).ravel(), dw)
                alpha, beta, sigma = 1.0, 0.5, 1e-6
                while alpha >= 1e-10:
                    w_try = cs.DM(w + alpha * dw)
                    if float(f_fun(w_try)) <= f_val + sigma * alpha * desc:
                        w = w_try
                        break
                    alpha *= beta
                else:
                    print(f"[gn] no improvement at iter={it}, stop.")
                    break
            else:
                w = w + dw

            last_norm_R = norm_R

        # # Plot A
        # csnlp.util.plot.spy(A)
        # plt.savefig("A.svg")
        # Pack solution
        G_final = np.array(g_fun(w)).ravel() if ncons > 0 else np.array([])
        sol = {
            "x": cs.DM(w),
            "f": float(f_fun(w)),
            "g": cs.DM(G_final.reshape((-1, 1)))
            if G_final.size > 0
            else cs.DM.zeros((0, 1)),
        }
        return sol
