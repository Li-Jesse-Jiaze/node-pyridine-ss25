from typing import Callable, Dict, Sequence, Any, Optional

import casadi as ca
import numpy as np

class ParameterEstimator:
    def __init__(
        self,
        ode: ca.Function,
        states: ca.MX,
        params: ca.MX,
        t_meas: Sequence[float],
        x_meas: np.ndarray,
        num_shooting: Optional[int] = None,
        p_init: Optional[Sequence[float]] = None,
        residual: Callable[[ca.MX], ca.MX] = lambda e: 0.5 * ca.dot(e, e),
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the estimator.

        Args:
            DAE: dictionary with keys 'x', 'p', and 'ode'.
            t_meas: measurement time points.
            x_meas: measured state values (N x nx array).
            num_shooting: number of shooting intervals.
            p_init: initial guess for parameters.
            p_lb: lower bounds for parameters.
            p_ub: upper bounds for parameters.
            residual: function to compute residual from error vector.
            options: dict of solver options.
        """
        if len(t_meas) != x_meas.shape[0]:
            raise ValueError("t_meas and x_meas have to be the same size.")

        self.ode = ode
        self.states = states
        self.params = params
        self.t_meas = list(t_meas)
        self.x_meas = x_meas
        self.N = len(self.t_meas)  # number of measurement instants

        # Default to one shooting node per measurement unless user overrides
        self.num_shooting = self.N if num_shooting is None else int(num_shooting)

        if self.num_shooting > self.N:
            print(
                f"\033[33mWARNING: num_shooting={self.num_shooting} > #measurements={self.N};"
                "falling back to one node per measurement.\033[0m"
            )
            self.num_shooting = self.N

        self.less_node = self.num_shooting < self.N
        if self.less_node:
            print(
                f"\033[33mWARNING: num_shooting={self.num_shooting} < #measurements={self.N}; "
                "estimation accuracy may degrade.\033[0m"
            )
        self.residual = residual

        # Merge user options with defaults
        opts = options or {}
        self.options = {
            "ipopt": {
                f"ipopt.{k}": v
                for k, v in opts.get("ipopt", {"print_level": 0}).items()
            },
            "gn": opts.get("gn", {}),
        }
        # Model dimensions
        self.n_x = int(states.size1())
        self.n_p = int(params.size1())
        # Parameter initial guess
        self.p_init = np.zeros(self.n_p) if p_init is None else np.array(p_init)

        # JIT compilation backend (greatly speeds up Jacobians/Hessians)
        if ca.Importer.has_plugin("clang"):
            self.with_jit = True
            self.compiler = "clang"
        elif ca.Importer.has_plugin("shell"):
            self.with_jit = True
            self.compiler = "shell"
        else:
            print("WARNING: running without JIT, might be slow")
            self.with_jit = False
            self.compiler = ""

        # Placeholders (will be filled by _build_cnlls)
        self.x0 = None
        self.cnlls = {"f": None, "x": None, "g": None}  # with g ≡ 0

        self._build_cnlls()

    def _build_cnlls(self) -> None:
        """Build the CNLLS using direct multiple-shooting.

        Simplifying assumptions:
            • g ≡ 0
            • r_2, r_3 ≡ 0
        """
        # Pre‑compute the time increments between successive measurements
        dt_meas = np.diff(self.t_meas).astype(float) # N‑1
        DT = ca.DM(dt_meas)

        # one‑step RK4 integrator
        dt = ca.MX.sym("dt")
        k1 = self.ode(self.states, self.params)
        k2 = self.ode(self.states + dt / 2 * k1, self.params)
        k3 = self.ode(self.states + dt / 2 * k2, self.params)
        k4 = self.ode(self.states + dt * k3, self.params)

        states_next = self.states + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        one_step_dt = ca.Function(
            "one_step_dt", [self.states, self.params, dt], [states_next]
        )

        # Decision variables
        Xv = ca.MX.sym("X", self.n_x, self.num_shooting)
        variables = ca.veccat(self.params, Xv)

        # Map the integrator in parallel over all *N‑1* sampling intervals
        f_map = one_step_dt.map(self.N - 1, "thread")
        Pmat = ca.repmat(self.params, 1, self.N - 1)  # broadcast parameters along columns
        if self.less_node:
            # measurement index → shooting node index
            idx_nodes = np.linspace(0, self.N - 1, self.num_shooting, dtype=int)
            # maps each step (interval) to the *shooting node* it belongs to
            idx_steps = np.searchsorted(
                idx_nodes[1:], np.arange(self.N - 1), side="right"
            )
            IDX_STEPS_DM = ca.DM(idx_steps.tolist())
            # Gather the shooting states corresponding to each step
            X0_steps = Xv[:, IDX_STEPS_DM]
            # Compute *per‑step* integration horizons (could be >1 Δt if we skipped nodes)
            C = ca.DM.zeros(IDX_STEPS_DM.numel())
            cum = 0.0

            for i in range(IDX_STEPS_DM.numel()):
                cum = (
                    (cum + DT[i])
                    if (i and IDX_STEPS_DM[i] == IDX_STEPS_DM[i - 1])
                    else DT[i]
                )
                C[i] = cum

            # Predict the states at measurement points
            X_pred = f_map(X0_steps, Pmat, C)

            # Defect constraints (“gaps”): enforce continuity at shooting nodes
            gap_list = []
            for s in range(self.num_shooting - 1):
                k_end = int(idx_nodes[s + 1]) # last interval that ends at node s+1
                gap_list.append(X_pred[:, k_end - 1] - Xv[:, s + 1])
            gaps = ca.hcat(gap_list)
            X_guess = ca.DM(self.x_meas[idx_nodes, :]).T
        else:
            X_pred = f_map(
                Xv[:, :-1], Pmat, ca.reshape(DT, 1, self.N - 1)
            )  # (n_p × N-1)
            gaps = X_pred - Xv[:, 1:]
            X_guess = ca.DM(self.x_meas).T

        errors = ca.vec(ca.DM(self.x_meas[1:, :]).T - X_pred)

        self.errors = errors # flat 1D
        self.variables = variables # flat 1D

        self.cnlls = {"x": variables, "f": self.residual(errors), "g": ca.vec(gaps)}
        self.x0 = ca.veccat(self.p_init, X_guess)

    def solve(self, strategy: str = "gn_fast") -> Dict[str, Any]:
        if strategy == "gn_fast":
            return self._solve_gn_fast()
        elif strategy == "ipopt":
            return self._solve_ipopt()
        elif strategy == "gn":
            return self._solve_gn()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _solve_ipopt(self) -> Dict[str, Any]:
        solver = ca.nlpsol("solver", "ipopt", self.cnlls)
        sol = solver(x0=self.x0, lbg=0, ubg=0)
        return sol

    def _solve_gn_fast(self) -> Dict[str, Any]:
        # Jacobian of residuals wrt *all* decision vars
        J = ca.jacobian(self.errors, self.variables)
        # Upper‑triangular part of JTJ
        H = ca.triu(ca.mtimes(J.T, J))
        sigma = ca.MX.sym("sigma")
        hessLag = ca.Function(
            "nlp_hess_l",
            {"x": self.variables, "lam_f": sigma, "hess_gamma_x_x": sigma * H},
            ["x", "p", "lam_f", "lam_g"],
            ["hess_gamma_x_x"],
            dict(jit=self.with_jit, compiler=self.compiler),
        )
        solver = ca.nlpsol(
            "solver", "ipopt", self.cnlls, dict(hess_lag=hessLag, jit=self.with_jit, compiler=self.compiler)
        )
        return solver(x0=self.x0, lbg=0, ubg=0)

    def _solve_gn(self) -> Dict[str, Any]:
        raise NotImplementedError("Plain Gauss‑Newton solver not implemented yet.")