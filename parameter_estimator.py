import casadi as ca
import numpy as np
from typing import Callable, Dict, Sequence, Any, Optional
from dataclasses import dataclass


@dataclass
class CNLLSProblem:
    # {'f': ..., 'x': ..., 'g': ...}
    prob: Dict[str, ca.MX]

    x0: np.ndarray
    lbx: np.ndarray
    ubx: np.ndarray

    lbg: np.ndarray
    ubg: np.ndarray


class ParameterEstimator:
    def __init__(
        self,
        DAE: Dict[str, ca.MX],
        t_meas: Sequence[float],
        x_meas: np.ndarray,
        num_shooting: Optional[int] = None,
        p_init: Optional[Sequence[float]] = None,
        p_lb: Optional[Sequence[float]] = None,
        p_ub: Optional[Sequence[float]] = None,
        residual: Callable[[ca.MX], ca.MX] = ca.sumsqr,
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
        if {"x", "p", "ode"} - set(DAE):
            raise ValueError("DAE dictionary must contain keys 'x', 'p', and 'ode'.")
        if len(t_meas) != x_meas.shape[0]:
            raise ValueError("t_meas and x_meas must match in length.")

        self.DAE = DAE
        self.t_meas = list(t_meas)
        self.x_meas = x_meas
        self.num_shooting = num_shooting or (len(self.t_meas) - 1)
        self.residual = residual
        # Merge user options with defaults
        opts = options or {}
        self.options = {
            "integrator": opts.get("integrator", {}),
            "ipopt": opts.get("ipopt", {"print_level": 0}),
            "gn": opts.get("gn", {}),
        }

        self.n_p = int(DAE["p"].size1())
        self.p_init = np.zeros(self.n_p) if p_init is None else np.array(p_init)
        self.p_lb = -np.inf * np.ones(self.n_p) if p_lb is None else np.array(p_lb)
        self.p_ub = np.inf * np.ones(self.n_p) if p_ub is None else np.array(p_ub)

        self.cnlls = None
        self._build_cnlls()

    def _build_cnlls(self) -> None:
        """Build the CNLLS using direct multiple-shooting.

        Simplifying assumptions:
            • g ≡ 0
            • r_2, r_3 ≡ 0
            • m = N - 1
        """
        # -- 1. basic dimensions --------------------------------------------
        nx = int(self.DAE["x"].size1())
        N = len(self.t_meas)
        dt = np.diff(self.t_meas).astype(float)

        # -- 2. integrator --------------------------------------------------
        integrators = []
        for k, h in enumerate(dt):
            F_k = ca.integrator(
                f"F_seg_{k}",
                "cvodes",
                self.DAE,
                0.0,
                float(h),
                self.options.get("integrator", {}),
            )
            integrators.append(F_k)

        # -- 3. decision variables ------------------------------------------
        X = [ca.MX.sym(f"X_{k}", nx) for k in range(N)]
        P = ca.MX.sym("P", self.n_p)
        w = ca.vertcat(*(X + [P]))

        # -- 4. continuity constraints --------------------------------------
        defects = []
        for k in range(N - 1):
            x_end = integrators[k](x0=X[k], p=P)["xf"]
            defects.append(X[k + 1] - x_end)
        g = ca.vertcat(*defects) if defects else ca.MX()

        # -- 5. residual vector & objective ---------------------------------
        R = ca.vertcat(*[X[k] - self.x_meas[k] for k in range(N)])
        J = self.residual(R)

        # -- 6. initial guess & variable bounds -----------------------------
        x0 = np.concatenate([self.x_meas[k] for k in range(N)] + [self.p_init])
        lbx = np.concatenate([np.full(nx, -np.inf) for _ in range(N)] + [self.p_lb])
        ubx = np.concatenate([np.full(nx, np.inf) for _ in range(N)] + [self.p_ub])

        lbg = np.zeros(g.shape) if g.numel() else np.array([])
        ubg = np.zeros_like(lbg)

        # -- 7. pack CNLLS --------------------------------------------------
        self.cnlls = CNLLSProblem(
            prob={"f": J, "x": w, "g": g},
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
        )

    def solve(self, strategy: str = "ipopt") -> Dict[str, Any]:
        if strategy == "ipopt":
            return self._solve_ipopt()
        elif strategy == "gn":
            return self._solve_gn()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _solve_ipopt(self) -> Dict[str, Any]:
        solver = ca.nlpsol("solver", "ipopt", self.cnlls.prob, {f"ipopt.{k}": v for k, v in self.options.get("ipopt", {}).items()})
        sol = solver(
            x0=self.cnlls.x0,
            lbx=self.cnlls.lbx,
            ubx=self.cnlls.ubx,
            lbg=self.cnlls.lbg,
            ubg=self.cnlls.ubg,
        )
        # TODO: may be necessary to split p from other
        return sol

    def _solve_gn(self) -> Dict[str, Any]:
        # TODO: generalized Gauß-Newton method
        pass
