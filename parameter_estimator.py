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

        self.n_p = len(self.DAE["p"])
        self.p_init = np.zeros(self.n_p) if p_init is None else np.array(p_init)
        self.p_lb = -np.inf * np.ones(self.n_p) if p_lb is None else np.array(p_lb)
        self.p_ub = np.inf * np.ones(self.n_p) if p_ub is None else np.array(p_ub)

        self.cnlls = CNLLSProblem()
        self._build_cnlls()

    def _build_cnlls(self) -> None:
        # TODO: multiple shooting
        pass

    def solve(self, strategy: str = "ipopt") -> Dict[str, Any]:
        if strategy == "ipopt":
            return self._solve_ipopt()
        elif strategy == "gn":
            return self._solve_gn()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _solve_ipopt(self) -> Dict[str, Any]:
        solver = ca.nlpsol("solver", "ipopt", self.cnlls.prob, self.options["ipopt"])
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
