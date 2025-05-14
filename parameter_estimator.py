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
            "ipopt": {f"ipopt.{k}": v for k, v in opts.get("ipopt", {"print_level": 0}).items()},
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
        w_sym = self.cnlls.prob["x"]
        f_sym = self.cnlls.prob["f"]
        g_sym = self.cnlls.prob["g"]
        w0    = self.cnlls.x0.copy()
        lbw   = self.cnlls.lbx
        ubw   = self.cnlls.ubx

        nvar  = w_sym.numel()
        ncons = g_sym.numel()

        N  = len(self.t_meas)
        nx = int(self.DAE["x"].size1())

        offset = 0
        X_blocks = []
        for k in range(N):
            Xk = w_sym[offset : offset+nx]
            X_blocks.append(Xk)
            offset += nx

        R_list = []
        for k in range(N):
            R_list.append(X_blocks[k] - self.x_meas[k])
        R_sym = ca.vertcat(*R_list)

        R_fun = ca.Function('R_fun', [w_sym],[R_sym])
        g_fun = ca.Function('g_fun', [w_sym],[g_sym])
        f_fun = ca.Function('f_fun', [w_sym],[f_sym])

        JR_sym = ca.jacobian(R_sym, w_sym)
        JG_sym = ca.jacobian(g_sym, w_sym)
        JR_fun = ca.Function('JR_fun',[w_sym],[JR_sym])
        JG_fun = ca.Function('JG_fun',[w_sym],[JG_sym])

        def solve_qp(H_, g_, A_, lbA_, ubA_, lbx_, ubx_):
            """
            min 0.5 dw^T H_ dw + g_^T dw
            s.t. A_*dw in [lbA_, ubA_],  dw in [lbx_, ubx_].
            """
            n_ = H_.shape[0]
            dw = ca.MX.sym("dw", n_, 1)

            obj = 0.5*ca.mtimes([dw.T, H_, dw]) + ca.dot(g_, dw)

            lhs = ca.mtimes(A_, dw) if A_.shape[0]>0 else ca.DM.zeros((0,1))

            qp_dict = {'x': dw, 'f': obj, 'g': lhs}
            solver = ca.qpsol("tmp_qp","qpoases", qp_dict, {"printLevel":"none"})
            sol = solver(lbg=lbA_, ubg=ubA_, lbx=lbx_, ubx=ubx_)
            return sol['x'].full().ravel()

        max_iter = self.options.get("max_iter", 15)
        tol      = self.options.get("tol", 1e-8)
        w = w0.copy()

        for it in range(max_iter):
            R_val = np.array(R_fun(w)).ravel()
            G_val = np.array(g_fun(w)).ravel() if ncons>0 else np.array([])
            f_val = float(f_fun(w))

            norm_R = np.linalg.norm(R_val,2)
            norm_G = np.linalg.norm(G_val,np.inf) if G_val.size>0 else 0.0

            if norm_R < tol and norm_G < tol:
                print(f"[gn] Converged at iter={it}, f={f_val:.3e}")
                break

            JR = np.array(JR_fun(w))
            H  = JR.T.dot(JR)
            g  = JR.T.dot(R_val)

            if ncons>0:
                JG = np.array(JG_fun(w))
                lbA_ = -G_val
                ubA_ = -G_val
                A_ = ca.DM(JG)
                lbA_dm = ca.DM(lbA_.reshape((-1,1)))
                ubA_dm = ca.DM(ubA_.reshape((-1,1)))
            else:
                A_ = ca.DM.zeros((0,nvar))
                lbA_dm = ca.DM.zeros((0,1))
                ubA_dm = ca.DM.zeros((0,1))

            lb_dw = lbw - w
            ub_dw = ubw - w
            lbx_dm = ca.DM(lb_dw.reshape((-1,1)))
            ubx_dm = ca.DM(ub_dw.reshape((-1,1)))

            H_ = ca.DM(H)
            g_ = ca.DM(g)
            if g_.shape == (nvar,):
                g_ = g_.reshape((nvar,1))

            dw = solve_qp(H_, g_, A_, lbA_dm, ubA_dm, lbx_dm, ubx_dm)
            if it > 0:
                desc_amount = np.dot(g, dw)  

                alpha     = 1.0
                beta      = 0.5
                sigma     = 1e-4
                alpha_min = 1e-4

                success = False
                
                while True:
                    w_try = w + alpha * dw
                    w_try = np.minimum(np.maximum(w_try, lbw), ubw)
                    f_try = float(f_fun(w_try))

                    lhs = f_try
                    rhs = f_val + sigma * alpha * desc_amount

                    if np.isfinite(lhs) and (lhs <= rhs):
                        w     = w_try
                        f_val = f_try
                        success = True
                        break
                    else:
                        alpha *= beta

                    if alpha < alpha_min:
                        print(f"[gn] no improvement at iter={it}, stop.")
                        break

                if not success:
                    break
            else:
                w = w + dw

        G_final = np.array(g_fun(w)).ravel() if ncons>0 else np.array([])
        sol = {
            "x": ca.DM(w),
            "f": float(f_fun(w)),
            "g": ca.DM(G_final.reshape((-1,1))) if G_final.size>0 else ca.DM.zeros((0,1)),
            "lam_g": ca.DM.zeros((G_final.size,1)),
            "lam_x": ca.DM.zeros((w.size,1))
        }
        return sol
