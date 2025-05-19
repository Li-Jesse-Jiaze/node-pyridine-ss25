from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from .parameter_estimator import ParameterEstimator
from .utils import timed


@dataclass
class ModelConfig:
    name: str
    build_integrator: Callable[[float], Tuple[cs.Function, dict]]
    true_p: List[float]
    x0: np.ndarray
    p_init: List[float]
    state_labels: List[str]


# Example systems
def lv_problem(dt: float):
    x1, x2 = cs.MX.sym("x1"), cs.MX.sym("x2")
    alpha, beta = cs.MX.sym("alpha"), cs.MX.sym("beta")
    rhs = cs.vertcat(alpha * x1 - beta * x1 * x2, 0.4 * x1 * x2 - 0.6 * x2)
    states = cs.vertcat(x1, x2)
    params = cs.vertcat(alpha, beta)
    ode = cs.Function("ode", [states, params], [rhs])
    dae = {"x": states, "p": params, "ode": rhs}
    integrator = cs.integrator("F", "cvodes", dae, 0.0, dt, {})
    return integrator, ode, states, params


def notorious_problem(dt: float, mu: float = 60.0):
    x1, x2, t = cs.MX.sym("x1"), cs.MX.sym("x2"), cs.MX.sym("t")
    p = cs.MX.sym("p")

    rhs = cs.vertcat(
        x2, 
        mu**2 * x1 - (mu**2 + p**2) * cs.sin(p * t), 
        1.0
    )
    X = cs.vertcat(x1, x2, t)

    ode = cs.Function("ode", [X, p], [rhs])

    dae = {"x": X, "p": p, "ode": rhs}
    F = cs.integrator("F", "cvodes", dae, 0.0, dt)
    return F, ode, X, p


def lorenz_problem(dt: float):
    x = cs.MX.sym("x")
    y = cs.MX.sym("y")
    z = cs.MX.sym("z")
    states = cs.vertcat(x, y, z)

    sigma = cs.MX.sym("sigma")
    rho = cs.MX.sym("rho")
    beta = cs.MX.sym("beta")
    params = cs.vertcat(sigma, rho, beta)

    rhs = cs.vertcat(
        sigma * (y - x), 
        x * (rho - z) - y, 
        x * y - beta * z
    )

    ode = cs.Function("ode_lorenz", [states, params], [rhs])

    dae = {"x": states, "p": params, "ode": rhs}
    integrator = cs.integrator("F_lorenz", "cvodes", dae, 0.0, dt, {})

    return integrator, ode, states, params


def pyridine_problem(dt: float):
    A, B, C, D, E, F, G = [cs.MX.sym(n) for n in "ABCDEFG"]
    p = cs.MX.sym("p", 11)
    rhs = cs.vertcat(
        -p[0] * A + p[8] * B,
        p[0] * A - p[1] * B - p[2] * B * C + p[6] * D - p[8] * B 
        + p[9] * D * F,
        p[1] * B - p[2] * B * C - 2 * p[3] * C * C - p[5] * C
        + p[7] * E + p[9] * D * F + 2 * p[10] * E * F,
        p[2] * B * C - p[4] * D - p[6] * D - p[9] * D * F,
        p[3] * C * C + p[4] * D - p[7] * E - p[10] * E * F,
        p[2] * B * C + p[3] * C * C + p[5] * C - p[9] * D * F 
        - p[10] * E * F,
        p[5] * C + p[6] * D + p[7] * E,
    )
    states = cs.vertcat(A, B, C, D, E, F, G)
    ode = cs.Function("ode", [states, p], [rhs])
    dae = {"x": states, "p": p, "ode": rhs}
    integrator = cs.integrator("F", "cvodes", dae, 0.0, dt, {})
    return integrator, ode, states, p


def generate_data(
    integrator: cs.Function,
    t_grid: np.ndarray,
    x0: np.ndarray,
    true_p: List[float],
    noise_std: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.zeros((len(t_grid), x0.size))
    data[0] = x0

    for k in range(len(t_grid) - 1):
        data[k + 1] = integrator(x0=data[k], p=true_p)["xf"].full().ravel()
    rng = np.random.default_rng(seed)
    meas = data + noise_std * rng.standard_normal(data.shape)
    return data, meas


@timed
def estimate(
    ode: cs.Function,
    states: cs.MX,
    params: cs.MX,
    t_grid: np.ndarray,
    meas: np.ndarray,
    p_init: List[float],
    num_shooting,
    strategy: str = "ipopt",
) -> np.ndarray:
    est = ParameterEstimator(
        ode,
        states,
        params,
        t_meas=t_grid,
        x_meas=meas,
        p_init=p_init,
        num_shooting=num_shooting,
        options={},
    )
    sol = est.solve(strategy)
    p = sol["x"][: len(p_init)].full().ravel()
    s = sol["x"][params.size1():].full().ravel().reshape(-1, states.size1())
    return p, s


def plot(
    t_grid: np.ndarray,
    meas: np.ndarray,
    est: np.ndarray,
    labels: List[str],
    true: Optional[np.ndarray] = None,
    show_every=1,
):
    idx = np.linspace(0, len(t_grid) - 1, len(est), dtype=int)
    _, ax = plt.subplots()
    for i, lbl in enumerate(labels):
        base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        if lbl == "t":
            continue
        if true is not None:
            ax.plot(t_grid, true[:, i], c=base_color, linestyle = "--", label=f"{lbl} true")
        ax.plot(t_grid[idx], est[:, i], c=base_color, label=f"{lbl} est")
        ax.scatter(t_grid[::show_every], meas[::show_every, i], s=8, alpha=0.4, c=base_color)
    ax.set(xlabel="time", ylabel="states", title="Measured vs Estimated")
    ax.legend(ncol=3, fontsize="small")
    plt.tight_layout()
    # plt.savefig("result.svg")
    # plt.show()
