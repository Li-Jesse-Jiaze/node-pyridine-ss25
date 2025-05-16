from dataclasses import dataclass
from typing import Callable, List, Tuple

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from .parameter_estimator import ParameterEstimator
from .utils import timed


@dataclass
class ModelConfig:
    name: str
    build_integrator: Callable[[float], Tuple[ca.Function, dict]]
    true_p: List[float]
    x0: np.ndarray
    p_init: List[float]
    state_labels: List[str]


# Example systems
def lv_problem(dt: float) -> Tuple[ca.Function, dict]:
    x1, x2 = ca.MX.sym("x1"), ca.MX.sym("x2")
    alpha, beta = ca.MX.sym("alpha"), ca.MX.sym("beta")
    rhs = ca.vertcat(alpha * x1 - beta * x1 * x2, 0.4 * x1 * x2 - 0.6 * x2)
    states = ca.vertcat(x1, x2)
    params = ca.vertcat(alpha, beta)
    ode = ca.Function("ode", [states, params], [rhs])
    dae = {"x": states, "p": params, "ode": rhs}
    integrator = ca.integrator("F", "cvodes", dae, 0.0, dt, {})
    return integrator, ode, states, params


def pyridine_problem(dt: float) -> Tuple[ca.Function, dict]:
    A, B, C, D, E, F, G = [ca.MX.sym(n) for n in "ABCDEFG"]
    p = ca.MX.sym("p", 11)
    rhs = ca.vertcat(
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
    states = ca.vertcat(A, B, C, D, E, F, G)
    ode = ca.Function("ode", [states, p], [rhs])
    dae = {"x": states, "p": p, "ode": rhs}
    integrator = ca.integrator("F", "cvodes", dae, 0.0, dt, {})
    return integrator, ode, states, p


def generate_data(
    integrator: ca.Function,
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
def estimate_p(
    ode: ca.Function,
    states: ca.MX,
    params: ca.MX,
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
    return sol["x"][: len(p_init)].full().ravel()


def simulate(
    F: ca.Function, t_grid: np.ndarray, x0: np.ndarray, p: List[float]
) -> np.ndarray:
    sim = np.zeros((len(t_grid), x0.size))
    sim[0] = x0
    for k in range(len(t_grid) - 1):
        sim[k + 1] = F(x0=sim[k], p=p)["xf"].full().ravel()
    return sim


def plot(
    t_grid: np.ndarray,
    # true: np.ndarray,
    meas: np.ndarray,
    est: np.ndarray,
    labels: List[str],
    show_every=1
):
    fig, ax = plt.subplots()
    for i, lbl in enumerate(labels):
        # ax.plot(t_grid, true[:, i], "--", label=f"{lbl} true")
        ax.plot(t_grid, est[:, i], label=f"{lbl} est")
        ax.scatter(t_grid[::show_every], meas[::show_every, i], s=8, alpha=0.4)
    ax.set(xlabel="time", ylabel="states", title="True vs Measured vs Estimated")
    ax.legend(ncol=3, fontsize="small")
    plt.tight_layout()
    plt.savefig("result.svg")
    # plt.show()
