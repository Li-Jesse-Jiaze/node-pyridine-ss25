import argparse
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Tuple
from parameter_estimator import ParameterEstimator

@dataclass
class ModelConfig:
    name: str
    build_integrator: Callable[[float], Tuple[ca.Function, dict]]
    true_p: List[float]
    x0: np.ndarray
    p_init: List[float]
    p_lb: List[float]
    p_ub: List[float]
    state_labels: List[str]


def build_lv_integrator(dt: float) -> Tuple[ca.Function, dict]:
    x1, x2 = ca.MX.sym('x1'), ca.MX.sym('x2')
    alpha, beta = ca.MX.sym('alpha'), ca.MX.sym('beta')
    ode = ca.vertcat(
        alpha*x1 - beta*x1*x2,
        0.4*x1*x2 - 0.6*x2
    )
    dae = {'x': ca.vertcat(x1, x2), 'p': ca.vertcat(alpha, beta), 'ode': ode}
    F = ca.integrator('F', 'cvodes', dae, 0.0, dt, {})
    return F, dae


def build_pyridine_integrator(dt: float) -> Tuple[ca.Function, dict]:
    A, B, C, D, E, Fm, G = [ca.MX.sym(n) for n in 'ABCDEFG']
    p = ca.MX.sym('p', 11)
    ode = ca.vertcat(
        -p[0]*A + p[8]*B,
        p[0]*A - p[1]*B - p[2]*B*C + p[6]*D - p[8]*B + p[9]*D*Fm,
        p[1]*B - p[2]*B*C - 2*p[3]*C*C - p[5]*C + p[7]*E + p[9]*D*Fm + 2*p[10]*E*Fm,
        p[2]*B*C - p[4]*D - p[6]*D - p[9]*D*Fm,
        p[3]*C*C + p[4]*D - p[7]*E - p[10]*E*Fm,
        p[2]*B*C + p[3]*C*C + p[5]*C - p[9]*D*Fm - p[10]*E*Fm,
        p[5]*C + p[6]*D + p[7]*E
    )
    dae = {'x': ca.vertcat(A, B, C, D, E, Fm, G), 'p': p, 'ode': ode}
    F = ca.integrator('F', 'cvodes', dae, 0.0, dt, {})
    return F, dae

# Model configurations
MODELS = {
    'lv': ModelConfig(
        name='Lotka–Volterra',
        build_integrator=build_lv_integrator,
        true_p=[0.8, 0.3],
        x0=np.array([10.0, 5.0]),
        p_init=[0.5, 0.5],
        p_lb=[0.0, 0.0],
        p_ub=[2.0, 2.0],
        state_labels=['x1', 'x2']
    ),
    'pyridine': ModelConfig(
        name='Pyridine Network',
        build_integrator=build_pyridine_integrator,
        true_p=[0.5, 0.4, 0.3, 0.2, 0.1,
                0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
        x0=np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.0, 0.0]),
        p_init=[0.4]*11,
        p_lb=[0.0]*11,
        p_ub=[2.0]*11,
        state_labels=list('ABCDEFG')
    )
}


def generate_data(F: ca.Function, t_grid: np.ndarray, x0: np.ndarray,
                  true_p: List[float], noise_std: float = 0.01,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    data = np.zeros((len(t_grid), x0.size))
    data[0] = x0
    for k in range(len(t_grid)-1):
        data[k+1] = F(x0=data[k], p=true_p)['xf'].full().ravel()
    rng = np.random.default_rng(seed)
    meas = data + noise_std * rng.standard_normal(data.shape)
    return data, meas


def estimate_p(dae: dict, t_grid: np.ndarray, meas: np.ndarray,
               p_init: List[float], p_lb: List[float], p_ub: List[float],
               ipopt_opts: dict = None) -> np.ndarray:
    est = ParameterEstimator(
        dae, t_meas=t_grid, x_meas=meas,
        p_init=p_init, p_lb=p_lb, p_ub=p_ub,
        options={'ipopt': ipopt_opts or {}}
    )
    sol = est.solve()
    return sol['x'][-len(p_init):].full().ravel()


def simulate(F: ca.Function, t_grid: np.ndarray, x0: np.ndarray,
             p: List[float]) -> np.ndarray:
    sim = np.zeros((len(t_grid), x0.size))
    sim[0] = x0
    for k in range(len(t_grid)-1):
        sim[k+1] = F(x0=sim[k], p=p)['xf'].full().ravel()
    return sim


def plot(t_grid: np.ndarray, true: np.ndarray, meas: np.ndarray,
         est: np.ndarray, labels: List[str]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, lbl in enumerate(labels):
        ax.plot(t_grid, true[:, i], label=f'{lbl} true')
        ax.plot(t_grid, est[:, i], '--', label=f'{lbl} est')
        ax.scatter(t_grid, meas[:, i], s=8, alpha=0.4,
                   label=f'{lbl} meas' if i == 0 else None)
    ax.set(xlabel='time', ylabel='states',
           title='True vs Measured vs Estimated')
    ax.legend(ncol=3, fontsize='small')
    plt.tight_layout()
    plt.savefig("result.svg")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS.keys(), default='lv',
                        help='lv or pyridine')
    args = parser.parse_args()

    cfg = MODELS[args.model]
    dt = 0.1
    t_grid = np.arange(0, 10 + 1e-9, dt)

    F, dae = cfg.build_integrator(dt)
    X_true, X_meas = generate_data(F, t_grid, cfg.x0, cfg.true_p)
    p_hat = estimate_p(dae, t_grid, X_meas,
                       cfg.p_init, cfg.p_lb, cfg.p_ub,
                       ipopt_opts={'print_level': 0})
    print(f"[{cfg.name}] Estimated parameters: {np.round(p_hat, 4)}")

    X_est = simulate(F, t_grid, cfg.x0, p_hat)
    plot(t_grid, X_true, X_meas, X_est, cfg.state_labels)

if __name__ == '__main__':
    main()