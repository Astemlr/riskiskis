import numpy as np
from typing import List
import pandas as pd
from sklearn.linear_model import LinearRegression

# глобальный шаг: один бизнес-день
DT = 1 / 252

# ──────────────────────────────── CIR ────────────────────────────────
def simulate_cir(k: float, theta: float, sigma: float,
                 r0: float, n: int,
                 dt: float = DT,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Векторизированная симуляция CIR-процесса.
    Возвращает массив длиной n+1 (включая r0).

    dr_t = k(θ − r_t)dt + σ√{r_t} dW_t
    """
    rng = rng or np.random.default_rng()
    z = rng.standard_normal(n)
    r = np.empty(n + 1, dtype=float)
    r[0] = r0

    # векторный супер-быстрый шаг Эйлера
    for t in range(n):
        sqrt_rt = np.sqrt(max(r[t], 0.0))
        r[t + 1] = (r[t]
                    + k * (theta - r[t]) * dt
                    + sigma * sqrt_rt * np.sqrt(dt) * z[t])
        # гарантируем неотрицательность
        r[t + 1] = max(r[t + 1], 0.0)

    return r


def ols_cir(series: np.ndarray,
            dt: float = DT) -> tuple[float, float, float]:
    """
    Оценка параметров CIR регрессией Чена-Скотта.
    Требует series > 0 (ставки/доходности).
    """
    series = np.asarray(series)
    if len(series) < 5:
        raise ValueError("Слишком мало наблюдений для оценки CIR")
    if np.any(series <= 0):
        raise ValueError("CIR не допускает отрицательные или нулевые значения")

    rt = series[:-1]
    rt1 = series[1:]

    y = (rt1 - rt) / np.sqrt(rt)
    X = np.column_stack((dt / np.sqrt(rt), -dt * np.sqrt(rt)))

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    beta1, beta2 = model.coef_

    if np.isclose(beta2, 0) or np.isnan(beta2):
        raise ValueError(f"Оценка beta2 некорректна (beta2={beta2:.5f})")

    k_hat = -beta2
    theta_hat = beta1 / k_hat
    sigma_hat = y.std(ddof=0) / np.sqrt(dt)

    return k_hat, theta_hat, sigma_hat


# ──────────────────────────────── GBM ────────────────────────────────
def simulate_gbm(mu: float, sigma: float,
                 s0: float, n: int,
                 dt: float = DT,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Геометрическое броуновское движение (лог-нормальный процесс цен).
    Возвращает массив длиной n+1 (включая s0).

    dS_t / S_t = μ dt + σ dW_t
    """
    rng = rng or np.random.default_rng()
    z = rng.standard_normal(n)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * z

    log_increments = drift + diffusion
    log_path = np.concatenate(([0.0], np.cumsum(log_increments)))
    return s0 * np.exp(log_path)


def estimate_gbm_params(prices: np.ndarray,
                        dt: float = DT) -> tuple[float, float]:
    """
    OLS-оценки μ и σ для GBM на основе лог-доходностей.
    """
    log_ret = np.diff(np.log(prices))
    mu_hat = log_ret.mean() / dt + 0.5 * log_ret.var(ddof=1) / dt
    sigma_hat = log_ret.std(ddof=1) / np.sqrt(dt)
    return mu_hat, sigma_hat



def get_table_trajectories(df: pd.DataFrame,
                           cir_assets: List[str],
                           num_trajectories: int,
                           num_steps: int,
                           rng: np.random.Generator) -> List[pd.DataFrame]:
    """
    Симулирует траектории по CIR и GBM активам.
    Возвращает список из num_trajectories DataFrame'ов с колонками, соответствующими df.columns.
    """
    assert all(col in df.columns for col in cir_assets), "CIR assets not in dataframe"

    result = []
    asset_columns = df.columns
    asset_types = {col: ('CIR' if col in cir_assets else 'GBM') for col in asset_columns}

    # последние значения
    last_values = df.iloc[-1]

    # оценка параметров по истории
    cir_params = {}
    gbm_params = {}
    for col in asset_columns:
        x = df[col].dropna().values
        if asset_types[col] == 'CIR':
            cir_params[col] = ols_cir(x, dt=DT)
        else:
            gbm_params[col] = estimate_gbm_params(x, dt=DT)

    # симуляции
    for _ in range(num_trajectories):
        traj = {}
        for col in asset_columns:
            if asset_types[col] == 'CIR':
                k, theta, sigma = cir_params[col]
                traj[col] = simulate_cir(k, theta, sigma,
                                         r0=last_values[col],
                                         n=num_steps, dt=DT, rng=rng)
            else:
                mu, sigma = gbm_params[col]
                traj[col] = simulate_gbm(mu, sigma,
                                         s0=last_values[col],
                                         n=num_steps, dt=DT, rng=rng)
        result.append(pd.DataFrame(traj))

    return result


def get_correlated_normals(cov_matrix: np.ndarray,
                            n_steps: int,
                            n_paths: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Генерация коррелированных нормальных шоков.
    Возвращает массив размера [n_paths, n_steps, n_assets]
    """
    L = np.linalg.cholesky(cov_matrix)
    n_assets = cov_matrix.shape[0]

    shocks = rng.standard_normal((n_paths, n_steps, n_assets))
    correlated = shocks @ L.T  # (n_paths, n_steps, n_assets)

    return correlated


def get_log_return_corr(df: pd.DataFrame) -> np.ndarray:
    log_returns = np.log(df / df.shift(1)).dropna()
    return log_returns.corr().values



def simulate_portfolio_correlated(df: pd.DataFrame,
                                   base_volumes: dict[str, float],
                                   num_steps: int,
                                   num_paths: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """
    Симулирует весь портфель сразу, с учётом коррелированных шоков.
    Возвращает массив: [n_paths, n_steps+1]
    """
    assets = list(base_volumes.keys())
    prices = df[assets]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    last_prices = prices.iloc[-1].values
    cov = get_log_return_corr(prices)
    mu = log_returns.mean().values / DT
    sigma = log_returns.std().values / np.sqrt(DT)

    # коррелированные нормальные шоки
    shocks = get_correlated_normals(cov, num_steps, num_paths, rng)  # shape [paths, steps, assets]

    # моделируем log-цены: S_{t+1} = S_t * exp(μΔt + σZ)
    dt = DT
    n_assets = len(assets)
    log_paths = np.zeros((num_paths, num_steps + 1, n_assets))
    log_paths[:, 0, :] = np.log(last_prices)

    for t in range(1, num_steps + 1):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * shocks[:, t - 1, :]
        log_paths[:, t, :] = log_paths[:, t - 1, :] + drift + diffusion

    # обратно в цены
    price_paths = np.exp(log_paths)  # shape [paths, steps+1, assets]

    # пересчёт в стоимость портфеля
    weighted = np.einsum('pna,a->pn', price_paths, np.array([base_volumes[a] for a in assets]))
    return weighted  # shape [n_paths, n_steps+1]


import numpy as np
import pandas as pd

def simulate_from_pca(factors_df: pd.DataFrame, loadings: pd.DataFrame, n_sim: int = 1000, horizon: int = 10, seed: int = 42):
    """
    Генерация симуляций исходных активов на основе PCA-факторов.

    :param factors_df: DataFrame с временным рядом главных компонент (PC1, PC2, ...)
    :param loadings: DataFrame с загрузками факторов (строки — активы, столбцы — PC1, PC2, ...)
    :param n_sim: число симуляций
    :param horizon: горизонт (количество шагов вперёд)
    :param seed: random seed
    :return: список DataFrame — симуляции исходных активов
    """
    np.random.seed(seed)

    factors = factors_df.columns
    mu = factors_df.mean().values
    sigma = factors_df.cov().values

    # Генерация симулированных PCA-факторов
    sims = np.random.multivariate_normal(mu, sigma, size=(n_sim, horizon))  # shape [n_sim, horizon, n_factors]
    sims = sims.transpose(0, 2, 1)  # shape [n_sim, n_factors, horizon]

    # Восстановим исходные активы через загрузки (активы x факторы)
    asset_names = loadings.index
    loadings_matrix = loadings.values  # shape [n_assets, n_factors]

    asset_sims = np.einsum("nfh,af->nah", sims, loadings_matrix)  # shape [n_sim, n_assets, horizon]

    # Создадим список DataFrame
    simulations = []
    for sim in asset_sims:
        df_sim = pd.DataFrame(sim.T, columns=asset_names)
        simulations.append(df_sim)

    return simulations
