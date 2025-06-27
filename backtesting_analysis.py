# backtesting_analysis.py

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt


def kupiec_test(num_violations: int, total_obs: int, alpha: float = 0.01) -> dict:
    x, n = num_violations, total_obs
    p_hat = x / n
    if p_hat in [0, 1]:  # избежание логарифма нуля
        return {"statistic": 0.0, "p_value": 1.0, "rejected": False}

    logL_H0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    logL_H1 = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    LR_uc = -2 * (logL_H0 - logL_H1)
    p_value = 1 - chi2.cdf(LR_uc, df=1)

    return {"statistic": LR_uc, "p_value": p_value, "rejected": p_value < 0.05}


def christoffersen_test(breaks: np.ndarray) -> dict:
    """
    Christoffersen (1998) Independence Test
    """
    if sum(breaks) <= 1:
        return {"statistic": 0.0, "p_value": 1.0, "rejected": False}

    # Переходы 00, 01, 10, 11
    t = breaks.astype(int)
    t_1 = t[:-1]
    t_2 = t[1:]

    n00 = np.sum((t_1 == 0) & (t_2 == 0))
    n01 = np.sum((t_1 == 0) & (t_2 == 1))
    n10 = np.sum((t_1 == 1) & (t_2 == 0))
    n11 = np.sum((t_1 == 1) & (t_2 == 1))

    pi01 = n01 / (n00 + n01 + 1e-8)
    pi11 = n11 / (n10 + n11 + 1e-8)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11 + 1e-8)

    logL_H0 = (n00 + n01) * np.log(1 - pi) + (n10 + n11) * np.log(pi)
    logL_H1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + \
              n10 * np.log(1 - pi11) + n11 * np.log(pi11)

    LR_ind = -2 * (logL_H0 - logL_H1)
    p_value = 1 - chi2.cdf(LR_ind, df=1)

    return {"statistic": LR_ind, "p_value": p_value, "rejected": p_value < 0.05}


def plot_backtesting_results(breaks: np.ndarray, alpha: float, label: str = "VaR"):
    expected = len(breaks) * alpha
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(breaks), label="Пробои")
    plt.axhline(y=expected, color='red', linestyle='--', label=f"Ожидаемый уровень ({alpha*100:.0f}%)")
    plt.title(f"Backtesting {label}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def expected_shortfall(losses: np.ndarray, alpha: float = 0.025) -> float:
    """
    Расчёт Expected Shortfall по данным убытков (негативные изменения стоимости).
    """
    var_threshold = np.quantile(losses, alpha)
    return losses[losses <= var_threshold].mean()


def run_backtesting_analysis(breaks: np.ndarray, alpha: float = 0.01, label: str = "VaR"):
    print(f"Пробои: {int(np.sum(breaks))} из {len(breaks)} дней")

    plot_backtesting_results(breaks, alpha, label)

    # Kupiec Test
    result_kupiec = kupiec_test(np.sum(breaks), len(breaks), alpha)
    print("\nKupiec test:")
    print(f"  Статистика: {result_kupiec['statistic']:.4f}")
    print(f"  p-value: {result_kupiec['p_value']:.4f}")
    print(f"  {'Гипотеза отвергается' if result_kupiec['rejected'] else 'Гипотеза НЕ отвергается'}")

    # Christoffersen Test
    result_chris = christoffersen_test(breaks)
    print("\nChristoffersen test (независимость пробоев):")
    print(f"  Статистика: {result_chris['statistic']:.4f}")
    print(f"  p-value: {result_chris['p_value']:.4f}")
    print(f"  {'Зависимы' if result_chris['rejected'] else 'Независимы'}")

    return {
        "kupiec": result_kupiec,
        "christoffersen": result_chris
    }


def compute_simulated_losses(simulations_df_list, base_volumes, portfolio_value_0):
    """
    Возвращает массив убытков (initial - terminal value) по симулированным траекториям портфеля.
    """
    losses = []
    for sim_df in simulations_df_list:
        final_values = sim_df[list(base_volumes.keys())].iloc[-1]  # последняя дата
        weighted_sum = (final_values * pd.Series(base_volumes)).sum()
        loss = portfolio_value_0 - weighted_sum
        losses.append(loss)
    return np.array(losses)



def plot_var_es_distribution(losses: np.ndarray, alpha: float = 0.025):
    """
    График распределения убытков с отображением VaR и ES
    """
    var = np.quantile(losses, alpha)
    es = expected_shortfall(losses, alpha)

    plt.figure(figsize=(6, 4))
    plt.hist(losses, bins=40, color='lightgray', edgecolor='black', density=True)
    plt.axvline(var, color='red', linestyle='--', label=f'VaR {alpha:.2%}')
    plt.axvline(es, color='blue', linestyle='--', label=f'ES {alpha:.2%}')
    plt.title("Распределение убытков, VaR и Expected Shortfall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def backtest_es(
    actual_losses: np.ndarray,
    simulated_losses: np.ndarray,
    alpha: float = 0.025
) -> dict:
    """
    Простая проверка ES: средний фактический убыток при пробоях VaR vs. ES по симуляциям.
    """
    var = np.quantile(simulated_losses, alpha)
    es_model = expected_shortfall(simulated_losses, alpha)
    actual_exceedances = actual_losses[actual_losses <= var]

    es_empirical = actual_exceedances.mean() if len(actual_exceedances) > 0 else np.nan

    return {
        "VaR": var,
        "ES_model": es_model,
        "ES_empirical": es_empirical,
        "n_exceedances": len(actual_exceedances)
    }





from models import get_table_trajectories
from backtesting_analysis import (
    compute_simulated_losses,
    plot_var_es_distribution,
    backtest_es
)
import numpy as np
import pandas as pd


def run_es_analysis(
    df: pd.DataFrame,
    base_volumes: dict,
    target_date: str = "2024-12-02",
    alpha: float = 0.025,
    horizon_days: int = 10,
    num_trajectories: int = 500,
    cir_assets: list = None,
    label: str = "Основной портфель"
):
    """
    Полный анализ Expected Shortfall (модельный и фактический) по заданному портфелю.
    """
    print(f"\nАнализ ES для портфеля: {label}")
    
    weights = pd.Series(base_volumes)

    # Найти ближайшую рабочую дату к целевой
    target_date = pd.Timestamp(target_date)
    if target_date not in df.index:
        actual_date = df.index[df.index.get_indexer([target_date], method="pad")[0]]
    else:
        actual_date = target_date

    # Начальная стоимость портфеля
    portfolio_value_0 = (df.loc[actual_date, weights.index] * weights).sum()
    print(f"Дата: {actual_date.date()}, Начальная стоимость портфеля: {portfolio_value_0:.2f} руб.")

    # CIR-моделируемые активы
    if cir_assets is None:
        cir_assets = []  # по умолчанию всё моделируется как GBM

    all_assets = list(base_volumes.keys())
    simulations_df_list = get_table_trajectories(
        df[all_assets],
        cir_assets=cir_assets,
        num_trajectories=num_trajectories,
        num_steps=horizon_days,
        rng=np.random.default_rng(42)
    )

    # Симулированные убытки
    losses = compute_simulated_losses(simulations_df_list, base_volumes, portfolio_value_0)

    # Фактическая стоимость через horizon_days
    future_date = df.index[df.index.get_indexer([actual_date + pd.Timedelta(days=horizon_days)], method="pad")[0]]
    portfolio_value_t = (df.loc[future_date, weights.index] * weights).sum()
    fact_losses = np.array([portfolio_value_0 - portfolio_value_t])

    print(f"Фактический убыток на {horizon_days} дней: {fact_losses[0]:.2f} руб.\n")

    # График и метрики
    plot_var_es_distribution(losses, alpha=alpha)

    results = backtest_es(
        actual_losses=fact_losses,
        simulated_losses=losses,
        alpha=alpha
    )

    print("Expected Shortfall Analysis:")
    print(f"  VaR({alpha:.0%}): {results['VaR']:.2f}")
    print(f"  ES (модель): {results['ES_model']:.2f}")
    print(f"  ES (реально при пробоях): {results['ES_empirical']:.2f}")
    print(f"  Пробоев: {results['n_exceedances']} шт.")

    return results
