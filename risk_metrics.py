from models import get_table_trajectories, simulate_portfolio_correlated
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def calc_var(df: pd.DataFrame,
             cir_assets: list[str],
             base_volumes: dict[str, float],
             num_steps: int = 10,
             num_trajectories: int = 1000,
             alpha: float = 0.01,
             rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Оценка Value-at-Risk портфеля на заданный горизонт.
    
    Параметры:
    - df: исторические данные по активам (цены/ставки)
    - cir_assets: список активов, которые симулируются через CIR
    - base_volumes: словарь с весами в портфеле
    - num_steps: горизонт в днях
    - num_trajectories: число симуляций
    - alpha: уровень VaR (например, 0.01 для 99%)
    - rng: генератор случайных чисел
    
    Возвращает:
    - фактическая стоимость портфеля (по df)
    - VaR-порог по симулированным значениям
    - список симулированных траекторий портфеля
    """
    assert set(base_volumes.keys()).issubset(set(df.columns))

    rng = rng or np.random.default_rng(42)
    assets = list(base_volumes.keys())

    # 1. Симуляции
    simulations_df_list = get_table_trajectories(
        df[assets],
        cir_assets=cir_assets,
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        rng=rng
    )

    # 2. Масштабируем траектории по базовым объемам
    portf_trajs = []
    for sim_df in simulations_df_list:
        sim_val = sum(sim_df[col] * base_volumes[col] for col in assets)
        portf_trajs.append(sim_val.to_numpy())

    portf_trajs = np.array(portf_trajs)  # shape = [num_trajectories, num_steps + 1]

    # 3. Реальные значения портфеля
    real_portf = sum(df[col] * base_volumes[col] for col in assets)
    real_portf = real_portf.values[:num_steps + 1]

    # 4. Оценка VaR и ES
    var_line = np.quantile(portf_trajs, alpha, axis=0)
    es_line = portf_trajs[portf_trajs <= var_line].mean(axis=0)  # опционально

    return real_portf, var_line, portf_trajs



def backtest_var_correlated(df: pd.DataFrame,
                             base_volumes: dict[str, float],
                             window_size: int = 250,
                             horizon_days: int = 10,
                             alpha: float = 0.01,
                             n_sim: int = 300,
                             rng: np.random.Generator = None) -> list[int]:
    """
    Backtest VaR с использованием коррелированной симуляции портфеля.
    
    Возвращает список из 0/1: 1 — пробой (реальная стоимость < VaR), 0 — нет.
    """
    rng = rng or np.random.default_rng(42)
    breaks = []
    assets = list(base_volumes.keys())

    for t in range(window_size, len(df) - horizon_days):
        try:
            # окно истории для калибровки
            hist_window = df.iloc[t - window_size:t][assets]
            # будущие реальные значения
            real_future = df.iloc[t: t + horizon_days + 1][assets]

            # симуляции
            sim_paths = simulate_portfolio_correlated(
                df=hist_window,
                base_volumes=base_volumes,
                num_steps=horizon_days,
                num_paths=n_sim,
                rng=rng
            )

            # реальная стоимость портфеля на горизонте
            real_portf = (real_future * pd.Series(base_volumes)).sum(axis=1).values
            VaR_line = np.quantile(sim_paths, alpha, axis=0)

            # сравниваем последний день
            is_break = real_portf[-1] < VaR_line[-1]
            breaks.append(int(is_break))

        except Exception as e:
            print(f"Ошибка на шаге {t}: {e}")
            continue

    return breaks


from backtesting_analysis import compute_simulated_losses, backtest_es, plot_var_es_distribution
def run_var_es_evaluation(
    df: pd.DataFrame,
    base_volumes: Dict[str, float],
    target_date: str,
    model_simulations: list[pd.DataFrame],
    alpha: float = 0.01,
    horizon_days: int = 10,
    plot: bool = True
):
    """
    Оценивает VaR и Expected Shortfall по симуляциям на заданную дату.

    Аргументы:
    - df: исходные данные
    - base_volumes: словарь {тикер: сумма в рублях}
    - target_date: дата оценки, строка 'YYYY-MM-DD'
    - model_simulations: список симулированных DataFrame
    - alpha: уровень значимости
    - horizon_days: горизонт в днях
    - plot: показывать ли график

    Возвращает:
    - словарь с VaR, ES, пробоями и пр.
    """

    target_date = pd.Timestamp(target_date)
    if target_date not in df.index:
        target_date = df.index[df.index.get_indexer([target_date], method='pad')[0]]

    weights = pd.Series(base_volumes)
    portfolio_value_0 = (df.loc[target_date, weights.index] * weights).sum()

    # Симулированные потери
    losses = compute_simulated_losses(model_simulations, base_volumes, portfolio_value_0)

    # Фактическая потеря на горизонте
    future_date = df.index[df.index.get_loc(target_date) + horizon_days]
    portfolio_value_future = (df.loc[future_date, weights.index] * weights).sum()
    fact_loss = portfolio_value_0 - portfolio_value_future

    if plot:
        plot_var_es_distribution(losses, alpha=alpha)

    results = backtest_es(
        actual_losses=[fact_loss],
        simulated_losses=losses,
        alpha=alpha
    )
    return {
        "target_date": target_date,
        "VaR": results['VaR'],
        "ES_model": results['ES_model'],
        "ES_empirical": results['ES_empirical'],
        "fact_loss": fact_loss,
        "n_exceedances": results['n_exceedances']
    }
