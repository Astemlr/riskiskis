import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def run_pca_analysis(df: pd.DataFrame, factor_cols: list[str], n_components: int = 11, plot: bool = True):
    """
    Выполняет PCA-анализ по лог-доходностям выбранных факторных колонок.

    Аргументы:
    - df: DataFrame с исходными ценами.
    - factor_cols: Список колонок, которые использовать как факторы.
    - n_components: Сколько компонент сохранить.
    - plot: Построить ли график объяснённой дисперсии.

    Возвращает:
    - pc_df: DataFrame с временными рядами PC (первыми n_components)
    - pca_model: обученная модель PCA
    - loadings_df: загрузки факторов в компоненты
    """
    returns = df[factor_cols].apply(lambda x: np.log(x) - np.log(x.shift(1))).dropna()

    scaler = StandardScaler()
    ret_scaled = scaler.fit_transform(returns)

    pca = PCA()
    pca_fit = pca.fit(ret_scaled)
    explained = pca_fit.explained_variance_ratio_

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(np.cumsum(explained)*100, marker='o')
        plt.xlabel("Число компонент")
        plt.ylabel("Накопленная объяснённая дисперсия (%)")
        plt.grid()
        plt.title("PCA — объяснённая дисперсия")
        plt.axhline(90, color='red', linestyle='--', label='90%')
        plt.legend()
        plt.show()

    # Преобразованные ряды
    pca_returns = pca.transform(ret_scaled)[:, :n_components]
    pc_df = pd.DataFrame(pca_returns, index=returns.index, columns=[f"PC{i+1}" for i in range(n_components)])

    # Загрузки
    loadings_df = pd.DataFrame(
        pca_fit.components_[:n_components].T,
        index=factor_cols,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return pc_df, pca, loadings_df


