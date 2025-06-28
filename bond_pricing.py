import numpy as np
import pandas as pd

def price_bond(cashflows: pd.DataFrame, discount_curve: pd.Series, today: pd.Timestamp) -> float:
    """
    Оценка справедливой стоимости облигации на дату today.

    Аргументы:
    - cashflows: DataFrame со столбцами ['date', 'amount'], где указаны даты и суммы выплат.
    - discount_curve: Series, индекс — дюрации (в днях или годах), значения — ставки (в долях, не процентах).
    - today: дата оценки (обычно pd.Timestamp).

    Возвращает:
    - float: приведённая стоимость облигации.
    """

    present_value = 0.0
    for _, row in cashflows.iterrows():
        date = row["date"]
        amount = row["amount"]

        t = (date - today).days / 365.0
        if t <= 0:
            continue  # пропускаем уже прошедшие выплаты

        # ищем ближайшую ставку (или интерполируем)
        if isinstance(discount_curve.index[0], (int, float)):
            curve = discount_curve
        else:
            # если кривая задана датами — пересчитаем в годы от today
            curve = pd.Series(
                discount_curve.values,
                index=[(d - today).days / 365.0 for d in discount_curve.index]
            )

        # линейная интерполяция ставки
        rate = np.interp(t, curve.index, curve.values)

        # дисконтируем
        discounted = amount / ((1 + rate) ** t)
        present_value += discounted

    return present_value
