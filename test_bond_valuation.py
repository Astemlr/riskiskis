
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bond_pricing import price_bond

# --- Параметры облигации ---
issue_date = pd.Timestamp("2023-01-01")
maturity_date = pd.Timestamp("2028-01-01")
coupon_rate = 0.078  # 7.8% годовых
nominal = 1_000
coupon_frequency = 2  # выплаты дважды в год
today = pd.Timestamp("2024-01-01")

# --- Генерация графика выплат ---
cashflow_dates = pd.date_range(start=issue_date, end=maturity_date, freq="6MS")
cashflow_dates = cashflow_dates[cashflow_dates > today]

coupon_payment = nominal * coupon_rate / coupon_frequency
amounts = [coupon_payment] * (len(cashflow_dates) - 1) + [coupon_payment + nominal]  # последний купон + номинал

cashflows = pd.DataFrame({
    "date": cashflow_dates,
    "amount": amounts
})

# --- Пример типовой кривой доходности ---
durations = np.arange(0.5, 10.5, 0.5)
rates = 0.07 + 0.01 * np.exp(-durations / 5)  # слегка выпуклая
discount_curve = pd.Series(rates, index=durations)

# --- Оценка стоимости облигации ---
price = price_bond(cashflows, discount_curve, today)
print(f"Справедливая стоимость облигации: {price:.2f} руб.")
