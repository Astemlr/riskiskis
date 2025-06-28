import numpy as np
import pandas as pd
from scipy.stats import norm

def black76_price(F, K, T, sigma, r, option_type='call'):
    """Black'76 pricing"""
    if T <= 0:
        return max(F-K, 0) if option_type=='call' else max(K-F, 0)
    d1 = (np.log(F/K)+(sigma**2)*T/2) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    df = np.exp(-r*T)
    if option_type=='call':
        return df*(F*norm.cdf(d1) - K*norm.cdf(d2))
    else:
        return df*(K*norm.cdf(-d2) - F*norm.cdf(-d1))

if __name__ == "__main__":
    # Параметры на 2023-10-01
    F = 85.0         # цена Brent, USD/barrel
    K = 80.0         # страйк: 5$ ниже — в деньгах для Call
    sigma = 0.20     # 20% годовых
    r = 0.07         # 7% безрисковая ставка
    T = 0.25         # квартал до экспирации

    call_price = black76_price(F, K, T, sigma, r, 'call')
    put_price = black76_price(F, K, T, sigma, r, 'put')

    print(f"Call Black'76: {call_price:.2f}")
    print(f"Put Black'76:  {put_price:.2f}")
