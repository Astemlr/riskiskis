# bond_option_valuation.py

import numpy as np

def bond_price_with_option(cf, dates, ytm, option_date, option_type='put'):
    def pv(cashflows, rates, times):
        return sum(c / (1 + r) ** t for c, r, t in zip(cashflows, rates, times))

    times = np.array([(d - dates[0]).days / 365 for d in dates])
    rates = [ytm] * len(cf)

    full_price = pv(cf, rates, times)

    option_idx = dates.index(option_date)
    option_cf = cf[:option_idx + 1]
    option_times = times[:option_idx + 1]
    option_rates = rates[:option_idx + 1]
    option_price = pv(option_cf, option_rates, option_times)

    if option_type == 'put':
        return min(full_price, option_price)
    elif option_type == 'call':
        return max(full_price, option_price)
    else:
        raise ValueError("Тип должен быть 'put' или 'call'")
