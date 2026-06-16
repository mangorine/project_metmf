import numpy as np
from scipy.stats import norm
from math import factorial, exp, sqrt, log

# Formules de BScholes ( pas BS lol ) pour option eu

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K * exp(-r * T), 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K * exp(-r * T) - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if option_type == 'call':
        return black_scholes_call(S, K, T, r, sigma)
    else:
        return black_scholes_put(S, K, T, r, sigma)
    
# partie 1

def prix_sauts_discrets(S, K, T, r, sigma, lam, a, b, p,
                        option_type='call', N=30):
    """
    Prix d'une option europeenne avecsauts a valeurs dans {a, b}. Params :
    S     : prix initial du sous-jacent
    K     : strike
    T     : maturite
    r     : taux sans risque
    sigma : volatilite brownienne
    lam   : intensite du processus de Poisson
    a, b  : valeurs des sauts relatifs (U_j in {a, b})
    """
    # Esperance des sauts et intensites decomposees
    E_U1    = p * a + (1 - p) * b
    lam_a  = lam * p
    lam_b  = lam * (1 - p)

    # Facteur de correction du spot
    correction = exp(-lam * T * E_U1)

    prix = 0.

    for k in range(N + 1):
        # Poids de Poisson(lambda_a * T) en k
        poids_k = exp(-lam_a * T) * (lam_a * T)**k / factorial(k)

        for m in range(N + 1):
            poids_m = exp(-lam_b * T) * (lam_b * T)**m / factorial(m)
            S_km = S * correction * (1 + a)**k * (1 + b)**m

            # Prix Black-Scholes avec ce spot modifie
            bs_km = black_scholes(S_km, K, T, r, sigma, option_type)
            prix += poids_k * poids_m * bs_km

    return prix