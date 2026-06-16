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
    Prix d'une option europeenne avecsauts dans {a, b}. Params :
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
    prix = 0.0

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

# partie 2 

def prix_sauts_lognormaux(S, K, T, r, sigma, lam, m, sigma_J,
                           option_type='call', N=50):
    """
    Prix d'une option europeenne avec sauts log-normaux. Params :
    m : moyenne de g (log du saut + 1)
    sigma_J : ecart-type de g
    """
    # Esperance du saut (formule log-normale)
    mu_J = exp(m + 0.5 * sigma_J**2) - 1
    prix = 0.0

    for n in range(N + 1):
        poids_n = exp(-lam * T) * (lam * T)**n / factorial(n)
        # Volatilite augmentee des sauts
        sigma_n = sqrt(sigma**2 + n * sigma_J**2 / T)
        # Taux modifie
        r_n = r - lam * mu_J + n * m / T
        bs_n = black_scholes(S, K, T, r_n, sigma_n, option_type)
        prix += poids_n * bs_n

    return prix



# test des fonctions

S = 100.0 # spot
K = 100.0 # strike (at-the-money)
T = 1.0 # maturite 1 an
r = 0.05 # taux sans risque
sigma = 0.20 # volatilite brownienne
lam = 1.0 # en moyenne 1 saut par an

# Bscholes sans sauts
bs_call = black_scholes_call(S, K, T, r, sigma)
bs_put  = black_scholes_put(S, K, T, r, sigma)
print(f"\nBlack-Scholes de reference :")
print(f"  Call ATM : {bs_call:.4f}")
print(f"  Put  ATM : {bs_put:.4f}")

# partie 1
print("\nPARTIE 1 : Sauts discrets U_1 in {a, b}")

a = 0.10
b = -0.10
p = 0.5   # equiprobable

call_discret = prix_sauts_discrets(S, K, T, r, sigma, lam, a, b, p, option_type='call', N=50)
put_discret  = prix_sauts_discrets(S, K, T, r, sigma, lam, a, b, p, option_type='put', N=50)

print(f"\nParametres : a={a}, b={b}, p={p}, lambda={lam}")
print(f"  E[U1] = {p*a + (1-p)*b:.4f}")
print(f"  Call ATM : {call_discret:.4f}")
print(f"  Put  ATM : {put_discret:.4f}")

parite = S - K * exp(-r * T)
print(f"  Parite call-put theorique C-P = {parite:.4f}")
print(f"  Parite call-put obtenue   C-P = {call_discret - put_discret:.4f}")

# Effet de l'intensite
print("\nEffet de l'intensite lambda :")
print(f"  {'lambda':>8} | {'Call':>10} | {'Put':>10}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}")
for l in [0.0, 0.5, 1.0, 2.0, 5.0]:
    if l == 0.0:
        c = bs_call
        pu = bs_put
    else:
        c  = prix_sauts_discrets(S, K, T, r, sigma, l, a, b, p, option_type='call', N=25)
        pu = prix_sauts_discrets(S, K, T, r, sigma, l, a, b, p, option_type='put',  N=25)
    print(f"  {l:>8.1f} | {c:>10.4f} | {pu:>10.4f}")