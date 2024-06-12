import numpy as np

def monte_carlo_option_pricing(S0, K, T, r, sigma, n, N):
    """
    Monte Carlo simulation for European call option pricing.
    
    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - n: Number of time steps
    - N: Number of simulations
    
    Returns:
    - Option price
    """
    dt = T / n
    price_paths = np.zeros((N, n + 1))
    price_paths[:, 0] = S0
    
    for t in range(1, n + 1):
        Z = np.random.standard_normal(N)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt +
                                                          sigma * np.sqrt(dt) * Z)
    
    payoffs = np.maximum(price_paths[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price

# Parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1     # Time to maturity in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
n = 100   # Number of time steps
N = 10000 # Number of simulations

option_price = monte_carlo_option_pricing(S0, K, T, r, sigma, n, N)
print(f"The estimated European call option price is: {option_price:.2f}")
