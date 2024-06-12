import numpy as np

def binomial_option_pricing(S0, K, T, r, sigma, n, option_type='call'):
    """
    Binomial pricing model for European options.
    
    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - n: Number of steps
    - option_type: 'call' or 'put'
    
    Returns:
    - Option price
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros(n + 1)
    option_values = np.zeros(n + 1)
    for i in range(n + 1):
        asset_prices[i] = S0 * (u ** (n - i)) * (d ** i)
    
    # Calculate option value at maturity
    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    else:
        option_values = np.maximum(0, K - asset_prices)
    
    # Backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = np.exp(-r * dt) * (p * option_values[j] + (1 - p) * option_values[j + 1])
    
    return option_values[0]

# Parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1     # Time to maturity in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
n = 100   # Number of steps

option_price = binomial_option_pricing(S0, K, T, r, sigma, n, option_type='call')
print(f"The estimated European call option price is: {option_price:.2f}")
