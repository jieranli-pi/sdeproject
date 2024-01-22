import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.10   # Drift coefficient
S0 = 50.0   # Initial stock price
sigma0 = 0.20  # Initial volatility
xi0 = 0.20   # Initial long-term volatility
T = 1.0     # Total time
N = 1000    # Number of time steps
dt = T / N   # Time step

# Values for p and alpha
p_values = [0, 0.5, 1, 10]
alpha_values = [0.15, 0.15, 0.15, 0.15]

# Arrays to store results
t_values = np.linspace(0, T, N+1)
S_values_sets = []
sigma_values_sets = []
xi_values_sets = []

# Simulation for different p and alpha values
for i in range(len(p_values)):
    S_values = np.zeros(N+1)
    sigma_values = np.zeros(N+1)
    xi_values = np.zeros(N+1)

    S_values[0] = S0
    sigma_values[0] = sigma0
    xi_values[0] = xi0

    for j in range(1, N+1):
        epsilon1 = np.random.normal(0, 1)
        epsilon2 = np.random.normal(0, 1)

        # Stochastic part for stock price
        S_values[j] = S_values[j-1] + mu * S_values[j-1] * dt + sigma_values[j-1] * S_values[j-1] * np.sqrt(dt) * epsilon1 + p_values[i] * sigma_values[j-1] * np.sqrt(dt) * epsilon2

        # Stochastic part for volatility (sigma) and long-term volatility (xi)
        sigma_values[j] = sigma_values[j-1] + (sigma_values[j-1] - xi_values[j-1]) * dt + p_values[i] * sigma_values[j-1] * np.sqrt(dt) * epsilon2
        xi_values[j] = xi_values[j-1] + (sigma_values[j-1] - xi_values[j-1]) * dt + dt / alpha_values[i] * (sigma_values[j-1] - xi_values[j-1])

    S_values_sets.append(S_values)
    sigma_values_sets.append(sigma_values)
    xi_values_sets.append(xi_values)

# Plot the results
plt.figure(figsize=(12, 12))

# Plot deterministic part without stochastic component for stock price
plt.subplot(3, 1, 1)
S_deterministic = S0 * np.exp(mu * t_values)
plt.plot(t_values, S_deterministic, label='Deterministic (Without Stochastic Part)', color='black', linestyle='--')
for i in range(len(p_values)):
    plt.plot(t_values, S_values_sets[i], label=f'Set {i+1} ($p={p_values[i]}, \\alpha={alpha_values[i]}$)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Euler Scheme: Stock Price Simulation with Different $p$ and $\\alpha$')
plt.legend()
plt.grid(True)

# Plot stochastic simulations for volatility (sigma) and long-term volatility (xi)
plt.subplot(3, 1, 2)
for i in range(len(p_values)):
    plt.plot(t_values, sigma_values_sets[i], label=f'Set {i+1} ($p={p_values[i]}, \\alpha={alpha_values[i]}$) - $\\sigma$')
plt.xlabel('Time')
plt.ylabel('$\\sigma$ (Volatility)')
plt.title('Euler Scheme: Volatility ($\\sigma$) Simulation with Different $p$ and $\\alpha$')
plt.legend()
plt.grid(True)

# Plot stochastic simulations for long-term volatility (xi)
plt.subplot(3, 1, 3)
for i in range(len(p_values)):
    plt.plot(t_values, xi_values_sets[i], label=f'Set {i+1} ($p={p_values[i]}, \\alpha={alpha_values[i]}$) - $\\xi$')
plt.xlabel('Time')
plt.ylabel('$\\xi$ (Long-term Volatility)')
plt.title('Euler Scheme: Long-term Volatility ($\\xi$) Simulation with Different $p$ and $\\alpha$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
