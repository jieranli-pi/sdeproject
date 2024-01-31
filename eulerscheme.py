import numpy as np
import matplotlib.pyplot as plt

def euler_scheme(mu, S0, sigma0, p, T, N, alpha):
    dt = T / N
    S_values = np.zeros(N+1)
    S_values[0] = S0

    for j in range(1, N+1):
        epsilon1 = np.random.normal(0, 1)
        epsilon2 = np.random.normal(0, 1)

        S_values[j] = S_values[j-1] + mu * S_values[j-1] * dt + sigma0 * S_values[j-1] * np.sqrt(dt) * epsilon1 + p * sigma0 * np.sqrt(dt) * epsilon2

    return S_values

# Parameters
mu = 0.10   # Drift coefficient
S0 = 50.0   # Initial stock price
sigma0 = 0.20  # Initial volatility
T = 1.0     # Total time
N = 1000    # Number of time steps

# Values for p and alpha
p_values = [0.0, 1, 0.2, -0.1, -0.2]
alpha_values = [0.1, 150, 0.2, 0.05, 0.25]

# Arrays to store results
t_values = np.linspace(0, T, N+1)
S_values_sets = []

# Simulation for different p and alpha values
for i in range(len(p_values)):
    S_values = euler_scheme(mu, S0, sigma0, p_values[i], T, N, alpha_values[i])
    S_values_sets.append(S_values)

# Plot the results
plt.figure(figsize=(10, 6))

for i in range(len(p_values)):
    plt.plot(t_values, S_values_sets[i], label=f'Set {i+1} (p={p_values[i]}, alpha={alpha_values[i]})')

plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Euler Scheme: Stock Price Simulation with Different p and alpha')
plt.legend()
plt.grid(True)
plt.show()
