import numpy as np
import matplotlib.pyplot as plt

def euler_scheme(mu, S0, sigma0, xi0, p, alpha, T, N):
    dt = T / N
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
        S_values[j] = S_values[j-1] + mu * S_values[j-1] * dt + sigma_values[j-1] * S_values[j-1] * np.sqrt(dt) * epsilon1 + p * sigma_values[j-1] * np.sqrt(dt) * epsilon2

        # Stochastic part for volatility (sigma) and long-term volatility (xi)
        sigma_values[j] = sigma_values[j-1] + (sigma_values[j-1] - xi_values[j-1]) * dt + p * sigma_values[j-1] * np.sqrt(dt) * epsilon2
        xi_values[j] = xi_values[j-1] + (sigma_values[j-1] - xi_values[j-1]) * dt + dt / alpha * (sigma_values[j-1] - xi_values[j-1])

    return S_values, sigma_values, xi_values

def milstein_scheme(mu, S0, sigma0, xi0, p, alpha, T, N):
    dt = T / N
    S_values = np.zeros(N+1)
    sigma_values = np.zeros(N+1)
    xi_values = np.zeros(N+1)

    S_values[0] = S0
    sigma_values[0] = sigma0
    xi_values[0] = xi0

    for j in range(1, N+1):
        epsilon1 = np.random.normal(0, 1)
        epsilon2 = np.random.normal(0, 1)

        # Milstein scheme for stock price
        S_values[j] = S_values[j-1] + mu * S_values[j-1] * dt + sigma_values[j-1] * S_values[j-1] * np.sqrt(dt) * epsilon1 + p * sigma_values[j-1] * np.sqrt(dt) * epsilon2
        S_values[j] += 0.5 * p * sigma_values[j-1] * sigma_values[j-1] * (epsilon2 * epsilon2 - 1) * dt  # Milstein correction term

        # Stochastic part for volatility (sigma) and long-term volatility (xi)
        sigma_values[j] = sigma_values[j-1] + (sigma_values[j-1] - xi_values[j-1]) * dt + p * sigma_values[j-1] * np.sqrt(dt) * epsilon2
        sigma_values[j] += 0.5 * p * sigma_values[j-1] * (epsilon2 * epsilon2 - 1) * dt  # Milstein correction term

        xi_values[j] = xi_values[j-1] + (sigma_values[j-1] - xi_values[j-1]) * dt + dt / alpha * (sigma_values[j-1] - xi_values[j-1])

    return S_values, sigma_values, xi_values


# Parameters
mu = 0.10   # Drift coefficient
S0 = 50.0   # Initial stock price
sigma0 = 0.20  # Initial volatility
xi0 = 0.20   # Initial long-term volatility
T = 1.0     # Total time
dt_ref = 0.5  # Reference time step (very small)
N_ref = int(T / dt_ref)  # Reference number of time steps
num_simulations = 500  # Number of simulations for each scenario

# Values for p and alpha, doesn't matter black schole
p_values = [0]
alpha_values = [3]

# Define time step factors
dt_factors = [1, 2, 4, 8, 16]

# Arrays to store results
euler_errors = []
milstein_errors = []

# Perform simulations for different p and alpha values
for i in range(len(p_values)):
    # Reference simulation with very small dt
    S_ref, _, _ = euler_scheme(mu, S0, sigma0, xi0, p_values[i], alpha_values[i], T, N_ref)

    # Arrays to store errors for each time step
    euler_dt_errors = np.zeros(len(dt_factors))
    milstein_dt_errors = np.zeros(len(dt_factors))

    # Iterate over different time steps
    for sim in range(num_simulations):
        for j, dt_factor in enumerate(dt_factors[::-1]):  # Reverse the order of dt_factors
            dt = dt_ref / dt_factor
            N = int(T / dt)

            # Perform Euler scheme simulation
            S_euler, _, _ = euler_scheme(mu, S0, sigma0, xi0, p_values[i], alpha_values[i], T, N)
            euler_error = np.max(np.abs(S_euler - np.mean(S_ref)))

            # Perform Milstein scheme simulation
            S_milstein, _, _ = milstein_scheme(mu, S0, sigma0, xi0, p_values[i], alpha_values[i], T, N)
            milstein_error = np.max(np.abs(S_milstein - np.mean(S_ref)))

            euler_dt_errors[j] += euler_error
            milstein_dt_errors[j] += milstein_error

    # Calculate average errors
    euler_dt_errors /= num_simulations
    milstein_dt_errors /= num_simulations

    euler_errors.append(euler_dt_errors)
    milstein_errors.append(milstein_dt_errors)

# Plot the results
plt.figure(figsize=(12, 8))

for i in range(len(p_values)):
    # Fit a line to the logarithm of the errors
    fit_euler = np.polyfit(np.log(dt_factors), np.log(euler_errors[i]), 1)
    fit_milstein = np.polyfit(np.log(dt_factors), np.log(milstein_errors[i]), 1)

    # Extract the slope (convergence rate)
    convergence_rate_euler = fit_euler[0]
    convergence_rate_milstein = fit_milstein[0]

    # Calculate the convergence order
    convergence_order_euler = convergence_rate_euler + 1
    convergence_order_milstein = convergence_rate_milstein + 1

    print(f'Euler Convergence Rate (p={p_values[i]}, alpha={alpha_values[i]}): {convergence_rate_euler}')
    print(f'Euler Convergence Order (p={p_values[i]}, alpha={alpha_values[i]}): {convergence_order_euler}')

    print(f'Milstein Convergence Rate (p={p_values[i]}, alpha={alpha_values[i]}): {convergence_rate_milstein}')
    print(f'Milstein Convergence Order (p={p_values[i]}, alpha={alpha_values[i]}): {convergence_order_milstein}')

    plt.semilogy(dt_factors, euler_errors[i], marker='o', label=f'Euler (p={p_values[i]}, alpha={alpha_values[i]})')
    plt.semilogy(dt_factors, milstein_errors[i], marker='o', label=f'Milstein (p={p_values[i]}, alpha={alpha_values[i]})')

plt.xlabel('Time Step Factor (dt_ref / dt)')
plt.ylabel('Maximum Absolute Error')
plt.title('Convergence Study: Euler vs Milstein')
plt.legend()
plt.grid(True)
plt.show()
