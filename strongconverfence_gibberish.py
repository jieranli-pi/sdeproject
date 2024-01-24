import numpy as np
import matplotlib.pyplot as plt

# Define parameters
S0 = 50
sigma0 = 0.20
xi0 = 0.20
mu = 0.10
T = 1  # Time horizon
alpha = 0.1  # Adjust as needed

# Function to generate Wiener increments
def generate_wiener_increments(num_steps, delta_t):
    return np.sqrt(delta_t) * np.random.normal(size=num_steps)

# Define the Black-Scholes SDE
def black_scholes_sde(S, sigma, mu, dt, dW):
    return mu * S * dt + sigma * S * dW

# Deterministic part of the solution
def deterministic_part(t):
    return 0.01 * S0 * np.exp(0.1 * t)

# Function to simulate the Black-Scholes model using Euler scheme
def euler_scheme_black_scholes(S0, sigma0, mu, T, alpha, num_steps):
    delta_t = T / num_steps
    S_values = np.zeros(num_steps + 1)
    S_values[0] = S0

    for i in range(num_steps):
        dW = generate_wiener_increments(1, delta_t)
        S_values[i + 1] = S_values[i] + black_scholes_sde(S_values[i], sigma0, mu, delta_t, dW)
    return S_values
# Function to simulate the Black-Scholes model using Milstein scheme
def milstein_scheme_black_scholes(S0, sigma0, mu, T, alpha, num_steps):
    delta_t = T / num_steps
    S_values = np.zeros(num_steps + 1)
    S_values[0] = S0

    for i in range(num_steps):
        dW = generate_wiener_increments(1, delta_t)
        S_values[i + 1] = S_values[i] + mu * S_values[i] * delta_t + sigma0 * S_values[i] * dW

        # Additional term for Milstein scheme (quadratic variation)
        S_values[i + 1] += 0.5 * sigma0 * alpha * S_values[i] * (dW**2 - delta_t)

    return S_values

# Function to perform the convergence study for the Black-Scholes model
def convergence_study_strong_convergence():
    # Choose a small initial time step Δt
    delta_t = 0.25

    # Generate numerical tracks for different values of Δt
    for _ in range(6):
        num_steps = int(T / delta_t)

        # Simulate with Euler scheme for Black-Scholes
        S_euler = euler_scheme_black_scholes(S0, sigma0, mu, T, alpha, num_steps)
        
        # Error given from means of variances
        error_strong=np.mean(S_euler-np.mean(S_euler))
        print("error is", error_strong, " when dt=", delta_t)

        # Plot the numerical tracks
        plt.plot(np.linspace(0, T, num_steps + 1), S_euler, label=f'Δt={delta_t}')

        # Plot the deterministic part
        t_values = np.linspace(0, T, num_steps + 1)

        # Reduce Δt for the next iteration
        delta_t /= 5
    plt.plot(t_values, S0 * np.exp(mu * t_values), '--', label='Ref')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Convergence Study in the Strong Sense Euler')
    plt.legend()
    plt.show()

# Perform the convergence study
convergence_study_strong_convergence()
