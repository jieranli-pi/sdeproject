import numpy as np
import matplotlib.pyplot as plt

def generate_wiener_process(T, N):
    """
    Generate a Wiener process (Brownian motion).

    Parameters:
    - T: Total time
    - N: Number of time steps

    Returns:
    - t: Time points
    - W: Wiener process values
    """
    dt = T / N
    t = np.linspace(0, T, N+1)
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    return t, W

# Set parameters
total_time = 1.0
num_steps = 500

# Generate Wiener process
time_points, wiener_process = generate_wiener_process(total_time, num_steps)

# Plot the Wiener process
plt.plot(time_points[:-1], wiener_process, label='Wiener Process')  # Remove the last element from time_points
plt.title('Wiener Process (Brownian Motion)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
