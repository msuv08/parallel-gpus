import numpy as np
import matplotlib.pyplot as plt

# Generate a more complex signal
frequencies = [1.0, 5.0, 10.0, 20.0, 50.0]  # Multiple frequencies
sample_rate = 1024.0  # Desired sample rate
duration = 180.0  # Extremely long duration (in seconds)

t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = np.zeros_like(t)

for frequency in frequencies:
    signal += np.sin(2 * np.pi * frequency * t)

# Add some noise to the signal
noise = np.random.normal(0, 0.1, signal.shape)
signal += noise

# Normalize the signal
signal /= np.max(np.abs(signal))

# Save the signal to a file
np.savetxt("input_signal.txt", signal, fmt="%.10f")

# Plot a small portion of the signal
plot_duration = 1.0  # Duration of the portion to plot (in seconds)
plot_samples = int(sample_rate * plot_duration)
plot_t = t[:plot_samples]
plot_signal = signal[:plot_samples]

plt.figure(figsize=(10, 4))
plt.plot(plot_t, plot_signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Generated Signal (Portion)")
plt.grid(True)
plt.tight_layout()
# Save the plot
plt.savefig("generated_signal_large.png")