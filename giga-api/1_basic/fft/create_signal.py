import numpy as np
import matplotlib.pyplot as plt

# Generate a more complex signal
frequencies = [1.0, 5.0, 10.0, 20.0, 50.0]  # Multiple frequencies
sample_rate = 44100.0  # Higher sample rate
duration = 10.0  # Longer duration

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

# Plot the signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Generated Signal")
plt.grid(True)
plt.tight_layout()


# save the plot
plt.savefig("generated_signal.png")