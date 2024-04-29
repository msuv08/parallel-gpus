# FOR USER: IF YOU ARE INTERESTED IN MAKING A SIGNAL,
# WE HIGHLY RECOMMEND YOU SAVE IT TO THE INPUT SIGNAL FILE!

import numpy as np

frequency = 1.0
sample_rate = 1024.0
duration = 1.0

t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = np.sin(2 * np.pi * frequency * t)


np.savetxt("input_signal.txt", signal, fmt="%.10f")