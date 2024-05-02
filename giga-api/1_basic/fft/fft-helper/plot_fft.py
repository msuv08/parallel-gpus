import numpy as np
import matplotlib.pyplot as plt


with open("fft_output.txt", "r") as file:
    data = file.readlines()

real_part = []
imag_part = []
for line in data:
    real, imag = map(float, line.split())
    real_part.append(real)
    imag_part.append(imag)

real_part = np.array(real_part)
imag_part = np.array(imag_part)

magnitude = np.sqrt(real_part**2 + imag_part**2)

num_samples = len(magnitude)
sample_rate = 1024.0  
freq_bins = np.fft.fftfreq(num_samples, 1 / sample_rate)[:num_samples // 2]

plt.figure(figsize=(8, 4))
plt.plot(freq_bins, magnitude[:num_samples // 2])

# use to potentially limit axes! :D
# plt.xlim(0, 25)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum")
plt.grid(True)
plt.savefig("fft_plot.png")

