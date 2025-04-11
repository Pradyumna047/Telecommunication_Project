# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:56:50 2025

@author: st241065
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

sampling_rate = 10000  
T = 1  
t = np.linspace(0, T, sampling_rate)   

def gen_sig(frequencies, amplitudes):
    signal = np.zeros_like(t)
    for f, A in zip(frequencies, amplitudes):
        signal += A * np.sin(2 * np.pi * f * t) 
    return signal

def plot_fft(signal, title):
    N = len(t)
    fft_values = fft(signal)  
    freqs = fftfreq(N, 1/sampling_rate)  

    plt.figure(figsize=(10, 4))
    plt.plot(freqs[:N//2], np.abs(fft_values[:N//2]) / N * 2)  
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'FFT of {title}')
    plt.grid()
    plt.show()
    
def plot_individual_components(frequencies, amplitudes, title):
    plt.figure(figsize=(12, 6))
    for i, (f, A) in enumerate(zip(frequencies, amplitudes), start=1):
        plt.plot(t, A * np.sin(2 * np.pi * f * t), label=f'{A}·sin(2π·{f}·t)')
    plt.title(f"Original Components of {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
# Scenario 1: Low-frequency 
freqs1 = [2, 4, 6, 10, 16, 26, 42, 68, 110] 
amps1 = [9, 8, 7, 6, 5, 4, 3, 2, 1] 
plot_individual_components(freqs1, amps1, "Scenario 1") 
signal1 = gen_sig(freqs1, amps1)

# Scenario 2: Mid-frequency dominant
freqs2 = [50, 100, 200, 300, 400, 500, 600, 700]  
amps2 = [34, 21, 13, 8, 5, 3, 2, 1]
plot_individual_components(freqs2, amps2, "Scenario 2")  
signal2 = gen_sig(freqs2, amps2)

# Scenario 3: High-frequency dominant
freqs3 = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]  
amps3 = [16, 14, 12, 10, 8, 6, 4, 2]
plot_individual_components(freqs3, amps3, "Scenario 3")  
signal3 = gen_sig(freqs3, amps3)


# Plot signals in time domain
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, signal1)
plt.title("Scenario 1: Low-Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(t, signal2)
plt.title("Scenario 2: Mid-Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(t, signal3)
plt.title("Scenario 3: High-Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

plot_fft(signal1, "Scenario 1")
plot_fft(signal2, "Scenario 2")
plot_fft(signal3, "Scenario 3")