# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 18:30:48 2025

@author: st241065
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal.windows import hann, hamming

# Sampling parameters
sampling_rate = 600
T = 2
t = np.linspace(0, T, int(T * sampling_rate), endpoint=False)
N = len(t)

def plot_individual_components(frequencies, amplitudes):
    plt.figure(figsize=(12, 6))
    for i, (f, A) in enumerate(zip(frequencies, amplitudes), start=1):
        plt.plot(t, A * np.sin(2 * np.pi * f * t), label=f'{f} Hz')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
# Create signal: combination of nearby frequencies
freqs = [2, 4, 6, 8, 10]
amps = [8, 5, 3, 2, 1]
plot_individual_components(freqs, amps) 
signal = sum(A * np.sin(2 * np.pi * f * t) for f, A in zip(freqs, amps))

# Function to apply a window, plot the window + signal, then FFT
def apply_window_and_plot(signal, window, name):
    windowed_signal = signal * window
    freqs_fft = fftfreq(N, 1/sampling_rate)
    fft_vals = fft(windowed_signal)

    # Plot window + signal
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, label='Original Signal', alpha=0.6)
    plt.plot(t, window, label=f'{name} Window', linewidth=2)
    plt.plot(t, windowed_signal, label='Windowed Signal', alpha=0.8)
    plt.title(f'Time Domain with {name} Window')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(freqs_fft[:N//2], np.abs(fft_vals[:N//2]) * 2 / N)
    plt.title(f'FFT after {name} Windowing')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

rect_window = np.ones(N)
hann_window = hann(N)
hamming_window = hamming(N)

apply_window_and_plot(signal, rect_window, "Rectangular")
apply_window_and_plot(signal, hann_window, "Hann")
apply_window_and_plot(signal, hamming_window, "Hamming")

