# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 18:30:48 2025

@author: st241065
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal.windows import hann, hamming, triang  

sampling_rate = 400
T = 1
t = np.linspace(0, T, int(T * sampling_rate), endpoint=False)
N = len(t)

def plot_individual_components(frequencies, amplitudes):
    plt.figure(figsize=(12, 6))
    for i, (f, A) in enumerate(zip(frequencies, amplitudes), start=1):
        plt.plot(t, A * np.sin(2 * np.pi * f * t), label=f'{f} Hz')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title('Original Signal')
    plt.legend(loc="upper right", fontsize=8)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
freqs = [10, 20, 30, 40]
amps = [8, 4, 2, 1]
plot_individual_components(freqs, amps) 
signal = sum(A * np.sin(2 * np.pi * f * t) for f, A in zip(freqs, amps))

def apply_window_and_plot(signal, window, name):
    windowed_signal = signal * window
    freqs_fft = fftfreq(N, 1/sampling_rate)
    fft_vals = fft(windowed_signal) 

    plt.figure(figsize=(8, 4))
    plt.plot(t, signal, label='Original Signal', alpha=0.6, color='green')
    plt.plot(t, window, label=f'{name} Window', linewidth=2, color='orange')
    plt.plot(t, windowed_signal, label='Windowed Signal', alpha=0.8, color='blue')
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
triangular_window = triang(N + 1)[:-1]
hann_window = hann(N)
hamming_window = hamming(N)

apply_window_and_plot(signal, rect_window, "Rectangular")
apply_window_and_plot(signal, triangular_window, "Triangular")
apply_window_and_plot(signal, hann_window, "Hann")
apply_window_and_plot(signal, hamming_window, "Hamming")


def add_noise_and_plot_fft(signal, window, name, noise_level=0.5):
    windowed_signal = signal * window
    noise = np.random.normal(0, noise_level, size=windowed_signal.shape)
    noisy_signal = signal + noise
    noisy_windowed_signal = noisy_signal * window
    fft_vals = fft(noisy_windowed_signal)
    freqs_fft = fftfreq(N, 1/sampling_rate)

    plt.figure(figsize=(8, 4))
    plt.plot(t, noisy_signal, label='Original Signal', alpha=0.6, color='green')
    plt.plot(t, window, label=f'{name} Window', linewidth=2, color='orange')
    plt.plot(t, noisy_windowed_signal, label='Noisy Windowed Signal', alpha=0.8, color='blue')
    plt.title(f'Time Domain with {name} Window + Noise')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(freqs_fft[:N//2], np.abs(fft_vals[:N//2]) * 2 / N)
    plt.title(f'FFT after {name} Windowing with Noise')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

add_noise_and_plot_fft(signal, rect_window, "Rectangular", noise_level=1.0)
add_noise_and_plot_fft(signal, triangular_window, "Triangular", noise_level=1.0)
add_noise_and_plot_fft(signal, hann_window, "Hann", noise_level=1.0)
add_noise_and_plot_fft(signal, hamming_window, "Hamming", noise_level=1.0)