# Telecommunication Project

A signal processing project exploring Fourier analysis, frequency domain analysis, and signal windowing techniques for telecommunications applications.

## Overview

This project demonstrates fundamental concepts in digital signal processing (DSP), including:
- **Frequency domain analysis** using Fast Fourier Transform (FFT)
- **Signal composition** from multiple frequency components
- **Window functions** and their effects on spectral analysis
- **Noise analysis** in signal processing

## Project Structure

The repository contains two main analysis scripts:

### `telecommunication_1.py`
Explores three different frequency scenarios through time-domain and frequency-domain analysis:

- **Scenario 1: Low-Frequency Signals**
  - Frequencies: 2, 4, 6, 10, 16, 26, 42, 68, 110 Hz
  - Demonstrates signal composition and FFT analysis of low-frequency components

- **Scenario 2: Mid-Frequency Dominant**
  - Frequencies: 50-700 Hz
  - Shows mid-range frequency content with varying amplitudes

- **Scenario 3: High-Frequency Dominant**
  - Frequencies: 1000-2400 Hz
  - Illustrates high-frequency signal characteristics

### `telecommunication_2.py`
Focuses on window functions and their impact on spectral analysis:

- **Window Functions Analyzed:**
  - Rectangular Window (no windowing)
  - Triangular Window
  - Hann Window
  - Hamming Window

- **Features:**
  - Visualization of window effects in time domain
  - FFT analysis with different window functions
  - Noise robustness analysis with varying noise levels

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib
