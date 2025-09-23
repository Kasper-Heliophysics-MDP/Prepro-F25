"""
===============================================================================
File: plot_spectrogram.py
Author: Callen Fields (fcallen@umich.edu), Aashi Mishra (aashim@umich.edu)
Date: 2025-09-22
Group: University of Michigan SunRISE Mission

Description:
This script loads a 2D spectrogram stored as a .npy file and plots it using 
matplotlib. The spectrogram is displayed with frequency on the y-axis and 
time on the x-axis, with intensity represented by color. 
"""
import sys
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrogram(big_array: np.ndarray, cmap="viridis"):
    """Plot a spectrogram: frequency (y) vs. time (x), intensity in color."""
    plt.figure(figsize=(12, 6))
    plt.imshow(
        big_array,
        aspect="auto",
        origin="lower",
        cmap=cmap
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("Time index")
    plt.ylabel("Frequency bin")
    plt.title("Spectrogram")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_spectrogram.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = np.load(file_path)
    plot_spectrogram(data)