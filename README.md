# Prepro-F25 – University of Michigan SunRISE Mission

**Authors:** Callen Fields, Aashi Mishra  

---

## Overview

This repository contains Python scripts for downloading, processing, and visualizing **eCallisto solar radio spectrograms**

## File Descriptions

### `one_day.py`

- **Purpose:** Download all recordings from a given eCallisto station for a specific day, concatenate them into a single array, and save as a `.npy` file.
- **Key Functions:**
  - `download_fits_from_gz(url: str) -> np.ndarray`: Downloads and reads a `.fit.gz` file into a NumPy array.
  - `circular_sort(files: List[str], offset: str) -> List[str]`: Sorts filenames by UTC time, starting from a given offset.
  - `one_day(station, year, month, day, time="000000")`: Downloads, orders, concatenates, and flips arrays for a single day.
- **Usage:**

     `python one_day.py <station> <month> <day> <year>` 

- **Example:**

    `python one_day.py ALASKA-ANCHORAGE 5 13 2025`

    This saves a file called spec-ALASKA-ANCHORAGE-5-13-2025.npy.

### `plot_spectrogram.py`

- **Purpose:** Load a `.npy` spectrogram file and display it using matplotlib.
- **Key Function:**
    - `plot_spectrogram(big_array: np.ndarray, cmap="viridis")`: Plots a 2D spectrogram with frequency vs. time.

- **Usage:**

    `python plot_spectrogram.py <file_path>`

- **Example:**

    `python plot_spectrogram.py spec-CALLISTO_Alaska-5-13-2025.npy`

## Requirements

Install dependencies with:
`pip install -r requirements.txt`

