"""
===============================================================================
File: one_day.py
Author: Callen Fields (fcallen@umich.edu)
Date: 2025-09-22
Group: University of Michigan SunRISE Mission

Description:
This script downloads all eCallisto recordings from a given station for a 
specific day, orders them in time (circularly starting from a specified UTC 
offset), concatenates them into a single 2D numpy array (frequency x time), 
and saves the resulting spectrogram as a .npy file. 
===============================================================================
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
from astropy.io import fits
import gzip
import re
from typing import List
import io
from tqdm import tqdm
import sys

BASE_URL = "https://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/"

def download_fits_from_gz(url: str) -> np.ndarray:
    """Download a .fit.gz URL and return the FITS data as a numpy array."""
    r = requests.get(url, stream=True)
    r.raise_for_status()

    # Decompress in memory
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
        with fits.open(io.BytesIO(gz.read())) as hdul:
            data = hdul[0].data
            return np.array(data)
        
def circular_sort(files: List[str], offset: str) -> List[str]:
    """
    The times on eCallisto are in UTC. So the beginning of the day locally
    is at an offset time, and the times are sequential from then.
    Circularly sort files by time (HHMMSS in filename) starting from offset.
    
    Args:
        files: list of filenames
        offset: string "HHMMSS" (UTC offset start)
        
    Returns:
        List[str]: circularly sorted list of filenames
    """
    
    def hhmmss_to_seconds(hhmmss: str) -> int:
        h, m, s = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
        return h * 3600 + m * 60 + s

    # regex to extract the middle part with time
    time_re = re.compile(r"_(\d{6})_")
    
    # map files to (time_in_seconds, filename)
    time_file_pairs = []
    for f in files:
        match = time_re.search(f)
        if match:
            t = hhmmss_to_seconds(match.group(1))
            time_file_pairs.append((t, f))
    
    # sort by time normally
    time_file_pairs.sort(key=lambda x: x[0])
    
    # compute offset in seconds
    offset_sec = hhmmss_to_seconds(offset)
    
    # rotate list so it starts from the first time >= offset
    times = [t for t, _ in time_file_pairs]
    idx = next((i for i, t in enumerate(times) if t >= offset_sec), 0)
    
    sorted_files = [f for _, f in time_file_pairs[idx:]] + [f for _, f in time_file_pairs[:idx]]
    
    return sorted_files

def one_day(station: str, year: int, month: int, day: int, time: str = "000000"):
    """
    Collects all eCallisto recordings from a given station on a given day
    Puts them in order, concatenates them, and returns them as a numpy array
    
    Args:
        station (str): The name of the eCallisto station 
        year (int): 
        month (int):
        day (int): The date to look at
        time (str): UTC time that designates the beginning of the day in "HHMMSS" format
        
    Returns:
        numpy arr: 2D spectrogram over all files found
    """
    #get the url from the date
    path = f"{year:04d}/{month:02d}/{day:02d}/"
    url = urljoin(BASE_URL.rstrip("/") + "/", path)

    #extract a list of all files
    response = requests.get(url)
    response.raise_for_status()  # raise error if bad request
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    files = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        # skip going back to parent dir
        if href in ("../", "./"):
            continue
        # construct absolute URL
        full_url = urljoin(url, href)
        files.append(full_url)

    station_files = [f for f in files if station in f] #extract only the files at this station
    sorted_files = circular_sort(station_files, time) #put them in order

    arrays = []
    for url in tqdm(sorted_files, desc="Downloading FITS files"):
        arr = download_fits_from_gz(url)
        if arr is not None:
            arrays.append(arr)

    if not arrays:
        raise ValueError("No valid FITS data found.")
    
    # Concatenate along time axis (axis=0)
    big_array = np.concatenate(arrays, axis=1)
    big_array = np.flipud(big_array)
    return big_array

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python one_day.py <station> <month> <day> <year>")
        sys.exit(1)

    station = sys.argv[1]
    year = sys.argv[4]
    month = sys.argv[2]
    day = sys.argv[3]
    data = one_day(station, year, month, day, "130000")
    file_name = "spec-" + station + "-" + str(month) + "-" + str(day) + "-" + str(year) + ".npy"
    np.save(file_name, data)


