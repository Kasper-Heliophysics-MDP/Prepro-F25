import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/"

import re
from typing import List

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

def list_files(station: str, year: int, month: int, day: int, time: str = "000000"):
    """
    Fetch all file links from a given URL (assuming it's a directory listing).
    
    Args:
        station (str): The name of the eCallisto station 
        year (int): 
        month (int):
        day (int): The date to look at
        time (str): UTC time that designates the beginning of the day in "HHMMSS" format
        
    Returns:
        list[str]: A list of absolute file URLs found on the page.
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

    return sorted_files

# Example usage:
if __name__ == "__main__":
    print(list_files("ALASKA-ANCHORAGE", 2025, 5, 13, "130000"))

