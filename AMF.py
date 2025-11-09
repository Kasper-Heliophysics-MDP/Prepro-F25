'''
===============================================================================
File: AMF.py
Author: Callen Fields (fcallen@umich.edu), Aashi Mishra (aashim@umich.edu)
Date: 2025-10-6
Group: University of Michigan SunRISE Mission

Description:
Performs adaptive medium filtering over a given spectrogram. At each pixel,
adaptively forms a kernel based off the angle of least change and a maximum
pixel distance of 15. Replaces each pixel value with the median of this kernel.

https://www.swsc-journal.org/articles/swsc/pdf/2018/01/swsc170092.pdf
===============================================================================
'''
import numpy as np
from scipy.ndimage import sobel, map_coordinates
import sys
from tqdm import tqdm

def AMF(spec, radius=15):
    """
    Adaptive Median Filtering
    Partial derivatives are calculated with a Sobel kernel.
    instead of using integer pixel coordinates, map_coordinates interpolates
    values between pixels to make the math work out better.
    For example, if you are at pixel (5, 5) and the angle is 30 degrees,
    the first pixel added to the kernel is (5 + (sqrt(3)/2), 5 + (1/2)).
    Bilinear interpolation is used to index into images at float coordinates.
    
    Args:
        spec (np.Arr): Frequency x Time spectrogram
        radius (int): maximum euclidean distance between the edge of the kernel and the starting pixel
        
    Returns:
        numpy arr: processed 2D spectrogram
    """    
    spec = np.asarray(spec, dtype=np.float32)  # reduce memory footprint

    # Gradients
    Gx = sobel(spec, axis=1, mode='reflect')
    Gy = sobel(spec, axis=0, mode='reflect')
    theta = np.arctan2(Gy, Gx) + np.pi / 2  # normal direction

    nrows, ncols = spec.shape
    filtered = np.zeros_like(spec, dtype=np.float32)

    # Precompute displacements along the normal direction
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    dx = np.cos(theta)
    dy = np.sin(theta)

    # Process by chunks to avoid excessive memory
    chunk_size = 2000  # adjust based on available RAM
    for start in tqdm(range(0, nrows, chunk_size), desc="AMF filtering"):
        end = min(start + chunk_size, nrows)
        rows = np.arange(start, end)[:, None]
        cols = np.arange(ncols)[None, :]

        # We'll collect samples for median computation, but only in this chunk
        local_vals = []

        for k in offsets:
            rr = rows + k * dy[start:end, :]
            cc = cols + k * dx[start:end, :]
            vals = map_coordinates(spec, [rr, cc], order=1, mode='reflect')
            local_vals.append(vals)

        # Stack and compute median for this chunk only
        stack_chunk = np.stack(local_vals, axis=-1)
        filtered[start:end, :] = np.median(stack_chunk, axis=-1)

        # Explicitly free memory before next chunk
        del stack_chunk, local_vals, rr, cc, vals

    return filtered

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python AMF.py <spectrogram.npy>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".npy", "-AMF.npy")
    spec = np.load(input_file)

    processed_spec = AMF(spec)
    np.save(output_file, processed_spec)

    print(f"Processed spectrogram saved to {output_file}")