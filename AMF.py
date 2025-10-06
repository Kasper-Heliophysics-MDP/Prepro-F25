import numpy as np
from scipy.ndimage import sobel, map_coordinates
import sys
from tqdm import tqdm

def AMF(spec, radius=15):
    """
    Apply a directional median filter along the local gradient (normal) direction.

    Parameters
    ----------
    spec : np.ndarray
        2D array (frequency x time)
    max_dist : int
        Maximum Euclidean distance (in pixels) to include in kernel.
    angle_tol : float
        Angular tolerance (radians) for points considered "along" the normal vector.

    Returns
    -------
    filtered : np.ndarray
        Directionally median-filtered spectrogram.
    """
    # Compute gradients (Sobel kernel for partial derivative estimates)
    Gx = sobel(spec, axis=1, mode='reflect')
    Gy = sobel(spec, axis=0, mode='reflect')

    # Compute normal vector angles
    theta = np.arctan2(Gy, Gx) + np.pi / 2  # normal direction (angle of least change)

    nrows, ncols = spec.shape
    filtered = np.zeros_like(spec)

    # Precompute displacements along the normal direction
    offsets = np.linspace(-radius, radius, 2 * radius + 1)
    dx = np.cos(theta)
    dy = np.sin(theta)

    # For speed: iterate coarsely over the image grid
    rows, cols = np.indices(spec.shape)

    for k in tqdm(range(-radius, radius + 1)):
        # Displace coordinates along the normal vector
        rr = rows + k * dy
        cc = cols + k * dx

        # Sample using bilinear interpolation
        values = map_coordinates(spec, [rr, cc], order=1, mode='reflect')

        if k == -radius:
            stack = np.expand_dims(values, axis=-1)
        else:
            stack = np.concatenate((stack, np.expand_dims(values, axis=-1)), axis=-1)

    # Compute median across the kernel dimension
    filtered = np.median(stack, axis=-1)

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