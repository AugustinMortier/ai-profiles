import numpy as np
import utils
import xarray as xr
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

def run(ds, save):
    backscatter = ds['attenuated_backscatter_0']

    # Apply the simple saturation correction
    corrected_backscatter = utils.correct_saturation_abs(backscatter)

    # Apply a Gaussian filter to smooth the backscatter signal
    filtered_backscatter_np = gaussian_filter(corrected_backscatter, sigma=(0, 0))

    # Convert back to xarray DataArray
    filtered_backscatter = xr.DataArray(filtered_backscatter_np, dims=backscatter.dims, coords=backscatter.coords)

    # Thresholding to identify cloud regions in filtered backscatter
    cloud_threshold = 1.  # Increase the threshold to reduce noise
    cloud_mask = filtered_backscatter > cloud_threshold

    # Apply morphological operations to clean up the noise
    cloud_mask = binary_erosion(cloud_mask, structure=np.ones((3, 3)))  # Erosion to remove small noise
    cloud_mask = binary_dilation(cloud_mask, structure=np.ones((3, 3)))  # Dilation to restore cloud regions
    ds['cloud mask'] = xr.DataArray(cloud_mask, dims=backscatter.dims, coords=backscatter.coords)
    
    # plot that
    left = {'variable': 'attenuated_backscatter_0', 'title': 'Attenuated Backscatter', 'vmin': -1, 'vmax': 1, 'cmap': 'bwr'}
    right = {'variable': 'cloud_mask','title': 'cloud mask','vmin': 0,'vmax': 1,'cmap': 'Grays'}
    utils.plot_pannel(ds, left, right, save, method="method3")