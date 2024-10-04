import inspect
import os
import numpy as np
import utils
import xarray as xr
from sklearn.cluster import DBSCAN

def run(ds, options, save):
    
    # Set parameters
    thr_cloud = options['thr_cloud']
    thr_snr = options['thr_snr']
    eps = options['eps']
    min_samples = options['min_samples']
    
    # Apply SNR computation
    ds = utils.snr(ds, 'attenuated_backscatter_0', step=2, log=False)
    
    # Cloud detection
    cloud_mask = ds.where(ds.attenuated_backscatter_0 >= thr_cloud).where(ds.snr >= thr_snr)['attenuated_backscatter_0'].data
    
    # Remove NaN values (only keep valid points)
    valid_mask = ~np.isnan(cloud_mask)
    cloud_mask_valid = cloud_mask[valid_mask]
    
    # Flatten the time-altitude grid into a 2D array of points (only for valid points)
    altitudes = np.arange(cloud_mask.shape[1])
    times = np.arange(cloud_mask.shape[0])
    X, Y = np.meshgrid(times, altitudes)
    
    # Filter out points where cloud_mask is NaN
    X_valid = X.ravel()[valid_mask.ravel()]
    Y_valid = Y.ravel()[valid_mask.ravel()]
    
    # Create a dataset where points have (time, altitude, cloud_mask intensity)
    points = np.column_stack([X_valid, Y_valid, cloud_mask_valid.ravel()])
    
    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    
    # Reshape labels to match the original data dimensions (using NaN mask)
    clustered_cloud_mask = np.full(cloud_mask.shape, -1)  # Initialize with -1 for noise
    clustered_cloud_mask[valid_mask] = labels  # Assign labels to valid positions
    
    # Save the clustered result to the dataset
    ds['cloud_mask'] = xr.DataArray(clustered_cloud_mask, dims=ds.dims, coords=ds.coords)
    
    # Plot that
    left = {'variable': 'attenuated_backscatter_0', 'title': 'Attenuated Backscatter', 'vmin': -1, 'vmax': 5, 'cmap': 'bwr'}
    right = {'variable': 'cloud_mask', 'title': 'cloud mask', 'vmin': 0, 'vmax': 1, 'cmap': 'Grays'}
    path = os.path.relpath(inspect.getfile(run))
    method = path.split('methods/')[1].split('.py')[0]
    str_options = utils.str_options(options)
    utils.plot_panel(ds, left, right, save, method=f'{method}-{str_options}')