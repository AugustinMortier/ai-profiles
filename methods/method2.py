import inspect
import os
import utils
import numpy as np
import xarray as xr
from scipy.ndimage import median_filter, gaussian_filter, binary_erosion, binary_dilation

def run(ds, options, save):
    
    # set parameters
    thr_cloud = options['thr_cloud']
    thr_snr = options['thr_snr']
    
    ds = utils.snr(ds, 'attenuated_backscatter_0', step=2, log=False)
    backscatter = ds['attenuated_backscatter_0'].data
    backscatter = median_filter(backscatter, size=(3, 3))
    ds['attenuated_backscatter_0'] = xr.DataArray(backscatter, dims=ds.dims, coords=ds.coords)
    
    cloud_mask = ds.where(ds.attenuated_backscatter_0>=thr_cloud).where(ds.snr>=thr_snr)['attenuated_backscatter_0'].data
    #cloud_mask = median_filter(cloud_mask, size=(3, 3))
    
    # Apply morphological operations to clean up the noise
    #binary_mask = cloud_mask > 0
    #cloud_mask = binary_erosion(binary_mask, structure=np.ones((2, 2)))  # Erosion to remove small noise
    #cloud_mask = binary_dilation(binary_mask, structure=np.ones((2, 2)))  # Dilation to restore cloud regions
    
    #ds['cloud_mask'] = ds.attenuated_backscatter_0.where(cloud_mask == True)
    ds['cloud_mask'] = xr.DataArray(cloud_mask, dims=ds.dims, coords=ds.coords)
    #print(ds['cloud_mask'])
    

    # plot that
    left = {'variable': 'attenuated_backscatter_0', 'title': 'Attenuated Backscatter', 'vmin': -1, 'vmax': 5, 'cmap': 'bwr'}
    right = {'variable': 'cloud_mask','title': 'cloud mask','vmin': 0,'vmax': 1,'cmap': 'Grays'}
    path = os.path.relpath(inspect.getfile(run))
    method = path.split('methods/')[1].split('.py')[0]
    str_options = utils.str_options(options)
    utils.plot_panel(ds, left, right, save, method=f'{method}-{str_options}')
