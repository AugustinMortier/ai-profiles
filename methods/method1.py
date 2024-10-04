import inspect
import os
import utils
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

def run(ds, options, save):
    
    # set parameters
    thr_cloud = options['thr_cloud']
    thr_snr = options['thr_snr']
    
    ds = utils.snr(ds, 'attenuated_backscatter_0', step=2, log=False)

    cloud_mask = ds.where(ds.attenuated_backscatter_0>=thr_cloud).where(ds.snr>=thr_snr)['attenuated_backscatter_0'].data
    ds['cloud_mask'] = xr.DataArray(cloud_mask, dims=ds.dims, coords=ds.coords)

    # plot that
    left = {'variable': 'attenuated_backscatter_0', 'title': 'Attenuated Backscatter', 'vmin': -1, 'vmax': 5, 'cmap': 'bwr'}
    right = {'variable': 'cloud_mask','title': 'cloud mask','vmin': 0,'vmax': 1,'cmap': 'Grays'}
    path = os.path.relpath(inspect.getfile(run))
    method = path.split('methods/')[1].split('.py')[0]
    str_options = utils.str_options(options)
    utils.plot_panel(ds, left, right, save, method=f'{method}-{str_options}')

