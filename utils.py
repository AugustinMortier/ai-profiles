import copy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from rich.progress import track
from pathlib import Path

def snr(ds: xr.Dataset, variable, step=4, verbose=False, log=True) -> xr.Dataset:
    """
    Method that calculates the Signal to Noise Ratio.
    
    Args:
        ds: Dataset.
        variable: variable to compute the SNR with
        step (int, optional): Number of steps around we calculate the SNR for a given altitude.
        verbose (bool, optional): Verbose mode.
        
    Returns:
        (xarray.DataArray).
    """

    # Get the dimensions of the data array
    time_len, altitude_len = ds[variable].shape
    
    # Preallocate the SNR array with zeros
    snr_array = np.zeros((time_len, altitude_len))
    
    for t in track(range(time_len), description="snr   ", disable=not verbose):
        # Extract 1D slice for current time step
        if log:
            array = np.log10(ds[variable].data[t, :])
        else:
            array = ds[variable].data[t, :]
        
        # Create a sliding window view for the rolling calculation
        sliding_windows = np.lib.stride_tricks.sliding_window_view(array, window_shape=2*step+1, axis=0)
        
        # Calculate mean and std across the window axis
        means = np.nanmean(sliding_windows, axis=1)
        stds = np.nanstd(sliding_windows, axis=1)
        
        # Handle the edges (where sliding window can't be applied due to boundary)
        means = np.pad(means, pad_width=(step,), mode='constant', constant_values=np.nan)
        stds = np.pad(stds, pad_width=(step,), mode='constant', constant_values=np.nan)
        
        # Avoid division by zero
        stds = np.where(stds == 0, np.nan, stds)
        
        # Compute SNR
        snr_array[t, :] = np.divide(means, stds, where=(stds != 0))
    
    # Create the DataArray
    ds["snr"] = (('time', 'altitude'), snr_array)
    ds["snr"] = ds.snr.assign_attrs({
        'long_name': 'Signal to Noise Ratio',
        'units': '',
        'step': step
    })
    
    return ds

def gaussian_filter(
        ds, sigma=0.25, var="attenuated_backscatter_0", inplace=False
    ):
        """
        Applies a 2D gaussian filter in order to reduce high frequency noise.

        Args:
            sigma (scalar or sequence of scalars, optional): Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.
            var (str, optional): variable name of the Dataset to be processed.
            inplace (bool, optional): if True, replace the instance of the (ProfilesData): class.

        Returns:
            (xarray.DataArray).
        """
        import copy
        from scipy.ndimage import gaussian_filter

        # apply gaussian filter
        filtered_data = gaussian_filter(ds[var].data, sigma=sigma)

        if inplace:
            ds[var].data = filtered_data
            new_ds = ds
        else:
            copied_dataset = copy.deepcopy(ds)
            copied_dataset[var].data = filtered_data
            new_ds = copied_dataset
        # add attribute
        new_ds[var].attrs["gaussian_filter"] = sigma
        return new_ds

def plot_panel(ds: xr.Dataset, left, right, save, method):
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot attenuated_backscatter_0 on the first subplot
    ds[left['variable']].transpose().plot(
        ax=axes[0], 
        vmin=left['vmin'], 
        vmax=left['vmax'], 
        cmap=left['cmap']
    )
    axes[0].set_title(left['title'])

    # Plot cloud filter on the second subplot
    ds[right['variable']].transpose().plot(
        ax=axes[1], 
        vmin=right['vmin'], 
        vmax=right['vmax'], 
        cmap=right['cmap']
    )
    axes[1].set_title(right['title'])

    # Adjust layout for better spacing
    plt.tight_layout()
    
    if save:
        station = f'{ds.attrs["wigos_station_id"]}-{ds.attrs["instrument_id"]}'
        date = str(ds.time[0].data).split('T')[0]
        Path('output','main').mkdir(parents=True, exist_ok=True)
        plt.savefig(Path('output', 'main', f'{station}-{date}-{method}.png'))
        plt.close()
    else:
        plt.show()

def correct_saturation_abs(backscatter, altitude_threshold=4000):
    """
    Apply a simple saturation correction by taking absolute values of the backscatter signal
    below the specified altitude threshold (default 4000m).
    
    Parameters:
        backscatter (xarray.DataArray): The backscatter signal to correct.
        altitude_threshold (float): Altitude threshold for applying the absolute correction.
    
    Returns:
        corrected_backscatter (xarray.DataArray): The backscatter signal with saturation correction.
    """
    # Apply absolute value below 4000m
    corrected_backscatter = backscatter.where(backscatter.altitude >= altitude_threshold, abs(backscatter))

    return corrected_backscatter

    """
    Apply an advanced saturation correction by detecting and adjusting negative values in
    the backscatter signal below the specified altitude threshold (default 4000m).
    
    Parameters:
        backscatter (xarray.DataArray): The backscatter signal to correct.
        altitude_threshold (float): Altitude threshold for applying the correction.
    
    Returns:
        corrected_backscatter (xarray.DataArray): The backscatter signal with advanced saturation correction.
    """
    # Detect negative values below 4000m and adjust them
    saturation_mask = (backscatter < 0) & (backscatter.altitude < altitude_threshold)
    
    # Apply a proportional correction (for example, we add a bias equal to the negative value)
    bias = -backscatter.where(saturation_mask, 0)
    corrected_backscatter = backscatter + bias
    
    return corrected_backscatter
    
def get_parameters_from_url(url):
    date = url.split('date=')[1].split('&')[0]
    yyyy, mm, dd = date[0:4], date[5:7], date[8:10]
    station_id = url.split('station_id=')[1]
    return yyyy, mm, dd, station_id

def str_options(dict):
    str_options = []
    for key in dict.keys():
        str_options.append(f'{key}:{dict[key]}')
    return '-'.join(str_options)