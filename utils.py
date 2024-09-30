import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from rich.progress import track

def snr(ds: xr.Dataset, variable, step=4, verbose=False) -> xr.Dataset:
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

def plot_pannel(ds: xr.Dataset, left, right):
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot attenuated_backscatter_0 on the first subplot
    ds[left['variable']].transpose().plot(
        ax=axes[0], 
        vmin=left['vmin'], 
        vmax=left['vmax'], 
        cmap=left['cmap']
    )
    axes[0].set_title(left['title'])

    # Plot snr on the second subplot
    ds[right['variable']].transpose().plot(
        ax=axes[1], 
        vmin=right['vmin'], 
        vmax=right['vmax'], 
        cmap=right['cmap']
    )
    axes[1].set_title(right['title'])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()