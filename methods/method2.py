import utils

def run(ds, save):
    ds = utils.gaussian_filter(ds, sigma=0.5)
    ds = utils.snr(ds, 'attenuated_backscatter_0', step=2, log=True)

    thr_cloud = 1.5; thr_snr = 0.5
    ds['cloud_mask'] = ds['attenuated_backscatter_0'].where(ds.attenuated_backscatter_0>=thr_cloud).where(ds.snr>=thr_snr)

    # plot that
    left = {'variable': 'attenuated_backscatter_0', 'title': 'Attenuated Backscatter', 'vmin': -1, 'vmax': 1, 'cmap': 'bwr'}
    right = {'variable': 'cloud_mask','title': 'cloud mask','vmin': 0,'vmax': 1,'cmap': 'Grays'}
    utils.plot_pannel(ds, left, right, save, method="method2")