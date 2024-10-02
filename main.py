import xarray as xr
import utils, methods.method1 as method1, methods.method2 as method2

urls = [
    '?date=2024-06-06&station_id=0-20008-0-UGR-A', # strong dust event
    '?date=2024-09-30&station_id=0-20008-0-UGR-A', # aloft dust layer
    '?date=2024-09-28&station_id=0-250-1001-07151-B', # low and high clouds
    '?date=2024-09-28&dev&lr=IFS&station_id=0-620-3704-5480-B', # high clouds
    '?date=2024-09-28&dev&lr=IFS&station_id=0-20000-0-01492-A', # low clouds
    '?date=2024-09-28&dev&lr=IFS&station_id=0-20000-0-01311-A', # clouds and precipitations
    '?date=2024-09-30&dev&lr=IFS&station_id=0-20000-0-01311-A', # low and mid clouds
    '?date=2024-09-30&dev&lr=IFS&station_id=0-20000-0-01001-A', #thin low clouds
]
vars = ['attenuated_backscatter_0']

for url in urls:
    print(f'https://vprofiles.met.no/{url}')
    yyyy, mm, dd, station_id = utils.get_parameters_from_url(url)
    path = f'../data/{yyyy}/{mm}/{dd}/AP_{station_id}-{yyyy}-{mm}-{dd}.nc'
    
    # read file
    ds = xr.open_dataset(path)[vars].load()
    
    # prerequesite: desaturate attenuated backscatter file
    #ds['attenuated_backscatter_0'] = utils.correct_saturation_abs(ds['attenuated_backscatter_0'])
    
    # run methods
    method1.run(ds, thr_cloud=1, thr_snr=0.5, save=True)
    method2.run(ds, thr_cloud=1, thr_snr=0.5, save=True)
    #method2.run(ds, thr_cloud=1.5, save=True)
