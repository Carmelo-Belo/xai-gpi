import numpy as np
import pandas as pd
import xarray as xr
import os
import glob

def main(basin_dir, var_name, std_name, long_name, units):
    
    # Check if var_name is a surface, a level variable or a diff level variable
    if var_name in ['msl', 'mpi', 'sst', 'ssta20', 'ssta30', 'w']:
        lev_var = False
        diff_lev_var = False
    elif var_name in ['abs_vo', 'mgu', 'mgv', 'r', 'vo', 'zgu', 'zgv']:
        lev_var = True
        diff_lev_var = False
    elif var_name in ['vws', 'vws_u', 'vws_v']:
        lev_var = False
        diff_lev_var = True
    else:
        raise ValueError(f"Variable {var_name} not recognized")
    
    # Get levels values if lev_var is True
    if lev_var:
        if var_name in ['abs_vo', 'vo']:
            var_levels = [1000., 850., 600., 500.]
        elif var_name in ['mgu', 'mgv', 'zgu', 'zgv']:
            var_levels = [850., 600., 500., 250., 200., 50.]
        elif var_name in ['r']:
            var_levels = [1000., 900., 850., 800., 700., 600., 500., 400., 300., 200., 100., 50.]
        lev_coord = 'level'
    elif diff_lev_var:
        var_levels = ['850-250', '850-200', '600-250', '500-250', '500-200', '250-50']
        lev_coord = 'diff_level'

    # Load an average, centroid and label file to create the xarray.Dataset
    avg_filepaths = glob.glob(os.path.join(basin_dir, f'averages_{var_name}*.csv'))
    avg_filepaths.sort()
    avgs = pd.read_csv(avg_filepaths[0], index_col=0, parse_dates=True)
    centr_filepaths = glob.glob(os.path.join(basin_dir, f'centroids_{var_name}*.csv'))
    centr_filepaths.sort()
    centr = pd.read_csv(centr_filepaths[0], index_col=0, parse_dates=True)
    label_filepaths = glob.glob(os.path.join(basin_dir, f'labels_{var_name}*.csv'))
    label_filepaths.sort()
    labels = pd.read_csv(label_filepaths[0], index_col=0)

    # If variable is 'vws' need to drop paths containing 'u' or 'v'
    if var_name == 'vws':
        avg_filepaths = [path for path in avg_filepaths if ('vws_u' not in path) and ('vws_v' not in path)]
        centr_filepaths = [path for path in centr_filepaths if ('vws_u' not in path) and ('vws_v' not in path)]
        label_filepaths = [path for path in label_filepaths if ('vws_u' not in path) and ('vws_v' not in path)]

    # Create the xarray.Dataset
    lats = np.unique(labels['nodes_lat'].to_numpy())
    lons = np.unique(labels['nodes_lon'].to_numpy())
    if not lev_var and not diff_lev_var:
        ds = xr.Dataset(
            coords = {
                'time': avgs.index,
                'cluster': np.arange(1, len(avgs.columns)+1),
                'latitude': lats,
                'longitude': lons
            },
            data_vars = {
                'averages': (['time', 'cluster'], np.zeros((len(avgs.index), len(avgs.columns)))),
                'centroids': (['time', 'cluster'], np.zeros((len(centr.index), len(centr.columns)))),
                'labels': (['latitude', 'longitude'], np.zeros((len(lats), len(lons))))
            }
        )
    else:
        ds = xr.Dataset(
            coords = {
                'time': avgs.index,
                'cluster': np.arange(1, len(avgs.columns)+1),
                lev_coord: var_levels,
                'latitude': lats,
                'longitude': lons
            },
            data_vars = {
                'averages': (['time', 'cluster', lev_coord], np.zeros((len(avgs.index), len(avgs.columns), len(var_levels)))),
                'centroids': (['time', 'cluster', lev_coord], np.zeros((len(centr.index), len(centr.columns), len(var_levels)))),
                'labels': (['latitude', 'longitude'], np.zeros((len(lats), len(lons))))
            }
        )

    # Set attributes of the dataset and of the variables
    ds.attrs.update({
        "Conventions": "CF-1.6",
        "title": f"North Atlantic cluster data for {long_name}",
        "source": "ERA5",
        "creator": "Filippo Dainelli",
        "institution": "Politecnico di Milano",
        "creation_date": "2025-09-02"
    })
    ds['averages'].attrs.update({
        "long_name": f"Area-weighted spatial means of each cluster for {long_name}",
        "standard_name": f"area_weighted_mean_{std_name}",
        "units": units
    })
    ds['centroids'].attrs.update({
        "long_name": f"Centroid points values of each cluster for {long_name}",
        "standard_name": f"centroid_{std_name}",
        "units": units
    })
    ds['labels'].attrs.update({
        "long_name": "cluster number to which the point of latitude and longitude coordinates belongs",
        "standard_name": "cluster_number",
        "units": "1"
    })

    # Fill the dataset with the data
    for averages_filepath, centroids_filepath, labels_filepath in zip(avg_filepaths, centr_filepaths, label_filepaths):
        # Load csvs
        avgs = pd.read_csv(averages_filepath, index_col=0, parse_dates=True)
        centr = pd.read_csv(centroids_filepath, index_col=0, parse_dates=True)
        labels = pd.read_csv(labels_filepath, index_col=0)
        # Fill the dataset with the data
        if diff_lev_var:
            lev = averages_filepath.split(var_name)[-1].split('.')[0]
            for c, col in enumerate(avgs.columns):
                ds['averages'].loc[dict(time=slice(None), cluster=c+1, diff_level=lev)] = avgs[col].to_numpy()
                ds['centroids'].loc[dict(time=slice(None), cluster=c+1, diff_level=lev)] = centr[col].to_numpy()
            for r, row in labels.iterrows():
                ds['labels'].loc[dict(latitude=row['nodes_lat'], longitude=row['nodes_lon'])] = row['cluster']    
        elif lev_var:
            lev = averages_filepath.split(var_name)[-1].split('.')[0]
            if lev_var:
                lev = float(lev)
            for c, col in enumerate(avgs.columns):
                ds['averages'].loc[dict(time=slice(None), cluster=c+1, level=lev)] = avgs[col].to_numpy()
                ds['centroids'].loc[dict(time=slice(None), cluster=c+1, level=lev)] = centr[col].to_numpy()
            for r, row in labels.iterrows():
                ds['labels'].loc[dict(latitude=row['nodes_lat'], longitude=row['nodes_lon'])] = row['cluster']
        else:
            for c, col in enumerate(avgs.columns):
                ds['averages'].loc[dict(time=slice(None), cluster=c+1)] = avgs[col].to_numpy()
                ds['centroids'].loc[dict(time=slice(None), cluster=c+1)] = centr[col].to_numpy()
            for r, row in labels.iterrows():
                ds['labels'].loc[dict(latitude=row['nodes_lat'], longitude=row['nodes_lon'])] = row['cluster']

    # Save the dataset
    ds.to_netcdf(os.path.join(basin_dir, f'{var_name}_1970-2022.nc'))

basin_dir = '/work/bk1318/b382153/FS_TCG/data_final/SP_9clusters'

# Create dataset for surface variables
var_name = 'msl'
std_name = 'mean_sea_level_pressure'
long_name = 'mean sea level pressure'
units = 'hPa'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'mpi'
std_name = 'max_potential_intensity'
long_name = 'maximum potential intensity'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'sst'
std_name = 'sea_surface_temperature'
long_name = 'sea surface temperature'
units = '°C'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'ssta20'
std_name = 'sea_surface_temperature_anomaly_20'
long_name = 'sea surface temperature anomaly from (20°N-20°S) mean'
units = '°C'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'ssta30'
std_name = 'sea_surface_temperature_anomaly_30'
long_name = 'sea surface temperature anomaly from (30°N-30°S) mean'
units = '°C'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'w'
std_name = 'vertical_velocity_500hPa'
long_name = 'vertical velocity at 500hPa'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

# Create dataset for level variables
var_name = 'abs_vo'
std_name = 'absolute_vorticity'
long_name = 'absolute vorticity'
units = 's-1'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'mgu'
std_name = 'meridional_gradient_u'
long_name = 'meridional gradient of zonal wind'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'mgv'
std_name = 'meridional_gradient_v'
long_name = 'meridional gradient of meridional wind'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'zgu'
std_name = 'zonal_gradient_u'
long_name = 'zonal gradient of zonal wind'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'zgv'
std_name = 'zonal_gradient_v'
long_name = 'zonal gradient of meridional wind'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'r'
std_name = 'relative_humidity'
long_name = 'relative humidity'
units = '%'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'vo'
std_name = 'vorticity'
long_name = 'vorticity'
units = 's-1'
main(basin_dir, var_name, std_name, long_name, units)

# Create dataset for diff level variables
var_name = 'vws'
std_name = 'vertical_wind_shear'
long_name = 'vertical wind shear between pressure levels'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'vws_u'
std_name = 'vertical_wind_shear_u'
long_name = 'zonal component of vertical wind shear between pressure levels'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

var_name = 'vws_v'
std_name = 'vertical_wind_shear_v'
long_name = 'meridional component of vertical wind shear between pressure levels'
units = 'm/s'
main(basin_dir, var_name, std_name, long_name, units)

# Change the target file format from csv to netcdf
target_csv = pd.read_csv(os.path.join(basin_dir, 'target_1970-2022_2.5x2.5.csv'), index_col=0, parse_dates=True)

# Create xarray Dataset
target_ds = xr.Dataset(
    coords={
        'time': target_csv.index
    },
    data_vars={
        'tcg': (['time'], target_csv['tcg'].to_numpy())
    }
)

# Add global attributes
target_ds.attrs.update({
    "Conventions": "vr04r01",
    "title": "North Atlantic monthly number of tropical cyclogenesis",
    "source": "International Best Track Archive for Climate Stewardship (IBTrACS)",
    "creator": "Filippo Dainelli",
    "institution": "Politecnico di Milano",
    "creation_date": "2025-09-02"
})

# Add variable attributes
target_ds['tcg'].attrs.update({
    "long_name": "number of cyclogenesis events in the basin",
    "standard_name": "number_of_cyclogenesis",
    "units": "1"
})

# Save to NetCDF
target_ds.to_netcdf(os.path.join(basin_dir, 'tcg_1970-2022.nc'))
