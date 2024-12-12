import os
import argparse
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/huripari/Documents/PhD/TCs_Genesis')
from utils import *

def main(res):
    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    save_data_dir = os.path.join(project_dir, 'FS_TCG', 'data')

    basins = ['GLB', 'NA', 'NEP', 'NWP', 'NI', 'SI', 'SP']

    # Compute the GPIS time series for each basin
    for b, basin in enumerate(basins):
        # Set the basin domain coordinates
        if basin == 'NWP':
            lon1, lon2, lat1, lat2 = 100, 180, 0, 40
        elif basin == 'NEP':
            lon1, lon2, lat1, lat2 = -180, -75, 0, 40
        elif basin == 'NA':
            lon1, lon2, lat1, lat2 = -100, 0, 0, 40
        elif basin == 'NI':
            lon1, lon2, lat1, lat2 = 45, 100, 0, 40
        elif basin == 'SP':
            lon1, lon2, lat1, lat2 = 135, -70, -40, 0
        elif basin == 'SI':
            lon1, lon2, lat1, lat2 = 35, 135, -40, 0
        elif basin == 'GLB':
            lon1, lon2, lat1, lat2 = -181, 181, -40, 40
        else:
            raise ValueError('basin not recognized')
        # Set the GPIS time series dataframe
        years = np.arange(1965, 2023) # dataset goes from 1965 to 2022
        time_index = pd.date_range(start=f'{years[0]}-01-01', end=f'{years[-1]}-12-01', freq='MS').astype('datetime64[ns]')
        df_gpis = pd.DataFrame(columns=['', 'engpi', 'ogpi'])
        df_gpis[''] = time_index
        df_gpis.set_index('', inplace=True)
        pd.to_datetime(df_gpis.index)
        # Cycle over the years, compute the monthly sum of GPIS and store it in a dataframe
        for y, year in enumerate(years):
            engpi = compute_engpi_rescaled(year, 2.5, lon1, lon2, lat1, lat2)
            ogpi = compute_ogpi_rescaled(year, 2.5, lon1, lon2, lat1, lat2)
            engpi_sum = engpi.sum(dim=['longitude', 'latitude'])
            ogpi_sum = ogpi.sum(dim=['longitude', 'latitude'])
            df_gpis.loc[df_gpis.index.isin(engpi_sum.time.values), 'engpi'] = engpi_sum.values
            df_gpis.loc[df_gpis.index.isin(ogpi_sum.time.values), 'ogpi'] = ogpi_sum.values

            print(f'\rBasin: {basin} ({b+1}/{len(basins)}) - Year: {year} ({y+1}/{len(years)})', end='')
        # Save the dataframe
        df_gpis.to_csv(os.path.join(save_data_dir, f'{basin}_{res}x{res}_gpis_time_series.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the GPIS time series for each basin')
    parser.add_argument('--res', type=int, default=2.5, help='Resolution of the dataset')
    args = parser.parse_args()
    main(args.res)