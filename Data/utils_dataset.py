import numpy as np
import pandas as pd
import xarray as xr
import os

def check_consecutive_repeats(df,col):
    repeats = df.shift(1) == df
    if repeats.sum() > 0:
        print('Consecutive values repeated found at',col)
        print(repeats[repeats].index)

def build_dataset(cluster_variables, index_variables, cluster_path, indexes_path, target_path, first_year, last_year, month_col=True):
    
    # Create a dataframe containing the data for the climate indeces
    date_range = pd.date_range(start=f'{first_year}-01-01', end=f'{last_year}-12-01', freq='MS')
    df_indeces = pd.DataFrame(index=date_range, columns=index_variables)
    for climate_index in index_variables:
        filename = os.path.join(indexes_path, climate_index + '.txt')
        data = pd.read_table(filename, sep='\s+', header=None)
        for r, row in enumerate(df_indeces.iterrows()):
            idx = df_indeces.index[r]
            month = idx.month
            year = idx.year
            df_indeces.loc[idx, climate_index] = data[(data[0] == year)][month].values[0]

    # Load the cluster data and merge it in a single dataframe
    for v, var in enumerate(cluster_variables):
        filename = f'averages_{var}.csv'
        path = os.path.join(cluster_path, filename)
        if v == 0:
            dataset_cluster = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            dataset_cluster = pd.concat([dataset_cluster, pd.read_csv(path, index_col=0, parse_dates=True)], axis=1)

    # Merge the cluster and index dataframes
    dataset = pd.concat([dataset_cluster, df_indeces], axis=1)

    # Add a column containing the month of the year
    if month_col:
        dataset['month'] = dataset.index.month
    
    # Check if any data is missing, repeated in consecutive days, or is above the average+7*std
    for col in dataset.columns:
        if dataset[col].isnull().sum() > 0:
            print('Warning: Missing values in', col)
        check_consecutive_repeats(dataset[col],col)
        mean = dataset[col].mean()
        std = dataset[col].std()
        if (np.abs(dataset[col]) > mean + 7*std).sum() > 0:
            print('Warning: Values above the average+7*std in', col)

    # Build the dataframe for the target variable -> number of tropical cyclone genesis events per month
    years = np.arange(first_year, last_year+1, 1)
    tcg_ds = xr.concat([xr.open_dataset(target_path + f'_{year}.nc') for year in years], dim='time')
    target = pd.DataFrame(index=date_range)
    target['tcg'] = tcg_ds.tcg.sum(dim=['latitude', 'longitude']).values.astype(int)

    return dataset, target