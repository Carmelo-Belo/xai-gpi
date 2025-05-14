import os
import argparse
from utils_dataset import build_dataset

def main(basin, n_clusters, res, first_year, last_year, remove_seasonality, remove_trend):
    # List of variables to include for the feature selection process
    cluster_variables = ['abs_vo850', 'mpi', 'msl', 'r700', 'sst', 'vo850', 'vws850-200', 'w']
    climate_indexes = ['AMM', 'ENSO3.4', 'NAO', 'PDO', 'PNA', 'SOI', 'TNA', 'TSA', 'WP'] # EP-NP is equal to -99.9 on december

    # Return error if asking to build a dataset with both detrending and deseasonalization
    if remove_trend == 'y' and remove_seasonality == 'y':
        raise ValueError('To build dataset removing trend and seasonality, use the the script build_dataset_noTS.py')

    # Directories to consider to build the dataset for feature selection
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    if remove_seasonality == 'y':
        cluster_path = os.path.join(project_dir, 'tcg_fsM', 'data', f'{basin}_{n_clusters}clusters_deseason')
        deseasonalize = True
        detrend = False
    elif remove_trend == 'y':
        cluster_path = os.path.join(project_dir, 'tcg_fsM', 'data', f'{basin}_{n_clusters}clusters_detrend')
        deseasonalize = False
        detrend = True
    else:
        cluster_path = os.path.join(project_dir, 'tcg_fsM', 'data', f'{basin}_{n_clusters}clusters')
        deseasonalize = False
        detrend = False
    indexes_path = os.path.join(project_dir, 'data', 'CI')
    resolution = '{}x{}'.format(res, res)
    target_path = os.path.join(project_dir, 'data', 'IBTrACS', resolution, 'TCG', f'TCG_{resolution}')
    # Folder to save the dataset
    save_path = cluster_path

    # Build the dataset and save it to file
    dataset_filename = f'predictors_{first_year}-{last_year}_{n_clusters}clusters_{len(cluster_variables)}vars_{len(climate_indexes)}idxs.csv'
    target_filename = f'target_{first_year}-{last_year}_2.5x2.5.csv'
    deseason_target_filename = f'target_deseasonal_{first_year}-{last_year}_2.5x2.5.csv'
    detrend_target_filename = f'target_detrend_{first_year}-{last_year}_2.5x2.5.csv'
    seasonal_filename = f'target_seasonality_{first_year}-{last_year}_2.5x2.5.csv'
    trend_filename = f'target_trend_{first_year}-{last_year}_2.5x2.5.csv'
    if deseasonalize:
        dataset, target, deseason_target, seasonal = build_dataset(basin, cluster_variables, climate_indexes, cluster_path, indexes_path, target_path, first_year, last_year, deseasonalize, detrend, month_col=True)
        dataset.to_csv(os.path.join(save_path, dataset_filename))
        target.to_csv(os.path.join(save_path, target_filename))
        deseason_target.to_csv(os.path.join(save_path, deseason_target_filename))
        seasonal.to_csv(os.path.join(save_path, seasonal_filename))
    elif detrend:
        dataset, target, detrend_target, trend = build_dataset(basin, cluster_variables, climate_indexes, cluster_path, indexes_path, target_path, first_year, last_year, deseasonalize, detrend, month_col=True)
        dataset.to_csv(os.path.join(save_path, dataset_filename))
        target.to_csv(os.path.join(save_path, target_filename))
        detrend_target.to_csv(os.path.join(save_path, detrend_target_filename))
        trend.to_csv(os.path.join(save_path, trend_filename))
    else:
        dataset, target = build_dataset(basin, cluster_variables, climate_indexes, cluster_path, indexes_path, target_path, first_year, last_year, deseasonalize, detrend, month_col=True)
        dataset.to_csv(os.path.join(save_path, dataset_filename))
        target.to_csv(os.path.join(save_path, target_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset for feature selection')
    parser.add_argument('--basin', type=str, help='Basin')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters')
    parser.add_argument('--res', type=float, default=2.5, help='Resolution')
    parser.add_argument('--first_year', type=int, default=1980, help='First year')
    parser.add_argument('--last_year', type=int, default=2022, help='Last year')
    parser.add_argument('--remove_seasonality', type=str, default='n', help='If y remove seasonality')
    parser.add_argument('--remove_trend', type=str, default='n', help='If y remove trend')
    args = parser.parse_args()
    main(args.basin, args.n_clusters, args.res, args.first_year, args.last_year, args.remove_seasonality, args.remove_trend)