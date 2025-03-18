import os
import argparse
from utils_dataset import build_dataset_noTS

def main(basin, n_clusters, res, first_year, last_year):
    # List of variables to include for the feature selection process
    cluster_variables = ['abs_vo850', 'mpi', 'msl', 'r700', 'sst', 'vo850', 'vws850-200', 'w']
    climate_indexes = ['AMM', 'ENSO3.4', 'NAO', 'PDO', 'PNA', 'SOI', 'TNA', 'TSA', 'WP'] # EP-NP is equal to -99.9 on december

    # Directories to consider to build the dataset for feature selection
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    cluster_path = os.path.join(project_dir, 'FS_TCG', 'data', f'{basin}_{n_clusters}clusters_noTS')
    indexes_path = os.path.join(project_dir, 'data', 'CI')
    resolution = '{}x{}'.format(res, res)
    target_path = os.path.join(project_dir, 'data', 'IBTrACS', resolution, 'TCG', f'TCG_{resolution}')
    # Folder to save the dataset
    save_path = cluster_path

    # Build the dataset and save it to file
    dataset_filename = f'predictors_{first_year}-{last_year}_{n_clusters}clusters_{len(cluster_variables)}vars_{len(climate_indexes)}idxs.csv'
    target_filename = f'target_residual_{first_year}-{last_year}_2.5x2.5.csv'
    seasonal_filename = f'target_seasonality_{first_year}-{last_year}_2.5x2.5.csv'
    trend_filename = f'target_trend_{first_year}-{last_year}_2.5x2.5.csv'

    dataset, residual, trend, seasonal = build_dataset_noTS(basin, cluster_variables, climate_indexes, cluster_path, indexes_path, target_path, first_year, last_year, month_col=True)
    dataset.to_csv(os.path.join(save_path, dataset_filename))
    residual.to_csv(os.path.join(save_path, target_filename))
    seasonal.to_csv(os.path.join(save_path, seasonal_filename))
    trend.to_csv(os.path.join(save_path, trend_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset for feature selection')
    parser.add_argument('--basin', type=str, help='Basin')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters')
    parser.add_argument('--res', type=float, default=2.5, help='Resolution')
    parser.add_argument('--first_year', type=int, default=1980, help='First year')
    parser.add_argument('--last_year', type=int, default=2022, help='Last year')
    args = parser.parse_args()
    main(args.basin, args.n_clusters, args.res, args.first_year, args.last_year)