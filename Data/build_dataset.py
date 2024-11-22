import os
import argparse
from utils_dataset import build_dataset

def main(basin, n_clusters, anomaly_clustering, res, first_year, last_year):
    # List of variables to include for the feature selection process
    cluster_variables = ['abs_vo850', 'mpi', 'msl', 'r700', 'sst', 'vo850', 'vws850-200', 'w']
    climate_indexes = ['EA-WR', 'ENSO3.4', 'EP-NP', 'NAO', 'PDO', 'PNA', 'SOI', 'TNA', 'TSA', 'WP']

    # Directories to consider to build the dataset for feature selection
    project_dir = '/home/simul6/Documentos/Filippo/'
    if anomaly_clustering == 'y':
        cluster_path = os.path.join(project_dir, 'FS_TCG', 'data', f'{basin}_{n_clusters}clusters_anomaly')
    else:
        cluster_path = os.path.join(project_dir, 'FS_TCG', 'data', f'{basin}_{n_clusters}clusters')
    indexes_path = os.path.join(project_dir, 'data', 'CI')
    resolution = '{}x{}'.format(res, res)
    target_path = os.path.join(project_dir, 'data', 'IBTrACS', resolution, 'TCG', f'TCG_{resolution}')
    # Folder to save the dataset
    save_path = cluster_path

    # Build the dataset and save it to file
    dataset, target = build_dataset(basin, cluster_variables, climate_indexes, cluster_path, indexes_path, target_path, first_year, last_year, month_col=True)
    dataset_filename = f'predictors_{first_year}-{last_year}_{n_clusters}clusters_{len(cluster_variables)}vars_{len(climate_indexes)}idxs.csv'
    dataset.to_csv(os.path.join(save_path, dataset_filename))
    target_filename = f'target_{first_year}-{last_year}_2.5x2.5.csv'
    target.to_csv(os.path.join(save_path, target_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset for feature selection')
    parser.add_argument('--basin', type=str, default='GLB', help='Basin')
    parser.add_argument('--n_clusters', type=int, default=8, help='Number of clusters')
    parser.add_argument('--anomaly_clustering', type=str, default='y', help='If y retrieve dataset of anomaly clustering')
    parser.add_argument('--res', type=float, default=2.5, help='Resolution')
    parser.add_argument('--first_year', type=int, default=1965, help='First year')
    parser.add_argument('--last_year', type=int, default=2022, help='Last year')
    args = parser.parse_args()
    main(args.basin, args.n_clusters, args.res, args.first_year, args.last_year)