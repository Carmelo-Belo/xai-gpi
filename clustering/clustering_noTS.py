import pandas as pd
import os
import argparse
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
from utils_clustering import perform_clustering_noTS

def main(basin, n_clusters, res, train_yearI, train_yearF):

    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    data_dir = os.path.join(project_dir, 'data')
    fs_data_dir = os.path.join(project_dir, 'xai-gpi', 'data')
    clustering_dir = os.path.join(project_dir, 'xai-gpi', 'clustering')
    # Create output directory
    path_output = os.path.join(fs_data_dir, f'{basin}_{n_clusters}clusters_noTS')
    os.makedirs(path_output, exist_ok=True)
    # Load dataframe containing information of variables to be clustered
    df_cluster_vars = pd.read_csv(os.path.join(clustering_dir, 'vars_dict.csv'))
    # Set resolution
    resolution = '{}x{}'.format(res, res)
    # Perform clustering
    for v, vars_info in df_cluster_vars.iterrows():
        var = vars_info['var']
        dataset = vars_info['dataset']
        vars_group = vars_info['vars_group']
        level = vars_info['level']
        if dataset == 'ERA5':
            path_predictor = os.path.join(data_dir, dataset, resolution, vars_group, dataset + '_' + vars_group)
        else:
            path_predictor = os.path.join(data_dir, dataset, resolution, vars_group, vars_group)

        # Clusters
        print(f'Clustering {var}, {level}')
        centroids, centroids_dataframe, clusters_av_dataframe, labels_dataframe = perform_clustering_noTS(var, level, basin, n_clusters, False, train_yearI, train_yearF, 
                                                                                                          resolution, path_predictor, path_output)

        # Update the var name for saving
        if (level != 'sfc') and (len(level) < 5):
            var = var + str(level)
        # If variable is defined between the difference of two pressure levels, select the difference level specified in the inputs
        elif (level != 'sfc') and (len(level) > 4):
            var = var + level
        # Save the data
        centroids_dataframe.to_csv(os.path.join(path_output, f'centroids_{var}.csv'))
        clusters_av_dataframe.to_csv(os.path.join(path_output, f'averages_{var}.csv'))
        labels_dataframe.to_csv(os.path.join(path_output, f'labels_{var}.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering of variables')
    parser.add_argument('--basin', type=str, help='Basin')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters')
    parser.add_argument('--res', type=float, default=2.5, help='Resolution')
    parser.add_argument('--train_yearI', type=int, default=1980, help='Initial year for training')
    parser.add_argument('--train_yearF', type=int, default=2013, help='Final year for training')

    args = parser.parse_args()
    main(args.basin, args.n_clusters, args.res, args.train_yearI, args.train_yearF)