import pandas as pd
import os
import argparse
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"
from utils_clustering import perform_clustering

def main(basin, n_clusters, res, train_yearI, train_yearF, remove_seasonality, remove_trend):

    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    data_dir = os.path.join(project_dir, 'data')
    fs_data_dir = os.path.join(project_dir, 'xai-gpi', 'data')
    clustering_dir = os.path.join(project_dir, 'xai-gpi', 'clustering')

    # Return error if asking to cluster the dataset with both detrending and deseasonalization
    if remove_trend == 'y' and remove_seasonality == 'y':
        raise ValueError('To cluster dataset removing trend and seasonality, use the the script clustering_noTS.py')

    # Create output directory
    if remove_seasonality == 'y':
        path_output = os.path.join(fs_data_dir, f'{basin}_{n_clusters}clusters_deseason')
        deseasonalize = True
        detrend = False
    elif remove_trend == 'y':
        path_output = os.path.join(fs_data_dir, f'{basin}_{n_clusters}clusters_detrend')
        deseasonalize = False
        detrend = True
    else:  
        path_output = os.path.join(fs_data_dir, f'{basin}_{n_clusters}clusters')
        deseasonalize = False
        detrend = False
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
        centroids, centroids_dataframe, clusters_av_dataframe, labels_dataframe = perform_clustering(var, level, basin, n_clusters, train_yearI, train_yearF, resolution, 
                                                                                                     path_predictor, path_output, deseasonalize, detrend)

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
    parser.add_argument('--remove_seasonality', type=str, default='n', help='If y remove seasonality from the data')
    parser.add_argument('--remove_trend', type=str, default='n', help='If y remove trend from the data')

    args = parser.parse_args()
    main(args.basin, args.n_clusters, args.res, args.train_yearI, args.train_yearF, args.remove_seasonality, args.remove_trend)