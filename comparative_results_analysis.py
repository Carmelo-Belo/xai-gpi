import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils_results as ut

def main(basin, n_vars, n_idxs):
    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    results_dir = os.path.join(fs_dir, 'results')
    basin_results_dir = os.path.join(results_dir, basin)
    # Set the acronyms names of the cluster variables and the indexes
    atm_vars = ['abs_vo850', 'mpi', 'msl', 'r700', 'sst', 'vo850', 'vws850-200', 'w']
    idx_vars = ['EA-WR', 'ENSO3.4', 'EP-NP', 'NAO', 'PDO', 'PNA', 'SOI', 'TNA', 'TSA', 'WP']

    ## Selection trends without distinguishing in clusters (comparing all experiments, experiments filtered by cluster method, or by model fitted) ## 
    # List all the experiments containing the same candidates dataset
    all_subfolders = os.listdir(basin_results_dir)
    experiments_folders = [f for f in all_subfolders if f'nv{n_vars}_nd{n_idxs}' in f]
    experiments_folders.sort()
    # Define list to store experiments folders according to their charasteristics (clustering method, model kind)
    experiments_folders_NC = []
    experiments_folders_AC = []
    experiments_folders_linreg = []
    experiments_folders_lgbm = []
    experiments_folders_linreg_NC = []
    experiments_folders_lgbm_NC = []
    experiments_folders_linreg_AC = []
    experiments_folders_lgbm_AC = [] 
    # Define lists to store the selected variables dataframes
    selected_vars_df_list = []
    selected_vars_df_list_NC = []
    selected_vars_df_list_AC = []
    selected_vars_df_list_linreg = []
    selected_vars_df_list_lgbm = []
    selected_vars_df_list_linreg_NC = []
    selected_vars_df_list_lgbm_NC = []
    selected_vars_df_list_linreg_AC = []
    selected_vars_df_list_lgbm_AC = []
    # Extract variable selection
    for experiment_name in experiments_folders:
        model_kind = experiment_name.split('_')[1]
        nc_string = experiment_name.split('_')[2]
        numbers = filter(str.isdigit, nc_string)
        nc = ''.join(numbers)
        experiment_filename = f'1965-2022_{nc}clusters_{n_vars}vars_{n_idxs}idxs.csv'
        best_solution = pd.read_csv(os.path.join(basin_results_dir, experiment_name, f'best_solution_{model_kind}_{experiment_filename}'), sep=',', header=None)
        if "A" in nc_string:
            data_dir = os.path.join(fs_dir, 'data', f'{basin}_{nc}clusters_anomaly')
        else:
            data_dir = os.path.join(fs_dir, 'data', f'{basin}_{nc}clusters')
        predictors_df = pd.read_csv(os.path.join(data_dir, f'predictors_{experiment_filename}'), index_col=0)
        predictors_df.index = pd.to_datetime(predictors_df.index)
        selected_vars_df = ut.df_selected_vars(predictors_df, best_solution)
        selected_vars_df_list.append(selected_vars_df)
        if "linreg" in experiment_name:
            experiments_folders_linreg.append(experiment_name)
            selected_vars_df_list_linreg.append(selected_vars_df)
        elif "lgbm" in experiment_name:
            experiments_folders_lgbm.append(experiment_name)
            selected_vars_df_list_lgbm.append(selected_vars_df)
        if "A" in nc_string:
            experiments_folders_AC.append(experiment_name)
            selected_vars_df_list_AC.append(selected_vars_df)
            if "linreg" in experiment_name:
                experiments_folders_linreg_AC.append(experiment_name)
                selected_vars_df_list_linreg_AC.append(selected_vars_df)
            elif "lgbm" in experiment_name:
                experiments_folders_lgbm_AC.append(experiment_name)
                selected_vars_df_list_lgbm_AC.append(selected_vars_df)
        else:
            experiments_folders_NC.append(experiment_name)
            selected_vars_df_list_NC.append(selected_vars_df)
            if "linreg" in experiment_name:
                experiments_folders_linreg_NC.append(experiment_name)
                selected_vars_df_list_linreg_NC.append(selected_vars_df)
            elif "lgbm" in experiment_name:
                experiments_folders_lgbm_NC.append(experiment_name)
                selected_vars_df_list_lgbm_NC.append(selected_vars_df)    
    # Plot heatmaps for variable selection considering all experiments, and then all experiments for linear regression and LightGBM
    comp_results_dir = os.path.join(fs_dir, 'results', 'comparative_analysis')
    os.makedirs(comp_results_dir, exist_ok=True)
    basin_comp = os.path.join(comp_results_dir, basin)
    os.makedirs(basin_comp, exist_ok=True)
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders, selected_vars_df_list, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'all_heatmaps_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders_linreg, selected_vars_df_list_linreg, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'linreg_heatmaps_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders_lgbm, selected_vars_df_list_lgbm, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'lgbm_heatmaps_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    # Plot heatmaps for variable selection considering all experiments, and then all experiments for linear regression and LightGBM when adopting non-anomaly clustering
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders_NC, selected_vars_df_list_NC, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'all_heatmaps_NC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders_linreg_NC, selected_vars_df_list_linreg_NC, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'linreg_heatmaps_NC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders_lgbm_NC, selected_vars_df_list_lgbm_NC, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'lgbm_heatmaps_NC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    # Plot heatmaps for variable selection considering all experiments, and then all experiments for linear regression and LightGBM when adopting anomaly clustering
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders_AC, selected_vars_df_list_AC, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'all_heatmaps_AC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders_linreg_AC, selected_vars_df_list_linreg_AC, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'linreg_heatmaps_AC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    heatmap = ut.vars_selection_heatmaps_no_cluster(experiments_folders_lgbm_AC, selected_vars_df_list_lgbm_AC, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(basin_comp, f'lgbm_heatmaps_AC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)

    ## Selection trends taking the clusters into account (comparing all experiments for each number of clusters, but also filtering by clustering method, and by model fitted) ##
    n_clusters  = [5, 6, 7, 8, 9, 10, 11, 12]
    for n_cluster in n_clusters:
        # List all the experiments containing the same number fo clusters
        experiments_folders = [f for f in all_subfolders if f'nc{n_cluster}' in f]
        experiments_folders.sort()
        # Define list to store experiments folders according to their charasteristics (clustering method, model kind)
        experiments_folders_NC = []
        experiments_folders_AC = []
        experiments_folders_linreg_NC = []
        experiments_folders_lgbm_NC = []
        experiments_folders_linreg_AC = []
        experiments_folders_lgbm_AC = []
        # Define lists to store the selected variables dataframes
        selected_vars_df_list_NC = []
        selected_vars_df_list_AC = []
        selected_vars_df_list_linreg_NC = []
        selected_vars_df_list_lgbm_NC = []
        selected_vars_df_list_linreg_AC = []
        selected_vars_df_list_lgbm_AC = []
        # Extract variable selection
        for experiment_name in experiments_folders:
            model_kind = experiment_name.split('_')[1]
            nc_string = experiment_name.split('_')[2]
            numbers = filter(str.isdigit, nc_string)
            nc = ''.join(numbers)
            experiment_filename = f'1965-2022_{nc}clusters_{n_vars}vars_{n_idxs}idxs.csv'
            best_solution = pd.read_csv(os.path.join(basin_results_dir, experiment_name, f'best_solution_{model_kind}_{experiment_filename}'), sep=',', header=None)
            if "A" in nc_string:
                data_dir = os.path.join(fs_dir, 'data', f'{basin}_{nc}clusters_anomaly')
            else:
                data_dir = os.path.join(fs_dir, 'data', f'{basin}_{nc}clusters')
            predictors_df = pd.read_csv(os.path.join(data_dir, f'predictors_{experiment_filename}'), index_col=0)
            predictors_df.index = pd.to_datetime(predictors_df.index)
            selected_vars_df = ut.df_selected_vars(predictors_df, best_solution)
            if "A" in nc_string:
                experiments_folders_AC.append(experiment_name)
                selected_vars_df_list_AC.append(selected_vars_df)
                if "linreg" in experiment_name:
                    experiments_folders_linreg_AC.append(experiment_name)
                    selected_vars_df_list_linreg_AC.append(selected_vars_df)
                elif "lgbm" in experiment_name:
                    experiments_folders_lgbm_AC.append(experiment_name)
                    selected_vars_df_list_lgbm_AC.append(selected_vars_df)
            else:
                experiments_folders_NC.append(experiment_name)
                selected_vars_df_list_NC.append(selected_vars_df)
                if "linreg" in experiment_name:
                    experiments_folders_linreg_NC.append(experiment_name)
                    selected_vars_df_list_linreg_NC.append(selected_vars_df)
                elif "lgbm" in experiment_name:
                    experiments_folders_lgbm_NC.append(experiment_name)
                    selected_vars_df_list_lgbm_NC.append(selected_vars_df)
        # Plot heatmaps for variable selection considering all experiments, and then all experiments for linear regression and LightGBM when adopting non-anomaly clustering
        sub_comp_results_dir = os.path.join(basin_comp, f'{n_cluster}clusters')
        os.makedirs(sub_comp_results_dir, exist_ok=True)
        heatmap = ut.vars_selection_heatmaps(experiments_folders_NC, n_cluster, selected_vars_df_list_NC, atm_vars, idx_vars, display_percentage=True)
        heatmap.savefig(os.path.join(sub_comp_results_dir, f'all_heatmaps_NC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
        plt.close(heatmap)
        heatmap = ut.vars_selection_heatmaps(experiments_folders_linreg_NC, n_cluster, selected_vars_df_list_linreg_NC, atm_vars, idx_vars, display_percentage=True)
        heatmap.savefig(os.path.join(sub_comp_results_dir, f'linreg_heatmaps_NC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
        plt.close(heatmap)
        heatmap = ut.vars_selection_heatmaps(experiments_folders_lgbm_NC, n_cluster, selected_vars_df_list_lgbm_NC, atm_vars, idx_vars, display_percentage=True)
        heatmap.savefig(os.path.join(sub_comp_results_dir, f'lgbm_heatmaps_NC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
        plt.close(heatmap)
        # Plot heatmaps for variable selection considering all experiments, and then all experiments for linear regression and LightGBM when adopting anomaly clustering
        heatmap = ut.vars_selection_heatmaps(experiments_folders_AC, n_cluster, selected_vars_df_list_AC, atm_vars, idx_vars, display_percentage=True)
        heatmap.savefig(os.path.join(sub_comp_results_dir, f'all_heatmaps_AC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
        plt.close(heatmap)
        heatmap = ut.vars_selection_heatmaps(experiments_folders_linreg_AC, n_cluster, selected_vars_df_list_linreg_AC, atm_vars, idx_vars, display_percentage=True)
        heatmap.savefig(os.path.join(sub_comp_results_dir, f'linreg_heatmaps_AC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
        plt.close(heatmap)
        heatmap = ut.vars_selection_heatmaps(experiments_folders_lgbm_AC, n_cluster, selected_vars_df_list_lgbm_AC, atm_vars, idx_vars, display_percentage=True)
        heatmap.savefig(os.path.join(sub_comp_results_dir, f'lgbm_heatmaps_AC_{n_vars}vars_{n_idxs}idxs.pdf'), format='pdf', dpi=300)
        plt.close(heatmap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparative analysis of the results of the feature selection for the TCs genesis dataset')
    parser.add_argument('--basin', type=str, help='Name of the basin to analyze')
    parser.add_argument('--n_vars', type=int, default=8, help='Number of candidate atmospheric variables')
    parser.add_argument('--n_idxs', type=int, default=10, help='Number of candidate climate indexes')
    args = parser.parse_args()
    main(args.basin, args.n_vars, args.n_idxs)



