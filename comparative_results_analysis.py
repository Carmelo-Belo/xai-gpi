import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils_results as ut

def main(basin, n_clusters):
    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    results_dir = os.path.join(fs_dir, 'results')
    data_dir = os.path.join(fs_dir, 'data', f'{basin}_{n_clusters}clusters')
    # List all the experiments for that number of clusters in the results directory 
    all_subfolders = os.listdir(results_dir)
    experiments_folders = [f for f in all_subfolders if f'nc{n_clusters}' in f]
    experiments_folders.sort()
    experiments_folders_linreg = [f for f in experiments_folders if 'LinReg' in f]
    experiments_folders_lgbm = [f for f in experiments_folders if 'LGBM' in f]
    # Load predicors dataframe
    experiment_filename = f'1965-2022_{n_clusters}clusters_7vars_10idxs.csv'
    predictors_df = pd.read_csv(os.path.join(data_dir, f'predictors_{experiment_filename}'), index_col=0)
    predictors_df.index = pd.to_datetime(predictors_df.index)
    # For each experiment in the folder, load the best solution and save the selected variables in a list of dataframes
    selected_vars_df_list = []
    selected_vars_df_list_linreg = []
    selected_vars_df_list_lgbm = []
    for experiment_name in experiments_folders:
        model_kind = experiment_name.split('_')[1]
        best_solution = pd.read_csv(os.path.join(results_dir, experiment_name, f'best_solution_{model_kind}_{experiment_filename}'), sep=',', header=None)
        selected_vars_df = ut.df_selected_vars(predictors_df, best_solution)
        selected_vars_df_list.append(selected_vars_df)
        if "LinReg" in experiment_name:
            selected_vars_df_list_linreg.append(selected_vars_df)
        elif "LGBM" in experiment_name:
            selected_vars_df_list_lgbm.append(selected_vars_df)
    # Set the names of the atmospheric variables (with clusters) and the indexes
    vars = predictors_df.columns.to_numpy()
    atm_vars =  np.unique([var.split('_cluster')[0] for var in vars if 'cluster' in var]).tolist()
    idx_vars = [var for var in vars if 'cluster' not in var]

    # Set the folder to save the plots for comparitive analysis
    comp_results_dir = os.path.join(fs_dir, 'results', 'comparative_analysis')
    os.makedirs(comp_results_dir, exist_ok=True)
    sub_results_dir = os.path.join(comp_results_dir, f'{basin}_{n_clusters}clusters')
    os.makedirs(sub_results_dir, exist_ok=True)
    # All experiments
    figs = ut.models_shares_vars_selection_spyder_plot(experiments_folders, n_clusters, selected_vars_df_list, atm_vars, idx_vars, display_percentage=False)
    for f, fig in enumerate(figs):
        fig.savefig(os.path.join(sub_results_dir, f'all_model_spyder_plot_lag{f}.pdf'), format='pdf', dpi=300)
    plt.close(fig)
    heatmap = ut.vars_selection_heatmaps(experiments_folders, n_clusters, selected_vars_df_list, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(sub_results_dir, f'all_models_heatmaps.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    # Linear Regression
    fig = ut.vars_selection_spyder_plot(experiments_folders_linreg, n_clusters, selected_vars_df_list_linreg, atm_vars, idx_vars, display_percentage=False)
    fig.savefig(os.path.join(sub_results_dir, f'linreg_spyder_plot.pdf'), format='pdf', dpi=300)
    plt.close(fig)
    heatmap = ut.vars_selection_heatmaps(experiments_folders_linreg, n_clusters, selected_vars_df_list_linreg, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(sub_results_dir, f'linreg_heatmaps.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)
    # LightGBM
    fig = ut.vars_selection_spyder_plot(experiments_folders_lgbm, n_clusters, selected_vars_df_list_lgbm, atm_vars, idx_vars, display_percentage=False)
    fig.savefig(os.path.join(sub_results_dir, f'lgbm_spyder_plot.pdf'), format='pdf', dpi=300)
    plt.close(fig)
    heatmap = ut.vars_selection_heatmaps(experiments_folders_lgbm, n_clusters, selected_vars_df_list_lgbm, atm_vars, idx_vars, display_percentage=True)
    heatmap.savefig(os.path.join(sub_results_dir, f'lgbm_heatmaps.pdf'), format='pdf', dpi=300)
    plt.close(heatmap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparative analysis of the results of the feature selection for the TCs genesis dataset')
    parser.add_argument('--basin', type=str, default='GLB', help='Name of the basin to analyze')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters to analyze')
    args = parser.parse_args()
    main(args.basin, args.n_clusters)



