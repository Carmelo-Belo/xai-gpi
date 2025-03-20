import os 
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import zip_longest
import matplotlib.pyplot as plt
from utils_plots import plot_selected_variables_clusters

def main():
    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    results_dir = os.path.join(fs_dir, 'results')
    best_model_dir = os.path.join(fs_dir, 'best_model_analysis_noTS')
    fig_dir = os.path.join(best_model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    # Set lists of basins, model kinds, cluster types, and number of clusters
    basin_dict = {
        'NEP': ('North East Pacific', 'linreg', 9), 
        'NWP': ('North West Pacific', 'linreg', 8), 
        'NA': ('North Atlantic', 'linreg', 12), 
        'NI': ('North Indian', 'linreg', 9), 
        'SI': ('South Indian', 'linreg', 10), 
        'SP': ('South Pacific', 'linreg', 11)
        }
    FINAL_MODEL = 'mlp'
    # Loop over basins
    for basin, (basin_name, model_kind, n_clusters) in basin_dict.items():
        # Get the predictors for the basin and the number of clusters
        cluster_data = f'{basin}_{n_clusters}clusters_noTS'
        data_dir = os.path.join(fs_dir, 'data', cluster_data)
        experiment_filename = f'1980-2022_{n_clusters}clusters_8vars_9idxs.csv'
        predictor_file = 'predictors_' + experiment_filename
        predictors_path = os.path.join(fs_dir, 'data', cluster_data, predictor_file)
        predictors_df = pd.read_csv(predictors_path, index_col=0)
        predictors_df.index = pd.to_datetime(predictors_df.index)
        candidate_variables = predictors_df.columns.to_numpy()
        # Load the performance file for the basin and filter to get the simulation with the best performance
        track_file = os.path.join(results_dir, f'sim_performance_{basin}_noTS.csv')
        track_df = pd.read_csv(track_file, index_col=0)
        track_df = track_df[track_df['n_clusters'] == n_clusters]
        # Build the df containing the number of selection of each predictors
        df_tier_sel_perc_fsmodels = pd.DataFrame(0, columns=candidate_variables, index=['linreg', 'lgbm', 'pi-lgbm'])
        for r, run_name in enumerate(track_df.index):
            model_kind = run_name.split('_')[1]
            sol_filename = f'{model_kind}_{experiment_filename}'
            output_dir = os.path.join(fs_dir, 'results', basin, run_name)
            best_sol_path = os.path.join(output_dir, f'best_solution_{sol_filename}')
            best_solution = pd.read_csv(best_sol_path, sep=',', header=None)
            best_solution = best_solution.to_numpy().flatten()
            # get feature selection model from run name
            model_kind = track_df.loc[run_name, 'model']
            # Select the variables from the best solutions
            column_names = predictors_df.columns.tolist()
            feat_sel = best_solution[2*len(column_names):]
            df_tier_sel_perc_fsmodels.loc[model_kind] = df_tier_sel_perc_fsmodels.loc[model_kind] + feat_sel
        # Get the percentage of selection
        df_tier_sel_perc_fsmodels = (df_tier_sel_perc_fsmodels / 5) * 100
        # Set the figure for the heatmap percentage of selection
        plt.figure(figsize=(3*n_clusters, 10))
        ax = sns.heatmap(df_tier_sel_perc_fsmodels, cmap="Blues", linewidths=0.5, linecolor="gray", square=True,
                        cbar_kws={'orientation': 'horizontal', 'label': '% of selection', 'shrink': 0.2, 'aspect': 20})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('% of selection', fontsize=14)
        # Set xticks labels
        features_clustered = [var for var in candidate_variables if 'cluster' in var]
        features_non_clustered = [var for var in candidate_variables if 'cluster' not in var]
        cluster_numbers = [col.split("cluster")[-1] if "cluster" in col else "" for col in features_clustered]
        variables = [col.split("_cluster")[0] for col in features_clustered]
        variable_positions = [variables.index(var) for var in sorted(set(variables), key=variables.index)]
        xticks_labels = cluster_numbers + features_non_clustered
        ax.set_xticks(np.arange(len(candidate_variables)) + 0.5)  
        ax.set_xticklabels(xticks_labels, rotation=40, ha="right", fontsize=12)
        for i, var in enumerate(sorted(set(variables), key=variables.index)):
            xpos = variable_positions[i] + (variable_positions[i+1] - variable_positions[i]) / 2 if i < len(variable_positions) - 1 else variable_positions[i] + n_clusters/2
            ax.text(xpos, len(df_tier_sel_perc_fsmodels) + 2, var, ha='center', va='center', fontsize=14, fontweight="bold")
        # Set the vertical lines between the different variables a bit thicker 
        thick_line_pos = [i+1 for i, var in enumerate(candidate_variables) if var.split('_cluster')[-1] == str(n_clusters)]
        for pos in thick_line_pos:
            ax.vlines(x=pos, ymin=-0.5, ymax=len(df_tier_sel_perc_fsmodels), linewidth=2.5, color="black")
        # Overlay red blocks at the bottom for zero columns
        zero_columns = (df_tier_sel_perc_fsmodels == 0).all(axis=0)
        for idx, is_zero in enumerate(zero_columns):
            if is_zero:
                ax.add_patch(plt.Rectangle((idx, len(df_tier_sel_perc_fsmodels) - 0.55), 1, 0.5, color='red', clip_on=False))
        # Set the yticks labels fontsize
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        # Save the figure
        basin_dir = os.path.join(fig_dir, basin)
        os.makedirs(basin_dir, exist_ok=True)
        fig_path = os.path.join(basin_dir, f'feat_sel_3fsmodels.pdf')
        fig.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Load the performance file for the basin and filter to get the simulation with the best performance
        track_file = os.path.join(results_dir, f'sim_performance_{basin}_noTS_extra.csv')
        track_df_extra = pd.read_csv(track_file, index_col=0)
        track_df_extra = track_df_extra[track_df_extra['model'] == 'linreg']
        track_df_extra = track_df_extra[track_df_extra['n_clusters'] == n_clusters]
        # Get the simulation with the best performance
        performance_col = f'MSE_{FINAL_MODEL}'
        sorted_df_extra = track_df_extra.sort_values(performance_col, ascending=True)
        ## Plot the heatmap of % of selection in the Top20%, Top-Mid20%, Mid20%, Bot-Mid20%, and Bot20% of the best models with the additional 85 runs ##
        df_tier_sel_perc_extra = pd.DataFrame(0, columns=candidate_variables, index=['Top20%', 'UpMid20%', 'Mid20%', 'BotMid20%', 'Bot20%'])
        n_sim = len(sorted_df_extra)
        for r, run_name in enumerate(sorted_df_extra.index):
            model_kind = run_name.split('_')[1]
            sol_filename = f'{model_kind}_{experiment_filename}'
            output_dir = os.path.join(fs_dir, 'results', basin, run_name)
            best_sol_path = os.path.join(output_dir, f'best_solution_{sol_filename}')
            best_solution = pd.read_csv(best_sol_path, sep=',', header=None)
            best_solution = best_solution.to_numpy().flatten()
            # Select the variables from the best solutions
            column_names = predictors_df.columns.tolist()
            feat_sel = best_solution[2*len(column_names):]
            if r < int(n_sim * 0.2):
                df_tier_sel_perc_extra.loc['Top20%'] = df_tier_sel_perc_extra.loc['Top20%'] + feat_sel
            elif r < int(n_sim * 0.4):
                df_tier_sel_perc_extra.loc['UpMid20%'] = df_tier_sel_perc_extra.loc['UpMid20%'] + feat_sel
            elif r < int(n_sim * 0.6):
                df_tier_sel_perc_extra.loc['Mid20%'] = df_tier_sel_perc_extra.loc['Mid20%'] + feat_sel
            elif r < int(n_sim * 0.8):
                df_tier_sel_perc_extra.loc['BotMid20%'] = df_tier_sel_perc_extra.loc['BotMid20%'] + feat_sel
            else:
                df_tier_sel_perc_extra.loc['Bot20%'] = df_tier_sel_perc_extra.loc['Bot20%'] + feat_sel
        df_tier_sel_perc_extra = (df_tier_sel_perc_extra / (n_sim * 0.2)) * 100
        fig = plt.figure(figsize=(3*n_clusters, 10))
        ax = sns.heatmap(df_tier_sel_perc_extra, cmap="Blues", linewidths=0.5, linecolor="gray", square=True, figure=fig,
                        cbar_kws={'orientation': 'horizontal', 'label': '% of selection', 'shrink': 0.2, 'aspect': 20})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('% of selection', fontsize=14)
        # Set xticks labels
        features_clustered = [var for var in candidate_variables if 'cluster' in var]
        features_non_clustered = [var for var in candidate_variables if 'cluster' not in var]
        cluster_numbers = [col.split("cluster")[-1] if "cluster" in col else "" for col in features_clustered]
        variables = [col.split("_cluster")[0] for col in features_clustered]
        variable_positions = [variables.index(var) for var in sorted(set(variables), key=variables.index)]
        xticks_labels = cluster_numbers + features_non_clustered
        ax.set_xticks(np.arange(len(candidate_variables)) + 0.5)  
        ax.set_xticklabels(xticks_labels, rotation=40, ha="right", fontsize=12)
        for i, var in enumerate(sorted(set(variables), key=variables.index)):
            xpos = variable_positions[i] + (variable_positions[i+1] - variable_positions[i]) / 2 if i < len(variable_positions) - 1 else variable_positions[i] + n_clusters/2
            ax.text(xpos, len(df_tier_sel_perc_extra) + 2, var, ha='center', va='center', fontsize=14, fontweight="bold")
        # Set the vertical lines between the different variables a bit thicker 
        thick_line_pos = [i+1 for i, var in enumerate(candidate_variables) if var.split('_cluster')[-1] == str(n_clusters)]
        for pos in thick_line_pos:
            ax.vlines(x=pos, ymin=-0.5, ymax=len(df_tier_sel_perc_extra), linewidth=2.5, color="black")
        # Overlay red blocks at the bottom for zero columns
        zero_columns = (df_tier_sel_perc_extra == 0).all(axis=0)
        for idx, is_zero in enumerate(zero_columns):
            if is_zero:
                ax.add_patch(plt.Rectangle((idx, len(df_tier_sel_perc_extra) - 0.55), 1, 0.5, color='red', clip_on=False))
        # Set the title
        # ax.set_title(f'{basin_name}', fontsize=16)
        # Set the yticks labels fontsize
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        # Save the figure
        fig_path = os.path.join(basin_dir, f'feat_sel_20%.pdf')
        fig.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ## Save model percentages of selection in .csv file ##
        best_models_perc = df_tier_sel_perc_extra.loc['Top20%']
        df_perc_sel = pd.DataFrame(zip_longest(best_models_perc[best_models_perc >= 50].index.to_list(),
                                               best_models_perc[best_models_perc >= 60].index.to_list(),
                                               best_models_perc[best_models_perc >= 70].index.to_list(),
                                               best_models_perc[best_models_perc >= 75].index.to_list(),
                                               best_models_perc[best_models_perc >= 80].index.to_list(),
                                               best_models_perc[best_models_perc >= 90].index.to_list()),
                                               columns=['50', '60', '70', '75', '80', '90'])
        csv_path = os.path.join(results_dir, f'selected_features_best_models_{basin}_{n_clusters}_noTS.csv')
        df_perc_sel.to_csv(csv_path)
        ## Plot the selected variables clusters ##
        for c, column in enumerate(df_perc_sel.columns):
            var_list = df_perc_sel[column].dropna().to_list()
            perc_fig_dir = os.path.join(basin_dir, f'sel_var_perc{column}')
            os.makedirs(perc_fig_dir, exist_ok=True)
            figs, cluster_vars = plot_selected_variables_clusters(basin, n_clusters, data_dir, var_list)
            for i, fig in enumerate(figs):
                fig_path = os.path.join(perc_fig_dir, f'{cluster_vars[i]}.pdf')
                fig.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == '__main__':
    main()