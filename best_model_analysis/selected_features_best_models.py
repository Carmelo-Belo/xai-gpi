import os 
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import zip_longest
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import sys
from sklearn.metrics import mean_squared_error 
sys.path.insert(0, '/Users/huripari/Documents/PhD/TCs_Genesis/FS_TCG')
import utils_results as ut

def plot_selected_variables_clusters(basin, n_clusters, data_dir, var_list):
    # Get the cluster variables
    cluster_variables = list(set([var.split('_cluster')[0] for var in var_list if 'cluster' in var]))
    # Set the domain extension for the figures
    if basin == 'NWP':
            west, east, south, north = 100, 180, 0, 40
    elif basin == 'NEP':
        west, east, south, north = -180, -75, 0, 40
    elif basin == 'NA':
        west, east, south, north = -100, 0, 0, 40
    elif basin == 'NI':
        west, east, south, north = 45, 100, 0, 40
    elif basin == 'SP':
        west, east, south, north = 135, -70, -40, 0
    elif basin == 'SI':
        west, east, south, north = 35, 135, -40, 0
    elif basin == 'GLB':
        west, east, south, north = -181, 181, -40, 40
    else:
        raise ValueError('Basin not recognized')
    figures = []
    # Plot the clusters of each variable in the list
    for v, var in enumerate(cluster_variables):
        # Load the labels file
        label_file = os.path.join(data_dir, f'labels_{var}.csv')
        label_df = pd.read_csv(label_file, index_col=0)
        unique_clusters = np.arange(1, n_clusters+1)
        # Define a color map with fixed colors for each cluster and map the clusters to the colors index
        c_map = plt.get_cmap('tab20', n_clusters)
        colors = c_map(np.linspace(0, 1, n_clusters))
        full_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)
        cluster_to_color_index = {cluster: i for i, cluster in enumerate(unique_clusters)}
        
        # Determine the clusters and corresponding lags selected for the variable
        clusters_selected = np.asarray([int(long_name.split('_cluster')[1]) for long_name in var_list if long_name.split('_cluster')[0] == var])

        # Select the rows of the label file that correspond to the selected clusters
        label_df_selected = label_df[label_df['cluster'].isin(clusters_selected)]

        # Set the figure and gridlines of the map
        fig = plt.figure(figsize=(30, 6))
        if basin == 'NA':
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', linewidth=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
        gl.xlabel_style = {'size': 20} 
        gl.ylabel_style = {'size': 20}

        # Plot only the selected clusters using their index in the full color map
        scatter = ax.scatter(
            label_df_selected['nodes_lon'].values, 
            label_df_selected['nodes_lat'].values,
            c=[cluster_to_color_index[cluster] for cluster in label_df_selected['cluster']],
            cmap=full_cmap, norm=norm, s=400, transform=ccrs.PlateCarree()
        )

        # Create a colorbar showing all clusters
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', ticks=np.arange(n_clusters))
        cbar.set_ticklabels(unique_clusters)
        cbar.ax.tick_params(labelsize=22)
        cbar.set_label('Cluster', fontsize=26)

        ax.set_title(f'{var}', fontsize=30)
        plt.tight_layout()
        figures.append(fig)

    return figures, cluster_variables

def main():
    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    results_dir = os.path.join(fs_dir, 'results')
    best_model_dir = os.path.join(fs_dir, 'best_model_analysis')
    fig_dir = os.path.join(best_model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    # Set lists of basins, model kinds, cluster types, and number of clusters
    basin_names = ['Global', 'North East Pacific', 'North West Pacific', 'North Atlantic', 'North Indian', 'South Indian', 'South Pacific']
    basins = ['GLB', 'NEP', 'NWP', 'NA', 'NI', 'SI', 'SP']
    model_kinds = ['linreg'] * 7
    cluster_types = ['DSnc', 'DSnc', 'Anc', 'DSnc', 'DSnc', 'DSnc', '_nc']
    n_clusters = [12, 12, 10, 6, 12, 9, 7]
    FINAL_MODEL = 'mlp'
    # Loop over basins
    for bb, basin in enumerate(basins):
        # Load the performance file for the basin and filter to get the simulation with the best performance
        track_file = os.path.join(results_dir, f'sim_performance_{basin}.csv')
        track_df = pd.read_csv(track_file, index_col=0)
        track_df = track_df[track_df.index.str.contains('nd9')]
        track_df = track_df[track_df.index.str.contains(model_kinds[bb])]
        track_df = track_df[track_df.index.str.contains(cluster_types[bb])]
        track_df = track_df[track_df['n_clusters'] == n_clusters[bb]]
        # Get the simulation with the best performance
        performance_col = f'R_Y_{FINAL_MODEL}'
        sorted_df = track_df.sort_values(performance_col, ascending=False)
        best_sim = sorted_df.iloc[0]

        n_cluster = n_clusters[bb]

        # Load predictors and candidate variables of the corresponding simulation
        nc_string = best_sim.name.split('_')[2]
        if "A" in nc_string:
            cluster_data = f'{basin}_{n_cluster}clusters_anomaly'
        elif "DS" in nc_string:
            cluster_data = f'{basin}_{n_cluster}clusters_deseason'
            target_season = 'target_seasonality_1970-2022_2.5x2.5.csv'
        else:
            cluster_data = f'{basin}_{n_cluster}clusters'
        # Set the paths to files
        experiment_filename = f'1970-2022_{n_cluster}clusters_8vars_9idxs.csv'
        predictor_file = 'predictors_' + experiment_filename
        data_dir = os.path.join(fs_dir, 'data', cluster_data)
        predictors_path = os.path.join(data_dir, predictor_file)
        predictors_df = pd.read_csv(predictors_path, index_col=0)
        predictors_df.index = pd.to_datetime(predictors_df.index)
        candidate_variables = predictors_df.columns.to_numpy()

        ## Plot the heatmap of % of selection in the Top33%, Mid33%, and Bot33% of the best models in the first 15 runs ##
        df_tier_sel_perc = pd.DataFrame(0, columns=candidate_variables, index=['Top33%', 'Mid33%', 'Bot33%'])
        for r, run_name in enumerate(sorted_df.index):
            model_kind = run_name.split('_')[1]
            sol_filename = f'{model_kind}_{experiment_filename}'
            output_dir = os.path.join(fs_dir, 'results', basin, run_name)
            best_sol_path = os.path.join(output_dir, f'best_solution_{sol_filename}')
            best_solution = pd.read_csv(best_sol_path, sep=',', header=None)
            best_solution = best_solution.to_numpy().flatten()
            # Select the variables from the best solutions
            column_names = predictors_df.columns.tolist()
            feat_sel = best_solution[2*len(column_names):]
            if r < 5:
                df_tier_sel_perc.loc['Top33%'] = df_tier_sel_perc.loc['Top33%'] + feat_sel
            elif r < 10:
                df_tier_sel_perc.loc['Mid33%'] = df_tier_sel_perc.loc['Mid33%'] + feat_sel
            else:
                df_tier_sel_perc.loc['Bot33%'] = df_tier_sel_perc.loc['Bot33%'] + feat_sel
        df_tier_sel_perc = (df_tier_sel_perc / 5) * 100
        # Plot the heatmap
        fig = plt.figure(figsize=(3*n_cluster, 10))
        ax = sns.heatmap(df_tier_sel_perc, cmap="Blues", linewidths=0.5, linecolor="gray", square=True, figure=fig,
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
            xpos = variable_positions[i] + (variable_positions[i+1] - variable_positions[i]) / 2 if i < len(variable_positions) - 1 else variable_positions[i] + n_cluster/2
            ax.text(xpos, len(df_tier_sel_perc) + 2, var, ha='center', va='center', fontsize=14, fontweight="bold")
        # Set the vertical lines between the different variables a bit thicker 
        thick_line_pos = [i+1 for i, var in enumerate(candidate_variables) if var.split('_cluster')[-1] == str(n_cluster)]
        for pos in thick_line_pos:
            ax.vlines(x=pos, ymin=-0.5, ymax=len(df_tier_sel_perc), linewidth=2.5, color="black")
        # Overlay red blocks at the bottom for zero columns
        zero_columns = (df_tier_sel_perc == 0).all(axis=0)
        for idx, is_zero in enumerate(zero_columns):
            if is_zero:
                ax.add_patch(plt.Rectangle((idx, len(df_tier_sel_perc) - 0.55), 1, 0.5, color='red', clip_on=False))
        # Set yticks labels fontsize
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        # Set the title
        ax.set_title(f'{basin_names[bb]}', fontsize=16)
        # Save the figure
        basin_dir = os.path.join(fig_dir, basin)
        os.makedirs(basin_dir, exist_ok=True)
        fig_path = os.path.join(basin_dir, f'feat_sel_33%.pdf')
        fig.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Load the performance file for the basin and filter to get the simulation with the best performance
        track_file = os.path.join(results_dir, f'sim_performance_extra_{basin}.csv')
        track_df = pd.read_csv(track_file, index_col=0)
        track_df = track_df[track_df.index.str.contains('nd9')]
        track_df = track_df[track_df.index.str.contains(model_kinds[bb])]
        track_df = track_df[track_df.index.str.contains(cluster_types[bb])]
        track_df = track_df[track_df['n_clusters'] == n_clusters[bb]]
        # Get the simulation with the best performance
        performance_col = f'R_Y_{FINAL_MODEL}'
        sorted_df_extra = track_df.sort_values(performance_col, ascending=False)
        best_sim_extra = sorted_df_extra.iloc[0]
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
        fig = plt.figure(figsize=(3*n_cluster, 10))
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
            xpos = variable_positions[i] + (variable_positions[i+1] - variable_positions[i]) / 2 if i < len(variable_positions) - 1 else variable_positions[i] + n_cluster/2
            ax.text(xpos, len(df_tier_sel_perc_extra) + 2, var, ha='center', va='center', fontsize=14, fontweight="bold")
        # Set the vertical lines between the different variables a bit thicker 
        thick_line_pos = [i+1 for i, var in enumerate(candidate_variables) if var.split('_cluster')[-1] == str(n_cluster)]
        for pos in thick_line_pos:
            ax.vlines(x=pos, ymin=-0.5, ymax=len(df_tier_sel_perc_extra), linewidth=2.5, color="black")
        # Overlay red blocks at the bottom for zero columns
        zero_columns = (df_tier_sel_perc_extra == 0).all(axis=0)
        for idx, is_zero in enumerate(zero_columns):
            if is_zero:
                ax.add_patch(plt.Rectangle((idx, len(df_tier_sel_perc_extra) - 0.55), 1, 0.5, color='red', clip_on=False))
        # Set the title
        ax.set_title(f'{basin_names[bb]}', fontsize=16)
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
                                               best_models_perc[best_models_perc >= 75].index.to_list(),
                                               best_models_perc[best_models_perc >= 90].index.to_list()),
                                               columns=['50', '60', '75', '90'])
        if cluster_types[bb] == '_nc':
            csv_path = os.path.join(results_dir, f'selected_features_best_models_{basin}{cluster_types[bb]}{n_cluster}.csv')
        else:
            csv_path = os.path.join(results_dir, f'selected_features_best_models_{basin}_{cluster_types[bb]}{n_cluster}.csv')
        df_perc_sel.to_csv(csv_path)

        ## Plot the selected variables clusters ##
        for c, column in enumerate(df_perc_sel.columns):
            var_list = df_perc_sel[column].dropna().to_list()
            perc_fig_dir = os.path.join(basin_dir, f'sel_var_perc{column}')
            os.makedirs(perc_fig_dir, exist_ok=True)
            figs, cluster_vars = plot_selected_variables_clusters(basin, n_cluster, data_dir, var_list)
            for i, fig in enumerate(figs):
                fig_path = os.path.join(perc_fig_dir, f'{cluster_vars[i]}.pdf')
                fig.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == '__main__':
    main()