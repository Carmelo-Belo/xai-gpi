import os 
import argparse
import numpy as np
import pandas as pd
from itertools import zip_longest
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

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

def main(predictors_type):
    # Check predictors type to compute the selected features of the best model
    if predictors_type != 'original' and predictors_type != 'deseason' and predictors_type != 'detrend':
        raise ValueError('Predictors type not recognized. Choose between "original", "deseason" or "detrend".')
    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    results_dir = os.path.join(fs_dir, 'results')
    best_model_dir = os.path.join(fs_dir, 'best_model_analysis')
    fig_dir = os.path.join(best_model_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    # Set lists of basins, model kinds, and number of clusters depending on the predictors type
    if predictors_type == 'original':
        basin_dict = {
            'NEP': ('North East Pacific', 'lgbm', 7), 
            'NWP': ('North West Pacific', 'pi-lgbm', 10), 
            'NA': ('North Atlantic', 'linreg', 6), 
            'NI': ('North Indian', 'pi-lgbm', 7), 
            'SI': ('South Indian', 'linreg', 6), 
            'SP': ('South Pacific', 'pi-lgbm', 9)
            }
    elif predictors_type == 'deseason':
        basin_dict = {
            'NEP': ('North East Pacific', 'linreg', 6), 
            'NWP': ('North West Pacific', 'linreg', 10), 
            'NA': ('North Atlantic', 'linreg', 10), 
            'NI': ('North Indian', 'linreg', 8), 
            'SI': ('South Indian', 'linreg', 8), 
            'SP': ('South Pacific', 'linreg', 9)
            }
    elif predictors_type == 'detrend':
        basin_dict = {
            'NEP': ('North East Pacific', 'linreg', 12), 
            'NWP': ('North West Pacific', 'linreg', 10), 
            'NA': ('North Atlantic', 'linreg', 12), 
            'NI': ('North Indian', 'linreg', 11), 
            'SI': ('South Indian', 'linreg', 7), 
            'SP': ('South Pacific', 'linreg', 10)
            }
    else:
        raise ValueError("Invalid predictors type. Choose 'original', 'deseason', or 'detrend'.")
    FINAL_MODEL = 'mlp'
    # Loop over basins
    for basin, (basin_name, fs_model, n_clusters) in basin_dict.items():
        # Load the performance file for the basin and filter to get the simulation with the best performance
        track_file = os.path.join(results_dir, f'sim_performance_{basin}.csv')
        track_df = pd.read_csv(track_file, index_col=0)
        track_df = track_df[track_df['model'] == fs_model]
        track_df = track_df[track_df['n_clusters'] == n_clusters]
        if predictors_type == 'deseason':
            track_df = track_df[track_df.index.str.contains('DS')]
            str_pt = 'DSnc'
        elif predictors_type == 'detrend':
            track_df = track_df[track_df.index.str.contains('DT')]
            str_pt = 'DTnc'
        elif predictors_type == 'original':
            track_df = track_df[track_df.index.str.contains('_nc')]
            str_pt = 'nc'
        # Get the simulation with the best performance
        performance_col = f'R_Y_{FINAL_MODEL}'
        sorted_df = track_df.sort_values(performance_col, ascending=False)
        # Load predictors and candidate variables of the corresponding simulation
        # Get the predictors for the basin and the number of clusters
        if predictors_type == 'original':
            cluster_data = f'{basin}_{n_clusters}clusters'
        elif predictors_type == 'deseason':
            cluster_data = f'{basin}_{n_clusters}clusters_deseason'
        elif predictors_type == 'detrend':
            cluster_data = f'{basin}_{n_clusters}clusters_detrend'
        # Set the paths to files
        experiment_filename = f'1980-2022_{n_clusters}clusters_8vars_9idxs.csv'
        predictor_file = 'predictors_' + experiment_filename
        data_dir = os.path.join(fs_dir, 'data', cluster_data)
        predictors_path = os.path.join(data_dir, predictor_file)
        predictors_df = pd.read_csv(predictors_path, index_col=0)
        predictors_df.index = pd.to_datetime(predictors_df.index)
        candidate_variables = predictors_df.columns.to_numpy()
        basin_dir = os.path.join(fig_dir, basin)

        # Get the simulation with the best performance
        performance_col = f'R_Y_{FINAL_MODEL}'
        sorted_df = track_df.sort_values(performance_col, ascending=False)
        ## Compute the % of selection in the Top20%, Top-Mid20%, Mid20%, Bot-Mid20%, and Bot20% of the best models considering also the additional runs ##
        df_tier_sel_perc = pd.DataFrame(0, columns=candidate_variables, index=['Top20%', 'UpMid20%', 'Mid20%', 'BotMid20%', 'Bot20%'])
        n_sim = len(sorted_df)
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
            if r < int(n_sim * 0.2):
                df_tier_sel_perc.loc['Top20%'] = df_tier_sel_perc.loc['Top20%'] + feat_sel
            elif r < int(n_sim * 0.4):
                df_tier_sel_perc.loc['UpMid20%'] = df_tier_sel_perc.loc['UpMid20%'] + feat_sel
            elif r < int(n_sim * 0.6):
                df_tier_sel_perc.loc['Mid20%'] = df_tier_sel_perc.loc['Mid20%'] + feat_sel
            elif r < int(n_sim * 0.8):
                df_tier_sel_perc.loc['BotMid20%'] = df_tier_sel_perc.loc['BotMid20%'] + feat_sel
            else:
                df_tier_sel_perc.loc['Bot20%'] = df_tier_sel_perc.loc['Bot20%'] + feat_sel
        df_tier_sel_perc = (df_tier_sel_perc / (n_sim * 0.2)) * 100
        ## Save model percentages of selection in .csv file ##
        best_models_perc = df_tier_sel_perc.loc['Top20%']
        df_perc_sel = pd.DataFrame(zip_longest(best_models_perc[best_models_perc >= 50].index.to_list(),
                                               best_models_perc[best_models_perc >= 60].index.to_list(),
                                               best_models_perc[best_models_perc >= 70].index.to_list(),
                                               best_models_perc[best_models_perc >= 75].index.to_list(),
                                               best_models_perc[best_models_perc >= 80].index.to_list(),
                                               best_models_perc[best_models_perc >= 90].index.to_list()),
                                               columns=['50', '60', '70', '75', '80', '90'])
        csv_path = os.path.join(results_dir, f'selected_features_best_models_{basin}_{str_pt}{n_clusters}.csv')
        df_perc_sel.to_csv(csv_path)

        ## Plot the selected variables clusters ##
        for c, column in enumerate(df_perc_sel.columns):
            var_list = df_perc_sel[column].dropna().to_list()
            perc_fig_dir = os.path.join(basin_dir, f'sel_var_{str_pt}{n_clusters}_perc{column}')
            os.makedirs(perc_fig_dir, exist_ok=True)
            figs, cluster_vars = plot_selected_variables_clusters(basin, n_clusters, data_dir, var_list)
            for i, fig in enumerate(figs):
                fig_path = os.path.join(perc_fig_dir, f'{cluster_vars[i]}.pdf')
                fig.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computed dataframe of selected variables clusters for the best models.')
    parser.add_argument('--predictors_type', type=str, help='Type of predictors: original, deseason, or detrend.')
    args = parser.parse_args()
    predictors_type = args.predictors_type
    main(predictors_type)