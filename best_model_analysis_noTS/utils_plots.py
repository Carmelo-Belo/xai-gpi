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

def final_models_violins(metric, results_dir, basins, basin_names):
    # Create a figure for the violin plots
    fmod_violin_fig = plt.figure(figsize=(25, 10))
    fmod_gs = gridspec.GridSpec(2, 3, figure=fmod_violin_fig)
    for bb, basin in enumerate(basins):
        # Load performance tracking file
        track_file = os.path.join(results_dir, f'sim_performance_{basin}_noTS.csv')
        track_df = pd.read_csv(track_file, index_col=0)
        # Define performance columns and group models of the same type together
        model_pairs = [('mlp', 'mlp_noFS'), ('lgbm', 'lgbm_noFS'), ('pi-lgbm', 'pi-lgbm_noFS')]
        performance_data = []
        model_labels = []
        noFS_labels = []
        median_values = []
        hue_labels = []
        for model, model_noFS in model_pairs:
            sorted_df_model = track_df.sort_values(f'{metric}_{model}', ascending=False)
            sorted_df_model_noFS = track_df.sort_values(f'{metric}_{model_noFS}', ascending=False)
            performance_data.extend([sorted_df_model[f'{metric}_{model}'].values, sorted_df_model_noFS[f'{metric}_{model_noFS}'].values])
            model_labels.extend([model] * len(sorted_df_model))  # Assign label for the standard model
            noFS_labels.extend(["With FS"] * len(sorted_df_model))  # Label indicating feature selection
            model_labels.extend([model] * len(sorted_df_model_noFS))  # Assign label for noFS model
            noFS_labels.extend(["No FS"] * len(sorted_df_model_noFS))  # Label indicating no feature selection
            median_values.append(np.median(sorted_df_model[f'{metric}_{model}'].values))
            hue_labels.append("With FS")
            median_values.append(np.median(sorted_df_model_noFS[f'{metric}_{model_noFS}'].values))
            hue_labels.append("No FS")
        # Create DataFrame for seaborn
        plot_df = pd.DataFrame({
            f"{metric}": [value for sublist in performance_data for value in sublist],
            "Model": model_labels,
            "Feature Selection": noFS_labels
        })
        # Define subplot positions
        ax = fmod_violin_fig.add_subplot(fmod_gs[bb])
        # Violin plot with `split=True` to compare models with and without FS
        sns.violinplot(
            x="Model", y=f"{metric}", hue="Feature Selection",
            data=plot_df, ax=ax, inner="quartile", split=True, fill=False,
            palette={"With FS": "blue", "No FS": "red"}
        )
        x_positions = np.arange(len(model_pairs))
        # Add median values to the plot
        for i, (median, hue_label) in enumerate(zip(median_values, hue_labels)):
            color = "blue" if hue_label == "With FS" else "red"
            x_pos = x_positions[i // 2] + (-0.2 if hue_label == "With FS" else 0.2)  # Offset for split violins
            ax.text(x_pos, median, f'{median:.3f}', ha='center', va='bottom', fontsize=10, color=color, fontweight="bold")
        # Set title
        ax.set_title(f'{basin_names[bb]} - {metric}')
        ax.set_ylabel('')
        ax.set_xlabel('')
    # Show the plot and return the figure
    plt.show()
    return fmod_violin_fig

def fs_models_violins(metric, final_model, results_dir, basins, basin_names, show_noFS=True):
    # Create a figure for the violin plots
    model_violin_fig = plt.figure(figsize=(25, 10))
    mods_gs = gridspec.GridSpec(2, 3, figure=model_violin_fig)
    for bb, basin in enumerate(basins):
        # Load file tracking simulation performance
        track_file = os.path.join(results_dir, f'sim_performance_{basin}_noTS.csv')
        track_df = pd.read_csv(track_file, index_col=0)
        track_df = track_df[track_df.index.str.contains('nd9')]
        performance_col = f'{metric}_{final_model}'
        # Create a list to store data for each cluster count
        model_labels = []
        values = []
        median_values = []
        model_positions = []
        if show_noFS:
            fs_models = ['linreg', '_lgbm', 'pi-lgbm', 'noFS']
            xticks_labels = ['Linear Regression', 'LGBM', 'PI-LGBM', 'no FS']
        else:
            fs_models = ['linreg', '_lgbm', 'pi-lgbm']
            xticks_labels = ['Linear Regression', 'LGBM', 'PI-LGBM']
        for model in fs_models:
            mod_df = track_df[track_df.index.str.contains(model)] if model != 'noFS' else track_df
            vals = mod_df[performance_col].values if model != 'noFS' else mod_df[f'{metric}_{final_model}_noFS']
            values.extend(vals)
            model_labels.extend([model] * len(vals))
            # Store median value
            if len(vals) > 0:
                median_values.append(np.median(vals))
                model_positions.append(model)  # Store the cluster position for annotation
        # Create DataFrame for Seaborn
        plot_df = pd.DataFrame({
            f"{metric}": values,
            "Models": model_labels
        })
        # Define subplot positions
        ax = model_violin_fig.add_subplot(mods_gs[bb])
        # Create violin plot
        sns.violinplot(x="Models", y=f"{metric}", hue="Models", data=plot_df, ax=ax, inner="quartile", palette="Blues", legend=False, fill=True)
        # Annotate median values in black
        x_positions = np.arange(len(model_positions))
        for x, median in zip(x_positions, median_values):
            ax.text(x, median, f'{median:.3f}', ha='center', va='bottom', fontsize=12, color='black', fontweight="bold")
        # Set title
        ax.set_title(f'{basin_names[bb]} - {metric}', fontdict={'fontsize': 16})
        ax.set_ylabel('')
        ax.set_xlabel('')
        # Set xticks
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xticks_labels, fontdict={'fontsize': 14})
        # Set yticks
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontdict={'fontsize': 12})
    # Show the plot and return the figure
    plt.show()
    return model_violin_fig

def n_clusters_violins(metric, final_model, fs_model, results_dir, basins, basin_names):
    # Create a figure for the violin plots
    ncl_violin_fig = plt.figure(figsize=(25, 10))
    ncle_gs = gridspec.GridSpec(2, 3, figure=ncl_violin_fig)
    for bb, basin in enumerate(basins):
        # Load file tracking simulation performance
        track_file = os.path.join(results_dir, f'sim_performance_{basin}_noTS.csv')
        track_df = pd.read_csv(track_file, index_col=0)
        track_df = track_df[track_df.index.str.contains(fs_model)]
        performance_col = f'{metric}_{final_model}'
        # Create a list to store data for each cluster count
        cluster_labels = []
        values = []
        median_values = []
        cluster_positions = []
        for ncl in range(5, 13):
            ncl_df = track_df[track_df['n_clusters'] == ncl]
            vals = ncl_df[performance_col].values
            values.extend(vals)
            cluster_labels.extend([str(ncl)] * len(vals))
            # Store median value
            if len(vals) > 0:
                median_values.append(np.median(vals))
                cluster_positions.append(str(ncl))  # Store the cluster position for annotation
        # Create DataFrame for Seaborn
        plot_df = pd.DataFrame({
            f"{metric}": values,
            "Clusters": cluster_labels
        })
        # Define subplot positions
        ax = ncl_violin_fig.add_subplot(ncle_gs[bb])
        # Create violin plot
        sns.violinplot(x="Clusters", y=f"{metric}", hue="Clusters", data=plot_df, ax=ax, inner="quartile", palette="Blues", legend=False, fill=True)
        # Annotate median values in black
        x_positions = range(len(cluster_positions))
        for x, median in zip(x_positions, median_values):
            ax.text(x, median, f'{median:.3f}', ha='center', va='bottom', fontsize=12, color='black', fontweight="bold")
        # Set title
        ax.set_title(f'{basin_names[bb]} - {metric}', fontdict={'fontsize': 16})
        ax.set_ylabel('')
        ax.set_xlabel('')
        # Set xticks
        ax.set_xticks(x_positions)
        ax.set_xticklabels(cluster_positions, fontdict={'fontsize': 14})
        # Set yticks
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontdict={'fontsize': 12})
    # Show the plot and return the figure
    plt.show()
    return ncl_violin_fig

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