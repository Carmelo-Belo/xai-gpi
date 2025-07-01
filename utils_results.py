import os
import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from pdf2image import convert_from_path
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import ipywidgets as widgets
from keras.models import load_model
from sklearn.model_selection import KFold
from sklearn import preprocessing
import shap

# Function to create a board containing the information of the selected features
def create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel):
    board = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        start_index = int(final_sequence[i]) 
        end_index = int(final_sequence[i]) + int(sequence_length[i])
        if feat_sel[i] != 0:
            board[i, start_index:end_index] = 1
    
    return board

# Function to create a dataframe containing the information of the selected features
# First columns are the names of the features, then each followgin column is a time lag
def df_selected_vars(predictors_df, best_solution):
    best_solution = best_solution.to_numpy().flatten()
    column_names = predictors_df.columns.to_list()
    final_sequence = best_solution[len(column_names):2*len(column_names)]
    sequence_length = best_solution[:len(column_names)]
    feat_sel = best_solution[2*len(column_names):]
    n_rows = len(column_names)
    n_cols = int(((sequence_length + final_sequence)*feat_sel).max())
    board_best = create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel)
    df = pd.DataFrame({'column_names': column_names})
    for i in range(n_cols):
        df[f'lag_{i}'] = board_best[:, i]
    return df

# Functions to plot the board with the selected features, at which time lags and to higlight the non-selected features
# It also displays the correlation between the features and the target if requested
def get_text_color(background_color):
    # Calculate brightness (luminance) of the background color
    r, g, b = background_color[:3]  # Ignore alpha if present
    brightness = 0.299 * r + 0.587 * g + 0.114 * b  # Standard luminance calculation
    # Return white text if brightness is low, black if high
    return 'white' if brightness < 0.5 else 'black'

def plot_board(board, column_names, feat_sel, correlations_lag0, correlations_lag1, corr_report=False):
    fig, ax = plt.subplots(figsize=(5, np.rint(len(column_names) / 5)))
    cmap = plt.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=np.min(board), vmax=np.max(board))
    ax.imshow(np.flip(board, axis=0), cmap=cmap, origin='lower', aspect='auto')
    
    ax.xaxis.set_label_position("top") 
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(board.shape[1]))
    ax.set_xticklabels(np.arange(board.shape[1]), fontsize=11)

    ax.set_yticks(np.arange(len(column_names)))
    ax.set_yticklabels(np.flip(np.asarray(column_names)), fontsize=11)

    minor_locator = mticker.AutoMinorLocator(2)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.grid(which='minor',color='black', linewidth=1)
    ax.xaxis.grid(which='minor',color='black', linewidth=1)
    ax.set_xlabel('Time lags (months)', fontsize=15)

    for i in range(board.shape[0]):
        pos = board.shape[0] - i
        if corr_report:
            # Get the background color for the specific cell
            cell_value = board[i, 0]  # Assuming correlation text is on the first column
            background_color = cmap(norm(cell_value))
            text_color = get_text_color(background_color)
            # Add text with dynamic color
            ax.text(-0.1, pos - 1.25, f'{correlations_lag0[i]:.2f}', fontsize=10, color=text_color)
            # If lag_1 present, add text for the correlation at lag 1
            if board.shape[1] > 1:
                # Get the background color for the specific cell
                cell_value = board[i, 1]
                background_color = cmap(norm(cell_value))
                text_color = get_text_color(background_color)
                # Add text with dynamic color
                ax.text(0.9, pos - 1.25, f'{correlations_lag1[i]:.2f}', fontsize=10, color=text_color)
        if feat_sel[i] == 0:
            rect = plt.Rectangle((-0.5, pos - 1.5), 1, 1, color='red')
            ax.add_patch(rect)

    plt.tight_layout()
    return fig

# Function to plot the clusters selected for each variable at each time lag
def plot_selected_clusters(basin, n_clusters, label_selected_vars, data_dir, results_figure_dir):
    # Create the subfolder of figure results to store the cluster selection figures
    save_figure_dir = os.path.join(results_figure_dir, 'clusters_selected')
    os.makedirs(save_figure_dir, exist_ok=True)
    # Get the variable names from the selected variables
    variables_with_cluster = [var for var in label_selected_vars if 'cluster' in var]
    variable_names = [var.split('_cluster')[0] for var in variables_with_cluster]
    variable_names = list(set(variable_names))
    variable_names.sort()

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

    for v, var in enumerate(variable_names):
        # Load the labels file
        label_file = f'labels_{var}.csv'
        label_df = pd.read_csv(os.path.join(data_dir, label_file), index_col=0)
        unique_clusters = np.arange(1, n_clusters+1)

        # Define a color map with fixed colors for each cluster and map the clusters to the colors index
        cmap = plt.get_cmap('tab20', n_clusters)
        colors = cmap(np.linspace(0, 1, n_clusters))
        full_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)
        cluster_to_color_index = {cluster: i for i, cluster in enumerate(unique_clusters)}

        # Determine the clusters and corresponding lags selected for the variable
        clusters_selected = np.asarray([int(long_name.split('_cluster')[1].split('_lag')[0]) 
                                        for long_name in label_selected_vars if long_name.split('_cluster')[0] == var])
        time_lags_selected = np.asarray([int(long_name.split('_lag')[1]) 
                                        for long_name in label_selected_vars if long_name.split('_cluster')[0] == var])
        
        # Cycle through the lags and select the clusters for each lag
        for lag in np.arange(2):
            clusters_for_lag = clusters_selected[time_lags_selected == lag] 
            # Plot the clusters for the selected variable and lag only if there are clusters selected
            if len(clusters_for_lag) > 0:
                # Select the rows of the label file that correspond to the selected clusters
                label_df_selected = label_df[label_df['cluster'].isin(clusters_for_lag)]
                
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

                ax.set_title(f'{var} - time lag: {lag}', fontsize=30)
                plt.tight_layout()

                # Save the figure
                fig_name = f'{var}_lag{lag}_clusters_selected.pdf'
                plt.savefig(os.path.join(save_figure_dir, fig_name), format='pdf', bbox_inches='tight', dpi=300)
                plt.close()

# Function to plot the clusters of a variable that are contained in a list
def plot_clusters_variable(var, basin, n_clusters, label_selected_vars, data_dir):
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

    # Load the labels file
    label_file = f'labels_{var}.csv'
    label_df = pd.read_csv(os.path.join(data_dir, label_file), index_col=0)
    unique_clusters = np.arange(1, n_clusters+1)

    # Define a color map with fixed colors for each cluster and map the clusters to the colors index
    cmap = plt.get_cmap('tab20', n_clusters)
    colors = cmap(np.linspace(0, 1, n_clusters))
    full_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)
    cluster_to_color_index = {cluster: i for i, cluster in enumerate(unique_clusters)}

    # Determine the clusters and corresponding lags selected for the variable
    clusters_selected = np.asarray([int(long_name.split('_cluster')[1].split('_lag')[0]) 
                                    for long_name in label_selected_vars if long_name.split('_cluster')[0] == var])
    
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
    
    return fig

# Function to plot the training loss and the validation loss during the training of the model
def plot_train_val_loss(train_loss, val_loss, train_loss_noFS, val_loss_noFS, test_loss, test_loss_noFS):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_loss, label='FS loss', color='#1f77b4')
    ax.plot(val_loss, label='FS val loss', color='#ff7f0e')
    ax.plot(train_loss_noFS, label='NoFS loss', color='#1f77b4', linestyle='--')
    ax.plot(val_loss_noFS, label='NoFS val loss', color='#ff7f0e', linestyle='--')
    ax.axhline(y=test_loss, color='#2ca02c', linestyle=':', label='Test loss')
    ax.axhline(y=test_loss_noFS, color='#d62728', linestyle=':', label='NoFS Test loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    plt.close()
    return fig

# Function to plot the spyder plots on the variables selection across different experiments
def vars_selection_spyder_plot(experiments_folders, n_clusters, selected_vars_df_list, atm_vars, idx_vars, display_percentage=False):
    # Set figure and grid for plotting
    lags_number = len(selected_vars_df_list[0].columns) - 1
    fig = plt.figure(figsize=(15,15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    # Get the number of experiments for computing percentages + define cluster strings for axes
    experiments_considered = len(experiments_folders)
    cluster_strings = [f'cluster{i}' for i in range(1,n_clusters+1)]
    all_vars = atm_vars + idx_vars

    # Plot the spyder plots for each atmospheric variable 
    for i, var in enumerate(all_vars):
        if var in atm_vars:
            vars_selected = [f'{var}_{cluster}' for cluster in cluster_strings]
        else:
            vars_selected = idx_vars
        # Get the information of number of times the variable was selected for each cluster
        selected_vars_df = selected_vars_df_list[0]
        var_selection_info = selected_vars_df[selected_vars_df['column_names'].isin(vars_selected)]
        for seleceted_vars_df in selected_vars_df_list[1:]:
            sel_info = seleceted_vars_df[seleceted_vars_df['column_names'].isin(vars_selected)]
            var_selection_info.iloc[:,1:] = (var_selection_info.iloc[:,1:] + sel_info.iloc[:,1:])
        # Compute the percentage of selection if requested
        if display_percentage:
            var_selection_info.iloc[:,1:] = var_selection_info.iloc[:,1:] / experiments_considered * 100
        # Plot in the spyder plot according to the number of lags
        ax = fig.add_subplot(gs[i], polar=True)
        for l in range(lags_number):
            values = var_selection_info[f'lag_{l}'].to_numpy()
            angles = np.linspace(0, 2 * np.pi, len(var_selection_info), endpoint=False).tolist()
            values = np.concatenate((values,[values[0]]))
            angles += angles[:1]
            ax.plot(angles, values, linewidth=4, linestyle='solid', label=f'lag_{l}')
            ax.fill(angles, values, alpha=0.1)
        # Set plots properties
        ax.set_xticks(angles[:-1])
        if var in atm_vars:
            ax.set_xticklabels(np.arange(1,n_clusters+1), fontsize=16)
            ax.set_title(f'{var}', fontsize=18, fontweight='bold')
        else:
            ax.set_xticklabels(var_selection_info['column_names'], fontsize=16)
            ax.set_title(f'No cluster var', fontsize=18, fontweight='bold')
        if display_percentage:
            ax.set_yticks(np.arange(101)[::20])
            ax.set_yticklabels((np.arange(101)[::20]), fontsize=12)
        else:
            ax.set_yticks(np.arange(experiments_considered+1)[::2])
            ax.set_yticklabels((np.arange(experiments_considered+1)[::2]), fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)

        if var in idx_vars:
            break

    fig.set_tight_layout(True)
    return fig

# Function to plot the spyder plots on the variables selection across different experiments and showing the selection for each model
def models_shares_vars_selection_spyder_plot(experiments_folders, n_clusters, selected_vars_df_list, atm_vars, idx_vars, display_percentage=False):
    # Set figure and grid for plotting
    lags_number = len(selected_vars_df_list[0].columns) - 1
    figs = []
    gs = []
    for lag in range(lags_number):
        fig = plt.figure(figsize=(16,16))
        figs.append(fig)
        gs.append(gridspec.GridSpec(3, 3, figure=fig))
    # Get the number of experiments for computing percentages + define cluster strings for axes
    experiments_considered = len(experiments_folders)
    cluster_strings = [f'cluster{i}' for i in range(1,n_clusters+1)]
    all_vars = atm_vars + idx_vars

    # Plot the spyder plots for each atmospheric variable 
    for i, var in enumerate(all_vars):
        if var in atm_vars:
            vars_selected = [f'{var}_{cluster}' for cluster in cluster_strings]
        else:
            vars_selected = idx_vars
        # Get the information of number of times the variable was selected for each cluster
        selected_vars_df = selected_vars_df_list[0]
        var_selection_info = selected_vars_df[selected_vars_df['column_names'].isin(vars_selected)]
        var_selection_info.iloc[:,1:] = 0.0
        # Create dataframe with the information for each model
        var_selection_info_linreg = var_selection_info.copy()
        var_selection_info_lgbm = var_selection_info.copy()
        for s, seleceted_vars_df in enumerate(selected_vars_df_list):
            # Add the information of the selection for each model to the dataframe
            sel_info = seleceted_vars_df[seleceted_vars_df['column_names'].isin(vars_selected)]
            var_selection_info.iloc[:,1:] = (var_selection_info.iloc[:,1:] + sel_info.iloc[:,1:])
            # Add the information of the selection for each model separetely
            if 'linreg' in experiments_folders[s]:
                var_selection_info_linreg.iloc[:,1:] = (var_selection_info_linreg.iloc[:,1:] + sel_info.iloc[:,1:])
            elif 'lgbm' in experiments_folders[s]:
                var_selection_info_lgbm.iloc[:,1:] = (var_selection_info_lgbm.iloc[:,1:] + sel_info.iloc[:,1:])
            else:
                raise ValueError('Model not recognized')
        # Compute the percentage of selection if requested
        if display_percentage:
            var_selection_info.iloc[:,1:] = var_selection_info.iloc[:,1:] / experiments_considered * 100
            var_selection_info_linreg.iloc[:,1:] = var_selection_info_linreg.iloc[:,1:] / experiments_considered * 100
            var_selection_info_lgbm.iloc[:,1:] = var_selection_info_lgbm.iloc[:,1:] / experiments_considered * 100
        # Plot in the spyder plot
        angles = np.linspace(0, 2 * np.pi, len(var_selection_info), endpoint=False).tolist()
        angles += angles[:1]
        for l in range(lags_number):
            # Get the values for each model
            values_linreg = var_selection_info_linreg[f'lag_{l}'].to_numpy()
            values_lgbm = var_selection_info_lgbm[f'lag_{l}'].to_numpy() + values_linreg
            values_linreg = np.concatenate((values_linreg,[values_linreg[0]]))
            values_lgbm = np.concatenate((values_lgbm,[values_lgbm[0]]))
            # Plot the values for each model
            ax = figs[l].add_subplot(gs[l][i], polar=True)
            ax.fill(angles, values_linreg, color='coral', label='linreg') # cornflowerblue
            ax.fill_between(angles, values_linreg, values_lgbm, color='tomato', label='lgbm') # royalblue
            # Set plots properties
            ax.set_xticks(angles[:-1])
            if var in atm_vars:
                ax.set_xticklabels(np.arange(1,n_clusters+1), fontsize=16)
                ax.set_title(f'{var}', fontsize=18, fontweight='bold')
            else:
                ax.set_xticklabels(var_selection_info['column_names'], fontsize=16)
                ax.set_title(f'No cluster var', fontsize=18, fontweight='bold')
            if display_percentage:
                ax.set_yticks(np.arange(101)[::20])
                ax.set_yticklabels((np.arange(101)[::20]), fontsize=12)
            else:
                ax.set_yticks(np.arange(experiments_considered+1)[::2])
                ax.set_yticklabels((np.arange(experiments_considered+1)[::2]), fontsize=12)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)

        if var in idx_vars:
            break
    
    for f, fig in enumerate(figs):
        fig.suptitle(f'Lag {f} - Shares of variable selection', fontsize=20)
        fig.set_tight_layout(True)
    return figs

# Function to plot the heatmaps for the selected features across different experiments
def vars_selection_heatmaps(experiments_folders, n_clusters, selected_vars_df_list, atm_vars, idx_vars, display_percentage=False, sel_percentage=60):
    # Create the dataframe with the total selection for each variable
    max_columns = np.array([len(selected_vars_df.columns) for selected_vars_df in selected_vars_df_list]).max()
    n_lags = max_columns - 1
    selected_vars_df_tot = pd.DataFrame({'column_names': selected_vars_df_list[0]['column_names']})
    for l in range(n_lags):
        selected_vars_df_tot[f'lag_{l}'] = 0.0
    for s, selected_vars_df in enumerate(selected_vars_df_list):
        if len(selected_vars_df.columns) < max_columns:
            selected_vars_df_tot.iloc[:,1:max_columns-1] = selected_vars_df_tot.iloc[:,1:max_columns-1] + selected_vars_df.iloc[:,1:max_columns-1]
        else:
            selected_vars_df_tot.iloc[:,1:] = selected_vars_df_tot.iloc[:,1:] + selected_vars_df.iloc[:,1:]
    # Create the heatmap figure, if only on lag is considered plot the heatmaps on one row, otherwise plot on two rows
    if n_lags == 1:
        if n_clusters >= 10:
            fig = plt.figure(figsize=(14, 4))
        else:
            fig = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(1, 2, figure=fig)
    else:
        if n_clusters >= 10:
            fig = plt.figure(figsize=(14, 8))
        else:
            fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 4, figure=fig)
    # Dataframe to keep track of the variables that are chosen x% of the time, x is determined by sel_percentage
    sel_vars_xp = pd.DataFrame(columns=['lag', 'selected_vars'])
    sel_vars_xp.set_index('lag', inplace=True)
    # Plot heatmaps for each lag for the cluster variables
    for l in range(n_lags):
        # Create dataframe for the heatmap
        cluster_var_heatmap_df = pd.DataFrame({'variables': atm_vars})
        for n in range(1, n_clusters+1):
            cluster_var_heatmap_df[f'cluster{n}'] = np.nan
        for r, row in cluster_var_heatmap_df.iterrows():
            var = row['variables']
            for n in range(1, n_clusters+1): 
                row[f'cluster{n}'] = selected_vars_df_tot[selected_vars_df_tot['column_names'] == f'{var}_cluster{n}'][f'lag_{l}'].values[0]
                cluster_var_heatmap_df.loc[r] = row
        cluster_var_heatmap_df.set_index('variables', inplace=True)
        # Plot the heatmap
        if n_lags == 1:
            ax = fig.add_subplot(gs[l])
        else:
            ax = fig.add_subplot(gs[l*2:(l*2)+2])
        if display_percentage:
            cluster_var_heatmap_df = cluster_var_heatmap_df / len(experiments_folders) * 100
            vmax = 100
        else:
            vmax = len(experiments_folders)
        ax = sns.heatmap(cluster_var_heatmap_df, annot=True, cmap='coolwarm', cbar=False, fmt='.0f', center=0, vmin=0, vmax=vmax)
        ax.set_title(f'Lag {l}')
        ax.set_ylabel('')
        ax.set_xlabel('Clusters', fontsize=12)
        ax.set_xticks(np.arange(n_clusters)+0.5, np.arange(1,n_clusters+1), rotation=0, fontsize=12)
        ax.set_yticks(np.arange(len(cluster_var_heatmap_df))+0.5, cluster_var_heatmap_df.index, rotation=0, fontsize=12)
        if display_percentage:
            for text in ax.texts:
                text.set_text(f"{(float(text.get_text())):.0f}%")
        # Add to the sel_vars_xp df list the variables that are selected x% of the time for each lag
        matches_xp = cluster_var_heatmap_df >= sel_percentage
        result = [f'{index}_{col}' for index, row in matches_xp.iterrows() for col in row.index if row[col]]
        sel_vars_xp.loc[l] = [result]
    # Plot the indexes heatmap
    if n_lags == 1:
        ax = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(gs[5:7])
    idxs_var_heatmap_df = selected_vars_df_tot[selected_vars_df_tot['column_names'].isin(idx_vars)]
    idxs_var_heatmap_df.set_index('column_names', inplace=True)
    if display_percentage:
        idxs_var_heatmap_df = idxs_var_heatmap_df / len(experiments_folders) * 100
    ax = sns.heatmap(idxs_var_heatmap_df, annot=True, cmap='coolwarm', cbar=False, fmt='.0f', center=0, vmin=0, vmax=vmax)
    ax.set_title(f'No cluster vars')
    ax.set_ylabel('')
    ax.set_yticks(np.arange(len(idxs_var_heatmap_df))+0.5, idxs_var_heatmap_df.index, rotation=0)
    if display_percentage:
        for text in ax.texts:
            text.set_text(f"{(float(text.get_text())):.0f}%")

    fig.set_tight_layout(True)
    # Add to the sel_vars_60 df list the indexes that are selected 60% of the time for each lag
    matches_xp = idxs_var_heatmap_df >= sel_percentage
    for l in range(n_lags):
        lag_sel = idxs_var_heatmap_df[f'lag_{l}'][matches_xp[f'lag_{l}']].index.to_list()
        current_sel = sel_vars_xp.loc[l].values.tolist()[0]
        if len(lag_sel) > 0:
            new_sel = current_sel + lag_sel
        else:
            new_sel = current_sel
        sel_vars_xp.loc[l] = [new_sel] 

    return fig, sel_vars_xp

# Function to plot the heatmaps for variable selection without considering the clusters
# So in case of cluster variables, the selection is counted if at least one cluster is selected
def vars_selection_heatmaps_no_cluster(experiments_folders, selected_vars_df_list, atm_vars, idx_vars, display_percentage=False):
    # Create a dataframe containing the information of the selected variables
    max_columns = np.array([len(selected_vars_df.columns) for selected_vars_df in selected_vars_df_list]).max()
    n_lags = max_columns - 1
    var_selection_tot = pd.DataFrame({'variables': atm_vars + idx_vars})
    for l in range(n_lags):
        var_selection_tot[f'lag_{l}'] = 0.0
    for s, selected_vars_df in enumerate(selected_vars_df_list):
        for v, var in enumerate(var_selection_tot['variables']):
            sel_var = selected_vars_df[selected_vars_df['column_names'].str.contains(var)].iloc[:,1:]
            sel_var = sel_var.sum(axis=0)
            sel_var[sel_var > 0] = 1.0
            if len(selected_vars_df.columns) < max_columns:
                var_selection_tot.iloc[v,1:max_columns-1] = var_selection_tot.iloc[v,1:max_columns-1] + sel_var
            else:
                var_selection_tot.iloc[v,1:] = var_selection_tot.iloc[v,1:] + sel_var
    # Create the heatmap
    var_selection_tot.set_index('variables', inplace=True)
    fig, ax = plt.subplots(figsize=(4, 6))
    if display_percentage:
        var_selection_tot = var_selection_tot / len(experiments_folders) * 100
    ax = sns.heatmap(var_selection_tot, annot=True, cmap='coolwarm', cbar=False, fmt='.0f', center=0, vmin=0, vmax=100)
    ax.set_ylabel('')
    if display_percentage:
        for text in ax.texts:
            text.set_text(f"{(float(text.get_text())):.0f}%")

    fig.set_tight_layout(True)
    return fig

# Function to plot a box plot of the correlation between the selected features and the target variable
def plot_correlation_boxplot(track_df, n_clusters=4, cluster_type='all', fitness_model='all', corr='R_Y'):
    # Raise error if one of the input is not recognized
    if n_clusters < 4 or n_clusters > 12:
        raise ValueError('Number of clusters not valid: must be between 4 and 12, 4 means all clusters number')
    if cluster_type not in ['all', '_nc', 'Anc', 'DSnc']:
        raise ValueError('Cluster type not recognized: must be one of all, _nc, Anc, DSnc')
    if fitness_model not in ['all', 'linreg', 'lgbm', 'pi-lgbm']:
        raise ValueError('Fitness model not recognized: must be one of all, linreg, lgbm, pi-lgbm')
    if corr not in ['R', 'R_S', 'R_Y']:
        raise ValueError('Correlation type not recognized: must be one of R, R_S, R_Y')
    # Load the data
    df = track_df.copy()
    # Filter the data according to the input
    if n_clusters > 4:
        df = df[df['n_clusters'] == n_clusters]
    if cluster_type != 'all':
        df = df[df.index.str.contains(cluster_type)]
    if fitness_model != 'all':
        df = df[df.index.str.contains(fitness_model)]
    # Get the best correlation out of the fitted final models
    performance_col = [f'{corr}_mlp', f'{corr}_pi-mlp', f'{corr}_lgbm', f'{corr}_pi-lgbm']
    df[f'max_{corr}'] = df[performance_col].max(axis=1)
    performance_col_noFS = [f'{corr}_mlp_noFS', f'{corr}_pi-mlp_noFS', f'{corr}_lgbm_noFS', f'{corr}_pi-lgbm_noFS']
    df[f'max_{corr}_noFS'] = df[performance_col_noFS].max(axis=1)
    # Plot the boxplot
    BestR = df[f'max_{corr}'].to_numpy()
    BestR_noFS = df[f'max_{corr}_noFS'].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([BestR, BestR_noFS], labels=['FS', 'NoFS'], showfliers=True, showmeans=True, meanline=True)
    ax.set_ylabel(f'Best {corr} value')
    return fig, ax

# Functions to display saved pdf or matplotlib figure in the notebook or in dashboard
def load_pdf_convert_to_image(pdf_path):
    with open(pdf_path, "rb") as file:
        image = convert_from_path(pdf_path)[0]
    return image

def PIL_to_widget(pil_img, wdt, hgt):
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    # Create an Image widget from byte data
    image_widget = widgets.Image(
        value=img_byte_arr,
        format='png',
        width=f"{wdt}px",
        height=f"{hgt}px",
        align_self='center'
    )
    return image_widget 

def figure_to_PIL(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    pil_image = Image.frombuffer('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
    return pil_image

# Function that from the basin name and simualtion folder name returns several data to be used in the feature importance analysis
# + sensitivity analysis on the percentage of the selected features in the best models
def runs_info(basin, run_name):
    # Set some additional variables and parameters that generally stay constant
    years = np.arange(1980, 2022, 1) # from 1980 to 2021 included
    n_folds = 3
    n_clusters = int(run_name.split('nc')[1].split('_')[0])
    n_vars = int(run_name.split('nv')[1].split('_')[0])
    n_idxs = int(run_name.split('nd')[1].split('_')[0])
    model_kind = run_name.split('_')[1]
    # Set directories and files names
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    target_file = 'target_1970-2022_2.5x2.5.csv'
    experiment_filename = f'1970-2022_{n_clusters}clusters_{n_vars}vars_{n_idxs}idxs.csv'
    # Retrieve the clusters type of data from the results folder
    nc_string = run_name.split('_')[2]
    if "A" in nc_string:
        cluster_data = f'{basin}_{n_clusters}clusters_anomaly'
    elif "DS" in nc_string:
        cluster_data = f'{basin}_{n_clusters}clusters_deseason'
        target_season = 'target_seasonality_1970-2022_2.5x2.5.csv'
    else:
        cluster_data = f'{basin}_{n_clusters}clusters'
    # Set the paths to the files
    predictor_file = 'predictors_' + experiment_filename
    fs_dir = os.path.join(project_dir, 'xai-gpi')
    results_dir = os.path.join(fs_dir, 'results')
    output_dir = os.path.join(fs_dir, 'results', basin, run_name)
    data_dir = os.path.join(fs_dir, 'data', cluster_data)
    predictors_path = os.path.join(data_dir, predictor_file)
    final_analysis_dir = os.path.join(output_dir, 'final_analysis')
    target_path = os.path.join(data_dir, target_file)
    gpis_path = os.path.join(fs_dir, 'data', f'{basin}_2.5x2.5_gpis_time_series.csv')
    # Load the predictors and the target in a DataFrame
    predictors_df = pd.read_csv(predictors_path, index_col=0)
    predictors_df.index = pd.to_datetime(predictors_df.index)
    target_df = pd.read_csv(target_path, index_col=0)
    target_df.index = pd.to_datetime(target_df.index)
    if "DS" in nc_string:
        target_season_df = pd.read_csv(os.path.join(data_dir, target_season), index_col=0)
        target_season_df.index = pd.to_datetime(target_season_df.index)
    # Load the gpis time series dataframe and select the target GPIs for physical information to pass to the network
    gpis_df = pd.read_csv(gpis_path, index_col=0)
    gpis_df.index = pd.to_datetime(gpis_df.index)
    gpi_pi = gpis_df['ogpi']
    # Load the best solution file if it is a test run
    if "test" in run_name:
        sol_filename = f'{model_kind}_' + experiment_filename
        best_sol_path = os.path.join(output_dir, f'best_solution_{sol_filename}')
        # Load the solutions file in a DataFrame and the best solution found
        best_solution = pd.read_csv(best_sol_path, sep=',', header=None)
        best_solution = best_solution.to_numpy().flatten()
        # Select the variables from the best solutions
        column_names = predictors_df.columns.tolist()
        final_sequence = best_solution[len(column_names):2*len(column_names)]
        sequence_length = best_solution[:len(column_names)]
        feat_sel = best_solution[2*len(column_names):]
    
    # Create dataset according to solution and list the labels of the selected variables
    if "test" in run_name:
        variable_selection = feat_sel.astype(int)
        time_sequences = sequence_length.astype(int)
        time_lags = final_sequence.astype(int)
        dataset_opt = target_df.copy()
        for c, col in enumerate(predictors_df.columns):
            if variable_selection[c] == 0 or time_sequences[c] == 0:
                continue
            for j in range(time_sequences[c]):
                dataset_opt[str(col) +'_lag'+ str(time_lags[c]+j)] = predictors_df[col].shift(time_lags[c]+j)
    else:
        # features selected >= sel_perc% of the time in the top20% best models
        sel_feat_perc_path = os.path.join(results_dir, f'selected_features_best_models_{basin}_{nc_string}.csv')
        df_sel_feat_perc = pd.read_csv(sel_feat_perc_path, index_col=0)
        sel_perc = run_name.split('_')[0].split('selfeat')[1]
        selected_features = df_sel_feat_perc[sel_perc].dropna().to_list()
        dataset_opt = predictors_df[selected_features]
        dataset_opt.columns = [f'{feat}_lag0' for feat in dataset_opt.columns]
        dataset_opt = dataset_opt.assign(tcg=target_df['tcg'])
    # Compone the dataset to train the model using all predictors possible
    dataset_opt_noFS = target_df.copy()
    for l in range(1):
        for var in predictors_df.columns:
            col_df = pd.DataFrame(predictors_df[var].shift(l).values, index=dataset_opt_noFS.index, columns=[f'{var}_lag{l}'])
            dataset_opt_noFS = pd.concat([dataset_opt_noFS, col_df], axis=1)

    ## Make predictions with the best solution found ##
    # Cross-Validation for train and test years
    kfold = KFold(n_splits=n_folds)
    Y_column = 'tcg' # Target variable
    Y_pred = []
    Y_pred_noFS = []
    Y_test = []
    X_test_eval = []
    X_test_eval_noFS = []
    mlps = []
    mlps_noFS = []
    # List to store the results of feature permutation importance and SHAP values
    perm_importance_mlp = []
    perm_importance_mlp_noFS = []
    shap_values_mlp = []
    shap_values_mlp_noFS = []

    for n_fold, (train_index, test_index) in enumerate(kfold.split(years)):

        # Set the indices for the training and test datasets
        train_years = years[train_index]
        test_years = years[test_index]
        # Split the optimized dataset
        train_indices = dataset_opt.index.year.isin(train_years)
        test_indices = dataset_opt.index.year.isin(test_years)
        train_dataset = dataset_opt[train_indices]
        test_dataset = dataset_opt[test_indices]
        # Split the entire dataset 
        train_indices_noFS = dataset_opt_noFS.index.year.isin(train_years)
        test_indices_noFS = dataset_opt_noFS.index.year.isin(test_years)
        train_dataset_noFS = dataset_opt_noFS[train_indices_noFS]
        test_dataset_noFS = dataset_opt_noFS[test_indices_noFS]
        # Split the gpis dataset
        gpi_pi_train = gpi_pi[train_indices]
        gpi_pi_test = gpi_pi[test_indices]

        # Standardize the optimized dataset
        X_train = train_dataset[train_dataset.columns.drop([Y_column])]
        Y_train = train_dataset[Y_column]
        X_test_fold = test_dataset[test_dataset.columns.drop([Y_column])]
        Y_test_fold = test_dataset[Y_column]
        scaler = preprocessing.MinMaxScaler()
        X_std_train = scaler.fit(X_train)
        X_std_train = scaler.transform(X_train)
        X_std_test = scaler.transform(X_test_fold)
        X_train = pd.DataFrame(X_std_train, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_std_test, columns=X_test_fold.columns, index=X_test_fold.index)
        # Append X_test to a list to use it later for SHAP explainability
        feature_names = ['{}'.format(col.split('_l')[0]) for col in np.array(X_test.columns)]
        xt = X_test
        xt.columns = feature_names
        X_test_eval.append(xt)
        # Standardize the entire dataset
        X_train_noFS = train_dataset_noFS[train_dataset_noFS.columns.drop([Y_column])]
        X_test_fold_noFS = test_dataset_noFS[test_dataset_noFS.columns.drop([Y_column])]
        scaler_noFS = preprocessing.MinMaxScaler()
        X_std_train_noFS = scaler_noFS.fit(X_train_noFS)
        X_std_train_noFS = scaler_noFS.transform(X_train_noFS)
        X_std_test_noFS = scaler_noFS.transform(X_test_fold_noFS)
        X_train_noFS = pd.DataFrame(X_std_train_noFS, columns=X_train_noFS.columns, index=X_train_noFS.index)
        X_test_noFS = pd.DataFrame(X_std_test_noFS, columns=X_test_fold_noFS.columns, index=X_test_fold_noFS.index)
        # Append X_test_noFS to a list to use it later for SHAP explainability
        feature_names_noFS = ['{}'.format(col.split('_l')[0]) for col in np.array(X_test_noFS.columns)]
        xt_noFS = X_test_noFS
        xt_noFS.columns = feature_names_noFS
        X_test_eval_noFS.append(xt_noFS)
        # Load the models
        mlp = load_model(os.path.join(final_analysis_dir, 'models', f'mlp_fold{n_fold+1}.keras'))
        mlp_noFS = load_model(os.path.join(final_analysis_dir, 'models', f'mlp_noFS_fold{n_fold+1}.keras'))
        mlps.append(mlp)
        mlps_noFS.append(mlp_noFS)
        # Append the predictions to a list
        Y_pred_fold = mlp.predict(X_test, verbose=0)
        Y_pred_fold = pd.DataFrame(Y_pred_fold, index=Y_test_fold.index, columns=['tcg'])
        Y_pred.append(Y_pred_fold)
        Y_test.append(Y_test_fold)
        Y_pred_fold_noFS = mlp_noFS.predict(X_test_noFS, verbose=0)
        Y_pred_fold_noFS = pd.DataFrame(Y_pred_fold_noFS, index=Y_test_fold.index, columns=['tcg'])
        Y_pred_noFS.append(Y_pred_fold_noFS)
        # Load the permutation importance results
        perm_importance_mlp.append(np.load(os.path.join(final_analysis_dir, 'explain_data', f'perm_imp_mlp_fold{n_fold+1}.npz')))
        perm_importance_mlp_noFS.append(np.load(os.path.join(final_analysis_dir, 'explain_data', f'perm_imp_mlp_noFS_fold{n_fold+1}.npz')))
        # Load the SHAP values
        npz_mpl = np.load(os.path.join(final_analysis_dir, 'explain_data', f'shap_mlp_fold{n_fold+1}.npz'), allow_pickle=True)
        expl_mlp = shap.Explanation(values=npz_mpl["shap_values"], base_values=npz_mpl["base_values"], data=npz_mpl["data"], feature_names=npz_mpl["feature_names"])
        shap_values_mlp.append(expl_mlp)
        npz_mpl_noFS = np.load(os.path.join(final_analysis_dir, 'explain_data', f'shap_mlp_noFS_fold{n_fold+1}.npz'), allow_pickle=True)
        expl_mlp_noFS = shap.Explanation(values=npz_mpl_noFS["shap_values"], base_values=npz_mpl_noFS["base_values"], data=npz_mpl_noFS["data"], feature_names=npz_mpl_noFS["feature_names"])
        shap_values_mlp_noFS.append(expl_mlp_noFS)

    return dataset_opt, dataset_opt_noFS, Y_pred, Y_pred_noFS, Y_test, X_test_eval, X_test_eval_noFS, mlps, mlps_noFS, perm_importance_mlp, perm_importance_mlp_noFS, shap_values_mlp, shap_values_mlp_noFS