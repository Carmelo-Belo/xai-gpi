import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

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
def plot_selected_clusters(n_clusters, label_selected_vars, data_dir, results_figure_dir):
    # Create the subfolder of figure results to store the cluster selection figures
    save_figure_dir = os.path.join(results_figure_dir, 'clusters_selected')
    os.makedirs(save_figure_dir, exist_ok=True)
    # Get the variable names from the selected variables
    variables_with_cluster = [var for var in label_selected_vars if 'cluster' in var]
    variable_names = [var.split('_cluster')[0] for var in variables_with_cluster]
    variable_names = list(set(variable_names))
    variable_names.sort()

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
        
        # Set the domain extension of the figures
        north, south = label_df['nodes_lat'].iloc[0], label_df['nodes_lat'].iloc[-1]
        west, east = label_df['nodes_lon'].iloc[0], label_df['nodes_lon'].iloc[-1]
        
        # Cycle through the lags and select the clusters for each lag
        for lag in np.arange(2):
            clusters_for_lag = clusters_selected[time_lags_selected == lag] 
            # Plot the clusters for the selected variable and lag only if there are clusters selected
            if len(clusters_for_lag) > 0:
                # Select the rows of the label file that correspond to the selected clusters
                label_df_selected = label_df[label_df['cluster'].isin(clusters_for_lag)]
                
                # Set the figure and gridlines of the map
                fig = plt.figure(figsize=(30, 6))
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
def vars_selection_heatmaps(experiments_folders, n_clusters, selected_vars_df_list, atm_vars, idx_vars, display_percentage=False):
    # Create the dataframe with the total selection for each variable
    max_columns = np.array([len(selected_vars_df.columns) for selected_vars_df in selected_vars_df_list]).max()
    n_lags = max_columns - 1
    selected_vars_df_tot = pd.DataFrame({'column_names': selected_vars_df_list[0]['column_names']})
    for l in range(n_lags):
        selected_vars_df_tot[f'lag_{l}'] = 0
    for s, selected_vars_df in enumerate(selected_vars_df_list):
        if len(selected_vars_df.columns) < max_columns:
            selected_vars_df_tot.iloc[:,1:max_columns-1] = selected_vars_df_tot.iloc[:,1:max_columns-1] + selected_vars_df.iloc[:,1:max_columns-1]
        else:
            selected_vars_df_tot.iloc[:,1:] = selected_vars_df_tot.iloc[:,1:] + selected_vars_df.iloc[:,1:]

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
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
        ax = fig.add_subplot(gs[l])
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
    # Plot the indexes heatmap
    ax = fig.add_subplot(gs[3])
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
    return fig

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
