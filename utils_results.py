import os
import numpy as np
import pandas as pd
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

def plot_board(board, column_names, feat_sel, correlations, corr_report=False):
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
            ax.text(-0.1, pos - 1.25, f'{correlations[i]:.2f}', fontsize=10, color=text_color)
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
    fig = plt.figure(figsize=(15,15))
    gs = gridspec.GridSpec(3, 3)
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
        # Plot in the spyder plot
        ax = fig.add_subplot(gs[i], polar=True)
        values_lag_0 = var_selection_info['lag_0'].to_numpy()
        values_lag_1 = var_selection_info['lag_1'].to_numpy()
        angles = np.linspace(0, 2 * np.pi, len(var_selection_info), endpoint=False).tolist()
        values_lag_0 = np.concatenate((values_lag_0,[values_lag_0[0]]))
        values_lag_1 = np.concatenate((values_lag_1,[values_lag_1[0]]))
        angles += angles[:1]
        ax.plot(angles, values_lag_0, linewidth=4, linestyle='solid', label='lag_0')
        ax.fill(angles, values_lag_0, alpha=0.1)
        ax.plot(angles, values_lag_1, linewidth=4, linestyle='dashed', label='lag_1')
        ax.fill(angles, values_lag_1, alpha=0.1)
        # Set plots properties
        ax.set_xticks(angles[:-1])
        if var in atm_vars:
            ax.set_xticklabels(np.arange(1,n_clusters+1), fontsize=16)
            ax.set_title(f'{var}', fontsize=18, fontweight='bold')
        else:
            ax.set_xticklabels(var_selection_info['column_names'], fontsize=16)
            ax.set_title(f'No cluster var', fontsize=18, fontweight='bold')
        if display_percentage:
            ax.set_yticks(np.arange(101)[::10])
            ax.set_yticklabels((np.arange(101)[::10]), fontsize=12)
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
    fig1 = plt.figure(figsize=(16,16))
    gs1 = gridspec.GridSpec(3, 3, figure=fig1)
    fig2 = plt.figure(figsize=(16,16))
    gs2 = gridspec.GridSpec(3, 3, figure=fig2)
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
        var_selection_info_LinReg = var_selection_info.copy()
        var_selection_info_LGBM = var_selection_info.copy()
        var_selection_info_XGB = var_selection_info.copy()
        for s, seleceted_vars_df in enumerate(selected_vars_df_list):
            # Add the information of the selection for each model to the dataframe
            sel_info = seleceted_vars_df[seleceted_vars_df['column_names'].isin(vars_selected)]
            var_selection_info.iloc[:,1:] = (var_selection_info.iloc[:,1:] + sel_info.iloc[:,1:])
            # Add the information of the selection for each model separetely
            if 'LinReg' in experiments_folders[s]:
                var_selection_info_LinReg.iloc[:,1:] = (var_selection_info_LinReg.iloc[:,1:] + sel_info.iloc[:,1:])
            elif 'LGBM' in experiments_folders[s]:
                var_selection_info_LGBM.iloc[:,1:] = (var_selection_info_LGBM.iloc[:,1:] + sel_info.iloc[:,1:])
            elif 'XGB' in experiments_folders[s]:
                var_selection_info_XGB.iloc[:,1:] = (var_selection_info_XGB.iloc[:,1:] + sel_info.iloc[:,1:])
            else:
                raise ValueError('Model not recognized')
        # Compute the percentage of selection if requested
        if display_percentage:
            var_selection_info.iloc[:,1:] = var_selection_info.iloc[:,1:] / experiments_considered * 100
            var_selection_info_LinReg.iloc[:,1:] = var_selection_info_LinReg.iloc[:,1:] / experiments_considered * 100
            var_selection_info_LGBM.iloc[:,1:] = var_selection_info_LGBM.iloc[:,1:] / experiments_considered * 100
            var_selection_info_XGB.iloc[:,1:] = var_selection_info_XGB.iloc[:,1:] / experiments_considered * 100
        # Plot in the spyder plot
        ax1 = fig1.add_subplot(gs1[i], polar=True)
        ax2 = fig2.add_subplot(gs2[i], polar=True)
        # All models
        values_lag_0 = var_selection_info['lag_0'].to_numpy()
        values_lag_1 = var_selection_info['lag_1'].to_numpy()
        values_lag_0 = np.concatenate((values_lag_0,[values_lag_0[0]]))
        values_lag_1 = np.concatenate((values_lag_1,[values_lag_1[0]]))
        angles = np.linspace(0, 2 * np.pi, len(var_selection_info), endpoint=False).tolist()
        angles += angles[:1]
        ax1.plot(angles, values_lag_0, linewidth=4, linestyle='solid', color='red', label='All')
        ax2.plot(angles, values_lag_1, linewidth=4, linestyle='dashed', color='midnightblue', label='All')
        # LinReg
        values_lag_0_LinReg = var_selection_info_LinReg['lag_0'].to_numpy()
        values_lag_1_LinReg = var_selection_info_LinReg['lag_1'].to_numpy()
        values_lag_0_LinReg = np.concatenate((values_lag_0_LinReg,[values_lag_0_LinReg[0]]))
        values_lag_1_LinReg = np.concatenate((values_lag_1_LinReg,[values_lag_1_LinReg[0]]))
        ax1.fill(angles, values_lag_0_LinReg, color='coral', label='LinReg')
        ax2.fill(angles, values_lag_1_LinReg, color='cornflowerblue', label='LinReg')
        # LGBM
        values_lag_0_LGBM = var_selection_info_LGBM['lag_0'].to_numpy() + var_selection_info_LinReg['lag_0'].to_numpy()
        values_lag_1_LGBM = var_selection_info_LGBM['lag_1'].to_numpy() + var_selection_info_LinReg['lag_1'].to_numpy()
        values_lag_0_LGBM = np.concatenate((values_lag_0_LGBM,[values_lag_0_LGBM[0]]))
        values_lag_1_LGBM = np.concatenate((values_lag_1_LGBM,[values_lag_1_LGBM[0]]))
        ax1.fill_between(angles, values_lag_0_LinReg, values_lag_0_LGBM, color='tomato', label='LGBM')
        ax2.fill_between(angles, values_lag_1_LinReg, values_lag_1_LGBM, color='royalblue', label='LGBM')
        # XGB
        values_lag_0_XGB = var_selection_info_XGB['lag_0'].to_numpy() + var_selection_info_LGBM['lag_0'].to_numpy() + var_selection_info_LinReg['lag_0'].to_numpy()
        values_lag_1_XGB = var_selection_info_XGB['lag_1'].to_numpy() + var_selection_info_LGBM['lag_1'].to_numpy() + var_selection_info_LinReg['lag_1'].to_numpy()
        values_lag_0_XGB = np.concatenate((values_lag_0_XGB,[values_lag_0_XGB[0]]))
        values_lag_1_XGB = np.concatenate((values_lag_1_XGB,[values_lag_1_XGB[0]]))
        ax1.fill_between(angles, values_lag_0_LGBM, values_lag_0_XGB, color='orangered', label='XGB')
        ax2.fill_between(angles, values_lag_1_LGBM, values_lag_1_XGB, color='blue', label='XGB')
        # Set plots properties
        ax1.set_xticks(angles[:-1])
        ax2.set_xticks(angles[:-1])
        if var in atm_vars:
            ax1.set_xticklabels(np.arange(1,n_clusters+1), fontsize=16)
            ax2.set_xticklabels(np.arange(1,n_clusters+1), fontsize=16)
            ax1.set_title(f'{var}', fontsize=18, fontweight='bold')
            ax2.set_title(f'{var}', fontsize=18, fontweight='bold')
        else:
            ax1.set_xticklabels(var_selection_info['column_names'], fontsize=16)
            ax2.set_xticklabels(var_selection_info['column_names'], fontsize=16)
            ax1.set_title(f'No cluster var', fontsize=18, fontweight='bold')
            ax2.set_title(f'No cluster var', fontsize=18, fontweight='bold')
        if display_percentage:
            ax1.set_yticks(np.arange(101)[::10])
            ax1.set_yticklabels((np.arange(101)[::10]), fontsize=12)
            ax2.set_yticks(np.arange(101)[::10])
            ax2.set_yticklabels((np.arange(101)[::10]), fontsize=12)
        else:
            ax1.set_yticks(np.arange(experiments_considered+1)[::2])
            ax1.set_yticklabels((np.arange(experiments_considered+1)[::2]), fontsize=12)
            ax2.set_yticks(np.arange(experiments_considered+1)[::2])
            ax2.set_yticklabels((np.arange(experiments_considered+1)[::2]), fontsize=12)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)

        if var in idx_vars:
            break

    fig1.set_tight_layout(True)
    fig2.set_tight_layout(True)
    return fig1, fig2