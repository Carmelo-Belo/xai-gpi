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
    column_names = predictors_df.columns.to_list()
    final_sequence = best_solution[len(column_names):2*len(column_names)]
    sequence_length = best_solution[:len(column_names)]
    feat_sel = best_solution[2*len(column_names):]
    n_rows = len(column_names)
    n_cols = int(((sequence_length + final_sequence)*feat_sel).max())
    board_best = create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel)
    df = pd.DataFrame({'column_names': column_names, 'lag_0': board_best[:, 0], 'lag_1': board_best[:, 1]})
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