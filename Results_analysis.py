import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from scipy.stats import pearsonr
import utils_results as ut

def main(n_clusters, n_vars, n_idxs, results_folder, basin, model_kind, n_folds, start_year, end_year):
    
    # Set project directory and name of file containing the target variable
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    target_file = 'target_1965-2022_2.5x2.5.csv'

    # Set the paths to the files
    experiment_filename = f'1965-2022_{n_clusters}clusters_{n_vars}vars_{n_idxs}idxs.csv'
    sol_filename = f'{model_kind}_' + experiment_filename
    predictor_file = 'predictors_' + experiment_filename
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    output_dir = os.path.join(fs_dir, 'results', results_folder)
    sol_path = os.path.join(output_dir, sol_filename)
    # final_sol_path = os.path.join(output_dir, f'CRO_{sol_filename}')
    data_dir = os.path.join(fs_dir, 'data', f'{basin}_{n_clusters}clusters')
    predictors_path = os.path.join(data_dir, predictor_file)
    target_path = os.path.join(data_dir, target_file)
    results_figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(results_figure_dir, exist_ok=True)

    # Load the predictors and the target in a DataFrame
    predictors_df = pd.read_csv(predictors_path, index_col=0)
    predictors_df.index = pd.to_datetime(predictors_df.index)
    target_df = pd.read_csv(target_path, index_col=0)
    target_df.index = pd.to_datetime(target_df.index)

    # Load the labels files and plot the clusters for each atmospheric variable
    files_labels = os.listdir(data_dir)
    files_labels = [file for file in files_labels if file.startswith('label')]
    files_labels.sort()

    # for label_file in files_labels:
    #     label_df = pd.read_csv(os.path.join(data_dir, label_file), index_col=0)
        
    #     # Set the figure and domain extension
    #     north, south = label_df['nodes_lat'].iloc[0], label_df['nodes_lat'].iloc[-1]
    #     west, east = label_df['nodes_lon'].iloc[0], label_df['nodes_lon'].iloc[-1]
    #     fig = plt.figure(figsize=(30, 6))
    #     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    #     ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    #     ax.coastlines(resolution='110m', linewidth=2)
    #     ax.add_feature(cfeature.BORDERS, linestyle=':')

    #     # Set the gridlines of the map 
    #     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    #     gl.top_labels = False
    #     gl.right_labels = False
    #     gl.xformatter = LongitudeFormatter()
    #     gl.yformatter = LatitudeFormatter()
    #     gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    #     gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    #     gl.xlabel_style = {'size': 20} 
    #     gl.ylabel_style = {'size': 20}

    #     # Get data for the plot and plot the clusters
    #     n_clusters = len(np.unique(label_df['cluster']))
    #     cmap = plt.cm.get_cmap('tab20', n_clusters)
    #     scatter = ax.scatter(label_df['nodes_lon'].values, label_df['nodes_lat'].values, c=label_df['cluster'], cmap=cmap, s=400, transform=ccrs.PlateCarree())

    #     # Add colorbar
    #     bounds = np.arange(n_clusters + 1) - 0.5
    #     cbar = plt.colorbar(scatter, ticks=np.arange(n_clusters), boundaries=bounds, ax=ax, orientation='vertical')
    #     cbar.set_ticklabels(np.arange(n_clusters)+1)
    #     cbar.ax.tick_params(labelsize=22)
    #     cbar.set_label('Cluster',fontsize=26)

    #     ax.set_title(label_file[7:-9], fontsize=30)

    #     plt.tight_layout()
    #     plt.show()

    # Load the solutione file in a DataFrame
    sol_file_df = pd.read_csv(sol_path, sep=' ', header=0)

    # Find solutions with best performance both for Cross-Validation and Test metric
    CVbest = sol_file_df['CV'].idxmin() # metric is mean squared error
    Testbest = sol_file_df['Test'].idxmin()
    array_bestCV = np.fromstring(sol_file_df['Sol'][CVbest].replace('[', '').replace(']', '').replace('\n', ''), dtype=float, sep=' ')
    array_bestTest = np.fromstring(sol_file_df['Sol'][Testbest].replace('[', '').replace(']', '').replace('\n', ''), dtype=float, sep=' ')

    # Plot the evolution of the metric for each solution found per evaluation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sol_file_df.index, sol_file_df['CV'], label='Cross-Validation')
    ax.plot(sol_file_df.index, sol_file_df['Test'], label='Test')
    ax.scatter(sol_file_df.index[CVbest], sol_file_df['CV'][CVbest], color='black', label='Best CV')
    ax.scatter(sol_file_df.index[Testbest], sol_file_df['Test'][Testbest], color='green', label='Best Test')
    ax.legend()
    ax.set_xlabel('Solutions')
    ax.set_ylabel('MSE')
    plt.tight_layout()
    fig.savefig(os.path.join(results_figure_dir, f'CV_sol_evolution.pdf'), format='pdf', dpi=300)

    # Scatter plot of Cross-Validation and Test metric for the solutions found, highlightig the 2 bests, also adding the trendline
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sol_file_df['CV'], sol_file_df['Test'], label='Solutions')
    ax.scatter(sol_file_df['CV'][CVbest], sol_file_df['Test'][CVbest], color='black', marker='x', s=100, linewidths=2, label='Best CV')
    ax.scatter(sol_file_df['CV'][Testbest], sol_file_df['Test'][Testbest], color='green', marker='x', s=100, linewidths=2, label='Best Test')
    z = np.polyfit(sol_file_df['CV'], sol_file_df['Test'], 1)
    p = np.poly1d(z)
    ax.plot(sol_file_df['CV'], p(sol_file_df['CV']), 'r', label=f'Trendline {p}', linewidth=2)
    ax.legend()
    ax.grid()
    ax.set_xlabel('Cross-Validation MSE')
    ax.set_ylabel('Test MSE')
    plt.tight_layout()
    fig.savefig(os.path.join(results_figure_dir, f'CV_sol_scatter.pdf'), format='pdf', dpi=300)

    # Select the variables from the best solutions and plot it
    column_names = predictors_df.columns.tolist()
    final_sequence = array_bestCV[len(column_names):2*len(column_names)]
    sequence_length = array_bestCV[:len(column_names)]
    feat_sel = array_bestCV[2*len(column_names):]
    n_rows = len(column_names)
    n_cols = int(((sequence_length + final_sequence)*feat_sel).max())
    board_best = ut.create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel)
    fig_board = ut.plot_board(board_best, column_names, feat_sel)
    fig_board.savefig(os.path.join(results_figure_dir, f'best_sol.pdf'), format='pdf', dpi=300)

    ## Train MLPregressor with the best solution found ##
    # Create dataset according to solution
    variable_selection = feat_sel.astype(int)
    time_sequences = sequence_length.astype(int)
    time_lags = final_sequence.astype(int)
    dataset_opt = target_df.copy()
    for c, col in enumerate(predictors_df.columns):
        if variable_selection[c] == 0 or time_sequences[c] == 0:
            continue
        for j in range(time_sequences[c]):
            dataset_opt[str(col) +'_lag'+ str(time_lags[c]+j)] = predictors_df[col].shift(time_lags[c]+j)

    # Cross-Validation for train and test years
    years = np.arange(start_year, end_year, 1)
    kfold = KFold(n_splits=n_folds)
    Y_column = 'tcg' # Target variable
    obs_indices = dataset_opt.index.year.isin(years)
    obs_dataset = dataset_opt[obs_indices]
    Y_test = obs_dataset[Y_column]
    Y_pred = pd.DataFrame()

    for n_fold, (train_index, test_index) in enumerate(kfold.split(years)):

        # Set the indices for the training and test datasets and split the dataset
        train_years = years[train_index]
        test_years = years[test_index]
        train_indices = dataset_opt.index.year.isin(train_years)
        test_indices = dataset_opt.index.year.isin(test_years)
        train_dataset = dataset_opt[train_indices]
        test_dataset = dataset_opt[test_indices]

        # Standardize the dataset
        X_train = train_dataset[train_dataset.columns.drop([Y_column])]
        Y_train = train_dataset[Y_column]
        X_test_fold = test_dataset[test_dataset.columns.drop([Y_column])]
        Y_test_fold = test_dataset[Y_column]
        scaler = preprocessing.StandardScaler()
        X_std_train = scaler.fit(X_train)
        X_std_train = scaler.transform(X_train)
        X_std_test = scaler.transform(X_test_fold)
        X_train = pd.DataFrame(X_std_train, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_std_test, columns=X_test_fold.columns, index=X_test_fold.index)

        # Build and compile the multi layer perceptron model 
        n_predictors = len(X_train.columns)
        mlpreg = Sequential([
            Dense(units=n_predictors*2, activation='relu', input_shape=(n_predictors,)),
            Dense(units=n_predictors, activation='relu'),
            Dense(units=1)
        ])
        mlpreg.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss='mse',
            metrics=['mae']
        )

        # Train the model and predict the target variable
        X_t, X_v, Y_t, Y_v = train_test_split(X_train, Y_train, test_size=0.2)
        history = mlpreg.fit(x=X_t, y=Y_t, validation_data=(X_v, Y_v), epochs=100, batch_size=32, verbose=0)
        Y_pred_fold = mlpreg.predict(X_test)
        Y_pred_fold = pd.DataFrame(Y_pred_fold, index=Y_test_fold.index, columns=['tcg'])
        Y_pred = pd.concat([Y_pred, Y_pred_fold])

        # Evaluate the model
        loss, metric = mlpreg.evaluate(X_test, Y_test_fold)

        # Plot the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.axhline(y=loss, color='black', linestyle='--', label='Test Loss')
        plt.axhline(y=metric, color='red', linestyle='--', label='Test MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_figure_dir, f'FS-MLP_Train_Val_Loss_{n_fold}.pdf'), format='pdf', dpi=300)

    # Compare observations to predictions
    r_mlpreg, _ = pearsonr(Y_test, Y_pred['tcg'])
    xticks = pd.Series(Y_test.index).dt.strftime('%m-%Y').to_numpy()
    plt.figure(figsize=(20, 6))
    plt.plot(xticks, Y_test, label='Observed (IBTrACS)', color='#15E6CD',)
    plt.plot(xticks, Y_pred['tcg'], label=f'FS-MLP - R:{r_mlpreg:.3f}', color='#0CF574')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(ticks=np.arange(len(xticks))[::4], labels=xticks[::4], rotation=45)
    plt.xlabel('Months')
    plt.ylabel('# of TCs per month')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_figure_dir, f'FS-MLP_Monthly.pdf'), format='pdf', dpi=300)

    # Compare seasonal accumulated number of TCs
    Y_test_seasonal = Y_test.groupby(Y_test.index.month).sum()
    Y_pred_seasonal = Y_pred.groupby(Y_pred.index.month).sum()
    rS_mlpreg, _ = pearsonr(Y_test_seasonal, Y_pred_seasonal['tcg'])
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test_seasonal.index, Y_test_seasonal, label='Observed (IBTrACS)', color='#15E6CD', linewidth=3)
    plt.plot(Y_pred_seasonal.index, Y_pred_seasonal['tcg'], label=f'FS-MLP - R:{rS_mlpreg:.3f}', color='#0CF574', linewidth=3)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Months')
    plt.ylabel('# of TCs per month')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_figure_dir, f'FS-MLP_Seasonal.pdf'), format='pdf', dpi=300)

    # Compare annual accumulated number of TCs
    Y_test_annual = Y_test.resample('YE').sum()
    Y_pred_annual = Y_pred.resample('YE').sum()
    rY_mlpreg, _ = pearsonr(Y_test_annual, Y_pred_annual['tcg'])
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test_annual.index.year, Y_test_annual, label='Observed (IBTrACS)', color='#15E6CD', linewidth=3)
    plt.plot(Y_pred_annual.index.year, Y_pred_annual['tcg'], label=f'FS-MLP - {rY_mlpreg:.3f}', color='#0CF574', linewidth=3)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Years')
    plt.ylabel('# of TCs per year')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_figure_dir, f'FS-MLP_Annual.pdf'), format='pdf', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results of the feature selection and training')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters')
    parser.add_argument('--n_vars', type=int, help='Number of atmospheric variables considered in the FS process')
    parser.add_argument('--n_idxs', type=int, help='Number of climate indexes considered in the FS process')
    parser.add_argument('--results_folder', type=str, help='Name of experiment and of the output folder where to store the results')
    parser.add_argument('--basin', type=str, default='GLB', help='Basin name')
    parser.add_argument('--model_kind', type=str, default='LinReg', help='Model kind')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds for division in train and test sets')
    parser.add_argument('--start_year', type=int, default=1980, help='Initial year of the dataset to consider')
    parser.add_argument('--end_year', type=int, default=2021, help='Final year of the dataset to consider')
    args = parser.parse_args()
    main(args.n_clusters, args.n_vars, args.n_idxs, args.results_folder, args.basin, args.model_kind, args.n_folds, args.start_year, args.end_year)