import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from scipy.stats import pearsonr

# Set the parameters of experiment to retrieve and process the corresponding resutls
project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
n_clusters = 12
n_vars = 4
n_idxs = 10
target_file = 'target_1965-2022_2.5x2.5.csv'
output_folder = 'test'
model_kind = 'MLPReg' # 'LinReg' or 'LogReg' or 'MLPReg
train_yearI = 1980 # First year of the training dataset
train_yearF = 2013 # Last year of the training dataset
test_yearF = 2021 # Last year of the test dataset

# Set the paths to the files
experiment_filename = f'1965-2022_{n_clusters}clusters_{n_vars}vars_{n_idxs}idxs.csv'
fs_dir = os.path.join(project_dir, 'FS_TCG')
output_dir = os.path.join(fs_dir, 'results', output_folder)
sol_filename = f'{model_kind}_' + experiment_filename
sol_path = os.path.join(output_dir, sol_filename)
# final_sol_path = os.path.join(output_dir, f'CRO_{sol_filename}')
data_dir = os.path.join(fs_dir, 'data', f'{n_clusters}clusters')
predictor_file = 'predictors_' + experiment_filename
predictors_path = os.path.join(data_dir, predictor_file)
target_path = os.path.join(data_dir, target_file)

# Load the predictors and the target in a DataFrame
predictors_df = pd.read_csv(predictors_path, index_col=0)
predictors_df.index = pd.to_datetime(predictors_df.index)
target_df = pd.read_csv(target_path, index_col=0)
target_df.index = pd.to_datetime(target_df.index)

# Set the indices for the training and test datasets
train_indices = (predictors_df.index.year >= train_yearI) & (predictors_df.index.year <= train_yearF) 
test_indices = (predictors_df.index.year > train_yearF) & (predictors_df.index.year <= test_yearF)

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
# plt.show()

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

# Create and plot board to visualize which predictors are selected in the best solution and which time lags
def create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel):
    board = np.zeros((n_rows, n_cols))

    for i in range(n_cols):
        start_index = int(final_sequence[i]) 
        end_index = int(final_sequence[i])  + int(sequence_length[i])
        if feat_sel[i] != 0:
            board[start_index:end_index, i] = 1
    
    return board

def plot_board(board, column_names, feat_sel):
    fig, ax = plt.subplots(figsize=(15, 14))
    ax.imshow(board, cmap='Blues', origin='lower', aspect='auto')
    ax.set_xticks(np.arange(len(column_names)))
    ax.set_xticklabels(column_names, rotation=90, fontsize=11)
    ax.set_yticks(np.arange(board.shape[0])-0.5)
    ax.set_yticklabels(np.arange(board.shape[0]), fontsize=11)
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.grid(which='minor',color='black', linewidth=1)
    ax.yaxis.grid(which='minor',color='black', linewidth=1)
    ax.set_ylabel('Time lags', fontsize=15)

    for i in range(board.shape[1]):
        if feat_sel[i] == 0:
            rect = plt.Rectangle((i - 0.5, -0.5), 1, 1, color='red')
            ax.add_patch(rect)

    plt.tight_layout()

column_names = predictors_df.columns.tolist()
final_sequence = array_bestCV[len(column_names):2*len(column_names)]
sequence_length = array_bestCV[:len(column_names)]
feat_sel = array_bestCV[2*len(column_names):]

n_rows = int(((sequence_length + final_sequence)*feat_sel).max())+1
n_cols = len(column_names)

board_best = create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel)
plot_board(board_best, column_names, feat_sel)

## Train MLPregressor with the best solution found
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

# Split dataset into train and test
train_dataset = dataset_opt[train_indices]
test_dataset = dataset_opt[test_indices]

# Standardize the dataset
Y_column = 'tcg' 
X_train = train_dataset[train_dataset.columns.drop([Y_column]) ]
Y_train = train_dataset[Y_column]
X_test = test_dataset[test_dataset.columns.drop([Y_column]) ]
Y_test = test_dataset[Y_column]
scaler = preprocessing.StandardScaler()
X_std_train = scaler.fit(X_train)
X_std_train = scaler.transform(X_train)
X_std_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_std_train, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_std_test, columns=X_test.columns, index=X_test.index)

# Train the MLPRegressor
n_predictors = len(X_train.columns)
mlpreg = MLPRegressor(
    hidden_layer_sizes=(n_predictors*2, n_predictors),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    shuffle=True,
)

# Cross-Validation
score = cross_val_score(mlpreg, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
mlpreg.fit(X_train, Y_train)
Y_pred = mlpreg.predict(X_test)
Y_pred = pd.DataFrame(Y_pred, index=X_test.index, columns=['tcg'])
r_mlpreg, _ = pearsonr(Y_test, Y_pred['tcg'])

# Compare observations to predictions
xticks = pd.Series(Y_test.index).dt.strftime('%m-%Y').to_numpy()
plt.figure(figsize=(10, 6))
plt.plot(xticks, Y_test, label='Observed (IBTrACS)', color='#15E6CD',)
plt.plot(xticks, Y_pred['tcg'], label=f'FS-MLP - {r_mlpreg:.3f}', color='#0CF574')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(ticks=np.arange(len(xticks))[::4], labels=xticks[::4], rotation=45)
plt.xlabel('Months')
plt.ylabel('# of TCs per month')
plt.legend()
plt.tight_layout()

# Compare annual accumulated number of TCs
Y_test_annual = Y_test.resample('YE').sum()
Y_pred_annual = Y_pred.resample('YE').sum()
rY_mlpreg, _ = pearsonr(Y_test_annual, Y_pred_annual['tcg'])
plt.figure(figsize=(10, 6))
plt.plot(Y_test_annual.index.year, Y_test_annual, label='Observed (IBTrACS)', color='#15E6CD',)
plt.plot(Y_pred_annual.index.year, Y_pred_annual['tcg'], label=f'FS-MLP - {rY_mlpreg:.3f}', color='#0CF574')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Years')
plt.ylabel('# of TCs per year')
plt.legend()
plt.tight_layout()
plt.show()