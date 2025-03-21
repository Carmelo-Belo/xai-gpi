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
from keras.models import load_model
from sklearn.model_selection import KFold
from sklearn import preprocessing
import shap

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

# Function that from the basin name and simualtion folder name returns several data to be used in the feature importance analysis
# + sensitivity analysis on the percentage of the selected features in the best models
def runs_info(basin, run_name):
    # Set some additional variables and parameters that generally stay constant
    years = np.arange(1980, 2022, 1) # from 1980 to 2021 included
    n_folds = 3
    n_clusters = int(run_name.split('nc')[1].split('_')[0])
    model_kind = run_name.split('_')[1]
    # Set directories and files names
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    target_file = 'target_1980-2022_2.5x2.5.csv'
    # Retrieve the clusters type of data from the results folder
    cluster_data = f'{basin}_{n_clusters}clusters_noTS'
    # Set the paths to the files
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    output_dir = os.path.join(fs_dir, 'results', basin, run_name)
    cluster_data_dir = os.path.join(fs_dir, 'data', cluster_data)
    final_analysis_dir = os.path.join(output_dir, 'final_analysis')
    # predictors
    experiment_filename = f'1980-2022_{n_clusters}clusters_8vars_9idxs.csv'
    predictor_file = 'predictors_' + experiment_filename
    predictors_df = pd.read_csv(os.path.join(cluster_data_dir, predictor_file), index_col=0)
    predictors_df.index = pd.to_datetime(predictors_df.index)
    predictors_df = predictors_df.loc[predictors_df.index.year.isin(years)]
    # target
    target_file = 'target_residual_1980-2022_2.5x2.5.csv'
    target_df = pd.read_csv(os.path.join(cluster_data_dir, target_file), index_col=0)
    target_df.index = pd.to_datetime(target_df.index)
    target_df = target_df.loc[target_df.index.year.isin(years)]
    # Create dataset according to solution and list the labels of the selected variables
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
        sel_feat_perc_path = os.path.join(fs_dir, 'results', f'selected_features_best_models_{basin}_{n_clusters}_noTS.csv')
        df_sel_feat_perc = pd.read_csv(sel_feat_perc_path, index_col=0)
        sel_perc = run_name.split('_')[0].split('selfeat')[1]
        selected_features = df_sel_feat_perc[sel_perc].dropna().to_list()
        dataset_opt = predictors_df[selected_features]
        dataset_opt.columns = [f'{feat}_lag0' for feat in dataset_opt.columns]
        dataset_opt = dataset_opt.assign(resid=target_df['resid'])
    # Compone the dataset to train the model using all predictors possible
    dataset_opt_noFS = target_df.copy()
    for l in range(1):
        for var in predictors_df.columns:
            col_df = pd.DataFrame(predictors_df[var].shift(l).values, index=dataset_opt_noFS.index, columns=[f'{var}_lag{l}'])
            dataset_opt_noFS = pd.concat([dataset_opt_noFS, col_df], axis=1)

    ## Make predictions with the best solution found ##
    # Cross-Validation for train and test years
    kfold = KFold(n_splits=n_folds)
    Y_column = 'resid' # Target variable
    Y_pred = []
    Y_pred_noFS = []
    X_test_eval = []
    X_test_eval_noFS = []
    mlps = []
    mlps_noFS = []
    # List to store the results of feature permutation importance and SHAP values
    perm_importance_mlp = []
    perm_importance_mlp_noFS = []
    shap_values_mlp = []
    shap_values_mlp_noFS = []
    # Loop through the folds
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

        # Standardize the optimized dataset
        X_train = train_dataset[train_dataset.columns.drop([Y_column])]
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
        Y_pred_fold = pd.DataFrame(Y_pred_fold, index=Y_test_fold.index, columns=['resid'])
        Y_pred.append(Y_pred_fold)
        Y_pred_fold_noFS = mlp_noFS.predict(X_test_noFS, verbose=0)
        Y_pred_fold_noFS = pd.DataFrame(Y_pred_fold_noFS, index=Y_test_fold.index, columns=['resid'])
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

    return Y_pred, Y_pred_noFS, X_test_eval, X_test_eval_noFS, mlps, mlps_noFS, perm_importance_mlp, perm_importance_mlp_noFS, shap_values_mlp, shap_values_mlp_noFS

def plot_annual_time_series(obs, pred, pred_noFS, engpi, ogpi, r_pred, r_pred_noFS, r_engpi, r_ogpi):
    fig_annual = plt.figure(figsize=(16, 8))
    axY = fig_annual.add_subplot(111)
    # observations
    axY.plot(obs.index, obs, label='Observed (IBTrACS)', color='green', linewidth=3)
    # mlp predictions
    axY.plot(pred.index, pred, label=f'FS - R:{r_pred:.3f}', color='blue', linewidth=3)
    axY.plot(pred_noFS.index, pred_noFS, label=f'NoFS - R:{r_pred_noFS:.3f}', color='red', linewidth=3)
    # gpis
    axY.plot(engpi.index, engpi, label=f'ENGPI - R:{r_engpi:.3f}', color='orange', linewidth=3, linestyle='--')
    axY.plot(ogpi.index, ogpi, label=f'oGPI - R:{r_ogpi:.3f}', color='purple', linewidth=3, linestyle='--')
    # set figure parameters
    axY.grid(True, which='both', linestyle='--', linewidth=0.5)
    axY.set_xticks(obs.index[::2])
    axY.set_xticklabels(obs.index[::2], rotation=45, fontsize=14, ha='right')
    axY.set_yticks(axY.get_yticks())
    axY.set_yticklabels(axY.get_yticks(), fontsize=14)
    axY.set_xlabel('Years', fontsize=16)
    axY.set_ylabel('# of TCs per year', fontsize=16)
    axY.legend(fontsize=14, loc='upper center')
    # Finalize the figure
    fig_annual.set_tight_layout(True)
    plt.show()
    return fig_annual

def plot_monthly_time_series(obs, pred, pred_noFS, engpi, ogpi, r_pred, r_pred_noFS, r_engpi, r_ogpi):
    fig_ts = plt.figure(figsize=(60, 16))
    ## Monthly time series ##
    ax = fig_ts.add_subplot(111)
    xticks = pd.Series(obs.index).dt.strftime('%m-%Y').to_numpy()
    # observations
    ax.plot(xticks, obs, label='Observed (IBTrACS)', color='green', linewidth=4)
    # predictions
    ax.plot(xticks, pred, label=f'FS - R:{r_pred:.3f}', color='blue', linewidth=4)
    ax.plot(xticks, pred_noFS, label=f'NoFS - R:{r_pred_noFS:.3f}', color='red', linewidth=4)
    # gpis
    ax.plot(xticks, engpi, label=f'ENGPI - R:{r_engpi:.3f}', color='orange', linewidth=4, linestyle='--')
    ax.plot(xticks, ogpi, label=f'oGPI - R:{r_ogpi:.3f}', color='purple', linewidth=4, linestyle='--')
    # set figure parameters
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(ticks=np.arange(len(xticks))[::6])
    ax.set_xticklabels(xticks[::6], rotation=45, fontsize=26, ha='right')
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=26)
    ax.set_xlabel('Months', fontsize=36)
    ax.set_ylabel('# of TCs', fontsize=36)
    ax.legend(fontsize=36, loc='upper left')
    # Finalize the figure
    fig_ts.set_tight_layout(True)
    plt.show()
    return fig_ts

def plot_variables_clusters(basin, n_clusters, cluster_data_dir, variable_names_cluster, selected_features):
    # Set list of the variables of which there are possible clusters
    cluster_variables = ['abs_vo850', 'mpi', 'msl', 'r700', 'sst', 'vo850', 'vws850-200', 'w']
    # Set figure charateristics depending on the basin
    if basin == 'NWP':
        west, east, south, north = 100, 180, 0, 40
        fig_size=(13,15)
    elif basin == 'NEP':
        west, east, south, north = -180, -75, 0, 40
        fig_size=(15,15)
    elif basin == 'NA':
        west, east, south, north = -100, 0, 0, 40
        fig_size=(15,15)
    elif basin == 'NI':
        west, east, south, north = 45, 100, 0, 40
        fig_size=(9,15)
    elif basin == 'SP':
        west, east, south, north = 110, -45, -40, 0
        fig_size=(20,15)
    elif basin == 'SI':
        west, east, south, north = 35, 135, -40, 0
        fig_size=(15,15)
    # Set figure size and grid
    fig_basin = plt.figure(figsize=fig_size)
    gs_basin = gridspec.GridSpec(5, 2, figure=fig_basin, height_ratios=[1, 1, 1, 1, 0.05])
    # Define a color map with fixed colors for each cluster and map the clusters to the colors index
    unique_clusters = np.arange(1, n_clusters+1)
    cmap = plt.get_cmap('tab20', n_clusters)
    colors = cmap(np.linspace(0, 1, n_clusters))
    full_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)
    cluster_to_color_index = {cluster: i for i, cluster in enumerate(unique_clusters)}
    # Cycle through the different variables and plot the clusters
    for v, var in enumerate(cluster_variables):
        # Set the figure and gridlines of the map
        if basin == 'NA':
            ax = fig_basin.add_subplot(gs_basin[v], projection=ccrs.PlateCarree())
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        elif basin == 'SP':
            ax = fig_basin.add_subplot(gs_basin[v], projection=ccrs.PlateCarree(central_longitude=180))
            ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree(central_longitude=180))
        else:
            ax = fig_basin.add_subplot(gs_basin[v], projection=ccrs.PlateCarree(central_longitude=180))
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
        gl.xlabel_style = {'size': 16} 
        gl.ylabel_style = {'size': 16}
        # If the variable is in the selected features, plot the corrsponding clusters
        if var in variable_names_cluster:
            # Load the labels file
            label_file = f'labels_{var}.csv'
            label_df = pd.read_csv(os.path.join(cluster_data_dir, label_file), index_col=0)
            # Determine the clusters and corresponding lags selected for the variable
            clusters_selected = np.asarray([int(long_name.split('_cluster')[1].split('_lag')[0]) 
                                            for long_name in selected_features if long_name.split('_cluster')[0] == var])
            # Plot the clusters for the selected variable and lag only if there are clusters selected
            if len(clusters_selected) > 0:
                # Select the rows of the label file that correspond to the selected clusters
                label_df_selected = label_df[label_df['cluster'].isin(clusters_selected)]
                # Plot only the selected clusters using their index in the full color map
                scatter = ax.scatter(
                    label_df_selected['nodes_lon'].values, 
                    label_df_selected['nodes_lat'].values,
                    c=[cluster_to_color_index[cluster] for cluster in label_df_selected['cluster']],
                    cmap=full_cmap, norm=norm, s=125, transform=ccrs.PlateCarree()
                )
        ax.set_title(f'{var}', fontsize=30)         
    # Create a colorbar showing all clusters
    ax_cbar = fig_basin.add_subplot(gs_basin[8:])
    cbar = plt.colorbar(mappable=scatter, cax=ax_cbar, orientation='horizontal', ticks=np.arange(n_clusters))
    cbar.set_ticklabels(unique_clusters)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Clusters', fontsize=24)
    # Finalize the figure
    fig_basin.set_tight_layout(True)
    plt.show()
    
    return fig_basin