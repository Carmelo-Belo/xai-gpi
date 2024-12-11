import os
import argparse
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import utils_results as ut

def main(basin, n_clusters, clusters_type, n_vars, n_idxs, model_kind, n_folds, start_year, end_year):
    
    # Set project directory and name of file containing the target variable
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    target_file = 'target_1965-2022_2.5x2.5.csv'

    # Set directories according to the clusters type
    if clusters_type == 'AC':
        cluster_data = f'{basin}_{n_clusters}clusters_anomaly'
        figures_folder = f'msfm_fit{model_kind}_AC_nf{n_folds}_nv{n_vars}_nd{n_idxs}'
    elif clusters_type == 'NC':
        cluster_data = f'{basin}_{n_clusters}clusters'
        figures_folder = f'msfm_fit{model_kind}_NC_nf{n_folds}_nv{n_vars}_nd{n_idxs}'
    else:
        raise ValueError('The clusters type should be either AC or NC')

    # Set the paths to the files
    experiment_filename = f'1965-2022_{n_clusters}clusters_{n_vars}vars_{n_idxs}idxs.csv'
    predictor_file = 'predictors_' + experiment_filename
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    data_dir = os.path.join(fs_dir, 'data', cluster_data)
    predictors_path = os.path.join(data_dir, predictor_file)
    target_path = os.path.join(data_dir, target_file)
    output_dir = os.path.join(fs_dir, 'results', 'comparative_analysis', basin, f'{n_clusters}clusters', figures_folder)

    # Load the predictors and the target in a DataFrame
    predictors_df = pd.read_csv(predictors_path, index_col=0)
    predictors_df.index = pd.to_datetime(predictors_df.index)
    target_df = pd.read_csv(target_path, index_col=0)
    target_df.index = pd.to_datetime(target_df.index)

    # Load the dataset containing the most selected variables
    selected_vars_filename = f'{model_kind}_60%selvar_{clusters_type}_{n_vars}vars_{n_idxs}idxs.csv'
    selected_vars_path = os.path.join(fs_dir, 'results', 'comparative_analysis', basin, f'{n_clusters}clusters', selected_vars_filename)
    selected_vars = pd.read_csv(selected_vars_path, index_col="lag")
    for l in range(len(selected_vars)):
        selected_vars.loc[l] = [ast.literal_eval(selected_vars.loc[l].values[0])]

    # Compone the dataset to train the model from the predictors_df and the selected variables
    dataset_opt = target_df.copy()
    for r, row in selected_vars.iterrows():
        variables = row[0]
        for var in variables:
            dataset_opt[f'{var}_lag{r}'] = predictors_df[var].shift(r)
    
    # Compone the dataset to train the model using all predictors possible
    dataset_opt_noFS = target_df.copy()
    for l in range(len(selected_vars)):
        for var in predictors_df.columns:
            col_df = pd.DataFrame(predictors_df[var].shift(l).values, index=dataset_opt_noFS.index, columns=[f'{var}_lag{l}'])
            dataset_opt_noFS = pd.concat([dataset_opt_noFS, col_df], axis=1)

    ## Train MLPregressor with the best solution found ##
    # Cross-Validation for train and test years
    kfold = KFold(n_splits=n_folds)
    years = np.arange(start_year, end_year, 1)
    Y_column = 'tcg' # Target variable
    obs_indices = dataset_opt.index.year.isin(years)
    obs_dataset = dataset_opt[obs_indices]
    Y_test = obs_dataset[Y_column]
    Y_pred_MLP = pd.DataFrame()
    Y_pred_MLP_noFS = pd.DataFrame()
    Y_pred_XGB = pd.DataFrame()
    Y_pred_XGB_noFS = pd.DataFrame()
    Y_pred_LGBM = pd.DataFrame()
    Y_pred_LGBM_noFS = pd.DataFrame()

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
        Y_train = train_dataset[Y_column]
        X_test_fold = test_dataset[test_dataset.columns.drop([Y_column])]
        Y_test_fold = test_dataset[Y_column]
        scaler = preprocessing.MinMaxScaler()
        X_std_train = scaler.fit(X_train)
        X_std_train = scaler.transform(X_train)
        X_std_test = scaler.transform(X_test_fold)
        X_train = pd.DataFrame(X_std_train, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_std_test, columns=X_test_fold.columns, index=X_test_fold.index)
        # Standardize the entire dataset
        X_train_noFS = train_dataset_noFS[train_dataset_noFS.columns.drop([Y_column])]
        Y_train_noFS = train_dataset_noFS[Y_column]
        X_test_fold_noFS = test_dataset_noFS[test_dataset_noFS.columns.drop([Y_column])]
        Y_test_fold_noFS = test_dataset_noFS[Y_column]
        scaler_noFS = preprocessing.MinMaxScaler()
        X_std_train_noFS = scaler_noFS.fit(X_train_noFS)
        X_std_train_noFS = scaler_noFS.transform(X_train_noFS)
        X_std_test_noFS = scaler_noFS.transform(X_test_fold_noFS)
        X_train_noFS = pd.DataFrame(X_std_train_noFS, columns=X_train_noFS.columns, index=X_train_noFS.index)
        X_test_noFS = pd.DataFrame(X_std_test_noFS, columns=X_test_fold_noFS.columns, index=X_test_fold_noFS.index)

        # Split the training set in training and validation sets for all models and both datasets
        X_t, X_v, Y_t, Y_v = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
        X_t_noFS, X_v_noFS, Y_t_noFS, Y_v_noFS = train_test_split(X_train_noFS, Y_train_noFS, test_size=0.2, random_state=42)

        ## MLPregressor ##
        # Build, compile and train the multi layer perceptron model for the optimized dataset
        n_predictors = len(X_train.columns)
        mlpreg = Sequential([
            Dense(units=n_predictors*2, activation='relu', input_shape=(n_predictors,)),
            Dense(units=1)
        ])
        mlpreg.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        callback = EarlyStopping(monitor='val_loss', patience=10)
        history = mlpreg.fit(x=X_t, y=Y_t, validation_data=(X_v, Y_v), epochs=100, batch_size=32, verbose=0, callbacks=[callback])
        Y_pred_fold = mlpreg.predict(X_test)
        Y_pred_fold = pd.DataFrame(Y_pred_fold, index=Y_test_fold.index, columns=['tcg'])
        Y_pred_MLP = pd.concat([Y_pred_MLP, Y_pred_fold])
        # Evaluate the model for the optimized dataset
        loss = mlpreg.evaluate(X_test, Y_test_fold, verbose=0)
        # Build, compile and train the multi layer perceptron model for the entire dataset
        n_predictors_noFS = len(X_train_noFS.columns)
        mlpreg_noFS = Sequential([
            Dense(units=n_predictors_noFS*2, activation='relu', input_shape=(n_predictors_noFS,)),
            Dense(units=1)
        ])
        mlpreg_noFS.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        history_noFS = mlpreg_noFS.fit(x=X_t_noFS, y=Y_t_noFS, validation_data=(X_v_noFS, Y_v_noFS), epochs=100, batch_size=32, verbose=0, callbacks=[callback])
        Y_pred_fold_noFS = mlpreg_noFS.predict(X_test_noFS)
        Y_pred_fold_noFS = pd.DataFrame(Y_pred_fold_noFS, index=Y_test_fold_noFS.index, columns=['tcg'])
        Y_pred_MLP_noFS = pd.concat([Y_pred_MLP_noFS, Y_pred_fold_noFS])
        # Evaluate the model for the entire dataset
        loss_noFS = mlpreg_noFS.evaluate(X_test_noFS, Y_test_fold_noFS, verbose=0)
        # Plot the training and validation loss for the 2 models
        loss_figure_dir = os.path.join(output_dir, 'models_losses')
        os.makedirs(loss_figure_dir, exist_ok=True)
        fig = ut.plot_train_val_loss(history.history['loss'], history.history['val_loss'], history_noFS.history['loss'], history_noFS.history['val_loss'], loss, loss_noFS)
        fig.savefig(os.path.join(loss_figure_dir, f'MLP_Loss_{n_fold}.pdf'), format='pdf', dpi=300)

        ## XGBoost ##
        # Build, compile and train the xgboost regressor for the optimized dataset
        xgboost = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, objective='reg:squarederror', early_stopping_rounds=10)
        xgboost.fit(X_t, Y_t, eval_set=[(X_t, Y_t), (X_v, Y_v)], verbose=False)
        Y_pred_fold_xgb = xgboost.predict(X_test)
        Y_pred_fold_xgb = pd.DataFrame(Y_pred_fold_xgb, index=Y_test_fold.index, columns=['tcg'])
        Y_pred_XGB = pd.concat([Y_pred_XGB, Y_pred_fold_xgb])
        # Evaluate the model for the optimized dataset
        loss_xgb = np.sqrt(mean_squared_error(Y_test_fold, Y_pred_fold_xgb))
        # Build, compile and train the xgboost regressor for the entire dataset
        xgboost_noFS = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, objective='reg:squarederror', early_stopping_rounds=10)
        xgboost_noFS.fit(X_t_noFS, Y_t_noFS, eval_set=[(X_t_noFS, Y_t_noFS), (X_v_noFS, Y_v_noFS)], verbose=False)
        Y_pred_fold_xgb_noFS = xgboost_noFS.predict(X_test_noFS)
        Y_pred_fold_xgb_noFS = pd.DataFrame(Y_pred_fold_xgb_noFS, index=Y_test_fold_noFS.index, columns=['tcg'])
        Y_pred_XGB_noFS = pd.concat([Y_pred_XGB_noFS, Y_pred_fold_xgb_noFS])
        # Evaluate the model for the entire dataset
        loss_xgb_noFS = np.sqrt(mean_squared_error(Y_test_fold_noFS, Y_pred_fold_xgb_noFS))
        # Plot the training and validation loss for the 2 models
        fig = ut.plot_train_val_loss(xgboost.evals_result_['validation_0']['rmse'], xgboost.evals_result_['validation_1']['rmse'],
                                    xgboost_noFS.evals_result_['validation_0']['rmse'], xgboost_noFS.evals_result_['validation_1']['rmse'], loss_xgb, loss_xgb_noFS)
        fig.savefig(os.path.join(loss_figure_dir, f'XGB_Loss_{n_fold}.pdf'), format='pdf', dpi=300)

        ## LightGBM ##
        # Build, compile and train the lightgbm regressor for the optimized dataset
        lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, objective='regression', verbosity=-1, early_stopping_rounds=10)
        lgbm.fit(X_t, Y_t, eval_set=[(X_t, Y_t), (X_v, Y_v)])
        Y_pred_fold_lgbm = lgbm.predict(X_test)
        Y_pred_fold_lgbm = pd.DataFrame(Y_pred_fold_lgbm, index=Y_test_fold.index, columns=['tcg'])
        Y_pred_LGBM = pd.concat([Y_pred_LGBM, Y_pred_fold_lgbm])
        # Evaluate the model for the optimized dataset
        loss_lgbm = np.sqrt(mean_squared_error(Y_test_fold, Y_pred_fold_lgbm))
        # Build, compile and train the lightgbm regressor for the entire dataset
        lgbm_noFS = LGBMRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, objective='regression', verbosity=-1, early_stopping_rounds=10)
        lgbm_noFS.fit(X_t_noFS, Y_t_noFS, eval_set=[(X_t_noFS, Y_t_noFS), (X_v_noFS, Y_v_noFS)])
        Y_pred_fold_lgbm_noFS = lgbm_noFS.predict(X_test_noFS)
        Y_pred_fold_lgbm_noFS = pd.DataFrame(Y_pred_fold_lgbm_noFS, index=Y_test_fold_noFS.index, columns=['tcg'])
        Y_pred_LGBM_noFS = pd.concat([Y_pred_LGBM_noFS, Y_pred_fold_lgbm_noFS])
        # Evaluate the model for the entire dataset
        loss_lgbm_noFS = np.sqrt(mean_squared_error(Y_test_fold_noFS, Y_pred_fold_lgbm_noFS))
        # Plot the training and validation loss for the 2 models
        fig = ut.plot_train_val_loss(lgbm.evals_result_['training']['l2'], lgbm.evals_result_['valid_1']['l2'], 
                                    lgbm_noFS.evals_result_['training']['l2'], lgbm_noFS.evals_result_['valid_1']['l2'], loss_lgbm, loss_lgbm_noFS)
        fig.savefig(os.path.join(loss_figure_dir, f'LGBM_Loss_{n_fold}.pdf'), format='pdf', dpi=300)

    # Create a where to store the info of the trainings and the performances of the models in terms of correlation
    performance_df_file = os.path.join(fs_dir, 'results', 'comparative_analysis', f'msfm_performance_{basin}.csv')
    performance_columns = ['experiment', 'model', 'n_clusters', 'clusters_type', 'n_folds', 'n_vars', 'n_idxs', 'tl_mlp', 'tl_mlp_noFS', 'tl_lgbm', 'tl_lgbm_noFS',
                            'tl_xgb', 'tl_xgb_noFS', 'R_mlp', 'R_mlp_noFS', 'R_lgbm', 'R_lgbm_noFS', 'R_xgb', 'R_xgb_noFS', 'R_S_mlp', 'R_S_mlp_noFS', 'R_S_lgbm', 
                            'R_S_lgbm_noFS', 'R_S_xgb', 'R_S_xgb_noFS', 'R_Y_mlp', 'R_Y_mlp_noFS', 'R_Y_lgbm', 'R_Y_lgbm_noFS', 'R_Y_xgb', 'R_Y_xgb_noFS']
    experiment_name = f'msfm_fit{model_kind}_{clusters_type}{n_clusters}_nf{n_folds}_nv{n_vars}_nd{n_idxs}'
    if os.path.exists(performance_df_file):
        performance_df = pd.read_csv(performance_df_file, index_col=0)
    else:
        performance_df = pd.DataFrame(columns=performance_columns)
        performance_df.set_index('experiment', inplace=True)

    # Compare observations to predictions
    r_mlp, _ = pearsonr(Y_test, Y_pred_MLP['tcg'])
    r_xgb, _ = pearsonr(Y_test, Y_pred_XGB['tcg'])
    r_lgbm, _ = pearsonr(Y_test, Y_pred_LGBM['tcg'])
    r_mlp_noFS, _ = pearsonr(Y_test, Y_pred_MLP_noFS['tcg'])
    r_xgb_noFS, _ = pearsonr(Y_test, Y_pred_XGB_noFS['tcg'])
    r_lgbm_noFS, _ = pearsonr(Y_test, Y_pred_LGBM_noFS['tcg'])
    xticks = pd.Series(Y_test.index).dt.strftime('%m-%Y').to_numpy()
    plt.figure(figsize=(30, 12))
    plt.plot(xticks, Y_test, label='Observed (IBTrACS)', color='#1f77b4')
    plt.plot(xticks, Y_pred_MLP['tcg'], label=f'FS-MLP - R:{r_mlp:.3f}', color='#ff7f0e')
    plt.plot(xticks, Y_pred_MLP_noFS['tcg'], label=f'NoFS-MLP - R:{r_mlp_noFS:.3f}', color='#ff7f0e', linestyle='--')
    plt.plot(xticks, Y_pred_XGB['tcg'], label=f'FS-XGB - R:{r_xgb:.3f}', color='#2ca02c')
    plt.plot(xticks, Y_pred_XGB_noFS['tcg'], label=f'NoFS-XGB - R:{r_xgb_noFS:.3f}', color='#2ca02c', linestyle='--')
    plt.plot(xticks, Y_pred_LGBM['tcg'], label=f'FS-LGBM - R:{r_lgbm:.3f}', color='#d62728')
    plt.plot(xticks, Y_pred_LGBM_noFS['tcg'], label=f'NoFS-LGBM - R:{r_lgbm_noFS:.3f}', color='#d62728', linestyle='--')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(ticks=np.arange(len(xticks))[::4], labels=xticks[::4], rotation=45)
    plt.xlabel('Months')
    plt.ylabel('# of TCs per month')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'monthly_evolution.pdf'), format='pdf', dpi=300)

    # Compare seasonal accumulated number of TCs
    Y_test_seasonal = Y_test.groupby(Y_test.index.month).mean()
    Y_pred_MLP_seasonal = Y_pred_MLP.groupby(Y_pred_MLP.index.month).mean()
    Y_pred_XGB_seasonal = Y_pred_XGB.groupby(Y_pred_XGB.index.month).mean()
    Y_pred_LGBM_seasonal = Y_pred_LGBM.groupby(Y_pred_LGBM.index.month).mean()
    Y_pred_MLP_noFS_seasonal = Y_pred_MLP_noFS.groupby(Y_pred_MLP_noFS.index.month).mean()
    Y_pred_XGB_noFS_seasonal = Y_pred_XGB_noFS.groupby(Y_pred_XGB_noFS.index.month).mean()
    Y_pred_LGBM_noFS_seasonal = Y_pred_LGBM_noFS.groupby(Y_pred_LGBM_noFS.index.month).mean()
    rS_mlp, _ = pearsonr(Y_test_seasonal, Y_pred_MLP_seasonal['tcg'])
    rS_xgb, _ = pearsonr(Y_test_seasonal, Y_pred_XGB_seasonal['tcg'])
    rS_lgbm, _ = pearsonr(Y_test_seasonal, Y_pred_LGBM_seasonal['tcg'])
    rS_mlp_noFS, _ = pearsonr(Y_test_seasonal, Y_pred_MLP_noFS_seasonal['tcg'])
    rS_xgb_noFS, _ = pearsonr(Y_test_seasonal, Y_pred_XGB_noFS_seasonal['tcg'])
    rS_lgbm_noFS, _ = pearsonr(Y_test_seasonal, Y_pred_LGBM_noFS_seasonal['tcg'])
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test_seasonal.index, Y_test_seasonal, label='Observed (IBTrACS)', color='#1f77b4', linewidth=2)
    plt.plot(Y_pred_MLP_seasonal.index, Y_pred_MLP_seasonal['tcg'], label=f'FS-MLP - R:{rS_mlp:.3f}', color='#ff7f0e', linewidth=2)
    plt.plot(Y_pred_MLP_noFS_seasonal.index, Y_pred_MLP_noFS_seasonal['tcg'], label=f'NoFS-MLP - R:{rS_mlp_noFS:.3f}', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.plot(Y_pred_XGB_seasonal.index, Y_pred_XGB_seasonal['tcg'], label=f'FS-XGB - R:{rS_xgb:.3f}', color='#2ca02c', linewidth=2)
    plt.plot(Y_pred_XGB_noFS_seasonal.index, Y_pred_XGB_noFS_seasonal['tcg'], label=f'NoFS-XGB - R:{rS_xgb_noFS:.3f}', color='#2ca02c', linestyle='--', linewidth=2)
    plt.plot(Y_pred_LGBM_seasonal.index, Y_pred_LGBM_seasonal['tcg'], label=f'FS-LGBM - R:{rS_lgbm:.3f}', color='#d62728', linewidth=2)
    plt.plot(Y_pred_LGBM_noFS_seasonal.index, Y_pred_LGBM_noFS_seasonal['tcg'], label=f'NoFS-LGBM - R:{rS_lgbm_noFS:.3f}', color='#d62728', linestyle='--', linewidth=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Months')
    plt.ylabel('# of TCs per month')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'seasonality.pdf'), format='pdf', dpi=300)

    # Compare annual accumulated number of TCs
    Y_test_annual = Y_test.resample('A').sum()
    Y_pred_MLP_annual = Y_pred_MLP.resample('A').sum()
    Y_pred_XGB_annual = Y_pred_XGB.resample('A').sum()
    Y_pred_LGBM_annual = Y_pred_LGBM.resample('A').sum()
    Y_pred_MLP_noFS_annual = Y_pred_MLP_noFS.resample('A').sum()
    Y_pred_XGB_noFS_annual = Y_pred_XGB_noFS.resample('A').sum()
    Y_pred_LGBM_noFS_annual = Y_pred_LGBM_noFS.resample('A').sum()
    rY_mlp, _ = pearsonr(Y_test_annual, Y_pred_MLP_annual['tcg'])
    rY_xgb, _ = pearsonr(Y_test_annual, Y_pred_XGB_annual['tcg'])
    rY_lgbm, _ = pearsonr(Y_test_annual, Y_pred_LGBM_annual['tcg'])
    rY_mlp_noFS, _ = pearsonr(Y_test_annual, Y_pred_MLP_noFS_annual['tcg'])
    rY_xgb_noFS, _ = pearsonr(Y_test_annual, Y_pred_XGB_noFS_annual['tcg'])
    rY_lgbm_noFS, _ = pearsonr(Y_test_annual, Y_pred_LGBM_noFS_annual['tcg'])
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test_annual.index.year, Y_test_annual, label='Observed (IBTrACS)', color='#1f77b4', linewidth=2)
    plt.plot(Y_pred_MLP_annual.index.year, Y_pred_MLP_annual['tcg'], label=f'FS-MLP - R:{rY_mlp:.3f}', color='#ff7f0e', linewidth=2)
    plt.plot(Y_pred_MLP_noFS_annual.index.year, Y_pred_MLP_noFS_annual['tcg'], label=f'NoFS-MLP - R:{rY_mlp_noFS:.3f}', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.plot(Y_pred_XGB_annual.index.year, Y_pred_XGB_annual['tcg'], label=f'FS-XGB - R:{rY_xgb:.3f}', color='#2ca02c', linewidth=2)
    plt.plot(Y_pred_XGB_noFS_annual.index.year, Y_pred_XGB_noFS_annual['tcg'], label=f'NoFS-XGB - R:{rY_xgb_noFS:.3f}', color='#2ca02c', linestyle='--', linewidth=2)
    plt.plot(Y_pred_LGBM_annual.index.year, Y_pred_LGBM_annual['tcg'], label=f'FS-LGBM - R:{rY_lgbm:.3f}', color='#d62728', linewidth=2)
    plt.plot(Y_pred_LGBM_noFS_annual.index.year, Y_pred_LGBM_noFS_annual['tcg'], label=f'NoFS-LGBM - R:{rY_lgbm_noFS:.3f}', color='#d62728', linestyle='--', linewidth=2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Years')
    plt.ylabel('# of TCs per year')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'annual_evolution.pdf'), format='pdf', dpi=300)

    # Save the information of the models' performance in the performance DataFrame
    row_data = {
        'experiment': experiment_name,
        'model': model_kind,
        'n_clusters': n_clusters,
        'clusters_type': clusters_type,
        'n_folds': n_folds,
        'n_vars': n_vars,
        'n_idxs': n_idxs,
        'tl_mlp': loss,
        'tl_mlp_noFS': loss_noFS,
        'tl_lgbm': loss_lgbm,
        'tl_lgbm_noFS': loss_lgbm_noFS,
        'tl_xgb': loss_xgb,
        'tl_xgb_noFS': loss_xgb_noFS,
        'R_mlp': r_mlp,
        'R_mlp_noFS': r_mlp_noFS,
        'R_lgbm': r_lgbm,
        'R_lgbm_noFS': r_lgbm_noFS,
        'R_xgb': r_xgb,
        'R_xgb_noFS': r_xgb_noFS,
        'R_S_mlp': rS_mlp,
        'R_S_mlp_noFS': rS_mlp_noFS,
        'R_S_lgbm': rS_lgbm,
        'R_S_lgbm_noFS': rS_lgbm_noFS,
        'R_S_xgb': rS_xgb,
        'R_S_xgb_noFS': rS_xgb_noFS,
        'R_Y_mlp': rY_mlp,
        'R_Y_mlp_noFS': rY_mlp_noFS,
        'R_Y_lgbm': rY_lgbm,
        'R_Y_lgbm_noFS': rY_lgbm_noFS,
        'R_Y_xgb': rY_xgb,
        'R_Y_xgb_noFS': rY_xgb_noFS
    }
    performance_df.loc[experiment_name] = row_data
    performance_df.to_csv(performance_df_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results of the feature selection and training')
    parser.add_argument('--basin', type=str, help='Basin name')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters')
    parser.add_argument('--clusters_type', type=str, help='Type of clusters (AC or NC)')
    parser.add_argument('--n_vars', type=int, default=8, help='Number of atmospheric variables considered in the FS process')
    parser.add_argument('--n_idxs', type=int, default=10, help='Number of climate indexes considered in the FS process')
    parser.add_argument('--model_kind', type=str, help='Model kind (all, linreg, lgbm)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds for division in train and test sets')
    parser.add_argument('--start_year', type=int, default=1980, help='Initial year of the dataset to consider')
    parser.add_argument('--end_year', type=int, default=2021, help='Final year of the dataset to consider')
    args = parser.parse_args()
    main(args.basin, args.n_clusters, args.clusters_type, args.n_vars, args.n_idxs, args.model_kind, args.n_folds, args.start_year, args.end_year)