import os
import argparse
import pandas as pd
import numpy as np
from keras.models import Model
from keras import Input, Model, layers, regularizers, callbacks
from keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
import shap
import sys
sys.path.insert(0, '/Users/huripari/Documents/PhD/TCs_Genesis/tcg_fsM')
import utils_results as ut

def main(basin, run_name):
    # Set parameters for later use
    years = np.arange(1980, 2022, 1) # from 1980 to 2021 included
    n_folds = 3
    n_clusters = int(run_name.split('nc')[1].split('_')[0])
    n_vars = int(run_name.split('nv')[1].split('_')[0])
    n_idxs = int(run_name.split('nd')[1].split('_')[0])
    model_kind = run_name.split('_')[1]

    # Set project directory and name of file containing the target variable
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    target_file = 'target_residual_1980-2022_2.5x2.5.csv'
    seasonal_file = 'target_seasonality_1980-2022_2.5x2.5.csv'
    trend_file = 'target_trend_1980-2022_2.5x2.5.csv'
    # Retrieve the clusters type of data from the results folder
    cluster_data = f'{basin}_{n_clusters}clusters_noTS'
    # Set the paths to the files
    experiment_filename = f'1980-2022_{n_clusters}clusters_{n_vars}vars_{n_idxs}idxs.csv'
    predictor_file = 'predictors_' + experiment_filename
    fs_dir = os.path.join(project_dir, 'tcg_fsM')
    results_dir = os.path.join(fs_dir, 'results')
    output_dir = os.path.join(results_dir, basin, run_name)
    data_dir = os.path.join(fs_dir, 'data', cluster_data)
    predictors_path = os.path.join(data_dir, predictor_file)
    target_path = os.path.join(data_dir, target_file)
    final_analysis_dir = os.path.join(output_dir, 'final_analysis')
    os.makedirs(final_analysis_dir, exist_ok=True)
    # Load the predictors and the target in a DataFrame
    predictors_df = pd.read_csv(predictors_path, index_col=0)
    predictors_df.index = pd.to_datetime(predictors_df.index)
    predictors_df = predictors_df.loc[predictors_df.index.year.isin(years)]
    target_df = pd.read_csv(target_path, index_col=0)
    target_df.index = pd.to_datetime(target_df.index)
    target_df = target_df.loc[target_df.index.year.isin(years)]
    target_season_df = pd.read_csv(os.path.join(data_dir, seasonal_file), index_col=0)
    target_season_df.index = pd.to_datetime(target_season_df.index)
    target_season_df = target_season_df.loc[target_season_df.index.year.isin(years)]
    target_trend_df = pd.read_csv(os.path.join(data_dir, trend_file), index_col=0)
    target_trend_df.index = pd.to_datetime(target_trend_df.index)
    target_trend_df = target_trend_df.loc[target_trend_df.index.year.isin(years)]
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
    else:
        os.makedirs(output_dir, exist_ok=True)

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
        sel_feat_perc_path = os.path.join(results_dir, f'selected_features_best_models_{basin}_{n_clusters}_noTS.csv')
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

    ## Train MLPregressor with the best solution found ##
    # Cross-Validation for train and test years
    kfold = KFold(n_splits=n_folds)
    Y_column = 'resid' # Target variable
    Y_pred_mlp = pd.DataFrame()
    Y_pred_mlp_noFS = pd.DataFrame()

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
        X_test_fold_noFS = test_dataset_noFS[test_dataset_noFS.columns.drop([Y_column])]
        scaler_noFS = preprocessing.MinMaxScaler()
        X_std_train_noFS = scaler_noFS.fit(X_train_noFS)
        X_std_train_noFS = scaler_noFS.transform(X_train_noFS)
        X_std_test_noFS = scaler_noFS.transform(X_test_fold_noFS)
        X_train_noFS = pd.DataFrame(X_std_train_noFS, columns=X_train_noFS.columns, index=X_train_noFS.index)
        X_test_noFS = pd.DataFrame(X_std_test_noFS, columns=X_test_fold_noFS.columns, index=X_test_fold_noFS.index)

        # Split the training set in training and validation sets for all models and both datasets
        X_t, X_v, Y_t, Y_v, X_t_noFS, X_v_noFS = train_test_split(X_train, Y_train, X_train_noFS, test_size=0.2, random_state=42)

        ## Define common training parameters and callbacks for the mlp ##
        n_neurons = 64
        epo = 100 # Number of epochs
        lr = 0.001 # Learning rate
        l2_reg = 0.001
        batch_size = 32
        callback = callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        ## MLPregressor with Selected Features ##
        # Build and compile the multi layer perceptron model for the optimized dataset
        n_predictors = len(X_train.columns)
        inputs = Input(shape=(n_predictors,))
        x = layers.Dense(n_neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        output = layers.Dense(1, kernel_regularizer=regularizers.l2(l2_reg))(x)
        mlpreg = Model(inputs, output)
        mlpreg.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        # Train the model
        history = mlpreg.fit(X_t, Y_t, validation_data=(X_v, Y_v), epochs=epo, batch_size=batch_size, callbacks=[callback], verbose=0)
        # Evaluate the model
        Y_pred_fold = mlpreg.predict(X_test, verbose=0)
        Y_pred_fold = pd.DataFrame(Y_pred_fold, index=Y_test_fold.index, columns=['resid'])
        Y_pred_mlp = pd.concat([Y_pred_mlp, Y_pred_fold])
        loss = mlpreg.evaluate(X_test, Y_test_fold, verbose=0)
        ## MLPregressor with all Features ##
        # Build and compile the multi layer perceptron model for the entire dataset
        n_predictors_noFS = len(X_train_noFS.columns)
        inputs_noFS = Input(shape=(n_predictors_noFS,))
        x_noFS = layers.Dense(n_neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(inputs_noFS)
        output_noFS = layers.Dense(1, kernel_regularizer=regularizers.l2(l2_reg))(x_noFS)
        mlpreg_noFS = Model(inputs_noFS, output_noFS)
        mlpreg_noFS.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        # Train the model
        history_noFS = mlpreg_noFS.fit(X_t_noFS, Y_t, validation_data=(X_v_noFS, Y_v), epochs=epo, batch_size=batch_size, callbacks=[callback], verbose=0)
        # Evaluate the model
        Y_pred_fold_noFS = mlpreg_noFS.predict(X_test_noFS, verbose=0)
        Y_pred_fold_noFS = pd.DataFrame(Y_pred_fold_noFS, index=Y_test_fold.index, columns=['resid'])
        Y_pred_mlp_noFS = pd.concat([Y_pred_mlp_noFS, Y_pred_fold_noFS])
        loss_noFS = mlpreg_noFS.evaluate(X_test_noFS, Y_test_fold, verbose=0)
        ## Plot the training and validation loss for the 2 models ##
        loss_fig_dir = os.path.join(final_analysis_dir, 'models_losses')
        os.makedirs(loss_fig_dir, exist_ok=True)
        fig = ut.plot_train_val_loss(history.history['loss'], history.history['val_loss'], history_noFS.history['loss'], history_noFS.history['val_loss'], loss, loss_noFS)
        fig.savefig(os.path.join(loss_fig_dir, f'mlp_loss_{n_fold+1}.pdf'), format='pdf', dpi=300)
        ## Save the two models to file ##
        models_savedir = os.path.join(final_analysis_dir, 'models')
        os.makedirs(models_savedir, exist_ok=True)
        mlpreg.save(os.path.join(models_savedir, f'mlp_fold{n_fold+1}.keras'))
        mlpreg_noFS.save(os.path.join(models_savedir, f'mlp_noFS_fold{n_fold+1}.keras'))

        feature_names = ['{}'.format(col.split('_l')[0]) for col in np.array(X_test.columns)]
        feature_names_noFS = ['{}'.format(col.split('_l')[0]) for col in np.array(X_test_noFS.columns)]
        ## Compute the permutation feature importance for the 2 models ##
        imp_mlp = permutation_importance(mlpreg, X_test, Y_test_fold, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')
        imp_mlp_noFS = permutation_importance(mlpreg_noFS, X_test_noFS, Y_test_fold, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')
        # Save them to file in the results folder
        explain_savedir = os.path.join(final_analysis_dir, 'explain_data')
        os.makedirs(explain_savedir, exist_ok=True)
        np.savez(os.path.join(explain_savedir, f'perm_imp_mlp_fold{n_fold+1}.npz'),
                 importances_mean=imp_mlp.importances_mean,
                 importances_std=imp_mlp.importances_std,
                 importances=imp_mlp.importances,
                 feature_names=feature_names)
        np.savez(os.path.join(explain_savedir, f'perm_imp_mlp_noFS_fold{n_fold+1}.npz'),
                 importances_mean=imp_mlp_noFS.importances_mean,
                 importances_std=imp_mlp_noFS.importances_std,
                 importances=imp_mlp_noFS.importances,
                 feature_names=feature_names_noFS)
        ## Compute the SHAP values for the 2 models ##
        expl_mlp = shap.Explainer(mlpreg.predict, X_t)
        expl_mlp_noFS = shap.Explainer(mlpreg_noFS.predict, X_t_noFS)
        shapv_mlp = expl_mlp(X_test)
        shapv_mlp_noFS = expl_mlp_noFS(X_test_noFS)
        # Save them to file in the results folder
        np.savez(os.path.join(explain_savedir, f'shap_mlp_fold{n_fold+1}.npz'),
                 shap_values=shapv_mlp.values,      # SHAP values
                 base_values=shapv_mlp.base_values, # base values
                 data=shapv_mlp.data,               # original data
                 feature_names=feature_names)       # feature names
        np.savez(os.path.join(explain_savedir, f'shap_mlp_noFS_fold{n_fold+1}.npz'),
                 shap_values=shapv_mlp_noFS.values,
                 base_values=shapv_mlp_noFS.base_values,
                 data=shapv_mlp_noFS.data,
                 feature_names=feature_names_noFS)

        print(f'Fold {n_fold+1}/{n_folds} - Processed Neural Networks')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the final analysis for the specific simulation')
    parser.add_argument('--basin', type=str, help='Name of the basin to analyze')
    parser.add_argument('--run_name', type=str, help='Name of the run to analyze')
    args = parser.parse_args()
    main(args.basin, args.run_name)