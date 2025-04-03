from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateReal import *
from PyCROSL.SubstrateInt import *

from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import argparse

def main(basin, n_clusters, remove_trend, remove_seasonality, n_vars, n_idxs, output_folder, model_kind, train_yearI, train_yearF, test_yearF):

    # Set project directory
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    # Set directories and name of the target file
    if remove_trend == 'y' and remove_seasonality == 'y':
        raise ValueError('To run feature selection with dataset without trend and seasonality, use the the script CRO_SpatioFS_noTS.py')
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    if remove_seasonality == 'y':
        data_dir = os.path.join(fs_dir, 'data', f'{basin}_{n_clusters}clusters_deseason')
        target_file = 'target_deseasonal_1980-2022_2.5x2.5.csv'
    elif remove_trend == 'y':
        data_dir = os.path.join(fs_dir, 'data', f'{basin}_{n_clusters}clusters_detrend')
        target_file = 'target_detrend_1980-2022_2.5x2.5.csv'
    else:
        data_dir = os.path.join(fs_dir, 'data', f'{basin}_{n_clusters}clusters')
        target_file = 'target_1980-2022_2.5x2.5.csv'

    # Set path and name of the predictor dataset and target dataset
    experiment_filename = f'1980-2022_{n_clusters}clusters_{n_vars}vars_{n_idxs}idxs.csv'
    predictor_file = 'predictors_' + experiment_filename
    predictors_path = os.path.join(data_dir, predictor_file)
    target_path = os.path.join(data_dir, target_file)
    # Load the predictor dataset and target dataset
    predictors_df = pd.read_csv(predictors_path, index_col=0)
    predictors_df.index = pd.to_datetime(predictors_df.index)
    target_df = pd.read_csv(target_path, index_col=0)
    target_df.index = pd.to_datetime(target_df.index)

    # Set the file names to store the solutions provided by the algorithm
    output_dir = os.path.join(fs_dir, 'results', basin, output_folder)
    os.makedirs(output_dir, exist_ok=True)
    indiv_path = os.path.join(output_dir, model_kind + '_' + experiment_filename)
    solution_filename = 'CRO_' + model_kind + '_' + experiment_filename # this file stores the last solution found by the algorithm

    # Create an empty file to store the solutions provided by the algorithm
    sol_data = pd.DataFrame(columns=['CV', 'Test', 'Sol'])
    sol_data.to_csv(indiv_path, sep=' ', header=sol_data.columns, index=None)

    # Split the dataset into train and test
    train_indices = (predictors_df.index.year >= train_yearI) & (predictors_df.index.year <= train_yearF) 
    test_indices = (predictors_df.index.year > train_yearF) & (predictors_df.index.year <= test_yearF)

    """
    All the following methods will have to be implemented for the algorithm to work properly with the same inputs, except for the constructor 
    """
    class ml_prediction(AbsObjectiveFunc):
        """
        This is the constructor of the class, here is where the objective function can be setted up.
        In this case we will only add the size of the vector as a parameter.
        """
        def __init__(self, size):
            self.size = size
            self.opt = "min" # it can be "max" or "min"

            # We set the limits of the vector (window size, time lags and variable selection)
            # Maximum time sequences I can select is 2, maximum time lag is 1 month, and the last one is regarding the binary selection of a variable
            self.sup_lim = np.append(np.append(np.repeat(1, predictors_df.shape[1]), np.repeat(0, predictors_df.shape[1])), np.repeat(1, predictors_df.shape[1]))
            self.inf_lim = np.append(np.append(np.repeat(1, predictors_df.shape[1]), np.repeat(0, predictors_df.shape[1])), np.repeat(0, predictors_df.shape[1]))

            super().__init__(self.size, self.opt, self.sup_lim, self.inf_lim)
        
        """
        This will be the objective function, that will recieve a vector and output a number
        """
        def objective(self, solution):
            # Read data
            sol_file = pd.read_csv(indiv_path, sep=' ', header=0)

            # Read solution
            time_sequences = np.append(np.array(solution[:predictors_df.shape[1]]).astype(int), 1)
            time_lags = np.append(np.array(solution[predictors_df.shape[1]:(2*predictors_df.shape[1])]).astype(int), 1)
            variable_selection = np.array(solution[(2*predictors_df.shape[1]):]).astype(int)

            if sum(variable_selection) == 0:  # If no variables are selected, return a high value
                return 100000

            # Create dataset according to solution
            shifted_columns = []
            for c, col in enumerate(predictors_df.columns):
                if variable_selection[c] == 0 or time_sequences[c] == 0:
                    continue
                for j in range(time_sequences[c]):
                    lag = time_lags[c] + j
                    col_name = f'{col}_lag{lag}'
                    shifted_columns.append(predictors_df[col].shift(lag).rename(col_name))
            shifted_df = pd.concat(shifted_columns, axis=1)
            dataset_opt = pd.concat([target_df.copy(), shifted_df], axis=1)
    
            # Split dataset into train and test
            train_dataset = dataset_opt[train_indices]
            test_dataset = dataset_opt[test_indices]
            
            # Standardize data
            Y_column = 'tcg' 
            X_train=train_dataset[train_dataset.columns.drop([Y_column]) ]
            Y_train=train_dataset[Y_column]
            X_test=test_dataset[test_dataset.columns.drop([Y_column]) ]
            Y_test=test_dataset[Y_column]
                
            scaler = preprocessing.MinMaxScaler()
            X_std_train = scaler.fit(X_train)
            X_std_train = scaler.transform(X_train)
            X_std_test = scaler.transform(X_test)
            X_train=pd.DataFrame(X_std_train, columns=X_train.columns, index=X_train.index)
            X_test=pd.DataFrame(X_std_test, columns=X_test.columns, index=X_test.index)

            # Train model
            if model_kind == 'linreg':
                clf = LinearRegression()
            elif model_kind == 'lgbm':
                clf = LGBMRegressor(verbosity=-1, n_estimators=25, max_depth=3, num_leaves=7, learning_rate=0.1, n_jobs=4)
            else:
                raise ValueError("Model kind not recognized")
            # Apply cross validation
            cv_scores = cross_val_score(clf, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
            clf.fit(X_train, Y_train)
            # Evaluate model on the test set
            Y_pred = clf.predict(X_test)
            test_mse = mean_squared_error(Y_test, Y_pred)
            # Prepare solution to save
            sol_file = pd.concat([sol_file, pd.DataFrame({'CV': [-cv_scores.mean()], 'Test': [test_mse], 'Sol': [solution]})], ignore_index=True)
            # Save solution
            sol_file.to_csv(indiv_path, sep=' ', header=sol_file.columns, index=None)
            
            return -cv_scores.mean()
        
        """
        This will be the function used to generate random vectors for the initialization of the algorithm
        """
        def random_solution(self):
            return np.random.choice(self.sup_lim[0], self.size, replace=True)
        
        """
        This will be the function that will repair solutions, or in other words, makes a solution
        outside the domain of the function into a valid one.
        If this is not needed simply return "solution"
        """
        def repair_solution(self, solution):

            # unique = np.unique(solution)
            # if len(unique) < len(solution):
            #     pool = np.setdiff1d(np.arange(self.inf_lim[0], self.sup_lim[0]), unique)
            #     new = np.random.choice(pool, len(solution) - len(unique), replace=False)
            #     solution = np.concatenate((unique, new))
            return np.clip(solution, self.inf_lim, self.sup_lim)

    # Define the objective function of the CRO algorithm optimization
    objfunc = ml_prediction(3*predictors_df.shape[1])

    # Set parameters of the CRO algorithm
    params = {
        "popSize": 100, # play with this parameter
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.2,
        "Pd": 0.8,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "stop_cond": "Neval",
        "time_limit": 4000.0,
        "Ngen": 10000,
        "Neval": 5000, # normally use 15000
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 1,
        "Njobs": 1,

        "dynamic": True,
        "dyn_method": "success",
        "dyn_metric": "avg",
        "dyn_steps": 10,
        "prob_amp": 0.01,

        "prob_file": os.path.join(output_dir, 'prob_history_' + model_kind + '_' + experiment_filename),
        "popul_file": os.path.join(output_dir, 'last_population_' + model_kind + '_' + experiment_filename),
        "history_file": os.path.join(output_dir, 'fit_history_' + model_kind + '_' + experiment_filename),
        "solution_file": os.path.join(output_dir, 'best_solution_' + model_kind + '_' + experiment_filename),
        "indiv_file": os.path.join(output_dir, 'indiv_history_' + model_kind + '_' + experiment_filename),
    }

    # Define the operators of the CRO algorithm, they are differ
    operators = [
        SubstrateInt("BLXalpha", {"F":0.8}),
        SubstrateInt("Multipoint"),
        SubstrateInt("HS", {"F": 0.7, "Cr":0.8,"Par":0.2}),
        SubstrateInt("Xor"),
    ]

    # Create the CRO algorithm object
    cro_alg = CRO_SL(objfunc, operators, params)

    # Run the optimization and save the solution
    solution, obj_value = cro_alg.optimize()
    solution.tofile(os.path.join(output_dir, solution_filename), sep=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature selection with CRO')
    parser.add_argument('--basin', type=str, help='Basin')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters')
    parser.add_argument('--remove_trend', type=str, help='If y retrieve dataset where trend has been removed')
    parser.add_argument('--remove_seasonality', type=str, help='If y retrieve dataset where seasonality has been removed') 
    parser.add_argument('--n_vars', type=int, help='Number of atmospheric variables considered in the FS process')
    parser.add_argument('--n_idxs', type=int, help='Number of climate indexes considered in the FS process')
    parser.add_argument('--output_folder', type=str, help='Name of experiment and of the output folder where to store the results')
    parser.add_argument('--model_kind', type=str, help='ML model to train for the computation of the optimization metric')
    parser.add_argument('--train_yearI', type=int, default=1980, help='Initial year for training')
    parser.add_argument('--train_yearF', type=int, default=2013, help='Final year for training')
    parser.add_argument('--test_yearF', type=int, default=2021, help='Final year for testing')
    args = parser.parse_args()
    main(args.basin, args.n_clusters, args.remove_trend, args.remove_seasonality, args.n_vars, args.n_idxs, args.output_folder, args.model_kind, args.train_yearI, 
         args.train_yearF, args.test_yearF)