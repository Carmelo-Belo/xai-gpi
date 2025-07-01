from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateReal import *
from PyCROSL.SubstrateInt import *

from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import tensorflow as tf
from keras import Input, Model
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import argparse

def loss_gpi_informed(y_true, y_pred, gpi):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    gpi = tf.cast(gpi, tf.float32)
    # Compute the mse between the true and predicted values
    mse_pred = tf.reduce_mean(tf.square(y_true - y_pred))
    # Compute the mse between the true values and the gpis
    mse_gpi = tf.reduce_mean(tf.square(y_true - gpi))
    return mse_pred + mse_gpi

class PI_model(Model):
    def train_step(self, data):
        X, (y_true, gpi) = data
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = loss_gpi_informed(y_true, y_pred, gpi)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

# Custom loss function for the physical informed lgbm model 
def lgbm_pi_obj(y_true, y_pred, gpi):
    # Define first and second derivate of the loss function to train the model
    grad = 2 * (y_pred - y_true) + 2 * (y_pred - gpi)
    hess = 2 * np.ones_like(y_true)
    return grad, hess

# Custom evaluation function for the physical informed lgbm model
def lgbm_pi_eval(y_true, y_pred, gpi):
    # Compute the mse between the true and predicted values
    mse_pred = mean_squared_error(y_true, y_pred)
    # Compute the mse between the true values and the gpis
    mse_gpi = mean_squared_error(y_true, gpi)
    eval_metric = mse_pred + mse_gpi
    return 'pi-mse_eval', eval_metric, False

def main(basin, n_clusters, n_vars, n_idxs, output_folder, model_kind, train_yearI, train_yearF, test_yearF):

    # Set project directory and name of file containing the target variable
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    target_file = 'target_residual_1980-2022_2.5x2.5.csv'
    # Set directories
    fs_dir = os.path.join(project_dir, 'xai-gpi')
    data_dir = os.path.join(fs_dir, 'data', f'{basin}_{n_clusters}clusters_noTS')

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
    # Load the gpis time series dataframe and select the target GPIs for physical information to pass to the network
    gpis_path = os.path.join(fs_dir, 'data', f'{basin}_2.5x2.5_gpis_time_series.csv')
    gpis_df = pd.read_csv(gpis_path, index_col=0)
    gpis_df.index = pd.to_datetime(gpis_df.index)
    gpis_df = gpis_df[gpis_df.index.year.isin(np.arange(1980,2023,1))]
    gpi_pi = gpis_df['ogpi']

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
    test_indices = (predictors_df.index.year > train_yearF) & (predictors_df.index.year < test_yearF)

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

            # Get the physical information of the GPIS for the training and testing periods
            gpi_pi_train = gpi_pi[train_indices]
            gpi_pi_test = gpi_pi[test_indices]
            
            # Standardize data
            Y_column = 'resid' 
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

            ## Train the model with physical information ##
            # Perform 5-fold cross validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold = 1
            cv_scores = []
            test_losses = []
            # Y_pred_train = pd.Series(index=Y_train.index)
            Y_pred_test = pd.Series(index=Y_test.index)
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                Y_train_fold, Y_val_fold = Y_train.iloc[train_index], Y_train.iloc[val_index]
                gpi_pi_train_fold, gpi_pi_val_fold = gpi_pi_train.iloc[train_index], gpi_pi_train.iloc[val_index]
                # Train the model
                def lgbm_custom_obj(y_true, y_pred):
                    gpi = gpi_pi_train_fold
                    return lgbm_pi_obj(y_true, y_pred, gpi)
                def lgbm_custom_eval(y_true, y_pred):
                    gpi = gpi_pi_val_fold
                    return lgbm_pi_eval(y_true, y_pred, gpi)
                lgbm_model = LGBMRegressor(
                    num_leaves=7, 
                    max_depth=3, 
                    learning_rate=0.1, 
                    n_estimators=25, 
                    objective=lgbm_custom_obj, 
                    n_jobs=4, 
                    verbosity=-1, 
                    early_stopping_rounds=5
                )
                lgbm_model.fit(
                    X_train_fold,
                    Y_train_fold,
                    eval_set=[(X_val_fold, Y_val_fold)],
                    eval_names=['val'],
                    eval_metric=lgbm_custom_eval
                )
                # Save metrics 
                val_loss = lgbm_model._best_score['val']['pi-mse_eval']
                cv_scores.append(val_loss)
                # Y_pred_train = lgbm_model.predict(X_train_fold)
                Y_pred_test = lgbm_model.predict(X_test)
                test_loss = lgbm_pi_eval(Y_test, Y_pred_test, gpi_pi_test)[1]
                test_losses.append(test_loss)
                fold += 1
            
            # Compute accuracy metric combining correlation and cross-validated MSE
            cv_score = np.array(cv_scores).mean()
            # Evaluate model on the test set
            test_loss_mean = np.array(test_losses).mean()
            # Prepare solution to save
            sol_file = pd.concat([sol_file, pd.DataFrame({'CV': [cv_score], 'Test': [test_loss_mean], 'Sol': [solution]})], ignore_index=True)
            # Save solution
            sol_file.to_csv(indiv_path, sep=' ', header=sol_file.columns, index=None)
            
            return cv_score
        
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
    parser.add_argument('--n_vars', type=int, default=8, help='Number of atmospheric variables considered in the FS process')
    parser.add_argument('--n_idxs', type=int, default=9, help='Number of climate indexes considered in the FS process')
    parser.add_argument('--output_folder', type=str, default='test', help='Name of experiment and of the output folder where to store the results')
    parser.add_argument('--model_kind', type=str, default='pi-lgbm', help='ML model to train for the computation of the optimization metric')
    parser.add_argument('--train_yearI', type=int, default=1980, help='Initial year for training')
    parser.add_argument('--train_yearF', type=int, default=2013, help='Final year for training')
    parser.add_argument('--test_yearF', type=int, default=2021, help='Final year for testing')
    args = parser.parse_args()
    main(args.basin, args.n_clusters, args.n_vars, args.n_idxs, args.output_folder, args.model_kind, args.train_yearI, 
         args.train_yearF, args.test_yearF)