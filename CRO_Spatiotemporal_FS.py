from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateReal import *
from PyCROSL.SubstrateInt import *

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os

# Set directories
project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
fs_dir = os.path.join(project_dir, 'FS_TCG')
data_dir = os.path.join(fs_dir, 'data')

"""
Path and name of the predictor dataset and target dataset
"""
predictor_file = 'predictors_1965-2022_12clusters_4vars_10idxs.csv'
predictors_path = os.path.join(data_dir, predictor_file)
target_file = 'target_1965-2022_2.5x2.5.csv'
target_path = os.path.join(data_dir, target_file)

# Load the dataset and target
predictors_df = pd.read_csv(predictors_path, index_col=0)
predictors_df.index = pd.to_datetime(predictors_df.index)
target_df = pd.read_csv(target_path, index_col=0)
target_df.index = pd.to_datetime(target_df.index)

"""
File names to store the solutions provided by the algorithm
"""
experiment_filename = predictor_file[-37:]
output_dir = os.path.join(fs_dir, 'results', 'test')
os.makedirs(output_dir, exist_ok=True)
indiv_path = os.path.join(output_dir, experiment_filename)
model_kind = 'LinReg'
solution_filename = 'CRO_' + model_kind + '_' + experiment_filename

# Create an empty file to store the solutions provided by the algorithm
sol_data = pd.DataFrame(columns=['CV', 'Test', 'Sol'])
sol_data.to_csv(indiv_path, sep=' ', header=sol_data.columns, index=None)

# Split the dataset into train and test
train_yearI = 1980 # First year of the training dataset
train_yearF = 2013 # Last year of the training dataset
test_yearF = 2021 # Last year of the test dataset
train_indices = (predictors_df.index.year >= train_yearI) & (predictors_df.index.year <= train_yearF) 
test_indices = (predictors_df.index.year > train_yearF) & (predictors_df.index.year <= test_yearF)

"""
All the following methods will have to be implemented for the algorithm to work properly
with the same inputs, except for the constructor 
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
        # self.sup_lim = np.append(np.append(np.repeat(60, predictors_df.shape[1]),np.repeat(180, predictors_df.shape[1])),np.repeat(1, predictors_df.shape[1]))  # array where each component indicates the maximum value of the component of the vector
        # self.inf_lim = np.append(np.append(np.repeat(1, predictors_df.shape[1]),np.repeat(0, predictors_df.shape[1])),np.repeat(0, predictors_df.shape[1])) # array where each component indicates the minimum value of the component of the vector
        # we call the constructor of the superclass with the size of the vector
        # and wether we want to maximize or minimize the function 

        # We set the limits of the vector (window size, time lags and variable selection)
        # Maximum time sequences I can select is 2, maximum time lag is 1 month, and the last one is regarding the binary selection of a variable
        self.sup_lim = np.append(np.append(np.repeat(2, predictors_df.shape[1]), np.repeat(1, predictors_df.shape[1])), np.repeat(1, predictors_df.shape[1]))
        self.inf_lim = np.append(np.append(np.repeat(1, predictors_df.shape[1]), np.repeat(0, predictors_df.shape[1])), np.repeat(0, predictors_df.shape[1]))

        super().__init__(self.size, self.opt, self.sup_lim, self.inf_lim)
    
    """
    This will be the objective function, that will recieve a vector and output a number
    """
    def objective(self, solution):
        # print(solution)
        # Read data
        sol_file = pd.read_csv(indiv_path, sep=' ', header=0)

        # Read solution
        time_sequences = np.append(np.array(solution[:predictors_df.shape[1]]).astype(int), 1)
        time_lags = np.append(np.array(solution[predictors_df.shape[1]:(2*predictors_df.shape[1])]).astype(int), 1)
        variable_selection = np.array(solution[(2*predictors_df.shape[1]):]).astype(int)

        if sum(variable_selection) == 0:  # If no variables are selected, return a high value
            return 100000

        # Create dataset according to solution
        dataset_opt = target_df.copy()
        for c, col in enumerate(predictors_df.columns):
            if variable_selection[c] == 0 or time_sequences[c] == 0:
                continue
            for j in range(time_sequences[c]):
                dataset_opt[str(col) +'_lag'+ str(time_lags[c]+j)] = predictors_df[col].shift(time_lags[c]+j)
   
        # Split dataset into train and test
        train_dataset = dataset_opt[train_indices]
        test_dataset = dataset_opt[test_indices]
        
        # Standardize data
        Y_column = 'tcg' 
        X_train=train_dataset[train_dataset.columns.drop([Y_column]) ]
        Y_train=train_dataset[Y_column]
        X_test=test_dataset[test_dataset.columns.drop([Y_column]) ]
        Y_test=test_dataset[Y_column]
            
        scaler = preprocessing.StandardScaler()
        X_std_train = scaler.fit(X_train)
        X_std_train = scaler.transform(X_train)
        X_std_test = scaler.transform(X_test)
        X_train=pd.DataFrame(X_std_train, columns=X_train.columns, index=X_train.index)
        X_test=pd.DataFrame(X_std_test, columns=X_test.columns, index=X_test.index)

        # Train model
        # clf = LogisticRegression() -> gives Nan in the mean score of cross validation
        clf = LinearRegression()

        # Apply cross validation
        # score = cross_val_score(clf, X_train, Y_train, cv=5, scoring='f1')
        score = cross_val_score(clf, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        # print(score.mean(), f1_score(Y_pred, Y_test, average='weighted'))
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        print(f"Cross-validated MSE: {-score.mean()}")  # Negated to report positive MSE
        print(f"Test MSE: {mse}, Test R^2: {r2}")

        # Save solution
        # sol_file = pd.concat([sol_file, pd.DataFrame({'CV': [score.mean()], 'Test': [f1_score(Y_pred, Y_test, average='weighted')], 'Sol': [solution]})], ignore_index=True)
        sol_file = pd.concat([sol_file, pd.DataFrame({'CV': [-score.mean()], 'Test': [mse], 'Sol': [solution]})], ignore_index=True)
        sol_file.to_csv(indiv_path, sep=' ', header=sol_file.columns, index=None)
        
        return 1/score.mean()
    
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

objfunc = ml_prediction(3*predictors_df.shape[1])

params = {
    "popSize": 100,
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
    "Neval": 150,
    "fit_target": 1000,

    "verbose": True,
    "v_timer": 1,
    "Njobs": 1,

    "dynamic": True,
    "dyn_method": "success",
    "dyn_metric": "avg",
    "dyn_steps": 10,
    "prob_amp": 0.01,

    "prob_file": os.path.join(output_dir, "prob_history_" + experiment_filename),
    "popul_file": os.path.join(output_dir, "last_population_" + experiment_filename),
    "history_file": os.path.join(output_dir, "fit_history_" + experiment_filename),
    "solution_file": os.path.join(output_dir, "best_solution_" + experiment_filename),
    "indiv_file": os.path.join(output_dir, "indiv_hisotry_" + experiment_filename),
}

operators = [
    SubstrateInt("BLXalpha", {"F":0.8}),
    SubstrateInt("Multipoint"),
    SubstrateInt("HS", {"F": 0.7, "Cr":0.8,"Par":0.2}),
    SubstrateInt("Xor"),
]

cro_alg = CRO_SL(objfunc, operators, params)

solution, obj_value = cro_alg.optimize()

solution.tofile(os.path.join(output_dir, solution_filename), sep=',')

