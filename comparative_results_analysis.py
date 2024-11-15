import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils_results as ut

def main():
    n_clusters = 6
    basin = 'GLB'

    # Set directories
    project_dir = '/Users/huripari/Documents/PhD/TCs_Genesis'
    fs_dir = os.path.join(project_dir, 'FS_TCG')
    results_dir = os.path.join(fs_dir, 'results')
    data_dir = os.path.join(fs_dir, 'data', f'{basin}_{n_clusters}clusters')
    # List all the files in the results directory and select the one to analyze
    all_subfolders = os.listdir(results_dir)
    experiments_folders = [f for f in all_subfolders if f'nc{n_clusters}' in f]
    # Load dataset containing all candidate predictors
    predictor_file = '1965'
    predictors_path = os.path.join(data_dir, predictor_file)


