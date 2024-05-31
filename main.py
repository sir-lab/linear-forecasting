"""
Main file to accompany ICML 2024 paper: "An Analysis of Linear Time Series Forecasting Models"
    Authors: William Toner and Luke Darlow.

This is a minimal code repository to demonstrate and support the OLS solutions described
in the paper. Repositories for re-implementing the comparison methods are widely available.

Note: it is important to avoid the 'drop_last' error that has recently been discovered. We
worked with the Authors of 'FITS' to detect and remedy this issue. See
https://github.com/VEWOXIC/FITS for a thorough description.
"""
import numpy as np
import argparse
from data.datasets import dataset_selector
from model import OLS

if __name__  == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', action='store', type=str, default='ETTh1', help="Dataset selector")
    parser.add_argument('--root', action='store', type=str, default='data/', help="Root directory for standard csvs or custom file")
    parser.add_argument('--custom_csv_filename', action='store', type=str, default=None, help="Name of custom csv file")
    parser.add_argument('--custom_train_percentage', action='store', type=float, default=0.7, help="Only used for custom csv files")
    parser.add_argument('--custom_test_percentage', action='store', type=float, default=0.2, help="Only used for custom csv files")
    parser.add_argument('--context_length', action='store', type=int, default=720, help="Context length to train on")
    parser.add_argument('--horizon', action='store', type=int, default=720, help="Forecast horizon length")
    parser.add_argument('--alpha', action='store', type=float, default=0.00001, help="Regularisation param")
    parser.add_argument('--instance_norm', action=argparse.BooleanOptionalAction, default=True, help="Whether to use instance norm (+IN in paper)")
    parser.add_argument('--individual', action=argparse.BooleanOptionalAction, default=True, help="Whether to use instance norm (+IN in paper)")
    parser.add_argument('--seed', action='store', type=int, default=42, help="Random Seed")
    parser.add_argument('--max_train_N', action='store', type=int, default=None, help="Truncate training data (faster approximate fit, good for debugging)")
    args = parser.parse_args()   

    ########################################################### HOUSEKEEPING
    if args.seed is not None:
        np.random.seed(args.seed)

    ########################################################### DATASETS AND LOADERS
    dataset_train, dataset_val, dataset_test = dataset_selector(args.dataset, args.context_length, args.horizon,
                                                                root=args.root,
                                                                custom_csv_filename=args.custom_csv_filename,
                                                                custom_train_percentage=args.custom_train_percentage, 
                                                                custom_test_percentage=args.custom_test_percentage,
                                                                )  

    ########################################################### MODEL DEFINITIONS
    model = OLS(dataset_train.data, 
                            args.context_length, 
                            args.horizon, 
                            instance_norm=args.instance_norm, 
                            individual=args.individual, 
                            alpha=args.alpha,
                            verbose=False,
                            seed=args.seed,
                            max_train_N=args.max_train_N)
    
    # Builds sliding window dataset from test data
    test_instances = np.lib.stride_tricks.sliding_window_view(dataset_test.data, (args.context_length+args.horizon), axis=0)  
    X = test_instances[:,:,:args.context_length]
    y = test_instances[:,:,args.context_length:]
    preds = model.predict(X)
    print(f'{args.dataset}. IN={args.instance_norm}, individual={args.individual}, alpha={args.alpha}. Context/Horizon={args.context_length}/{args.horizon}. MSE={np.mean((y-preds)**2):0.3f}; MAE={np.mean(np.abs(y-preds)):0.3f}.')
    