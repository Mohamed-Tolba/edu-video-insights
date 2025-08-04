"""
Module: model_data.py
Author: Mohamed Tolba
Date Created: 04-08-2025
Last Updated: 04-08-2025

Description:
    Script that loads data and trains a model.
"""

import sys
import os
import copy, math  # Math library for mathematical functions
import numpy as np # NumPy, a popular library for scientific computing
np.set_printoptions(precision = 2)  # reduced display precision on numpy arrays
import matplotlib
matplotlib.use('QtAgg')  # or 'Qt5Agg', 'Qt6Agg', etc.
import matplotlib.pyplot as plt
# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

from core.csv_utils import CSVHandler  # Import the CSVHandler class for managing CSV files
from core.regression_core import MultipleLinearRegression  # Import the RegressionModel class for regression tasks
from scripts.create_temp_data_files import *

def populate_train_dataset_file(characs_file_path, metrics_file_path, train_dataset_file_path) -> None:
    """
    Populates the video_submission.csv file with video IDs and user inputs.
    This function reads video IDs from user_data.csv and creates a new video_submission.csv file.
    """
    characs_file_handler = CSVHandler(characs_file_path)
    metrics_file_handler = CSVHandler(metrics_file_path) 
    train_dataset_file_handler = CSVHandler(train_dataset_file_path) 

    # Clear all data in the video submission file, keeping only the header
    train_dataset_file_handler.clear_all_rows(msg="Any data in the train dataset file has been deleted")  # Clear all data in the video submission file, keeping only the header
    train_dataset_file_handler.clean_csv()  # Clean the video submission file by removing invalid rows and duplicates, and extra unnamed columns
    print("Creating train dataset file...")
    
    video_ids = characs_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file
    for video_id in video_ids:  # Loop through each video ID
        if video_id:
            train_dataset_fields = {
                "video_id": video_id,  # Store the video ID
                "dataset_tag": characs_file_handler.get_cell_value_by_match("video_id", video_id, "dataset_tag"),
                "duration_min": characs_file_handler.get_cell_value_by_match("video_id", video_id, "duration_min"),
                "speaking_words_count": characs_file_handler.get_cell_value_by_match("video_id", video_id, "speaking_words_count"),
                "avg_speaking_speed_wpm": characs_file_handler.get_cell_value_by_match("video_id", video_id, "avg_speaking_speed_wpm"),
                "scenes_count": characs_file_handler.get_cell_value_by_match("video_id", video_id, "scenes_count"),
                "avg_scene_change_per_min": characs_file_handler.get_cell_value_by_match("video_id", video_id, "avg_scene_change_per_min"),
                "average_percentage_viewed": metrics_file_handler.get_cell_value_by_match("video_id", video_id, "average_percentage_viewed")
            }
            train_dataset_file_handler.add_new_row(train_dataset_fields)  # Populate the row with the video user inputs
    train_dataset_file_handler.clean_csv()  # Clean the video submission file by removing invalid rows and duplicates, and extra unnamed columns

def model_fit(train_dataset_file_path, dataset_tag_train):
    """
    Function to fit the model with the training data.
    This function loads the data, processes it, and trains the model.
    """
    # An instance of the LinearUnivariateRegression class
    model = MultipleLinearRegression()

    train_dataset_file_handler = CSVHandler(train_dataset_file_path)

    video_ids = train_dataset_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file 
    i = 0; X_rows = []; y_train = []
    for target_dataset_tag in dataset_tag_train:  # Loop through each dataset tag
        if not target_dataset_tag:  # Skip empty dataset tags
            continue
        print(f"Processing dataset tag: {target_dataset_tag}")

        video_ids = train_dataset_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file 

        if not video_ids:  # If no video IDs found for the dataset tag, skip to the next iteration
            print(f"No video IDs found for dataset tag: {target_dataset_tag}")
            continue
        print(f"Found {len(video_ids)} video IDs for dataset tag: {target_dataset_tag}")
        # Loop through each video ID and extract the relevant features and target variable
        for video_id in video_ids:  # Loop through each video ID
            if video_id:  # Check if the video ID is not empty
                dataset_tag = train_dataset_file_handler.get_cell_value_by_match("video_id", video_id, "dataset_tag"), # Get the dataset tag of the video from the video submission file
                dataset_tag = dataset_tag[0] if dataset_tag else None  # Extract the first element from the tuple
                if dataset_tag != target_dataset_tag: # Check if the dataset tag matches the target dataset tag
                    continue
                duration_min = train_dataset_file_handler.get_cell_value_by_match("video_id", video_id, "duration_min")
                speaking_words_count = train_dataset_file_handler.get_cell_value_by_match("video_id", video_id, "speaking_words_count")
                avg_speaking_speed_wpm = train_dataset_file_handler.get_cell_value_by_match("video_id", video_id, "avg_speaking_speed_wpm")
                scenes_count = train_dataset_file_handler.get_cell_value_by_match("video_id", video_id, "scenes_count")
                avg_scene_change_per_min = train_dataset_file_handler.get_cell_value_by_match("video_id", video_id, "avg_scene_change_per_min")
                average_percentage_viewed = train_dataset_file_handler.get_cell_value_by_match("video_id", video_id, "average_percentage_viewed")
                X_rows.append([duration_min, speaking_words_count, avg_speaking_speed_wpm, scenes_count, avg_scene_change_per_min])
                y_train.append(average_percentage_viewed)
    
    X_train = np.array(X_rows)
    y_train = np.array(y_train)
    X_features = ['Duration in minutes','Total number of words','Average speaking speed (wpm)','Total number of scenes','Average scenes change rate (per minute)']
    m = X_train.shape[0] # Number of training examples
    n = X_train.shape[1] # Number of training features

    ## View the variables (Before starting on any task, it is useful to get more familiar with your dataset)
    # print x_train
    print("Type of x_train:",type(X_train))
    print("First five elements of x_train are:\n", X_train[:5]) 
    # print y_train
    print("Type of y_train:",type(y_train))
    print("First five elements of y_train are:\n", y_train[:5])

    ## Check the dimensions of your variables
    print ('The shape of x_train is:', X_train.shape)
    print ('The shape of y_train is: ', y_train.shape)
    print ('Number of training examples (m):', m)

    ## Visualize your data
    # The scatter plots below can provide some indication of which features have the strongest influence on the output
    # dlc = {"dlorange": "#FF9900"}    
    # fig,ax = plt.subplots(1,n,figsize=(12,3),sharey=True)
    # for i in range(len(ax)):
    #     ax[i].scatter(X_train[:,i], y_train, marker='x', c='r' ,label = 'target')
    #     ax[i].set_xlabel(X_features[i])
    # ax[0].set_ylabel("average_percentage_viewed"); ax[0].legend()
    # fig.suptitle("Visualising the trainning data (without normalisation)")
    # plt.show()

    ## Scale/normalize the training data
    X_norm = model.zscore_normalize_features(X_train)
    # X_norm = model.mean_normalize_features(X_train)  # Mean normalisation
    print(f"Peak to Peak range by column in Raw        X: {np.ptp(X_train,axis=0)}")   
    print(f"Peak to Peak range by column in Normalized X: {np.ptp(X_norm,axis=0)}")
    # The scatter plots below can provide some indication of which features have the strongest influence on the output
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(15, 8))
    fig.suptitle('First Row: Non-Normalised | Second Row: Normalised (Z-score Normalisation)', fontsize=16)
    # Plot non-normalised data (first row)
    for i in range(n):
        data = X_train[:,i]
        mu = np.mean(data); sigma  = np.std(data)                  
        axes[0, i].scatter(X_train[:,i], y_train, marker='x', c='r' ,label = 'target')
        axes[0, i].set_xlabel(X_features[i])
        # axes[0, i].set_title(f'Raw Data {i+1}')
        axes[0, i].grid(True)
        axes[0, i].text(0.95, 0.05, f'Mean = {mu:.2f}\nStd = {sigma:.2f}', 
                    transform=axes[0, i].transAxes,
                    horizontalalignment='right', verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0, 0].set_ylabel("Average Percentage Viewed (%)")
    # Plot normalised data (second row)
    for i in range(n):
        data = X_norm[:,i]
        mu = np.mean(data); sigma  = np.std(data)  
        axes[1, i].scatter(X_norm[:,i], y_train, marker='x', c='r' ,label = 'target')
        axes[1, i].set_xlabel(X_features[i]+" (normalised)")
        # axes[1, i].set_title(f'Normalised Data {i+1}')
        axes[1, i].grid(True)
        axes[1, i].text(0.95, 0.05, f'Mean = {mu:.2f}\nStd = {sigma:.2f}', 
                    transform=axes[1, i].transAxes,
                    horizontalalignment='right', verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        axes[1, i].set_xlim(-1, 1)
    axes[1, 0].set_ylabel("Average Percentage Viewed (%)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    ## Create and fit the regression model
    w_norm, b_norm, J_hist = model.gradient_descent(X = X_norm, y = y_train, w_in = np.zeros(n,), b_in = 0, 
                                                  cost_function = model.compute_cost, gradient_function = model.compute_gradient, 
                                                  alpha = 0.9, num_iters = 2000)
    print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

    ## Visualise the cost function change over iterations
    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
    plt.show()

    ## Make predictions
    y_pred = model.predict(X_norm, w_norm, b_norm)
    print(f"Prediction on training set:\n{y_pred[:4]}" )
    print(f"Target values \n{y_train[:4]}")

    ## Model Evaluation
    mse, rmse, mae, r2 = model.evaluate_model(y_train, y_pred)
    print(f"Evaluation Metrics:")
    print(f"- MSE  : {mse:.4f}")
    print(f"- RMSE : {rmse:.4f}")
    print(f"- MAE  : {mae:.4f}")
    print(f"- R^2  : {r2:.4f}")
    ## Plot Results
    # plot predictions and targets vs original features
    dlc = {"dlorange": "#FF9900"}    
    fig,ax=plt.subplots(1,n,figsize=(12,3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:,i],y_pred,color = dlc["dlorange"], label = 'predict')
    ax[0].set_ylabel("Average Percentage Viewed (%)"); ax[0].legend()
    fig.suptitle("Target versus prediction using z-score normalised model")
    plt.show()
   

if __name__ == "__main__":
    
    # characs_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_characs.csv'
    # metrics_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_metrics.csv'
    # train_dataset_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_train_dataset.csv'
    
    # characs_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_characs.csv'
    # metrics_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_metrics.csv'
    # train_dataset_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_train_dataset.csv'

    # create_train_dataset_csv(train_dataset_file_path)  # Create the train dataset file
    # populate_train_dataset_file(characs_file_path, metrics_file_path, train_dataset_file_path)  # Populate the train dataset file
    
    all_train_dataset_file_path = parent_dir + '/' + 'data/all_train_dataset.csv'

    dataset_tag_train = ["newcastle_mech1750_lecture_level1_agregg_2025", "newcastle_engg2440_lecture_level2_agregg_2025"]
    model_fit(all_train_dataset_file_path, dataset_tag_train)
    # model_fit(all_train_dataset_file_path, [dataset_tag_train[0]])
