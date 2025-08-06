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
import pandas as pd
import numpy as np # NumPy, a popular library for scientific computing
np.set_printoptions(precision = 2)  # reduced display precision on numpy arrays
import matplotlib
matplotlib.use('QtAgg')  # or 'Qt5Agg', 'Qt6Agg', etc.
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns  # Seaborn, a statistical data visualization library based on Matplotlib
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

def engineer_features(train_dataset_file_path, dataset_tag_train):
    train_dataset_file_handler = CSVHandler(train_dataset_file_path)

    video_ids = train_dataset_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file 
    i = 0; X_rows = []; y_train = []; X_rows_modified = []  # Initialize empty lists to store the features and target variable
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

                # X_rows_modified.append([duration_min, duration_min**2, duration_min**3,
                #                         speaking_words_count, speaking_words_count**2, speaking_words_count**3,
                #                         avg_speaking_speed_wpm, avg_speaking_speed_wpm**2, avg_speaking_speed_wpm**3, 
                #                         scenes_count, scenes_count**2, scenes_count**3,
                #                         avg_scene_change_per_min, avg_scene_change_per_min**2, avg_scene_change_per_min**3])
                y_train.append(average_percentage_viewed)
    
    X_train = np.array(X_rows)
    y_train = np.array(y_train)
    X_features = ['Duration in minutes','Total number of words','Average speaking speed (wpm)','Total number of scenes','Average scenes change rate (spm)']
    return X_train, y_train, X_features

def explore_data_trends(X, y, X_features, y_label):
    n = X.shape[1]  # Number of features        
    x_bins_number = 15

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 4*2), constrained_layout=True)
    # plt.figure(figsize=(3.5, 2.5))
    # Scatter Plot for the training data
    for i in range(n):
        data = X[:,i]
        mu = np.mean(data); sigma  = np.std(data)                  
        axes[0,i].scatter(X[:,i], y, marker='x', c='r' ,label = 'target')
        axes[0,i].set_xlabel(X_features[i])
        # axes[0, i].set_title(f'Raw Data {i+1}')
        axes[0,i].grid(True)
        axes[0,i].text(0.95, 0.05, f'Mean = {mu:.2f}\nStd = {sigma:.2f}', 
                    transform=axes[0,i].transAxes,
                    horizontalalignment='right', verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0,0].set_ylabel("Average Percentage Viewed (%)")
    plt.suptitle("Raw Training Data", fontsize=16)

    # Curve plot for average target value vs inputs bins
    # This plot shows the average target value for each bin of the feature, which can help
    # identify trends and relationships between the feature and the target variable.
    # It is useful for understanding how the target variable changes with respect to the feature.
    # X-axis: feature bins. Y-axis: average target value
    for i in range(n):
        x = X[:, i]
        x_bins = pd.cut(x, bins=x_bins_number)
        df = pd.DataFrame({'x_bin': x_bins, 'y': y})
        mean_y = df.groupby('x_bin', observed=True)['y'].mean() # Calculate the mean of y for each bin of x

        axes[1,i].plot(mean_y.index.astype(str), mean_y.values, marker='o')
        #axes[1,i].set_title(f'Feature: {X_features[i]}')
        axes[1,i].set_xlabel(X_features[i])
        axes[1,i].set_ylabel("Average Percentage Viewed (%)")
        axes[1,i].tick_params(axis='x', rotation=90)
        axes[1,i].grid(True)
    plt.savefig(f"Figure_1_All_{x_bins_number}bins.pdf", bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 2*1.75), constrained_layout=True)
    fig.suptitle('First Row: Box Plots | Second Row: Histograms', fontsize=16)
    # Boxplots (first row)
    for i in range(n):
        # A boxplot (box-and-whisker plot) shows the spread and summary of a numeric dataset using five key numbers:
        # minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum.
        # X-axis: feature values. Y-axis: target values
        data = X[:, i]
        x_bins = pd.cut(data, x_bins_number)
        df_temp = pd.DataFrame({'x_bin': x_bins, 'y': y})
    
        sns.boxplot(x='x_bin', y='y', data = df_temp, ax = axes[0, i])
        axes[0, i].set_xlabel(X_features[i])
        #axes[0, i].xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.1f}"))
        #axes[0, i].tick_params(axis='x', labelrotation=90)
        axes[0, i].grid(True)
    # Histograms (second row)
    for i in range(n):
        data = X[:,i]
        # A histogran is a plot that shows the distribution of a numeric variable by dividing the data into bins (intervals) 
        # and counting how many values fall into each bin. X-axis: value ranges (bins). Y-axis: frequency (counts)
        axes[1, i].hist(data, x_bins_number, facecolor='skyblue', edgecolor='black', linewidth=1.2)
        axes[1, i].set_xlabel(X_features[i])
        axes[1, i].set_ylabel("Number of videos")
        # axes[1, i].set_title(f'Normalised Data {i+1}')
        axes[1, i].grid(True)
    # axes[1, 0].set_ylabel("Number of videos")
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"Figure_2_All_{x_bins_number}bins.pdf", bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 1.75), constrained_layout=True)
    axes[0].hist(X[:,-1], bins = x_bins_number, facecolor='skyblue', edgecolor='black', linewidth=1.2)
    axes[0].set_xlabel(X_features[-1])
    axes[0].set_ylabel('Number of videos')
    axes[0].grid(True)

    axes[1].hist(y, bins = x_bins_number, facecolor='skyblue', edgecolor='black', linewidth=1.2)
    axes[1].set_xlabel(y_label)
    axes[1].set_ylabel('Number of videos')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"Figure_2_ctd_All_{x_bins_number}bins.pdf", bbox_inches='tight')
    plt.show()

def visualise_feature_trends(X, y, feature_names, y_label="Target Output"):
    """
    Visualises:
    1. Correlation of each feature with the target
    2. Binned average trends of y across each feature
    3. Smoothed LOESS curves per feature

    Parameters:
        X (np.ndarray): 2D array of shape (n_samples, n_features)
        y (np.ndarray): 1D array of target values (n_samples,)
        feature_names (list of str): Names of the features
        y_label (str): Label for the target variable
    """
    x_bins_number = 10

    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df[y_label] = y

    # 1. Correlation bar plot
    correlations = df.corr(numeric_only=True)[y_label].drop(y_label)
    # df.corr(numeric_only=True): Computes the correlation matrix between all numeric columns in the DataFrame.
    # [y_label]: Extracts the column of correlations between each variable and y_label.
    # .drop(y_label): Removes the correlation of y_label with itself (which is always 1.0).
    # The .corr() method (by default) computes the Pearson correlation coefficient between each pair of numeric columns.
    # Ranges from –1 to +1; Measures linear association; +1 = perfect positive linear correlation; 0 = no linear correlation; –1 = perfect negative linear correlation
    plt.figure(figsize=(3.5, 2.5))
    colors = ['blue' if val > 0 else 'red' for val in correlations]
    sns.barplot(x=correlations.values, y=correlations.index, palette=colors)
    plt.title(f'Correlation with {y_label}')
    plt.xlabel('Correlation Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Figure_3_All_{x_bins_number}bins.pdf", bbox_inches='tight')
    plt.show()

    # 2. Full correlation matrix
    plt.figure(figsize=(10, 8))
    full_corr = df[feature_names + [y_label]].corr(numeric_only=True)
    sns.heatmap(full_corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Full Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # 3. Binned average y per feature
    fig, axes = plt.subplots(1, len(feature_names), figsize=(5 * len(feature_names), 4), constrained_layout=True)
    for i, feature in enumerate(feature_names):
        x_bin = pd.cut(df[feature], x_bins_number)
        mean_y = df.groupby(x_bin, observed=True)[y_label].mean()
        axes[i].plot(mean_y.index.astype(str), mean_y.values, marker='o')
        axes[i].set_title(feature)
        axes[i].set_xlabel('Binned')
        axes[i].set_ylabel(f'Avg {y_label}')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True)
    plt.savefig(f"Figure_4_All_{x_bins_number}bins.pdf", bbox_inches='tight')
    plt.show()

    # 4. LOESS smoothed plots
    fig, axes = plt.subplots(1, len(feature_names), figsize=(5 * len(feature_names), 2), constrained_layout=True)
    for i, feature in enumerate(feature_names):
        x_vals = df[feature]
        mu = np.mean(x_vals)
        sigma = np.std(x_vals)

        loess_smoothed = lowess(endog=df[y_label], exog=x_vals, frac=0.3) # Locally Weighted Scatterplot Smoothing (LOESS) of y vesus x
        # It's a non-parametric regression technique that fits local linear regressions to smooth a curve through noisy data.
        axes[i].scatter(x_vals, df[y_label], s=10, marker='o', c='blue', alpha=0.3) # s: marker size
        axes[i].plot(loess_smoothed[:, 0], loess_smoothed[:, 1], color='red')
        axes[i].set_title(f"{feature} (LOESS)")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(y_label)
        axes[i].grid(True)
        axes[i].text(0.95, 0.05, f'Mean = {mu:.2f}\nStd = {sigma:.2f}', 
                    transform=axes[i].transAxes,
                    horizontalalignment='right', verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.savefig(f"Figure_5_All_{x_bins_number}bins.pdf", bbox_inches='tight')
    plt.show()

def scale_features(X, y, X_features):
    # An instance of the LinearUnivariateRegression class
    model = MultipleLinearRegression()
    n = X.shape[1]  # Number of features

    ## Scale/normalize the training data
    X_norm = model.zscore_normalize_features(X)
    # X_norm = model.mean_normalize_features(X_train)  # Mean normalisation
    print(f"Peak to Peak range by column in Raw        X: {np.ptp(X,axis=0)}")   
    print(f"Peak to Peak range by column in Normalized X: {np.ptp(X_norm,axis=0)}")
    # The scatter plots below can provide some indication of which features have the strongest influence on the output
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(15, 8))
    fig.suptitle('First Row: Non-Normalised | Second Row: Normalised (Z-score Normalisation)', fontsize=16)
    # Plot non-normalised data (first row)
    for i in range(n):
        data = X[:,i]
        mu = np.mean(data); sigma  = np.std(data)                  
        axes[0, i].scatter(X[:,i], y, marker='x', c='r' ,label = 'target')
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
        axes[1, i].scatter(X_norm[:,i], y, marker='x', c='r' ,label = 'target')
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

    return X_norm

def model_fit(X_train, y_train, X_features):
    """
    Function to fit the model with the training data.
    This function loads the data, processes it, and trains the model.
    """
    # An instance of the LinearUnivariateRegression class
    model = MultipleLinearRegression()

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

    ## Create and fit the regression model
    w, b, J_hist = model.gradient_descent(X = X_norm, y = y_train, w_in = np.zeros(n,), b_in = 0, 
                                                  cost_function = model.compute_cost, gradient_function = model.compute_gradient, 
                                                  alpha = 0.9, num_iters = 2000)
    print(f"model parameters:                   w: {w}, b:{b}")

    ## Visualise the cost function change over iterations
    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
    # plt.show()

    ## Make predictions
    y_pred = model.predict(X_train, w, b)
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
    fig,ax=plt.subplots(1,n,figsize=(5 * n, 1.75),sharey=True)
    for i in range(n):
        ax[i].scatter(X_train[:,i],y_train, s=10, label = 'target') # s=10, marker='o', c='blue', alpha=0.3
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:,i],y_pred,color = dlc["dlorange"], s=10, label = 'predict')
        ax[i].set_ylabel("Average Viewed (%)"); ax[0].legend()
        ax[i].grid(True)
    # fig.suptitle("Target versus prediction using z-score normalised model")
    plt.savefig(f"Figure_predicted_vs_actual_all.pdf", bbox_inches='tight')
    plt.show()
   

if __name__ == "__main__":
    
    all_train_dataset_file_path = parent_dir + '/' + 'data/all_train_dataset.csv'

    dataset_tag_train = ["newcastle_mech1750_lecture_level1_agregg_2025", "newcastle_engg2440_lecture_level2_agregg_2025"]
    
    X_train, y_train, X_features = engineer_features(all_train_dataset_file_path, dataset_tag_train)
    # explore_data_trends(X_train, y_train, X_features, y_label="Average Viewed (%)")
    visualise_feature_trends(X_train, y_train, X_features, y_label="Average Viewed (%)")
    # X_norm = scale_features(X_train, y_train, X_features)
    # model_fit(X_norm, y_train, X_features)  # Fit the model with the training data



    # model_fit(all_train_dataset_file_path, dataset_tag_train)
    # model_fit(all_train_dataset_file_path, [dataset_tag_train[0]])

        
    # characs_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_characs.csv'
    # metrics_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_metrics.csv'
    # train_dataset_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_train_dataset.csv'
    
    # characs_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_characs.csv'
    # metrics_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_metrics.csv'
    # train_dataset_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_train_dataset.csv'

    # create_train_dataset_csv(train_dataset_file_path)  # Create the train dataset file
    # populate_train_dataset_file(characs_file_path, metrics_file_path, train_dataset_file_path)  # Populate the train dataset file
