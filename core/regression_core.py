"""
Validate and test the K-fold cross-validation implementation
"""


import copy, math  # Math library for mathematical functions
import numpy as np # NumPy, a popular library for scientific computing
np.set_printoptions(precision = 2)  # reduced display precision on numpy arrays
import matplotlib
matplotlib.use('QtAgg')  # or 'Qt5Agg', 'Qt6Agg', etc.
import matplotlib.pyplot as plt

import os, sys
# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add to Python module search path
sys.path.append(parent_dir)

from core.csv_utils import CSVHandler  # Import the CSVHandler class for managing CSV files

class MultipleLinearRegression:
    def __init__(self):
        pass

    def zscore_normalize_features(self, X):
        """
        computes  X, zcore normalized by column

        Args:
          X (ndarray (m,n))     : input data, m examples, n features

        Returns:
          X_norm (ndarray (m,n)): input normalized by column
          mu (ndarray (n,))     : mean of each feature
          sigma (ndarray (n,))  : standard deviation of each feature
        """
        # find the mean of each column/feature
        mu = np.mean(X, axis=0)                 # mu will have shape (n,)
        # find the standard deviation of each column/feature
        sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
        # element-wise, subtract mu for that column from each example, divide by std for that column
        X_norm = (X - mu) / sigma      

        return (X_norm)
    
    def mean_normalize_features(self, X):
        """
        computes  X, mean normalized by column

        Args:
          X (ndarray (m,n))     : input data, m examples, n features

        Returns:
          X_norm (ndarray (m,n)): input normalized by column
          mu (ndarray (n,))     : mean of each feature
        """
        # find the mean of each column/feature
        mu = np.mean(X, axis=0)                 # mu will have shape (n,)
        max = np.max(X, axis=0)  # max of each column
        min = np.min(X, axis=0)  # min of each column
        # element-wise, subtract mu for that column from each example
        X_norm = (X - mu) / (max - min)

        return (X_norm)
    
    def predict(self, X, w, b): 
        """
        single predict using linear regression
        Args:
          x (ndarray): Shape (n,) example with multiple features
          w (ndarray): Shape (n,) model parameters   
          b (scalar):             model parameter 

        Returns:
          p (scalar):  prediction
        """
        p = np.dot(X, w) + b     
        return p
    
    def compute_cost(self, X, y, w, b): 
     """
     compute cost
     Args:
       X (ndarray (m,n)): Data, m examples with n features
       y (ndarray (m,)) : target values
       w (ndarray (n,)) : model parameters  
       b (scalar)       : model parameter
       
     Returns:
       cost (scalar): cost
     """
     m = X.shape[0]
     cost = 0.0
     for i in range(m):                                
         f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
         cost = cost + (f_wb_i - y[i])**2       #scalar
     total_cost = cost / (2 * m)                #scalar    
     return total_cost
    
    def compute_gradient(self, X, y, w, b): 
        """
        Computes the gradient for linear regression 
        Args:
          X (ndarray (m,n)): Data, m examples with n features
          y (ndarray (m,)) : target values
          w (ndarray (n,)) : model parameters  
          b (scalar)       : model parameter

        Returns:
          dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
          dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
        """
        m,n = X.shape           #(number of examples, number of features)
        dj_dw = np.zeros((n,))
        dj_db = 0.

        for i in range(m):                             
            err = (np.dot(X[i], w) + b) - y[i]   
            for j in range(n):                         
                dj_dw[j] = dj_dw[j] + err * X[i, j]    
            dj_db = dj_db + err                        
        dj_dw = dj_dw / m                                
        dj_db = dj_db / m                                

        return dj_db, dj_dw
    
    def gradient_descent(self, X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
        """
        Performs batch gradient descent to learn w and b. Updates w and b by taking 
        num_iters gradient steps with learning rate alpha

        Args:
          X (ndarray (m,n))   : Data, m examples with n features
          y (ndarray (m,))    : target values
          w_in (ndarray (n,)) : initial model parameters  
          b_in (scalar)       : initial model parameter
          cost_function       : function to compute cost
          gradient_function   : function to compute the gradient
          alpha (float)       : Learning rate
          num_iters (int)     : number of iterations to run gradient descent

        Returns:
          w (ndarray (n,)) : Updated values of parameters 
          b (scalar)       : Updated value of parameter 
          """

        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        w = copy.deepcopy(w_in)  #avoid modifying global w within function
        b = b_in

        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            # Save cost J at each iteration
            if i < 100000:      # prevent resource exhaustion 
                J_history.append(cost_function(X, y, w, b))
                
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

        return w, b, J_history #return final w,b and J history for graphing
    
    def evaluate_model(self, y_true, y_pred):
      m = len(y_true)
      mse = np.mean((y_true - y_pred) ** 2) # Mean Square Error
      rmse = np.sqrt(mse) # Root Mean Square Error
      mae = np.mean(np.abs(y_true - y_pred)) # Mean Absolute Error
      r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2) # R-squared (Coefficient of Determination)
      # R2 = 1 is perfect; R2 = 0 means tmodel is as good as the mean.
      return mse, rmse, mae, r2
    
    def k_fold_cv(self, X, y, k=5, alpha=0.92, num_iters=2000,
              shuffle=True, seed=None, normalise="zscore",
              plot=False, metric="RMSE"):
      rng = np.random.default_rng(seed)
      m = X.shape[0]
      idx = np.arange(m)
      if shuffle: rng.shuffle(idx)
      folds = np.array_split(idx, k)

      fold_metrics, fold_params = [], []

      for f in range(k):
          val_idx = folds[f]
          train_idx = np.hstack([folds[i] for i in range(k) if i != f])

          X_tr, y_tr = X[train_idx], y[train_idx]
          X_va, y_va = X[val_idx], y[val_idx]

          if normalise == "zscore":
              mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0); sigma[sigma == 0] = 1.0
              X_tr_n = (X_tr - mu) / sigma
              X_va_n = (X_va - mu) / sigma
          elif normalise == "mean":
              mu = X_tr.mean(axis=0); rng_ = (X_tr.max(axis=0) - X_tr.min(axis=0)); rng_[rng_ == 0] = 1.0
              X_tr_n = (X_tr - mu) / rng_
              X_va_n = (X_va - mu) / rng_
          else:
              X_tr_n, X_va_n = X_tr, X_va

          n = X.shape[1]
          w0 = np.zeros(n); b0 = 0.0
          w, b, _ = self.gradient_descent(
              X_tr_n, y_tr, w0, b0,
              cost_function=self.compute_cost,
              gradient_function=self.compute_gradient,
              alpha=alpha, num_iters=num_iters
          )

          y_hat = self.predict(X_va_n, w, b)
          mse, rmse, mae, r2 = self.evaluate_model(y_va, y_hat)
          fold_metrics.append({"fold": f+1, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})
          fold_params.append((w, b))

      keys = ["MSE","RMSE","MAE","R2"]
      means = {k_: float(np.mean([fm[k_] for fm in fold_metrics])) for k_ in keys}
      stds  = {k_: float(np.std( [fm[k_] for fm in fold_metrics], ddof=1)) for k_ in keys}

      result = {"per_fold": fold_metrics, "mean": means, "std": stds, "params": fold_params}

      if plot:
          scores = [fm[metric] for fm in fold_metrics]
          mean_score = np.mean(scores)

          # 1) Bar chart per fold + mean line
          fig_bar = plt.figure()
          plt.bar(range(1, len(scores)+1), scores)
          plt.axhline(mean_score, linestyle="--")
          plt.title(f"K-fold {metric}")
          plt.xlabel("Fold"); plt.ylabel(metric)

          # 2) Box plot of scores
          fig_box = plt.figure()
          plt.boxplot(scores, vert=True, labels=[metric])
          plt.title(f"{metric} distribution across folds")
          plt.ylabel(metric)

          result["figs"] = {"per_fold_bar": fig_bar, "boxplot": fig_box}

      return result
        
if __name__ == "__main__":
    # Example usage of the LinearUnivariateRegression class
    model = MultipleLinearRegression()
    
    # Load the data set
    characs_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_characs.csv'
    characs_file_handler = CSVHandler(characs_file_path)
    metrics_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_metrics.csv'
    metrics_file_handler = CSVHandler(metrics_file_path)
    dataset_tag_train = "newcastle_mech1750_lecture_level1_agregg_2025"
    # characs_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_characs.csv'
    # characs_file_handler = CSVHandler(characs_file_path)
    # metrics_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_metrics.csv'
    # metrics_file_handler = CSVHandler(metrics_file_path)
    # dataset_tag_train = "newcastle_engg2440_lecture_level2_agregg_2025"
    video_ids = characs_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file 

    i = 0; X_rows = []; y_train = []
    for video_id in video_ids:  # Loop through each video ID
        dataset_tag = characs_file_handler.get_cell_value_by_match("video_id", video_id, "dataset_tag"), # Get the dataset tag of the video from the video submission file
        if video_id and dataset_tag[0] == dataset_tag_train:  # Check if the video ID is not empty, and the dataset is targetted one
            duration_min = characs_file_handler.get_cell_value_by_match("video_id", video_id, "duration_min")
            speaking_words_count = characs_file_handler.get_cell_value_by_match("video_id", video_id, "speaking_words_count")
            avg_speaking_speed_wpm = characs_file_handler.get_cell_value_by_match("video_id", video_id, "avg_speaking_speed_wpm")
            scenes_count = characs_file_handler.get_cell_value_by_match("video_id", video_id, "scenes_count")
            avg_scene_change_per_min = characs_file_handler.get_cell_value_by_match("video_id", video_id, "avg_scene_change_per_min")
            average_percentage_viewed = metrics_file_handler.get_cell_value_by_match("video_id", video_id, "average_percentage_viewed")
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
    axes[1, 0].set_ylabel("Average Percentage Viewed (%)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
 
    ## Create and fit the regression model
    w_norm, b_norm, J_hist = model.gradient_descent(X = X_norm, y = y_train, w_in = np.zeros(n,), b_in = 0, 
                                                  cost_function = model.compute_cost, gradient_function = model.compute_gradient, 
                                                  alpha = 0.92, num_iters = 2000)
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

    cv = model.k_fold_cv(X_train, y_train, k=5, alpha=0.8, num_iters=2000, normalise="zscore")
    print("CV mean:", cv["mean"])
    print("CV std :", cv["std"])