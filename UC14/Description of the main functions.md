1. **data_loading_preprocessing()** function:
   - This function takes the path to a folder containing the data as input and performs several preprocessing steps on the data.
   - It reads the data from the specified path and stores it in a Pandas DataFrame.
   - The function assumes that the data has a column named 'time' that represents the timestamp of each data point.
   - It converts the 'time' column to a datetime format and sets it as the index of the DataFrame.
   - The function also calls another function named 'add_calendar_components()' to add additional calendar-related columns to the DataFrame.
   - Next, the function creates a dictionary of DataFrames, where each DataFrame corresponds to a specific house (identified by 'dataid').
   - If the parameter 'plot_consumption_per_house' is set to True, the function also generates a plot showing the consumption per house.
   - If the parameter 'to_csv_name' is provided, it saves the DataFrame to a CSV file with the specified name.
   - The function returns a tuple of two DataFrames: the original DataFrame and the dictionary of house-specific DataFrames.

2. **consumer_profile_features()** function:
This function takes a DataFrame with metadata as input and performs feature extraction related to consumer profiles.
   - The input DataFrame should contain the following columns: 'dataid', 'value', 'hour', 'weekday', 'month', 'dayofyear', and 'year'.
   - The function calculates various statistics and features based on the energy consumption data.
   - It first calculates summary statistics for each house, including the minimum, maximum, mean, variance, and standard deviation of the consumption.
   - Then, it calculates the mean consumption values separately for weekdays and weekends.
   - Next, it calculates the normalized load profiles for each house by dividing the consumption values by the mean consumption for that 'dataid'.
   - The function also calculates the total daily energy mean for each house.
   - It divides the daily energy mean into two separate values for weekdays and weekends.
   - The function further calculates maximum and minimum consumption values for each day, along with the mean consumption for each day.
   - It then calculates additional features related to different time periods (overnight, breakfast, daytime, evening) by grouping the consumption data accordingly.
   - Finally, the function returns three DataFrames: 'clustering_profiles', 'load_profiles', and 'clustering_profiles_weekend_weekday_difference_score'.
   - If the parameter 'to_csv_name' is provided, it saves the output to a CSV file with the specified name.

3. **time_series_clustering()** function: 
This function performs time series clustering on load profiles. It takes a dataframe of load profiles as input and returns a dataframe with an additional column indicating the cluster to which each profile belongs. The function accepts the following parameters:
   - `load_profiles`: The dataframe of load profiles.
   - `path_pickle_load`: The path to a pickle file where a pre-trained model will be loaded from (optional).
   - `path_pickle`: The path to the pickle file where the model will be saved (optional, default: "IC_timeseries_kmeans.pkl").
   - `visualise_KMeans_clustering`: If set to `True`, it plots the elbow curve to help choose the number of clusters (optional, default: `False`).
   - `to_csv_name`: The name of the CSV file to save the results (optional).

4. *non_time_series_clustering()** function:
 This function performs clustering on non-time series features of load profiles. It takes a dataframe of profiles as input and returns a dataframe with an additional column indicating the cluster to which each profile belongs. The function accepts the following parameters:
   - `clustering_profiles_weekend_weekday_difference_score`: The dataframe containing the data to be clustered.
   - `path_pickle_load`: The path to a pickle file where a pre-trained model will be loaded from (optional).
   - `path_pickle`: The path to the pickle file where the model will be saved (optional, default: "IC_non_timeseries_kmeans.pkl").
   - `visualise_KMeans_clustering`: If set to `True`, it plots the elbow curve to help choose the number of clusters (optional, default: `False`).
   - `to_csv_name`: The name of the CSV file to save the results (optional).

5. **clustering_profiles_and_metadata()** function:
 This function combines the clustering profiles with metadata from the dataset. It takes the dataset with metadata, clustering profiles for timeseries and non-timeseries, and returns a dataframe with the clustering profiles and metadata. The function accepts the following parameters:
   - `dataset_with_metadata`: The dataset with metadata.
   - `clustering_profiles_metadata`: A dataframe with the cluster assignments for each dataid.
   - `clustering_profiles_timeseries`: The clustering profiles for the timeseries data (optional).
   - `clustering_profiles_non_timeseries`: The non-timeseries clustering profiles (optional).
   - `calculate_centroids`: If set to `True`, it calculates centroid distances (optional, default: `True`).
   - `to_csv_name`: The name of the CSV file to save the dataframe (optional).


The provided code consists of two functions: `process_data` and `RNN_MLP_Model`, as well as a script that calls the `process_data` function and assigns its output to several variables.

6. **process_data()** function:
   - Parameters:
     - `dataset_with_clustering_profiles_and_metadata`: A dataset containing clustering profiles and metadata.
     - `inference` (optional): A boolean indicating whether the function is called during inference (default is `False`).
     - `shuffle` (optional): A boolean indicating whether to shuffle the data (default is `True`).
   - Description:
     - This function processes the input dataset by performing various data transformations and returns the processed data for training an RNN-MLP model.
   - Steps:
     1. Define a list of column names.
     2. Apply skewness transformation and Box-Cox transformation to the dataset, if `inference` is `False`. The function `skewness_and_box_cox` is called to perform these transformations.
     3. Normalize the input features (`X`) and the output target (`y`) using scaler_normalize function.
     4. Prepare the input data (`X_lstm`) and output data (`y_lstm`) for the RNN model by splitting the dataset into sequences of length `n_steps`.
     5. Shift the MLP input data (`normalizedX`) by `n_steps-1` hours so that it aligns with the RNN data.
     6. Concatenate the RNN data (`X_lstm`) and MLP data (`normalizedX`) to create `X_concatenated`.
     7. Split the concatenated data into training and testing sets using `train_test_split`.
     8. Split the input feature vector into separate inputs for the RNN (`X_train_lstm`) and MLP (`X_train_mlp`).
      9. Return the processed data and relevant information.

7. **RNN_MLP_Model()** function:
   - Parameters:
     - `features`: The number of features in the input data.
     - `X_train_concatenated`, `X_test_concatenated`, `y_train_concatenated`, `y_test_concatenated`: Training and testing data for the model.
     - `X_lstm`, `y_lstm`: Data specifically for the RNN part of the model.
     - `X_concatenated`: Concatenated data for both RNN and MLP parts of the model.
     - `scalerY`: Scaler object used for normalizing the output data.
     - `n_steps`: The number of time steps in the RNN.
     - `start_training` (optional): A boolean indicating whether to start training the model (default is `False`).
     - `shuffle` (optional): A boolean indicating whether to shuffle the data during training (default is `False`).
     - `model_grid_search` (optional): A boolean indicating whether to perform grid search for model hyperparameter tuning (default is `False`).
     - `train_grid_search` (optional): A boolean indicating whether to train the grid search models (default is `False`).
     - `param_grid` (optional): A dictionary containing the hyperparameter grid for grid search (default is an empty dictionary).
     - `plot_learning_curve` (optional): A boolean indicating whether to plot the learning curve (default is `False`).
   - Description:
     - This function defines an RNN-MLP model architecture and performs training and evaluation.
   - Steps:
     1. Define the architecture of the model using the Keras functional API.
     2. If `model_grid_search` is `True` and `param_grid` is provided, define a function
