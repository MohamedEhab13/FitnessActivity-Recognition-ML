import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("D://study//Machine Learning//Projects//Fitness_Tracker//data//interim//data_processed.pkl")


# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
sensors_data = list(df.columns[:6])

# Visualize outliers using boxplot 
df.boxplot(column=sensors_data[:3], by="label", figsize=(20,10), layout=(1,3))
plt.suptitle("Accelerometer Outliers")  

df.boxplot(column=sensors_data[3:], by="label", figsize=(20,10), layout=(1,3))
plt.suptitle("Gyroscope Outliers") 

#plt.show();


# Define plot binary outliers functions 
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )


    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()



# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

# Loop over all columns
# for col in sensors_data :
#     data_set = mark_outliers_iqr(df, col)
#     plot_binary_outliers(data_set, col, col + "_outlier", True)

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    
    # Create a list to store probability and outlier mask with correct length
    prob = np.zeros(len(dataset))
    mask = np.zeros(len(dataset), dtype=bool)

    # Pass all rows in the dataset.
    for i in range(len(dataset)):
        # Determine the probability of observing the point
        # Use .iloc to access elements by position instead of label
        prob[i] = 1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
        # And mark as an outlier when the probability is below our criterion.
        mask[i] = prob[i] < criterion
    
    # Assign the boolean mask to the new column
    dataset[col + "_outlier"] = mask
    
    return dataset


# Loop over all columns
# for col in sensors_data :
#     data_set = mark_outliers_chauvenet(df, col, 2)
#     plot_binary_outliers(data_set, col, col + "_outlier", True)



# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


# Loop over all columns
# data_set, outliers, x_scores = mark_outliers_lof(df, sensors_data, 20)
# for col in sensors_data :
#     plot_binary_outliers(data_set, col, "outlier_lof" , True)


# --------------------------------------------------------------
# Pick methode and deal with outliers 
# --------------------------------------------------------------
removed_outliers_df = df.copy()
for col in sensors_data:
    for label in df["label"].unique(): 
        data_set = mark_outliers_chauvenet(df[df["label"] == label], col, 2)
        data_set.loc[data_set[col + "_outlier"], col] = np.nan
        removed_outliers_df.loc[(removed_outliers_df["label"] == label) , col] = data_set[col]
        n_outliers = len(data_set) - len(data_set[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label} exercise")


# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
removed_outliers_df.to_pickle("D://study//Machine Learning//Projects//Fitness_Tracker//data//interim//outliers_removed_chauvenet.pkl")