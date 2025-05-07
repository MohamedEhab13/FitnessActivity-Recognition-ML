import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Read DataFrame
df = pd.read_pickle("D://study//Machine Learning//Projects//Fitness_Tracker//data//interim//added_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
train_df = df.drop(["participant","category","set","set_duration"], axis=1)

X = train_df.drop("label", axis=1)
Y = train_df["label"]

x_train , x_test , y_train , y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
squared_features = ['acc_r', 'gyr_r']
pca_features = ['pca_1', 'pca_2', 'pca_3']
mean_std_features = [col for col in train_df if "_temp_" in col]
frequency_features = [col for col in train_df if (("_freq" in col) or ("_pse" in col))]
cluster_features = ["cluster"]

print(f"basic features : {len(basic_features)}")
print(f"squared features : {len(squared_features)}")
print(f"pca features : {len(pca_features)}")
print(f"mean_std features : {len(mean_std_features)}")
print(f"frequency features : {len(frequency_features)}")
print(f"cluster_features : {len(cluster_features)}")

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(feature_set_1 + squared_features + pca_features))
feature_set_3 = list(set(feature_set_2 + mean_std_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))



# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
classifier = ClassificationAlgorithms()
max_features = 10 

selected_features, ordered_features, ordered_scores = classifier.forward_selection(max_features, x_train, y_train)

print(selected_features)
print(ordered_features)
print(ordered_scores)

plt.figure(figsize=(10,10))
plt.plot(range(1,max_features), ordered_scores)
plt.xlabel("Features number")
plt.ylabel("Accuracy")
plt.title("Model Accuracy at different number of optimal features")
plt.show()
# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------