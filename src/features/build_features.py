import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("D://study//Machine Learning//Projects//Fitness_Tracker//data//interim//outliers_removed_chauvenet.pkl")

columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in columns :
    df[col] = df[col].interpolate()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
for set_n in df["set"].unique() :
    start = df[df["set"] == set_n].index[0]
    end = df[df["set"] == set_n].index[-1]   
    set_duration = end - start 

    df.loc[(df["set"] == set_n), "set_duration"] = set_duration.seconds

#print(df)
duration_df = df.groupby(["category"])["set_duration"].mean()

heavy_reps_duration = duration_df.iloc[0] /5
medium_reps_duration = duration_df.iloc[1] /10

#print(heavy_reps_duration)


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
lowpass_df = df.copy()
lowpass = LowPassFilter()

fs = 1000 / 200 # sampling frequency
fc = 1.3 # cutoff frequency

for col in columns:
    lowpass_df = lowpass.low_pass_filter(lowpass_df , col , fs , fc , 5)
    lowpass_df[col] = lowpass_df[col + "_lowpass"]
    del lowpass_df[col + "_lowpass"]

#print(lowpass_df)

# fig , ax = plt.subplots(nrows=2 , sharex=True , figsize=(20,10))
# ax[0].plot(df[df["set"] == 4]["acc_y"].reset_index(drop=True), label="Raw data")
# ax[1].plot(lowpass_df[lowpass_df["set"] == 4]["acc_y"].reset_index(drop=True), label="Filtered data")
# ax[0].legend()
# ax[1].legend()

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
pca_df = lowpass_df.copy()

pca = PrincipalComponentAnalysis()

pca_variance = pca.determine_pc_explained_variance(pca_df , columns)

# Determine optimal number of principal components 
# pca component number at which variance is getting 
# almost smooth is the optimal pca number 
# plt.plot(range(1, len(columns) + 1) , pca_variance)
# plt.xlabel("Principal Components Number")
# plt.ylabel("Explained Variance")


# Apply PCA to the data frame
pca_df = pca.apply_pca(pca_df , columns , 3)

subset = pca_df[pca_df["set"] == 35]
subset[["pca_1","pca_2","pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
squared_df = pca_df.copy()

acc_r = squared_df["acc_x"] ** 2 + squared_df["acc_y"] ** 2 + squared_df["acc_z"] ** 2 
gyr_r = squared_df["gyr_x"] ** 2 + squared_df["gyr_y"] ** 2 + squared_df["gyr_z"] ** 2

squared_df["acc_r"] = np.sqrt(acc_r)
squared_df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
temp_df = squared_df.copy()

num_abs = NumericalAbstraction()
ws = int(1000/200) # window size in 1 second 

columns = columns + ['acc_r', 'gyr_r']
temp_list = []

for set_n in temp_df["set"].unique() :
    subset = temp_df[temp_df["set"] == set_n].copy()
    for col in columns :
        subset = num_abs.abstract_numerical(subset, [col], ws, "mean")
        subset = num_abs.abstract_numerical(subset, [col], ws, "std")
    temp_list.append(subset)

temp_df = pd.concat(temp_list)

#print(temp_df)

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
freq_df = temp_df.copy()
fourier_tr = FourierTransformation()

sampling_rate = int(1000/200)
ws = int(2800/200)

freq_list = []
for set_n in freq_df["set"].unique() :
    subset = freq_df[freq_df["set"] == set_n].reset_index().copy()
    subset = fourier_tr.abstract_frequency(subset, columns, ws, sampling_rate)
    freq_list.append(subset)

freq_df = pd.concat(freq_list).set_index("epoch (ms)", drop=True)



# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
freq_df = freq_df.dropna()
freq_df = freq_df[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
cluster_df = freq_df.copy()
cluster_columns = ["acc_x" , "acc_y" , "acc_z"]
k_values = range(2, 10)
inertias = [] 

# Determine optimal k by plotting againest variance
for k in k_values :
    subset = cluster_df[cluster_columns]
    kmeans = KMeans(n_clusters=k , n_init=20 , random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,10))
plt.plot(k_values,inertias)
plt.xlabel("K Values")
plt.ylabel("interia")
plt.show()

subset = cluster_df[cluster_columns]
kmeans = KMeans(n_clusters=5 , n_init=20 , random_state=0)
cluster_df["cluster"] = kmeans.fit_predict(subset)



# Plot clustered data (KMeans output)
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

for c in cluster_df['cluster'].unique():
    subset = cluster_df[cluster_df['cluster'] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.show()


# Plot true activity labels (e.g., squat, deadlift, etc.)
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

for l in cluster_df["label"].unique():
    subset = cluster_df[cluster_df["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.show()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
print(cluster_df)
cluster_df.to_pickle("D://study//Machine Learning//Projects//Fitness_Tracker//data//interim//added_features.pkl")