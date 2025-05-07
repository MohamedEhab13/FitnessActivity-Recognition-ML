import os
import pandas as pd
from glob import glob

## Load data files 
files = glob("D://study//Machine Learning//Projects//Fitness_Tracker//data//raw//MetaMotion//*.csv")

# --------------------------------------------------------------
# Define read data function 
#---------------------------------------------------------------
def read_data_from_files(files) :

    ## Empty data frames
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    ## Counters for data groups
    acc_set = 1
    gyr_set = 1

    for f in files : 

        filename = os.path.basename(f)
        
        ## Extract features 
        participant = filename.split("-")[0]
        label = filename.split("-")[1]
        category = filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
       
        ## Read excel sheet
        df = pd.read_csv(f)
        
        ## Add extracted features 
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        ## Append data sets in Accelerometer data frame
        if "Accelerometer" in f :
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        ## Append data sets in Gyroscope data frame
        if "Gyroscope" in f :
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    ## Convert index into datetime index to facilitate resampling 
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
    
    ## Delete unneeded timestamp columns 
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df , gyr_df

## Call the read data function
acc_df , gyr_df = read_data_from_files(files)

    
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set"
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

# Split by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] = data_resampled["set"].astype("int")

print(data_resampled)

# Export dataset
data_resampled.to_pickle("D://study//Machine Learning//Projects//Fitness_Tracker//data//interim//data_processed.pkl")



