import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("D://study//Machine Learning//Projects//Fitness_Tracker//data//interim//data_processed.pkl")


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
catagory_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

fig , ax = plt.subplots()
catagory_df.groupby(["category"])["acc_y"].plot(figsize=(10, 4),
                                                title=f"Participant (A) y-axis acceleration during squats for heavy and medium weights")
plt.xlabel("Samples")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

fig , ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot(figsize=(10, 4),
                                                      title="Comparison between participants in bench exercise")
plt.xlabel("Samples")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
Label = "ohp"
Participant = "B"
acc_df = df.query(f"label == '{Label}'").query(f"participant == '{Participant}'").reset_index()

fig , ax = plt.subplots()
acc_df[["acc_x" , "acc_y" , "acc_z"]].plot(figsize=(10, 4),
                                           title="Accelerometer data for participant (B) in ohp exercise")
plt.xlabel("Samples")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
for label in df["label"].unique():
    for participant in df["participant"].unique():

        combined_df = (
        df.query(f"label == '{label}'")
        .query(f"participant == '{participant}'")
        .reset_index()
        )
        
        if (len(combined_df) > 0):
        
            fig , ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

            combined_df[["acc_x" , "acc_y" , "acc_z"]].plot(ax=ax[0],
            title=f"Accelerometer data for participant ({participant}) in {label} exercise")

            combined_df[["gyr_x" , "gyr_y" , "gyr_z"]].plot(ax=ax[1],
            title=f"Gyroscope data for participant ({participant}) in {label} exercise")
            
            ax[0].set_xlabel("Samples")
            ax[0].set_ylabel("Acceleration (m/s²)")
            ax[1].set_xlabel("Samples")
            ax[1].set_ylabel("Acceleration (m/s²)")
            ax[0].legend()
            ax[0].grid(True)
            ax[1].legend()
            ax[1].grid(True)
              
            plt.savefig(f"D://study//Machine Learning//Projects//Fitness_Tracker//reports//figures//{label.title()}_({participant}).png")

            plt.show()

