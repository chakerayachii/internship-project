import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_geo = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion/"
f = files[0]


participant = f.split("-")[0].replace(data_path, "")  # Participant B
label = f.split("-")[1]  # Over head press
category = f.split("-")[2].rstrip("123")

df = pd.read_csv(f)
df["participant"] = participant
df["label"] = label
df["category"] = category

# for the geoscope data
f1 = files[3]
participant = f1.split("-")[0].replace(data_path, "")
label = f1.split("-")[1]
category = f1.split("-")[2].replace("_MetaWear_2019", "")
# ajouter les nouveau columns to th data set
df1 = pd.read_csv(f1)
df1["participant"] = participant


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion/"

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])


# acc_df containes 23578    , gyr_df containes 47218
# that's because the gyroscope frequenscy is highier than the accelerometer


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

# ----- epoch(ms)=> unix time since 1 jab 1970
acc_df.info()
# time is considered as an object so we need to transform it to date time
pd.to_datetime(df["epoch (ms)"], unit="ms")

df["time (01:00)"]

# after convertting the epochs (ms) we see of diiference of one day betweeen it and col times
# and that's because the difference between UTC time and CET winter time


acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
data_path = "../../data/raw/MetaMotion/"


def read_data_from_files(file):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
data_merged.head(30)
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
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
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
    "set": "last",
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

# split by days
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
days[8]
data_reasampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)
data_reasampled.info()
# set is defied as float

data_reasampled["set"] = data_reasampled["set"].astype("int")


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_reasampled.to_pickle("../../data/interim/01_data_processed.pkl")
