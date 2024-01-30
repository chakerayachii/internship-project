import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

df["set"]
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds  # 20 seconfd

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]
    duration = end - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds


duration_df = df.groupby(["category"])["duration"].mean()
# the duration of the sets is not the same for each category
# heavy       14.743501
# medium      24.942529
# sitting     33.000000
# standing    39.000000

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# the duration for a single rep is 2.948700 seconds for heavy and 2.494253 seconds for medium

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()
fs = 1000 / 200
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)
subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="row data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="buttterworth lowpass")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)


for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("pricipale component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 13]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
""" 
        To further exploit the data, the scalar magnitudes r of the accelerometer and gyroscope were calculated. 
        r is the scalar magnitude of the three combined data points: x, y, and z. The advantage of using r versus
        any particular data direction is that it is impartial to device orientation and can handle dynamic re-orientations.
        r is calculated by:
"""

# r_{magnitude} = \sqrt{x^2 + y^2 + z^2}

df_squared = df_pca.copy()
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


df_temporal = df_squared.copy()
df_temporal.isna().sum()
numAbs = NumericalAbstraction()
new_variables = ["acc_r", "gyr_r"]
predictor_columns += new_variables

ws = int(1000 / 200)  # 1000/200
# windows size of  1 second

for col in predictor_columns:
    df_temporal = numAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = numAbs.abstract_numerical(df_temporal, [col], ws, "std")


df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = numAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = numAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)


df_temporal = pd.concat(df_temporal_list)


subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()
fs = int(1000 / 200)
ws = int(2800 / 200)  # average length of the repition 2800 ms

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
df_freq.columns


# Visualisation
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_0.0_Hz_ws_14",
        "acc_y_freq_0.357_Hz_ws_14",
    ]
].plot()


df_freq_list = []
for s in df_freq["set"].unique():
    print(f"apllying fourier transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

df_freq[["duration"]]

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]
K_values = range(2, 10)
intertias = []
for k in K_values:
    subset = df_cluster[cluster_columns]
    Kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_laels = Kmeans.fit_predict(subset)
    intertias.append(Kmeans.inertia_)

# plot clusters
fig = plt.figure(figsize=(20, 20))
plt.plot(K_values, intertias, "bx-")
plt.xlabel("k")
plt.ylabel("intertia || sum of squard distance")
plt.show()

# best k value is 5
Kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = Kmeans.fit_predict(subset)  # add cluster labels to df


# plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"cluster {c}")
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
plt.legend()
plt.show()


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f" {c}")
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_data_fetures.pkl")
