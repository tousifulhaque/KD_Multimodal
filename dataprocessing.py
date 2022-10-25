import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("KU-HAR_time_domain_subsamples_20750x300.csv")
print(df.shape)

f = open("data_aug_KU_HAR.txt", "r")
all_lines = f.readlines()

pairs = []

for line in all_lines:
    line = line.rstrip().split(" ")

    # store pairs
    pairs.append([line[0], line[-1]])

CLASS_LABELS = np.array(
    [
        "Stand",
        "Sit",
        "Talk-sit",
        "Talk-stand",
        "Stand-sit",
        "Lay",
        "Lay-stand",
        "Pick",
        "Jump",
        "Push-up",
        "Sit-up",
        "Walk",
        "Walk-backward",
        "Walk-circle",
        "Run",
        "Stair-up",
        "Stair-down",
        "Table-tennis",
    ]
)

df = pd.read_csv("KU-HAR_time_domain_subsamples_20750x300.csv", header=None)

signals = df.values[:, 0:1800]
signals = np.array(signals, dtype=np.float32)
labels = df.values[:, 1800]
labels = np.array(labels, dtype=np.int64)

print(signals.shape, labels.shape, "\n")


# broken samples in original dataset
indexes = [
    6587,
    6588,
    6589,
    6590,
    6591,
    6592,
    6593,
    6594,
    6595,
    6596,
    6597,
    6598,
    6599,
    6600,
    6601,
    6602,
    6603,
    6604,
    6605,
    6606,
    6607,
    6660,
    6661,
    6662,
    6663,
    6664,
    6665,
    6666,
    6667,
    6668,
    6669,
    6670,
    6671,
    6672,
    6673,
    6674,
    6675,
    6676,
    6677,
    6678,
    6679,
    6680,
    6681,
    6682,
    6683,
    6684,
    6685,
    6686,
    6687,
    6716,
    6717,
    6718,
    6719,
    6720,
    6721,
    6722,
    6723,
    6724,
    6725,
    6726,
    6727,
    6728,
    6729,
    6730,
    6731,
    6732,
    6733,
    6734,
    6735,
    6736,
    6737,
    6738,
    6739,
    6740,
    6741,
    6742,
    6743,
    6750,
    6751,
    6752,
    6753,
    6754,
    6755,
    6756,
    6757,
    6758,
    6759,
    6760,
    6761,
    6762,
    6763,
    6764,
    6765,
    6766,
    6767,
]

# delete the bad samples
signals = np.delete(signals, indexes, 0)
labels = np.delete(labels, indexes, 0)

signals = np.stack(
    [
        signals[:, 0:300],  # ACC X
        signals[:, 300:600],  # ACC Y
        signals[:, 600:900],  # ACC Z
        signals[:, 900:1200],  # GYRO X
        signals[:, 1200:1500],  # GYRO Y
        signals[:, 1500:1800],  # GYRO Z
    ],
    axis=-1,
)
labels = np.repeat(labels.reshape(labels.shape[0], 1), signals.shape[1], axis=1)

new_signals = []
new_labels = []

for i in range(len(pairs)):


    first = np.where(CLASS_LABELS == pairs[i][0])[0]
    second = np.where(CLASS_LABELS == pairs[i][1])[0]


    first_indexes = np.unique(np.where(labels == first)[0])
    second_indexes = np.unique(np.where(labels == second)[0])


    # minimum pre vytvorenie absolutne neduplicitnych prikladov - zabranenie overfit
    count = min(first_indexes.shape[0], second_indexes.shape[0])

    merged_signals = np.concatenate(
        (signals[first_indexes[:count]], signals[second_indexes[:count]]), axis=1
    )


    merged_labels = np.concatenate(
        (labels[first_indexes[:count]], labels[second_indexes[:count]]), axis=1
    )

    downsample_signals = merged_signals[:, ::2, :]
    new_signals.append(downsample_signals)

    downsample_labels = merged_labels[:, ::2]

    new_labels.append(downsample_labels)

# merge all pairs into batch axis
new_signals = np.concatenate(new_signals, axis=0)
new_labels = np.concatenate(new_labels, axis=0)


final_signals = np.concatenate([signals, new_signals], axis=0)
final_labels = np.concatenate([labels, new_labels], axis=0)

np.savez_compressed("new_dataset", signals=final_signals, labels=final_labels)

