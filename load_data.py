import numpy as np
import os

def load_data(data_path = 'data/'):
    data = []
    labels = []

    file_list = sorted(os.listdir(data_path))           # load files' names

    for i, file_name in enumerate(file_list):           # load data
        if file_name.endswith('.npy'):
            file_path = os.path.join(data_path, file_name)
            file_data = np.load(file_path)

            data.append(file_data)
            labels.append(np.full(file_data.shape[0], i))

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    # signal filter (The zero-phase digital filtering)
    # Zero-phase filtering is considered in this paper, but it is not set here.
    
    # Z-normalization
    mean = np.mean(data)
    std = np.std(data)
    data_std = (data - mean) / std
    data_std = data_std.reshape((data_std.shape[0], 1, data_std.shape[1]))

    return data_std, labels


# data, labels = load_data()
