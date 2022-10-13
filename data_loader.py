# encoding=utf-8
import numpy as np
from torch.utils.data import Dataset, DataLoader


# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int32) - 1
    YY = np.eye(6)[data]
    return YY


def format_data_subject(datafile):
    data = np.loadtxt(datafile, dtype=np.int32)-1
    SS = np.eye(30)[data]
    return SS


# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data(data_folder):
    import os
    if os.path.isfile(data_folder + 'data_har.npz') == True:
        data = np.load(data_folder + 'data_har.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
    else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        str_folder = data_folder + 'UCI HAR Dataset/'
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_train_sub = str_folder + 'train/subject_train.txt'

        X_train = format_data_x(str_train_files)
        S_train = format_data_subject(str_train_sub)
        Y_train = format_data_y(str_train_y)

    return X_train, onehot_to_label(Y_train), onehot_to_label(S_train)


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


def normalize(x):
    x_min = x.min(axis=(0, 2, 3), keepdims=True, type=float)
    x_max = x.max(axis=(0, 2, 3), keepdims=True, type=float)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def load_training(data_folder, batch_size, subjects, kwargs):
    x_train, y_train, s_train = load_data(data_folder)
    x_train = x_train.reshape((-1, 9, 1, 128))
    transform = None
    train_loaders = []
    for i in subjects:
        train_set = data_loader(
            x_train[np.where(s_train == i)], y_train[np.where(s_train == i)], transform)
        train_loaders.append(DataLoader(train_set, batch_size=batch_size,
                             shuffle=True, drop_last=True, **kwargs))
    return train_loaders
