import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_dataset(dataset_path, sub_path):
    path = dataset_path
    x = np.array([]).reshape(0, 60, 23)
    y = np.array([]).reshape(0, 1)
    files = os.listdir(path + sub_path)
    for file in files:
        df = pd.read_csv(path+sub_path + "/" +file).to_numpy()
        x = np.append(x, df.reshape(1, 60, 23), axis=0)
        if sub_path == "\malware":
            y = np.append(y, [[1]], axis=0)
        else:
            y = np.append(y, [[0]], axis=0)
    return x, y


def data_normalization(data, standard=True):
    if standard:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    original_shape = data.shape
    data = scaler.fit_transform(data.reshape(original_shape[0] * 60, 23)).reshape(original_shape)
    return data


def load_data(path, ratio, norm=False):
    x_safe, y_safe = load_dataset(path, "\safe")
    x_mal, y_mal = load_dataset(path, "\malware")
    x_safe_train, x_safe_test, y_safe_train, y_safe_test = train_test_split(x_safe, y_safe, test_size=1.-ratio[0], random_state=8)
    x_mal_train, x_mal_test, y_mal_train, y_mal_test = train_test_split(x_mal, y_mal, test_size=1.-ratio[0], random_state=8)
    x_train = data_normalization(np.concatenate((x_safe_train, x_mal_train), axis=0), norm)
    y_train = np.concatenate((y_safe_train, y_mal_train), axis=0)
    x_test = np.concatenate((x_safe_test, x_mal_test), axis=0)
    y_test = np.concatenate((y_safe_test, y_mal_test), axis=0)
    x_test, y_test = shuffle(x_test, y_test)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=ratio[2]/(1.-ratio[0]), random_state=9)
    x_test = data_normalization(x_test, norm)
    x_val = data_normalization(x_val, norm)
    x_train, y_train = shuffle(x_train, y_train)
    return x_train, y_train, x_test, y_test, x_val, y_val
