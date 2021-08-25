import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_dataset(dataset_path):
    path = dataset_path
    sub_paths = ["\safe", "\malware"]
    x = np.array([]).reshape(0, 60, 23)
    y = np.array([]).reshape(0, 1)
    for sub_path in sub_paths:
        files = os.listdir(path + sub_path)
        for file in files:
            df = pd.read_csv(path+sub_path + "/" +file).to_numpy()
            x = np.append(x, df.reshape(1, 60, 23), axis=0)
            if sub_path == "\malware":
                y = np.append(y, [[1]], axis=0)
            else:
                y = np.append(y, [[0]], axis=0)
    x, y = shuffle(x, y)
    return x, y


def data_normalization(data, standard=True):
    if standard:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    original_shape = data.shape
    data = scaler.fit_transform(data.reshape(original_shape[0] * 60, 23)).reshape(original_shape)
    return data


def data_split(X, y, ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio[1], random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ratio[2]/(1. - ratio[1]), random_state=9)
    return X_train, X_test, X_val, y_train, y_test, y_val
    
