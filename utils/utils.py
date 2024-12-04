import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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
    x = np.concatenate((data_normalization(x_safe, False), data_normalization(x_mal, False)), axis=0)
    y = np.concatenate((y_safe, y_mal), axis=0)
    x, y = shuffle(x, y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio[1], random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ratio[2]/(1. - ratio[1]), random_state=9)
    return X_train, X_test, X_val, y_train, y_test, y_val


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            return None
        return directory_path

    
def plot(history):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

