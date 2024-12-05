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
from sklearn.metrics import classification_report

def load_dataset(dataset_path, sub_path):
 

    # c the full path
    dir_path = os.path.join(dataset_path, sub_path)
    print(f"Accessing: {dir_path}")  # Debugging

    # Check if the directory exists
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Initialize data arrays
    x = np.array([]).reshape(0, 60, 23)
    y = np.array([]).reshape(0, 1)

    # Load files
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        print(f"Processing file: {file_path}")  # Debugging
        df = pd.read_csv(file_path).to_numpy()
        x = np.append(x, df.reshape(1, 60, 23), axis=0)
        label = 1 if sub_path == "malware" else 0
        y = np.append(y, [[label]], axis=0)

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
    x_safe, y_safe = load_dataset(path, "safe")
    x_mal, y_mal = load_dataset(path, "malware")
    x = np.concatenate((data_normalization(x_safe, False), data_normalization(x_mal, False)), axis=0)
    y = np.concatenate((y_safe, y_mal), axis=0)
    x, y = shuffle(x, y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio[1], random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ratio[2]/(1. - ratio[1]), random_state=9)
    return X_train, X_test, X_val, y_train, y_test, y_val


# def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
#     res = pd.DataFrame(data=np.zeros((1, 4), dtype=float), index=[0], # np.float to float
#                        columns=['precision', 'accuracy', 'recall', 'duration'])
#     res['precision'] = precision_score(y_true, y_pred, average='macro')
#     res['accuracy'] = accuracy_score(y_true, y_pred)

#     if not y_true_val is None:
#         res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

#     res['recall'] = recall_score(y_true, y_pred, average='macro')
#     res['duration'] = duration
#     return res

from sklearn.metrics import precision_score, accuracy_score, recall_score
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    # Initialize the results DataFrame
    res = pd.DataFrame(data=np.zeros((1, 5), dtype=float), 
                       columns=['precision', 'accuracy', 'recall', 'duration', 'accuracy_val'])

    # Compute metrics
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration

    # Validation accuracy (if available)
    if y_true_val is not None and y_pred_val is not None:
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

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

"""
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
    """
    #def plot(history, model):
def plot(history, model):
    # Extract data from history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    # Plot training and validation loss
    plt.plot(epochs, val_loss, label='Val Loss', color='blue')
    plt.plot(epochs, train_loss, label='Train Loss', color='green')
    
    # Plot training and validation accuracy
    plt.plot(epochs, val_acc, label='Val Acc', color='orange')
    plt.plot(epochs, train_acc, label='Train Acc', color='red')
    
    # Plot training and validation loss with markers and line styles
    # plt.plot(epochs, val_loss, label='Val Loss', color='blue', linestyle='-', marker='o')  # Solid line with circles
    # plt.plot(epochs, train_loss, label='Train Loss', color='green', linestyle='--', marker='x')  # Dashed line with x
    # plt.plot(epochs, val_acc, label='Val Acc', color='orange', linestyle='-.', marker='s')  # Dash-dot line with squares
    # plt.plot(epochs, train_acc, label='Train Acc', color='red', linestyle=':', marker='d')  # Dotted line with diamonds
    
    # Add labels, legend, and title
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.ylim(0, 1)  # Adjust the y-axis limits if necessary
    plt.show()

    model.summary()