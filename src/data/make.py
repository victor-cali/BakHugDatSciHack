import os
import cv2
import zipfile
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def download():
    os.system('cmd /c "kaggle competitions download -c f7853596-822e-451e-b220-a94c809cddf3 -p ../data/external')
    with zipfile.ZipFile('../data/external/f7853596-822e-451e-b220-a94c809cddf3.zip', 'r') as zip_ref:
        zip_ref.extractall('../data/raw/')

def process():
    train_path = '../data/raw/Imagesv2/Train' 
    train_samples = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path,f))]
    train_samples.sort()

    to_predict_path = '../data/raw/Imagesv2/Test' 
    to_predict_samples = [f for f in os.listdir(to_predict_path) if os.path.isfile(os.path.join(to_predict_path,f))]
    to_predict_samples.sort()

    labels = pd.read_csv('../data/raw/Imagesv2/train_labels.csv')
    y = labels.loc[:,'Category'].to_numpy()
    y = to_categorical(y)

    X = np.empty((len(train_samples), 28, 28), dtype = np.float32)
    X_to_predict = np.empty((len(to_predict_samples), 28, 28), dtype = np.float32)

    for i in range(len(train_samples)):
        img = cv2. imread(os.path.join(train_path,train_samples[i]))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        X[i, :, :] = img[:,:,-1]
    X = X.reshape(X.shape[0], 28, 28, 1).astype('float32') / 255

    for i in range(len(to_predict_samples)):
        img = cv2. imread(os.path.join(to_predict_path,to_predict_samples[i]))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        X_to_predict[i, :, :] = img[:,:,-1]
    X_to_predict = X_to_predict.reshape(X_to_predict.shape[0], 28, 28, 1).astype('float32') / 255

    np.save('../data/processed/X.npy', X)
    np.save('../data/processed/y.npy', y)
    np.save('../data/processed/X_to_predict.npy', X_to_predict)

    return X, y, X_to_predict

def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify = y)
    return X_train, X_test, y_train, y_test