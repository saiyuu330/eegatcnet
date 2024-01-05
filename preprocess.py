import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

def load_BCI2a_data(data_path, subject, training, all_trials = True):
    n_channels = 22
    n_tests = 6*48 # 288
    window_Length = 7*250 # 1750

    fs = 250
    t1 = int(1.5 * fs) # 375
    t2 = int(6 * fs)   # 1500

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))
    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0,0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if(a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[NO_valid_trial, : , :] = np.transpose( a_X[int(a_trial[trial]) : (int(a_trial[trial])+window_Length), :22])
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1
    
    data_return = data_return[0 : NO_valid_trial, : , t1 : t2]
    class_return = class_return[0 : NO_valid_trial]
    class_return = (class_return - 1).astype(int)

    return data_return, class_return

def standardize_data(X_train, X_test, channels):
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
    
    return X_train, X_test

def get_data(path, subject, dataset = 'BCI2a', classes_labels = 'all', LOSO = False, isStandard = True, isShuffle = True):
    X_train, y_train = load_BCI2a_data(path, subject+1, True)
    X_test, y_test = load_BCI2a_data(path, subject+1, False)

    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)
    
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = F.one_hot(torch.tensor(y_train, dtype=torch.long))
    N_tr, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = F.one_hot(torch.tensor(y_test, dtype=torch.long))

    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)
    
    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot