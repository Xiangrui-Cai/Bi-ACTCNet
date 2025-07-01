from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal
import numpy as np
import os
from scipy.signal import butter, filtfilt
from scipy.io import loadmat

ROOT = os.getcwd()

def butter_notchstop(notch,Q,fs):
    b, a = signal.iirnotch(notch, Q,fs)
    return b, a

def preprocess_norm(eeg_data):
    scale_mean = np.mean(eeg_data, axis=-1, keepdims=True)
    scale_std = np.std(eeg_data, axis=-1, keepdims=True)
    eeg_data = (eeg_data - scale_mean) / (scale_std + 1e-8)

    return eeg_data

# 带通
def preprocess_filt(data, low_cut=0.1, high_cut=40, fs=500, order=4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    padlen = len(data) // 3
    proced = signal.filtfilt(b, a, data,padlen=padlen)
    return proced

def preprocess_bsfilt(data, low_cut=49, high_cut=51, fs=500):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(4, [low, high], btype='bandstop')
    padlen = len(data) // 3
    proced = signal.filtfilt(b, a, data,padlen=padlen)
    return proced

def preprocess_notch(data, notch=50, Q=35, fs=500):
    # notch, Q, fs = 50, 35, 200
    b, a = butter_notchstop(notch, Q, fs)
    filted_eeg_rawdata = filtfilt(b, a, data)
    return filted_eeg_rawdata

def preprocess_hpfilt(data, low_cut=0.1, fs=500):
    b, a = butter(11, low_cut, btype='hp', fs=fs)
    filted_data = filtfilt(b, a, data)
    return filted_data

def preprocess_resample(data: np.ndarray, fs: int = 500, refs: int = 250):
    up_factor = refs
    down_factor = fs
    proced = signal.resample_poly(data, up=up_factor, down=down_factor, axis=-1)
    return proced

#（40，32，1500）
def normalize_eeg(eeg_data):
    min_values = np.min(eeg_data, axis=(0, 2), keepdims=True)
    max_values = np.max(eeg_data, axis=(0, 2), keepdims=True)
    normalized_data = 2 * (eeg_data - min_values) / (max_values - min_values) - 1
    return normalized_data

class PreProcessSequential:
    def __init__(self):
        self.fs = 1000
        self.refs = 250
        self.bplow = 0.5
        self.bphigh = 40
        self.bslow = 49
        self.bshigh = 51
        self.notch = 50

    def __call__(self, data: np.ndarray):
        return self._sequential(data)

    def _sequential(self, x):
        x = preprocess_filt(x, low_cut=self.bplow, high_cut=self.bphigh, fs=self.fs)
        x = preprocess_resample(x,fs=self.fs,refs=self.refs)
        return x


def load_data_bci41_vis(data_path, subject, all_trials=True):

    if subject <= 4:   # motor 1
        data_path = data_path + str(subject) + '.mat'
        a = loadmat(data_path)
        data1 = a['cnt']
        data1 = data1.T
        time1 = a['mrk'][0][0]
        time2 = time1[0]
        labbel1 = time1[1]
        time3 = time2[0]
        labbel2 = labbel1[0]
        class1 = np.zeros((100, 59, 4000))
        class2 = np.zeros((100, 59, 4000))
        class3 = np.zeros((100, 59, 4000))
        class4 = np.zeros((100, 59, 4000))
        a = 0
        b = 0
        c = 0
        d = 0
        for i in range(200):
            if labbel2[i] == -1:
                data444 = data1[:, time3[i] - 4000:time3[i]]
                class1[a] = data444
                a += 1
                data555 = data1[:, time3[i]: time3[i] + 4000]
                class2[b] = data555
                b += 1
            if labbel2[i] == 1:
                data666 = data1[:, time3[i] - 4000:time3[i]]
                class3[c] = data666
                c += 1
                data777 = data1[:, time3[i]: time3[i] + 4000]
                class4[d] = data777
                d += 1

        class111 = class1[:50]
        class222 = class2[:50]
        class333 = class1[50:]
        class444 = class2[50:]

        preprocessor = PreProcessSequential()
        class3 = preprocessor(class111)
        class4 = preprocessor(class222)
        class5 = preprocessor(class333)
        class6 = preprocessor(class444)

        train_data_return = np.concatenate((class3, class4), axis=0)
        test_data_return = np.concatenate((class5, class6), axis=0)

        print("train_data_return:", train_data_return.shape)
        print("test_data_return:", test_data_return.shape)
        label1 = []
        label2 = []
        for i in range(50):
            label1.append(1)
        for i in range(50):
            label1.append(2)
        for i in range(50):
            label2.append(1)
        for i in range(50):
            label2.append(2)
        class_return = label1
        test_class_return = label2

    else:  # motor 2
        data_path = data_path + str(subject) + '.mat'
        a = loadmat(data_path)  # 训练
        data1 = a['cnt']
        data1 = data1.T
        time1 = a['mrk'][0][0]

        time2 = time1[0]
        labbel1 = time1[1]

        time3 = time2[0]
        labbel2 = labbel1[0]

        class1 = np.zeros((100, 59, 4000))
        class2 = np.zeros((100, 59, 4000))
        class3 = np.zeros((100, 59, 4000))
        class4 = np.zeros((100, 59, 4000))

        a = 0
        b = 0
        c = 0
        d = 0

        for i in range(200):
            if labbel2[i] == -1:
                data444 = data1[:, time3[i] - 4000:time3[i]]
                class1[a] = data444
                a += 1
                data555 = data1[:, time3[i]: time3[i] + 4000]
                class2[b] = data555
                b += 1
            if labbel2[i] == 1:
                data666 = data1[:, time3[i] - 4000:time3[i]]
                class3[c] = data666
                c += 1
                data777 = data1[:, time3[i]: time3[i] + 4000]
                class4[d] = data777
                d += 1

        class111 = class3[:50]
        class222 = class4[:50]
        class333 = class3[50:]
        class444 = class4[50:]

        print("class111:", class111.shape)
        print("class222:", class222.shape)
        print("class333:", class333.shape)
        print("class444:", class444.shape)

        preprocessor = PreProcessSequential()
        class3 = preprocessor(class111)
        class4 = preprocessor(class222)
        class5 = preprocessor(class333)
        class6 = preprocessor(class444)

        train_data_return = np.concatenate((class3, class4), axis=0)
        test_data_return = np.concatenate((class5, class6), axis=0)
        print("train_data_return:", train_data_return.shape)
        print("test_data_return:", test_data_return.shape)
        label1 = []
        label2 = []
        for i in range(50):
            label1.append(1)
        for i in range(50):
            label1.append(2)
        for i in range(50):
            label2.append(1)
        for i in range(50):
            label2.append(2)
        class_return = label1
        test_class_return = label2

    return train_data_return, test_data_return, class_return, test_class_return

def standardize_data(X_train, X_test, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
    return X_train, X_test


def get_data(path, subject, isStandard=True):
    X_train, X_test, y_train, y_test = load_data_bci41_vis(path, subject + 1)
    y_train = np.array(y_train)
    N_tr, N_ch, _ = X_train.shape
    X_train = X_train[:, :, :].reshape(N_tr, 1, N_ch, _)
    y_train_onehot = (y_train - 1).astype(int)
    y_train_onehot = to_categorical(y_train_onehot)
    y_test = np.array(y_test)
    N_test, N_ch, _ = X_test.shape
    X_test = X_test[:, :, :].reshape(N_test, 1, N_ch, _)
    y_test_onehot = (y_test - 1).astype(int)
    y_test_onehot = to_categorical(y_test_onehot)
    # Standardize the data
    if (isStandard == True):
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

