from DeepLibphys.utils.functions.common import get_fantasia_full_paths, remove_noise, segment_signal
import os
from DeepLibphys.sandbox.nunovb.conv_AE_graph import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import spectrogram
#from matplotlib.image import imread
from cv2.cv2 import cvtColor, COLOR_GRAY2BGR
import glob
from cv2.cv2 import resize
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from tensorflow.contrib.rnn import NASCell
from keras.models import Sequential
from keras.layers import Dense, Conv2D, ConvLSTM2D, Dropout, Activation, BatchNormalization, MaxPooling2D, Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import SGD
import keras.regularizers as rgl
from keras.constraints import max_norm
from scipy.signal import butter, lfilter
from itertools import repeat
import multiprocessing as mp
from skimage.measure import compare_ssim as ssim
#import keras.backend as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/ECG-ID'


#def encode(signal):
    # Encodes a signal using a pretrained autoencoder


def normalize(sig):
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig)) * 2 - 1

def get_signals(n_samples, window_size=1024, train_ratio=0.67, overlap=0.5):
    # Gets training and testing segments
    files = sorted(glob.iglob(os.path.join(DATASET_DIRECTORY + '/*', '*_[1-2]m.mat'), recursive=True))
    signals_train = []
    signals_test = []
    labels_train = []
    labels_test = []
    n_windows = int(n_samples / (window_size * overlap))
    train_windows = int(n_windows * train_ratio)
    test_windows = int(n_windows * (1-train_ratio))
    for filename in files:
        # print(filename)
        original_signal = np.array(loadmat(os.path.join(DATASET_DIRECTORY, filename))['val'][0][:n_samples])  # 160000:160000+n_samples
        # print(original_signal.shape)
        # Signal Scaling
        #centered = original_signal - np.mean(original_signal)
        #original_signal = (original_signal - np.mean(original_signal)) / (np.max(original_signal) - np.min(original_signal))
        original_signal = remove_noise(original_signal)
        original_signal = normalize(original_signal)

        original_signal = segment_signal(original_signal, window_size, overlap=overlap)[0]
        # print("Orig:", original_signal.shape)
        # print(train_windows, test_windows)
        # exit()
        train_signal = original_signal[:train_windows]
        test_signal = original_signal[train_windows:]
        signals_train.append(train_signal)
        signals_test.append(test_signal)
        labels_train.append([filename.split('/')[-2] for i in range(train_windows)])
        labels_test.append([filename.split('/')[-2] for i in range(test_windows)])

    return np.array(signals_train), np.array(signals_test), np.array(labels_train), np.array(labels_test)
    
def convertw(list, n):
    new_list = []
    for element in list:
        new_list += element[n].tolist()

    return np.array(new_list)



# SPEC_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Spectrograms/Fantasia"
# os.chdir(SPEC_DIRECTORY)
MODEL_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/bento"
os.chdir(MODEL_DIRECTORY)

#
n_samples = 5000 # number of samples from each subject/person.  10000 for ECG-ID ;  9000000 for Fantasia

signals_train, signals_test, labels_train, labels_test = get_signals(n_samples)
# signals_train, signals_test, labels_train, labels_test = convertw(returned, 0),\
#                                                        convertw(returned, 1),\
#                                                        convertw(returned, 2),\
#                                                        convertw(returned, 3),


# np.savez("filtered_spec_ecgid", signals_train=signals_train, signals_test=signals_test, labels_train=labels_train, labels_test=labels_test)
# file = np.load("filtered_spec_ecgid.npz")
# signals_train = file['signals_train'].astype(np.float32)
# signals_test = file['signals_test'].astype(np.float32)
# labels_train = file['labels_train']
# labels_test = file['labels_test']

print(signals_train.shape)
print(signals_train.dtype)

signals_train = signals_train.reshape((-1, signals_train.shape[2], 1)) #(179,18,1024,1)
signals_test = signals_test.reshape((-1, signals_test.shape[2], 1))
# signals_train = signals_train.reshape((signals_train.shape[0], signals_train.shape[1], signals_train.shape[2], 1)) #(179,18,1024,1)
# signals_test = signals_test.reshape((signals_test.shape[0], signals_test.shape[1], signals_test.shape[2], 1))
print(signals_test.shape)
# print(signals_train[0,0])
# plt.imshow(signals_train[0,0], cmap=plt.cm.Reds)
# plt.show()
# plt.plot(signals_train[0,0])
# plt.show()
# exit()

# Load, Reshape and Binarize labels
labels_train = LabelEncoder().fit_transform(labels_train.flatten())
labels_test = LabelEncoder().fit_transform(labels_test.flatten())
print(labels_train.shape)
print(labels_test.shape)

model = Autoencoder()

model.fit(signals_train, n_epochs=1, learning_rate=0.003, batch_size=64, load=False, save=True, name='1/CAE_class')
# Best: CAE4.2 (256,128);CAE4.2.ld (4,1);

# Porque é que o sinal fica invertido?
# Porque não fazer overfit?

signals_train = model.get_latent(signals_train)
print("S train:",signals_train.shape)

# print(preds.shape)
signals_test = model.get_latent(signals_test)
print("S_test:",signals_test.shape)

signals_train = signals_train.reshape((-1, signals_train.shape[1] * signals_train.shape[2]))
signals_test = signals_test.reshape((-1, signals_test.shape[1]))
print(signals_train.shape)

model = LogisticRegression()
model.fit(signals_train, labels_train)

preds = model.predict(signals_test)

#np.save('/home/bento/pred.npy', pred)
#pred = np.load('/home/bento/pred.npy')
#lr = model.get_adapt_lr()
#np.save('/home/bento/lr.npy', lr)
#lr = np.load('/home/bento/lr.npy')
#lr = (np.max(lr) - lr)/(np.max(lr) - np.min(lr))
# plt.subplot(211)
# plt.title("Test Signal")
# plt.ylabel('Normalized Voltage')
# plt.xlabel('Samples')
# plt.plot(signals_test[3][3].flatten())
# plt.subplot(212)
# plt.title("Predicted Signal")
# plt.ylabel('Normalized Voltage')
# plt.xlabel('Samples')
# plt.plot(preds[3][3].flatten())
# plt.show()

print("Accuracy:", accuracy_score(labels_test, preds))
cm = confusion_matrix(labels_test, preds)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

sensitivity = np.mean(TP / (TP + FN))
print('Sensitivity : ', sensitivity)

specificity = np.mean(TN / (TN + FP))
print('Specificity : ', specificity)

# training: 25955
# test: 6981
plt.matshow(cm)
plt.show()