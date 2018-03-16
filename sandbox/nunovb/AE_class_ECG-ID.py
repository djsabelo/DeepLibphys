from DeepLibphys.utils.functions.common import get_fantasia_full_paths, remove_noise, segment_signal
from DeepLibphys.sandbox.ConvNets import CNN
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

def read_signals(n_samples=900000, window_size=1024, train_ratio=0.67, overlap=0.5):

    files = sorted(glob.iglob(os.path.join(DATASET_DIRECTORY + '/*', '*_[1-2]m.mat'), recursive=True))
    signals_train = []
    signals_test = []
    labels_train = []
    labels_test = []
    n_windows = n_samples / (window_size * overlap)

    for filename in files:
    
        train_length = int(n_samples * train_ratio)

        # print(filename)
        original_signal = np.array(loadmat(os.path.join(DATASET_DIRECTORY, filename))['val'][0][:n_samples])  # 160000:160000+n_samples

        # Signal Scaling
        #centered = original_signal - np.mean(original_signal)
        #original_signal = (original_signal - np.mean(original_signal)) / (np.max(original_signal) - np.min(original_signal))

        #original_signal = remove_noise(original_signal)
        x = segment_signal(original_signal, window_size, overlap=overlap)[0]
        train_signal = original_signal[:train_length]
        test_signal = original_signal[train_length:]
        signals_train.append(train_signal)
        signals_test.append(test_signal)

        labels_train.append([filename.split('/')[-2] for i in range(n_windows * train_ratio)])
        labels_test.append([filename.split('/')[-2] for i in range(n_windows * (1-train_ratio))])

    return signals_train, signals_test, labels_train, labels_test
    
def convertw(list, n):
    new_list = []
    for element in list:
        new_list += element[n].tolist()

    return np.array(new_list)



# SPEC_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Spectrograms/Fantasia"
# os.chdir(SPEC_DIRECTORY)
MODEL_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/bento"
os.chdir(MODEL_DIRECTORY)


n_samples = 10000 # number of samples from each subject/person

returned = read_signals(n_samples)
signals_train, signals_test, labels_train, labels_test = convertw(returned, 0),\
                                                       convertw(returned, 1),\
                                                       convertw(returned, 2),\
                                                       convertw(returned, 3),


np.savez("filtered_spec_ecgid", signals_train=signals_train, signals_test=signals_test, labels_train=labels_train, labels_test=labels_test)
file = np.load("filtered_spec_ecgid.npz")
signals_train = file['signals_train'].astype(np.float32)
signals_test = file['signals_test'].astype(np.float32)
labels_train = file['labels_train']
labels_test = file['labels_test']

print(signals_train.shape)
print(signals_train.dtype)


signals_train = signals_train.reshape((signals_train.shape[0], signals_train.shape[1], signals_train.shape[2], 1))
signals_test = signals_test.reshape((signals_test.shape[0], signals_test.shape[1], signals_test.shape[2], 1))
print(signals_test.shape)
# print(signals_train[0,0])
# plt.imshow(signals_train[0,0], cmap=plt.cm.Reds)
# plt.show()
# plt.plot(signals_train[0,0])
# plt.show()
# exit()

# Load, Reshape and Binarize labels
labels_train = LabelBinarizer().fit_transform(labels_train)
labels_test = LabelEncoder().fit_transform(labels_test)
#print(labels_train.shape)

# Best 1024 ##########################
# Accuracy: 0.9067468844005157
# Sensitivity :  0.9072649572649574
# Specificity :  0.9989521464097736

# l1 = [768, 1152, 1280]
# l3 = 0
# best = 0
# for i in range(3):
#     model = Sequential()
#     # print(signals_tr_t[1:].shape)
#     # # Try selu, hard_sigmoid, linear
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
#                      input_shape=(signals_train.shape[1], signals_train.shape[2], 1)))
#     model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
#                      padding='same'))  # , input_shape=(signals_tr.shape[1], signals_tr.shape[2], 1)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
#     model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', padding='same'))
#     model.add(Flatten())
#     # #model.add(Dropout(0.3))
#     model.add(Dense(l1[i], activation='relu'))  # Try 1000
#     #model.add(Dense(l2[i], activation='relu'))# 0.9119771085979016 700
#     # model.add(Dropout(0.2))
#     model.add(Dense(90, activation='softmax'))
#
#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
#     #                              patience=5, min_lr=0.0001)
#     # model = load_model('CNN_fantasia_clean.h5')# CNN_fantasia
#     # Transfer learning
#     # 0.9161095338400952 from CNN_fantasia.h5 (100ep trained w/ artifacts) -> worse
#     # print(model.summary())
#     # model.pop()
#     # model.pop()
#     # print(model.summary())
#     # model.add(Dense(700, activation='relu'))
#     # model.add(Dense(90, activation='softmax'))#, name='dense_2'))
#     # 0.9274658851543182 -> 2 epochs
#     # 0.9368073999450499 -> 100ep
#     # 0.9134794442057012 -> 2048 2eps
#
#     # 2 eps (600, 300)
#     # Accuracy: 0.8253831829250824
#     # Sensitivity :  0.8239316239316239
#     # Specificity :  0.9980378901282854
#     model.compile(loss='binary_crossentropy', optimizer='adam')  #
#     model.fit(signals_train, labels_train, epochs=5, batch_size=16, verbose=0)  # , callbacks=[reduce_lr])
#     # model.save('CNN_ECGID.h5')
#
#
#
#     # noise = 0.1*np.max(signals_test[0])*np.random.normal(size=(30,30))
#     preds = model.predict(signals_test).argmax(axis=1)  # signals_test[0]
#     print(preds)
#     print(labels_test)
#     print(preds.shape)
#     print(labels_test.shape)
#     # Try average between different filter shapes
#     # Try batch norm and max norm
#     # Try to remove outlier windows
#
#     acc = accuracy_score(labels_test, preds)
#     print(i, " Accuracy:", acc)
#     if acc > best:
#         l3 = l1[i]
#         #l4 = l2[i]
#
#     cm = confusion_matrix(labels_test, preds)
#
#     FP = cm.sum(axis=0) - np.diag(cm)
#     FN = cm.sum(axis=1) - np.diag(cm)
#     TP = np.diag(cm)
#     TN = cm.sum() - (FP + FN + TP)
#
#     sensitivity = np.mean(TP / (TP + FN))
#     print('Sensitivity : ', sensitivity)
#
#     specificity = np.mean(TN / (TN + FP))
#     print('Specificity : ', specificity)

model = Sequential()
# print(signals_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                 input_shape=(signals_train.shape[1], signals_train.shape[2], 1)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 padding='same'))  # , input_shape=(signals_tr.shape[1], signals_tr.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(Flatten())
# #model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))  # Try 1000
#model.add(Dense(l4, activation='relu'))# 0.9119771085979016 700
model.add(Dense(90, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
#                              patience=5, min_lr=0.0001)

model.compile(loss='binary_crossentropy', optimizer='adam')  #
model.fit(signals_train, labels_train, epochs=2, batch_size=16, verbose=0)#, callbacks=[reduce_lr])
model.save('CNN_ECGID_fin_2.h5')


# noise = 0.1*np.max(signals_test[0])*np.random.normal(size=(30,30))
preds = model.predict(signals_test).argmax(axis=1)  # signals_test[0]
print(preds)
print(labels_test)
print(preds.shape)
print(labels_test.shape)
# Try average between different filter shapes
# Try batch norm and max norm
# Try to remove outlier windows


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

model.fit(signals_train, labels_train, epochs=8, batch_size=16, verbose=0)#, callbacks=[reduce_lr])
model.save('CNN_ECGID_fin_10.h5')

# Accuracy: 0.9213579716373013
# Sensitivity :  0.9186609686609687
# Specificity :  0.9991163262349705

# noise = 0.1*np.max(signals_test[0])*np.random.normal(size=(30,30))
preds = model.predict(signals_test).argmax(axis=1)  # signals_test[0]
print(preds)
print(labels_test)
print(preds.shape)
print(labels_test.shape)
# Try average between different filter shapes
# Try batch norm and max norm
# Try to remove outlier windows


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
###############################

model = Sequential()
# print(signals_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same',
                 input_shape=(signals_train.shape[1], signals_train.shape[2], 1)))
model.add(Conv2D(48, kernel_size=(5, 5), activation='relu',
                 padding='same'))  # , input_shape=(signals_tr.shape[1], signals_tr.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
#model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Flatten())
# #model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))  # Try 1000
#model.add(Dense(l4, activation='relu'))# 0.9119771085979016 700
# model.add(Dropout(0.5))
model.add(Dense(90, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
#                              patience=5, min_lr=0.0001)

model.compile(loss='binary_crossentropy', optimizer='adam')  #
model.fit(signals_train, labels_train, epochs=2, batch_size=16, verbose=0)#, callbacks=[reduce_lr])
model.save('CNN_ECGID_try.h5')


# noise = 0.1*np.max(signals_test[0])*np.random.normal(size=(30,30))
preds = model.predict(signals_test).argmax(axis=1)  # signals_test[0]
print(preds)
print(labels_test)
print(preds.shape)
print(labels_test.shape)
# Try average between different filter shapes
# Try batch norm and max norm
# Try to remove outlier windows


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

##################################
model = Sequential()
# print(signals_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                 input_shape=(signals_train.shape[1], signals_train.shape[2], 1)))
# model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
#                  padding='same'))  # , input_shape=(signals_tr.shape[1], signals_tr.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Flatten())
# #model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))  # Try 1000
#model.add(Dense(l4, activation='relu'))# 0.9119771085979016 700
model.add(Dropout(0.5))
model.add(Dense(90, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
#                              patience=5, min_lr=0.0001)

model.compile(loss='binary_crossentropy', optimizer='adam')  #
model.fit(signals_train, labels_train, epochs=2, batch_size=16, verbose=0)#, callbacks=[reduce_lr])
model.save('CNN_ECGID_fin_2.h5')


# noise = 0.1*np.max(signals_test[0])*np.random.normal(size=(30,30))
preds = model.predict(signals_test).argmax(axis=1)  # signals_test[0]
print(preds)
print(labels_test)
print(preds.shape)
print(labels_test.shape)
# Try average between different filter shapes
# Try batch norm and max norm
# Try to remove outlier windows


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

#############################

model.fit(signals_train, labels_train, epochs=8, batch_size=16, verbose=0)
model.save('CNN_ECGID_fin_10.h5')



# noise = 0.1*np.max(signals_test[0])*np.random.normal(size=(30,30))
preds = model.predict(signals_test).argmax(axis=1)  # signals_test[0]
print(preds)
print(labels_test)
print(preds.shape)
print(labels_test.shape)
# Try average between different filter shapes
# Try batch norm and max norm
# Try to remove outlier windows


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