from DeepLibphys.utils.functions.common import get_fantasia_full_paths, remove_noise
from DeepLibphys.sandbox.ConvNets import CNN
import os
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
from itertools import repeat
import multiprocessing as mp
from skimage.measure import compare_ssim as ssim
#import keras.backend as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/ECG-ID'


def create_spectrograms(n_samples, window_size=2048, train_ratio=0.67, nperseg=512, noverlap=480):
    #plt.ion()
    pool = mp.Pool(mp.cpu_count())
    return pool.starmap(get_spec, zip(sorted(glob.iglob(os.path.join(DATASET_DIRECTORY+'/*', '*_[1-2]m.mat'), recursive=True)),
                                    repeat(n_samples), repeat(window_size), repeat(train_ratio), repeat(nperseg),
                                    repeat(noverlap)))
# Before 0.8841397849462366
# 0.8887096774193548 50x50
# 0.918010752688172 cut to 25 50x50
# 0.9236559139784947 30; 50x50
# 0.9260752688172043 30;60x60
# 0.9303763440860215 30;60x60 (32,48,500)
def get_spec(filename, n_samples, window_size, train_ratio, nperseg, noverlap):
    images_train = []
    images_test = []
    labels_train = []
    labels_test = []
    windows_per_subject = n_samples // (nperseg - noverlap) - window_size // (nperseg - noverlap)  # total number of windows
    train_windows = int(windows_per_subject * train_ratio)
    print(filename)
    original_signals = np.array(loadmat(os.path.join(DATASET_DIRECTORY, filename))['val'][0][:n_samples])#loadmat(get_fantasia_full_paths()[i])['val'][0][:n_samples]
    #original_signals = process_cnn_signal(original_signals)
    # print(original_signals.shape)
    original_signals = remove_noise(original_signals)

    for k in range(train_windows):
        # plt.plot(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size])
        # plt.pause(0.3)
        # plt.clf()
        # plt.show()
        f, t, Sxx = spectrogram(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size],
                                500, nperseg=nperseg, noverlap=noverlap, window=('tukey',.5))#, mode='complex')

        #print(Sxx)#.shape)
        Sxx = resize(Sxx[:30, :], (60,60)) # try 60x60 and cutting less
        #Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
        # Try to normalize

        Sxx = (Sxx / np.max(Sxx))#.astype('df')
        ###Sxx = np.round(Sxx, 2) * 256
        # plt.imshow(Sxx)
        # plt.axis('off')
        # plt.pause(0.3)
        # plt.show()

        labels_train.append(filename.split('/')[-2])
        images_train.append(Sxx)
    for j in range(train_windows, windows_per_subject):
        f, t, Sxx = spectrogram(original_signals[j * (nperseg - noverlap): j * (nperseg - noverlap) + window_size],
                                500, nperseg=nperseg, noverlap=noverlap, window=('tukey',.5))#, mode='complex')

        Sxx = resize(Sxx[:30, :], (60,60))
        # Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
        Sxx = (Sxx / np.max(Sxx))#.astype('f') # TRY * 100 & 256
        #Sxx = np.round(Sxx, 2) * 256
        labels_test.append(filename.split('/')[-2])
        images_test.append(Sxx)
        #plt.savefig("test" + '_' + str(i) + '_' + str(k))
        #plt.close()
        #labels[i, k] = i  # person identifier
    return np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)

def convertw(list, n):
    new_list = []
    for element in list:
        new_list += element[n].tolist()

    return np.array(new_list)



# SPEC_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Spectrograms/Fantasia"
# os.chdir(SPEC_DIRECTORY)
MODEL_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/bento"
os.chdir(MODEL_DIRECTORY)


n_samples = 10000# # number of samples from each subject/person

returned = create_spectrograms(n_samples)
images_train, images_test, labels_train, labels_test = convertw(returned, 0),\
                                                       convertw(returned, 1),\
                                                       convertw(returned, 2),\
                                                       convertw(returned, 3),


#np.savez("filtered_spec_ecgid", images_train=images_train, images_test=images_test, labels_train=labels_train, labels_test=labels_test)
# file = np.load("filtered_spec_ecgid.npz")
# images_train = file['images_train'].astype(np.float32)
# images_test = file['images_test'].astype(np.float32)
# labels_train = file['labels_train']
# labels_test = file['labels_test']

print(images_train.shape)
print(images_train.dtype)


images_tr = images_train.reshape((images_train.shape[0], images_train.shape[1], images_train.shape[2], 1))
images_test = images_test.reshape((images_test.shape[0], images_test.shape[1], images_test.shape[2], 1))
print(images_test.shape)
# print(images_train[0,0])
# plt.imshow(images_train[0,0], cmap=plt.cm.Reds)
# plt.show()
# plt.plot(images_train[0,0])
# plt.show()
# exit()

# Load, Reshape and Binarize labels
labels_train = LabelBinarizer().fit_transform(labels_train)
labels_test = LabelEncoder().fit_transform(labels_test)
#print(labels_train.shape)

# Train the ConvNet
model = Sequential()
# print(images_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(images_tr.shape[1], images_tr.shape[2], 1)))
#model.add(Conv2D(32, kernel_size=(1,1), activation='relu', padding='same'))#, input_shape=(images_tr.shape[1], images_tr.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))#, dim_ordering="th"))
model.add(Conv2D(48, kernel_size=(3,3), activation='relu', padding='same'))
model.add(Flatten())
# #model.add(Dropout(0.3))
model.add(Dense(700, activation='relu'))# 0.9119771085979016 700
#model.add(Dropout(0.2))
model.add(Dense(90, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
                              patience=5, min_lr=0.0001)
#model = load_model('CLSTM_class.h5')


# 0.8242926653451073 -> overlap 87.5
# 0.865471256082177 -> overlap 93.75 -> Used
# 0.8678512934419241 -> overlap 96.875
model.compile(loss='binary_crossentropy', optimizer='adam')#
model.fit(images_tr, labels_train, epochs=2, batch_size=32, callbacks=[reduce_lr])
#model.save('CNN_ECGID.h5')

#noise = 0.1*np.max(images_test[0])*np.random.normal(size=(30,30))
preds = model.predict(images_test).argmax(axis=1)#images_test[0]
print(preds)
print(labels_test)
print(preds.shape)
print(labels_test.shape)
# Try average between different filter shapes
# Try batch norm and max norm
# Try to remove outlier windows


print("Accuracy:", accuracy_score(labels_test,preds))

plt.matshow(confusion_matrix(labels_test, preds))
plt.show()