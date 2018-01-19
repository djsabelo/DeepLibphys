from DeepLibphys.utils.functions.common import get_fantasia_full_paths
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
from keras.layers import Dense, Conv2D, ConvLSTM2D, Dropout, Activation, BatchNormalization, MaxPooling2D, Flatten, TimeDistributed, LSTM
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import SGD
import keras.regularizers as rgl
from keras.constraints import max_norm
from skimage.measure import compare_ssim as ssim
#import keras.backend as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/Fantasia/ECG/mat'

# Try SSIM as loss function
'''def custom_loss(y_true, y_pred):
    #mean, var = tf.nn.moments(x, axes=[1])
    y_pred = tf.Session()
    y_true = y_true.reshape((y_pred.shape(2), y_pred.shape[3]))
    y_pred = y_pred.reshape((y_pred.shape(2), y_pred.shape[3]))
    print(y_true.shape)
    print(y_pred.shape)
    #exit()
    return ssim(y_true,y_pred)'''

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#def process_cnn_signal(sig):

# (240520, 30, 30)
def create_spectrograms(n_samples, window_size=1024, train_ratio=0.67, nperseg=256, noverlap=224):#window_size=256,nperseg=128

    windows_per_subject = n_samples // (nperseg - noverlap) - window_size // (nperseg - noverlap)  # total number of windows
    train_windows = int(windows_per_subject * train_ratio)
    images_train = []
    images_test = []
    labels_train = []
    labels_test =[]
    #plt.ion()
    for filename in sorted(glob.glob(os.path.join(DATASET_DIRECTORY, '*.mat'))):#for i in range(n_subjects):
        print(filename)
        original_signals = np.array(loadmat(os.path.join(DATASET_DIRECTORY, filename))['val'][0][:])#loadmat(get_fantasia_full_paths()[i])['val'][0][:n_samples]
        #original_signals = process_cnn_signal(original_signals)

        for k in range(train_windows):
            # plt.plot(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size])
            # plt.pause(0.3)
            # plt.clf()
            # plt.show()
            f, t, Sxx = spectrogram(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size],
                                    250, nperseg=nperseg, noverlap=noverlap)#, mode='complex')

            #print(Sxx)#.shape)
            Sxx = resize(Sxx[:15, :], (30,30))
            #Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
            Sxx = (Sxx / np.max(Sxx))#.astype('df')
            Sxx = np.round(Sxx, 2) * 256
            #plt.imshow(Sxx)
            #plt.axis('off')
            #plt.pause(0.3)
            #plt.show()

            labels_train.append(filename)
            images_train.append(Sxx)
        for j in range(train_windows, windows_per_subject):
            f, t, Sxx = spectrogram(original_signals[j * (nperseg - noverlap): j * (nperseg - noverlap) + window_size],
                                    250, nperseg=nperseg, noverlap=noverlap)#, mode='complex')

            Sxx = resize(Sxx[:15, :], (30,30))
            # Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
            Sxx = (Sxx / np.max(Sxx))#.astype('f') # TRY * 100 & 256
            #Sxx = np.round(Sxx, 2) * 256
            labels_test.append(filename)
            images_test.append(Sxx)
            #plt.savefig("test" + '_' + str(i) + '_' + str(k))
            #plt.close()
            #labels[i, k] = i  # person identifier
    return np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)

# SPEC_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Spectrograms/Fantasia"
# os.chdir(SPEC_DIRECTORY)
MODEL_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/bento"
os.chdir(MODEL_DIRECTORY)


n_samples = 10000#00 # number of samples from each subject/person
#n_subjects = 40
#window_size = 256
#noverlap = 120  # number of windows to overlap


# Read and rescale all images to (60,80,3)
# images_train = np.array([np.array([resize(imread("spec" + '_' + str(i) + '_' + str(k) + '.png')[:,:,:3], (60,60))
#                                    for k in range(n_windows_train)]) for i in range(n_subjects)])
#
# images_test = np.array([np.array([resize(imread("test" + '_' + str(i) + '_' + str(k) + '.png')[:,:,:3], (60,60))
#                                   for k in range(n_windows_test)])for i in range(n_subjects)])



# Reshape to the theano cnn input shape: (batch size, input channels, input rows, input cols)
#images_train = images_train.reshape((images_train.shape[0] * images_train.shape[1], images_train.shape[4], images_train.shape[2], images_train.shape[3]))
#images_test = images_test.reshape((images_test.shape[0] * images_test.shape[1], images_test.shape[4], images_test.shape[2], images_test.shape[3]))

images_train, images_test, labels_train, labels_test = create_spectrograms(n_samples)
print(images_train.shape)
print(images_train.dtype)


images_tr = images_train.reshape((images_train.shape[0], 1, images_train.shape[1], images_train.shape[2], 1))
images_test = images_test.reshape((images_test.shape[0], 1, images_test.shape[1], images_test.shape[2], 1))
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
model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3),padding='same'), input_shape=(None, 30, 30, 1)))
#model.add(Conv2D(32, kernel_size=(1,1), activation='relu', padding='same'))#, input_shape=(images_tr.shape[1], images_tr.shape[2], 1)))
#model.add(BatchNormalization())
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))#, input_shape=(None, 30, 30, 32)))#, dim_ordering="th"))
#model.add(ConvLSTM2D(32, kernel_size=(3, 3),padding='same', return_sequences=True))
model.add(TimeDistributed(Flatten()))#input_shape=(30,30,1)))
# #model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=False))
model.add(Dense(500, activation='relu'))
model.add(Dense(40, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
                              patience=5, min_lr=0.0001)
#model = load_model('CLSTM.h5')

model.compile(loss='binary_crossentropy', optimizer='adam')#
model.fit(images_tr, labels_train, epochs=20, batch_size=32, callbacks=[reduce_lr])
model.save('CLSTM.h5')

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
