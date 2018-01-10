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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Dropout, Activation
from keras.models import load_model

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/Fantasia/ECG/mat'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#def process_cnn_signal(sig):


def create_spectrograms(n_samples, window_size=1024, train_ratio=0.67, nperseg=1024, noverlap=768):#window_size=256,nperseg=128

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
            f, t, Sxx = spectrogram(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size],
                                    250, nperseg=nperseg, noverlap=noverlap)

            # print(Sxx[:15, :].shape)
            Sxx = resize(Sxx[:15, :], (30,30))
            #Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
            Sxx = (Sxx / np.max(Sxx))#.astype('df')
            #Sxx = np.round(Sxx, 2) * 256
            # plt.imshow(Sxx)
            # plt.axis('off')
            # plt.pause(0.8)
            # plt.show()
            labels_train.append(filename)
            images_train.append(Sxx)
        for j in range(train_windows, windows_per_subject):
            f, t, Sxx = spectrogram(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size],
                                    250, nperseg=nperseg, noverlap=noverlap)

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

SPEC_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Spectrograms/Fantasia"
os.chdir(SPEC_DIRECTORY)

n_samples = 10000#00 # number of samples from each subject/person
#n_subjects = 40
window_size = 256
noverlap = 120  # number of windows to overlap


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




images_train = images_train.reshape((images_train.shape[0], 1, 1, images_train.shape[1], images_train.shape[2]))
images_test = images_test.reshape((images_test.shape[0], 1, 1, images_test.shape[1], images_test.shape[2]))
print(images_test.shape)
# print(images_train[0,0])
# plt.imshow(images_train[0,0], cmap=plt.cm.Reds)
# plt.show()
# plt.plot(images_train[0,0])
# plt.show()
# exit()

# Load, Reshape and Binarize labels
#labels_train = LabelBinarizer().fit_transform(labels_train)
#labels_test = LabelBinarizer().fit_transform(labels_test)
#print(labels_train.shape)

# Train the ConvNet
model = Sequential()

model.add(ConvLSTM2D(6,kernel_size=(3,3), activation='tanh', padding='same', input_shape=(1, 1, images_train.shape[1], images_train.shape[2])))#,
#model.add(Dropout(0.5))
#model.add(ConvLSTM2D(1,kernel_size=(3,3), activation='linear'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',optimizer='adam')#"categorical_crossentropy"
model.fit(images_train[:-1], images_train[1:], batch_size=20)
model.save('CLSTM.h5')
#model = load_model('CLSTM.h5')
pred = model.predict(images_test[0])

plt.subplot(211)
plt.imshow(images_test[0,...,30,30])
plt.subplot(211)
plt.imshow(pred)
plt.show()
# Como saber se os espetrogramas estao bem criados? Maior janela?
# plt.imshow(results[0]) # for predicted spectrograms
# plt.axis('off')
# plt.show()
