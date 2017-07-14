from DeepLibphys.utils.functions.common import process_cnn_signal, get_fantasia_full_paths
from DeepLibphys.sandbox.ConvNets import CNN
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import spectrogram
#from matplotlib.image import imread
from cv2.cv2 import cvtColor, COLOR_GRAY2BGR
from cv2.cv2 import resize
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def create_spectrograms(n_samples, window_size=256, train_ratio=0.7, nperseg=128, noverlap=64):
    n_subjects = len(get_fantasia_full_paths())
    windows_per_subject = n_samples // (nperseg - noverlap) - window_size // (nperseg - noverlap)  # total number of windows
    train_windows = int(windows_per_subject * train_ratio)
    images_train = []
    images_test = []
    labels_train = []
    labels_test =[]
    for i in range(n_subjects):
        original_signals = loadmat(get_fantasia_full_paths()[i])['val'][0][:n_samples]
        original_signals = process_cnn_signal(original_signals)

        for k in range(train_windows):
            f, t, Sxx = spectrogram(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size],
                                    250, nperseg=nperseg, noverlap=noverlap)

            # print(Sxx[:15, :].shape)
            Sxx = resize(Sxx[:15, :], (30,30))
            #Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
            Sxx = Sxx / np.max(Sxx)
            #Sxx = np.round(Sxx, 2) * 256
            # plt.imshow(Sxx)
            # plt.axis('off')
            # plt.show()
            labels_train.append(i)
            images_train.append(Sxx)
        for j in range(train_windows, windows_per_subject):
            f, t, Sxx = spectrogram(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size],
                                    250, nperseg=nperseg, noverlap=noverlap)

            Sxx = resize(Sxx[:15, :], (30,30))
            # Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
            Sxx = Sxx / np.max(Sxx) # TRY * 100 & 256
            #Sxx = np.round(Sxx, 2) * 256
            labels_test.append(i)
            images_test.append(Sxx)
            #plt.savefig("test" + '_' + str(i) + '_' + str(k))
            #plt.close()
            #labels[i, k] = i  # person identifier
    return np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)

SPEC_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Spectrograms/Fantasia"
os.chdir(SPEC_DIRECTORY)

n_samples = 90000 # number of samples from each subject/person
n_subjects = len(get_fantasia_full_paths())
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


images_train = images_train.reshape((images_train.shape[0], 1, images_train.shape[1], images_train.shape[2]))
images_test = images_test.reshape((images_test.shape[0], 1, images_test.shape[1], images_test.shape[2]))

print(images_train[0,0])
plt.imshow(images_train[0,0], cmap=plt.cm.Reds)
plt.show()
plt.plot(images_train[0,0])
plt.show()

# Load, Reshape and Binarize labels
labels_train = LabelBinarizer().fit_transform(labels_train)

# Train the ConvNet
model = CNN.CNN()

model.fit(images_train, labels_train, images_test, labels_test, batch_size=20)

results = model.predict(images_test)
print(labels_test)
print(results)
print(accuracy_score(labels_test, results))

plt.matshow(confusion_matrix(labels_test, results))
plt.colorbar()
plt.show()
# Como saber se os espetrogramas estao bem criados? Maior janela?
# plt.imshow(results[0]) # for predicted spectrograms
# plt.axis('off')
# plt.show()
