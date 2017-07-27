from DeepLibphys.utils.functions.common import process_cnn_signal, get_fantasia_full_paths
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import spectrogram
from matplotlib.image import imread
from sklearn.neighbors import KNeighborsClassifier
from cv2.cv2 import resize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
#from sklearn.preprocessing import LabelEncoder

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def create_spectrograms(n_samples, window_size=1024, nperseg=128, noverlap=64):
    plt.ion()
    plt.axis('off')
    n_subjects = len(get_fantasia_full_paths())
    windows_per_subject = n_samples // (nperseg - noverlap) - window_size // (nperseg - noverlap)
    print(windows_per_subject)


    for i in range(n_subjects):
        original_signals = loadmat(get_fantasia_full_paths()[i])['val'][0][:n_samples]
        # all signals = np.array([loadmat(get_fantasia_full_paths()[i])['val'][0] for i in range(len(get_fantasia_full_paths()))])
        original_signals = process_cnn_signal(original_signals)

        for k in range(windows_per_subject):
            f, t, Sxx = spectrogram(original_signals[k * (nperseg - noverlap) : k * (nperseg - noverlap) + window_size], 250, nperseg=nperseg, noverlap=noverlap)

            Sxx = (Sxx - np.mean(Sxx)) / np.max(Sxx)
            Sxx = np.round(Sxx, 2)
            Sxx = resize(Sxx[:15, :], (15, 15))
            plt.imshow(Sxx)
            plt.draw()
            plt.pause(0.2)
            #plt.savefig("test" + '_' + str(i) + '_' + str(k))
            #plt.close()
            #labels[i, k] = i  # person identifier

    #np.save('labels_train', labels) # change name for test


def PCA_comps(img, nComp = 10):
    # For dimensionality reduction. Performance gets worse
    Xhat = np.empty((len(img), nComp * img.shape[2]))
    for i in range(len(img)):
        # mu = np.mean(images_train, axis=0)
        pca = PCA(nComp)
        pca.fit(img[i, :, :])
        # 3 comps (25,32,3)
        # print(Xhat.shape)

        Xhat[i] = pca.components_.flatten()
    return np.array(Xhat)

SPEC_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Spectrograms/Fantasia"
os.chdir(SPEC_DIRECTORY)

n_samples_train = 60000 # number of samples from each subject/person
n_subjects = len(get_fantasia_full_paths())
window_size = 1024
noverlap = 250  # number of windows to overlap

# n_windows_train = n_samples_train // window_size
# n_windows_test = (n_samples_train // 2) // window_size
create_spectrograms(5000)
exit()
#images_train = np.array([np.array([rgb2gray(imread("spec" + '_' + str(i) + '_' + str(k) + '.png')[:,:,:3]).flatten() for k in range(25)]) for i in range(n_subjects)])

#images_test = np.array([np.array([rgb2gray(imread("test" + '_' + str(i) + '_' + str(k) + '.png')[:,:,:3]).flatten() for k in range(10)]) for i in range(n_subjects)])

# Read and rescale all images to (60,80,3)
images_train = np.array([np.array([rgb2gray(resize(imread("spec" + '_' + str(i) + '_' + str(k) + '.png')[:,:,:3], (60,60)))
                                   for k in range(n_windows_train)]) for i in range(n_subjects)])

images_test = np.array([np.array([rgb2gray(resize(imread("test" + '_' + str(i) + '_' + str(k) + '.png')[:,:,:3], (60,60)))
                                  for k in range(n_windows_test)])for i in range(n_subjects)])


images_train = images_train.reshape((images_train.shape[0] * images_train.shape[1], images_train.shape[2]))
images_test = images_test.reshape((images_test.shape[0] * images_test.shape[1], images_test.shape[2]))

print(images_train.shape)
print(np.mean(images_train[0]))

# plt.imshow(images_train[0,0])
# plt.axis('off')
# plt.show()
# plt.imshow(images_train[0,0])
# plt.show()


labels_train = np.load('labels_train.npy')[:,:25]
labels_test = np.load('labels_test.npy')[:,:10]
labels_train = labels_train.reshape((labels_train.shape[0] * labels_train.shape[1]))
labels_test = labels_test.reshape((labels_test.shape[0] * labels_test.shape[1]))
# plt.plot(labels_train)
# plt.show()

model = KNeighborsClassifier()
model.fit(images_train, labels_train)

results = model.predict(images_test)

print("Accuracy:", accuracy_score(labels_test, results))

plt.matshow(confusion_matrix(labels_test, results))
plt.colorbar()

#imread()
# plt.subplot(211)
# plt.plot(np.arange(len(original_signals)) / 250, original_signals)
# plt.subplot(212)
# plt.pcolormesh(t, f, Sxx)
# plt.show()

