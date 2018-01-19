from DeepLibphys.utils.functions.common import get_fantasia_full_paths
from DeepLibphys.sandbox.ConvNets import CNN
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.io import loadmat
from scipy.signal import spectrogram
#from matplotlib.image import imread
from cv2.cv2 import cvtColor, COLOR_GRAY2BGR
import glob
from cv2.cv2 import resize
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
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
images_train = images_train.reshape(920,-1)


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
#images_train = np.random.shuffle(images_train)
#print(np.max(images_train))
tsne_results = tsne.fit_transform(images_train)
print(tsne_results.shape)

#images_tr = images_train.reshape((images_train.shape[0], 1, 1, images_train.shape[1], images_train.shape[2]))
#images_tr_t = images_train.reshape((images_train.shape[0], 1, images_train.shape[1], images_train.shape[2]))
#images_test = images_test.reshape((images_test.shape[0],1, 1, images_test.shape[1], images_test.shape[2]))
#print(images_test.shape)
# print(images_train[0,0])
# plt.imshow(images_train[0,0], cmap=plt.cm.Reds)
# plt.show()
# plt.plot(images_train[0,0])
# plt.show()
# exit()

# Load, Reshape and Binarize labels
labels_train_enc = LabelEncoder().fit_transform(labels_train)
#labels_test_enc = LabelBinarizer().fit_transform(labels_test)
#print(labels_train.shape)
hsv = plt.get_cmap('rainbow')

# labels_color = [hsv(i/40) for i in range(40)]
#colors = hsv(np.linspace(0, 1.0, 40))
print(np.array(sns.color_palette(n_colors=40)[1]))

labels_color = [list(sns.color_palette(palette=sns.color_palette("rainbow", 40), n_colors=40))[c] for c in labels_train_enc]
print(np.unique(labels_color).shape)
#exit()
#print(tsne_results[:,0])
tsne_df = pd.DataFrame(dict(imgx=tsne_results[:,0], imgy=tsne_results[:,1], color=labels_color, labels=labels_train_enc))
# tsne_df = pd.DataFrame(dict(imgx=tsne_results[:,0], imgy=tsne_results[:,1], color=labels_color, labels=labels_train_enc))
#, color=hsv(tsne_df.color)
#print(tsne_df.color)
#print(tsne_df.color.values.shape)
# f, ax = plt.subplots(1, 1)
facets = sns.lmplot(data=tsne_df, x='imgx', y='imgy', hue='color', legend=False, fit_reg=False)
ax = facets.ax
ax.set_title('TSNE Spectrogram')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [str(i) for i in range(len(handles))])
plt.show()
# Como saber se os espetrogramas estao bem criados? Maior janela?
# plt.imshow(results[0]) # for predicted spectrograms
# plt.axis('off')
# plt.show()
