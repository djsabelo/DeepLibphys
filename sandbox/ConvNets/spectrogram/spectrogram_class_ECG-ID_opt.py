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
import torch
from torch.autograd import Variable
from torch import nn, optim
from DeepLibphys.sandbox.ConvNets.spectrogram.Densenet import DenseNet

from scipy.signal import butter, lfilter
from itertools import repeat
import multiprocessing as mp
#from skimage.measure import compare_ssim as ssim
#import keras.backend as tf

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/ECG-ID'

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# def cnn(self, in_t, reuse=False):
#     self.l1 = tf.layers.conv2d(in_t, 32, [8, 8], strides=(4, 4), reuse=reuse, activation=tf.nn.relu, name='Conv1')
#     # self.nan1 = tf.isnan(self.l1)
#     self.l2 = tf.layers.conv2d(self.l1, 64, [4, 4], strides=(2, 2), reuse=reuse, activation=tf.nn.relu,
#                                name='Conv2')
#     # print(self.l2)
#     # self.nan2 = tf.isnan(self.l2)
#     self.l3 = tf.layers.conv2d(self.l2, 64, [3, 3], strides=(1, 1), activation=tf.nn.relu, name='Conv3')
#     # self.l4 = tf.layers.flatten(self.l2)
#     return self.l3  # tf.reshape(self.l2, [1,1,1152])

def create_spectrograms(n_samples, window_size=2048, train_ratio=0.67, nperseg=512, noverlap=480):
    #plt.ion()
    pool = mp.Pool(mp.cpu_count()-2)
    return pool.starmap(get_spec, zip(sorted(glob.iglob(os.path.join(DATASET_DIRECTORY+'/*', '*_[1-2]m.mat'), recursive=True)),
                                    repeat(n_samples), repeat(window_size), repeat(train_ratio), repeat(nperseg),
                                    repeat(noverlap)))
# Before 0.8841397849462366
# 0.8887096774193548 50x50
# 0.918010752688172 cut to 25 50x50
# 0.9236559139784947 30; 50x50
# 0.9260752688172043 30;60x60
# 0.9303763440860215 30;60x60 (32,48,500)
def get_spec(filename, n_samples, window_size, train_ratio, nperseg, noverlap, spec_size=100, im_size=60):
    images_train = []
    images_test = []
    labels_train = []
    labels_test = []
    train_length = int(n_samples * train_ratio)
    test_length = n_samples - train_length
    train_windows = train_length // (nperseg - noverlap) - window_size // (nperseg - noverlap) # total number of windows
    test_windows = test_length // (nperseg - noverlap) - window_size // (nperseg - noverlap)
    # print(filename)
    original_signals = np.array(loadmat(os.path.join(DATASET_DIRECTORY, filename))['val'][0][:n_samples])  # 160000:160000+n_samples

    # original_signals = process_cnn_signal(original_signals)
    # original_signals = butter()

    # Signal Scaling
    #centered = original_signals - np.mean(original_signals)
    original_signals = (original_signals - np.mean(original_signals)) / (np.max(original_signals) - np.min(original_signals))

    original_signals = remove_noise(original_signals)
    #original_signals = butter_bandpass_filter(original_signals, 0.5, 30, 500, order=3)
    train_signals = original_signals[:train_length]
    test_signals = original_signals[train_length:]
    for k in range(train_windows):
        # plt.plot(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size])
        # plt.pause(0.3)
        # plt.clf()
        # plt.show()
        f, t, Sxx = spectrogram(train_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size],
                                500, nperseg=nperseg, noverlap=noverlap, window=('tukey',.5))#, mode='complex')

        #print(Sxx)#.shape)
        Sxx = resize(Sxx[:spec_size, :], (im_size,im_size)) # Try log
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
    for j in range(test_windows):
        f, t, Sxx = spectrogram(test_signals[j * (nperseg - noverlap): j * (nperseg - noverlap) + window_size],
                                500, nperseg=nperseg, noverlap=noverlap, window=('tukey',.5))#, mode='complex')

        Sxx = resize(Sxx[:spec_size, :], (im_size,im_size))
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


n_samples = 10000 # number of samples from each subject/person

# Optimize spec_size(15-500?), im_size(30-100) - fitness: acc - 0.2 * time taken
returned = create_spectrograms(n_samples)
images_train, images_test, labels_train, labels_test = convertw(returned, 0),\
                                                       convertw(returned, 1),\
                                                       convertw(returned, 2),\
                                                       convertw(returned, 3),


np.savez("filtered_spec_ecgid", images_train=images_train, images_test=images_test, labels_train=labels_train, labels_test=labels_test)
file = np.load("filtered_spec_ecgid.npz")
images_train = file['images_train'].astype(np.float32)
images_test = file['images_test'].astype(np.float32)
labels_train = file['labels_train']
labels_test = file['labels_test']

print(images_train.shape)
print(images_train.dtype)


images_train = images_train.reshape((images_train.shape[0], images_train.shape[1], images_train.shape[2], 1))
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

max_iter = 1000 # Maximum iterations of GA
n_pop = 20
n_param = 4
max_hidden_c = 64
max_hidden_fc = 1024
max_steps = 50
m_8_c = np.arange(24,max_hidden_c, 8)
print(m_8_c)
m_32_fc = np.arange(32,max_hidden_fc, 32)
n_layers_c = [np.random.randint(5,15) for i in range(n_pop)]
n_hidden_c = [[np.random.choice(m_8_c) for l in range(n_layers_c[p])] for p in range(n_pop)]
k_sizes = [[np.random.randint(3,8) for l in range(n_layers_c[p])] for p in range(n_pop)]
n_layers_fc = [np.random.randint(1,4) for i in range(n_pop)]
n_hidden_fc = [[np.random.choice(m_32_fc) for l in range(n_layers_fc[p])] for p in range(n_pop)]
lr = [np.random.choice([1e-2,5e-3,1e-3,5e-4,1e-4,5e-5]) for i in range(n_pop)]
steps = [np.random.randint(5,max_steps)for i in range(n_pop)]
batch_size = [np.random.choice([16, 32, 48, 64, 96, 128, 160, 256]) for i in range(n_pop)] # Is there an option to increase batch size?
#arch_type = np.random.choice(['reg','res','dense'])
# Better than arch type is for any layer to have the ability to concatenate with any other. This allows the choice of an architecture
# to become a more accurate process, i.e. less dependent on random choices and the experience of the practitioner.

#layer_connections = [np.random.choice(range(n_layers_c[p]), size=np.random.randint(n_layers_c[p]-1)) for p in range(n_pop)]
# 'or' condition to include the various connections between networks
for it in range(max_iter):

    for p in range(n_pop):

model = Sequential()
# print(images_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                 input_shape=(images_train.shape[1], images_train.shape[2], 1)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 padding='same'))  # , input_shape=(images_tr.shape[1], images_tr.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', padding='same'))
print(model.summary())
model.add(Flatten())
# #model.add(Dropout(0.3))
model.add(Dense(500, activation='relu'))  # Try 1000
#model.add(Dense(l4, activation='relu'))# 0.9119771085979016 700
model.add(Dense(90, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
#                              patience=5, min_lr=0.0001)

model.compile(loss='binary_crossentropy', optimizer='adam')  #
model.fit(images_train, labels_train, epochs=2, batch_size=16, verbose=0)#, callbacks=[reduce_lr])
model.save('CNN_ECGID_fin_2.h5')


# noise = 0.1*np.max(images_test[0])*np.random.normal(size=(30,30))
preds = model.predict(images_test).argmax(axis=1)  # images_test[0]
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