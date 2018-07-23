from DeepLibphys.utils.functions.common import get_fantasia_full_paths, remove_noise
#from DeepLibphys.sandbox.ConvNets import CNN
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
from keras.layers import Input,Dense, Conv2D, ConvLSTM2D, Dropout, Activation, BatchNormalization, MaxPooling2D, Flatten, concatenate
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model, Model
from keras.optimizers import SGD
import keras.regularizers as rgl
from keras.constraints import max_norm
from scipy.signal import butter, lfilter
from itertools import repeat
import multiprocessing as mp
#from skimage.measure import compare_ssim as ssim
#import keras.backend as tf

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""

#DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/ECG-ID'
DATASET_DIRECTORY = '/home/nvb/Data/ECG-ID'

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

def normalize(x):
    #return ((x - np.min(x)) / (np.max(x) - np.min(x))) * 2 - 1
    return (x - np.mean(x))/ np.std(x)

def create_spectrograms(n_samples, window_size=2048, train_ratio=0.5, nperseg=512, noverlap=480, sxx_samples=30, im_size=60, log_spec=True):
    #plt.ion()
    pool = mp.Pool(mp.cpu_count()-2) # -2 if working
    return pool.starmap(get_spec, zip(sorted(glob.iglob(os.path.join(DATASET_DIRECTORY+'/*', '*_[1-2]m.mat'), recursive=True)),
                                    repeat(n_samples), repeat(window_size), repeat(train_ratio), repeat(nperseg),
                                    repeat(noverlap), repeat(sxx_samples), repeat(im_size), repeat(log_spec)))
# Before 0.8841397849462366
# 0.8887096774193548 50x50
# 0.918010752688172 cut to 25 50x50
# 0.9236559139784947 30; 50x50
# 0.9260752688172043 30;60x60
# 0.9303763440860215 30;60x60 (32,48,500)
def get_spec(filename, n_samples, window_size, train_ratio, nperseg, noverlap, sxx_samples, im_size, log_spec):
    images_train = []
    images_test = []
    labels_train = []
    labels_test = []
    train_length = int(n_samples * train_ratio)
    test_length = n_samples - train_length
    train_windows = train_length // (nperseg - noverlap) - window_size // (nperseg - noverlap)  # total number of windows
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


        #Sxx[Sxx == None] = 0#np.zeros(Sxx.shape[1])

        if sxx_samples == 'all':
            if log_spec:
                Sxx = resize(np.log(Sxx + 1e-10), (im_size, im_size)) # + 1e-10
            else:
                Sxx = resize(Sxx, (im_size, im_size))
        else:
            if log_spec:
                Sxx = resize(np.log(Sxx[:sxx_samples, :] + 1e-10), (im_size, im_size))
            else:
                Sxx = resize(Sxx[:sxx_samples, :], (im_size,im_size)) # try 60x60 and cutting less
        #Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
        # Try to normalize

        #Sxx = (Sxx / np.max(Sxx))#.astype('df')
        Sxx = normalize(Sxx)
        #print(Sxx)

        #if log_spec:
        #    Sxx = np.log(Sxx)
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

        # if sxx_samples == 'all':
        #     Sxx = resize(Sxx, (im_size, im_size))
        # else:
        #     Sxx = resize(Sxx[:sxx_samples, :], (im_size,im_size)) #
        #Sxx[Sxx == None] = 0
        if sxx_samples == 'all':
            if log_spec:
                Sxx = resize(np.log(Sxx + 1e-10), (im_size, im_size))
            else:
                Sxx = resize(Sxx, (im_size, im_size))
        else:
            if log_spec:
                Sxx = resize(np.log(Sxx[:sxx_samples, :] + 1e-10), (im_size, im_size))
            else:
                Sxx = resize(Sxx[:sxx_samples, :], (im_size,im_size))
        # Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
        Sxx = normalize(Sxx)
        #Sxx = (Sxx / np.max(Sxx))#.astype('f') # TRY * 100 & 256
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
#MODEL_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/bento"
#os.chdir(MODEL_DIRECTORY)


n_samples = 5000 # number of samples from each subject/person


# np.savez("filtered_spec_ecgid", images_train=images_train, images_test=images_test, labels_train=labels_train, labels_test=labels_test)
# file = np.load("filtered_spec_ecgid.npz")
# images_train = file['images_train'].astype(np.float32)
# images_test = file['images_test'].astype(np.float32)
# labels_train = file['labels_train']
# labels_test = file['labels_test']

#print(images_train.shape)
#print(images_train.dtype)


# images_train = images_train.reshape((images_train.shape[0], images_train.shape[1], images_train.shape[2], 1))
# images_test = images_test.reshape((images_test.shape[0], images_test.shape[1], images_test.shape[2], 1))
#print(images_test.shape)
# print(images_train[0,0])
# plt.imshow(images_train[0,0], cmap=plt.cm.Reds)
# plt.show()
# plt.plot(images_train[0,0])
# plt.show()
# exit()

# Load, Reshape and Binarize labels
# labels_train = LabelBinarizer().fit_transform(labels_train)
# labels_test = LabelEncoder().fit_transform(labels_test)
#print(labels_train.shape)

# Best 1024 ##########################
# Accuracy: 0.9067468844005157
# Sensitivity :  0.9072649572649574
# Specificity :  0.9989521464097736

#l1 = [768, 1152, 1280]
#l3 = 0

# Use the 3 best pairs of (log/ not log) ('all',150,30) for the 3 channels of the pretrained nets

sxx_samples_list = 120#[100, 80, 120] # Try to change cnn model

# Best  0.7170790103750998 -> 80 + no log; 0.7509976057462091 -> 100 + no log
# 0.8024740622505986 120 nl

#120 - 3
#80 -3

# best : 0.5215482841181165 -> 150
# best : 0.6065442936951316 -> 100
# best 5 ep: 0.88? -> 60 std norm
# .8367 l

# Bad 100+no log

# Best log: 0.7509976057462091 30 samps
best = 0

for i in range(1):# [i % 3]
    log_spec = False#bool(np.abs(i % 2 - 1))
    samples = sxx_samples_list#[i % 3]
    print("Log:", log_spec)
    print("samples:", samples)
    returned = create_spectrograms(n_samples, sxx_samples=samples, log_spec=log_spec)# T, F, T...
    images_train, images_test, labels_train, labels_test = convertw(returned, 0), \
                                                           convertw(returned, 1), \
                                                           convertw(returned, 2), \
                                                           convertw(returned, 3)

    #print("Labels:", labels_train.shape)
    if i == 0:
        print(images_train.shape)
        print(images_test.shape)
    images_train = images_train.reshape((images_train.shape[0], images_train.shape[1], images_train.shape[2], 1))
    images_test = images_test.reshape((images_test.shape[0], images_test.shape[1], images_test.shape[2], 1))

    labels_train = LabelBinarizer().fit_transform(labels_train)
    labels_test = LabelEncoder().fit_transform(labels_test)

    #model = Sequential()
    # print(images_tr_t[1:].shape)
    # # Try selu, hard_sigmoid, linear
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', #try 7x7
    #                  input_shape=(images_train.shape[1], images_train.shape[2], 1)))
    # model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
    #                  padding='same'))  # , input_shape=(images_tr.shape[1], images_tr.shape[2], 1)))
    input_img = Input(shape=(images_train.shape[1], images_train.shape[2], 1))

    # x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    # y = Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(input_img)
    # z = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(input_img)
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)

    #tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
    #tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)

    output = concatenate([tower_1, tower_3], axis=1)
    output = MaxPooling2D(pool_size=(2, 2))(output)

    tower_1b = Conv2D(48, (1, 1), padding='same', activation='relu')(output)
    tower_1b = Conv2D(48, (3, 3), padding='same', activation='relu')(tower_1b)

    #tower_2b = Conv2D(64, (1, 1), padding='same', activation='relu')(output)
    #tower_2b = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2b)

    tower_3b = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(output)
    tower_3b = Conv2D(48, (1, 1), padding='same', activation='relu')(tower_3b)

    output2 = concatenate([tower_1b, tower_3b], axis=1) # try concat with previous
    output2 = Flatten()(output2)
    fc = Dense(500, activation='relu')(output2)
    output = Dense(90, activation='softmax')(fc)


    model = Model(input_img, output)

    #model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
    # model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')) # was strides=1
    # model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', padding='same'))
    # #print(model.summary())
    # model.add(Flatten())
    # # #model.add(Dropout(0.3))
    # model.add(Dense(1024, activation='relu'))  # Try 1000
    # # model.add(Dense(l2[i], activation='relu'))# 0.9119771085979016 700
    # # model.add(Dropout(0.2))
    # model.add(Dense(90, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
    #                              patience=5, min_lr=0.0001)
    # model = load_model('CNN_fantasia_clean.h5')# CNN_fantasia
    # Transfer learning
    # 0.9161095338400952 from CNN_fantasia.h5 (100ep trained w/ artifacts) -> worse
    # print(model.summary())
    # model.pop()
    # model.pop()
    # print(model.summary())
    # model.add(Dense(700, activation='relu'))
    # model.add(Dense(90, activation='softmax'))#, name='dense_2'))
    # 0.9274658851543182 -> 2 epochs
    # 0.9368073999450499 -> 100ep
    # 0.9134794442057012 -> 2048 2eps

    # 2 eps (600, 300)
    # Accuracy: 0.8253831829250824
    # Sensitivity :  0.8239316239316239
    # Specificity :  0.9980378901282854
    model.compile(loss='binary_crossentropy', optimizer='adam')  #
    model.fit(images_train, labels_train, epochs=5, batch_size=16, verbose=0)  # , callbacks=[reduce_lr])
    # model.save('CNN_ECGID.h5')



    # noise = 0.1*np.max(images_test[0])*np.random.normal(size=(30,30))
    preds = model.predict(images_test).argmax(axis=1)  # images_test[0]

    del model
    print(preds)
    print(labels_test)
    print(preds.shape)
    print(labels_test.shape)
    # Try average between different filter shapes
    # Try batch norm and max norm
    # Try to remove outlier windows

    acc = accuracy_score(labels_test, preds)
    print(i, " Accuracy:", acc)
    #if acc > best:
    #    #l3 = l1[i]
    #    #l4 = l2[i]

    cm = confusion_matrix(labels_test, preds)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    sensitivity = np.mean(TP / (TP + FN))
    print('Sensitivity : ', sensitivity)

    specificity = np.mean(TN / (TN + FP))
    print('Specificity : ', specificity)

# Train the best
'''model = Sequential()
# print(images_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                 input_shape=(images_train.shape[1], images_train.shape[2], 1)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 padding='same'))  # , input_shape=(images_tr.shape[1], images_tr.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))  # , dim_ordering="th"))
model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(48, kernel_size=(5, 5), activation='relu', padding='same'))
print(model.summary())
model.add(Flatten())
# #model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))  # Try 1000
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

model.fit(images_train, labels_train, epochs=8, batch_size=16, verbose=0)#, callbacks=[reduce_lr])
model.save('CNN_ECGID_fin_10.h5')

# Accuracy: 0.9213579716373013
# Sensitivity :  0.9186609686609687
# Specificity :  0.9991163262349705

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
print('Specificity : ', specificity)'''
###############################
'''
model = Sequential()
# print(images_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same',
                 input_shape=(images_train.shape[1], images_train.shape[2], 1)))
model.add(Conv2D(48, kernel_size=(5, 5), activation='relu',
                 padding='same'))  # , input_shape=(images_tr.shape[1], images_tr.shape[2], 1)))
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
model.fit(images_train, labels_train, epochs=2, batch_size=16, verbose=0)#, callbacks=[reduce_lr])
model.save('CNN_ECGID_try.h5')


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

##################################
model = Sequential()
# print(images_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                 input_shape=(images_train.shape[1], images_train.shape[2], 1)))
# model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
#                  padding='same'))  # , input_shape=(images_tr.shape[1], images_tr.shape[2], 1)))
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

#############################

model.fit(images_train, labels_train, epochs=8, batch_size=16, verbose=0)
model.save('CNN_ECGID_fin_10.h5')



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
'''
# training: 25955
# test: 6981
plt.matshow(cm)
plt.show()