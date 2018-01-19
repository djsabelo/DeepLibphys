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
from keras.layers import Dense, ConvLSTM2D, Dropout, Activation, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import SGD
import keras.regularizers as rgl
from keras.constraints import max_norm
from skimage.measure import compare_ssim as ssim
#import keras.backend as tf

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/ECG_MotionArtifact'

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
def create_spectrograms(window_size=2048, nperseg=512, noverlap=448):#window_size=256,nperseg=128

    #windows_per_subject = n_samples // (nperseg - noverlap) - window_size // (nperseg - noverlap)  # total number of windows
    #train_windows = int(windows_per_subject * train_ratio)
    images_train = []
    images_test = []
    labels_train = []
    labels_test = []
    # plt.ion()
    for filename in sorted(glob.iglob(os.path.join(DATASET_DIRECTORY+'/*', '*.mat'), recursive=True)):#for i in range(n_subjects):
        print(filename)
        original_signals = np.array(loadmat(os.path.join(DATASET_DIRECTORY, filename))['val'][0][:])#loadmat(get_fantasia_full_paths()[i])['val'][0][:n_samples]
        original_signals = (original_signals - np.min(original_signals)) / (np.max(original_signals) - np.min(original_signals))
        windows_per_subject = len(original_signals) // (nperseg - noverlap) - window_size // (nperseg - noverlap)
        # plt.plot(original_signals)
        # plt.pause(0.5)
        # plt.clf()
        # plt.show()
        # original_signals = process_cnn_signal(original_signals)
        images_subj = []
        for k in range(windows_per_subject):
            # plt.plot(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size])
            # plt.show()
            # plt.pause(0.8)
            # plt.clf()
            sig = original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size]
            #if len(sig) < window_size:
            #    break
            f, t, Sxx = spectrogram(sig, 500, nperseg=nperseg, noverlap=noverlap)#, mode='complex')

            #print(Sxx.shape)
            Sxx = resize(Sxx[:15, :], (30,30))
            #Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
            Sxx = (Sxx / np.max(Sxx))#.astype('df')
            #Sxx = np.round(Sxx, 2) * 256
            # plt.imshow(Sxx)
            # plt.axis('off')
            # plt.pause(0.5)
            # plt.clf()
            # plt.show()

            labels_test.append(filename)
            images_subj.append(Sxx)
        images_test.append(np.array(images_subj))
        # for j in range(train_windows, windows_per_subject):
        #     sig = original_signals[j * (nperseg - noverlap): j * (nperseg - noverlap) + window_size]
        #     if len(sig) < window_size:
        #         break
        #     f, t, Sxx = spectrogram(sig, 500, nperseg=nperseg, noverlap=noverlap)#, mode='complex')
        #
        #     Sxx = resize(Sxx[:15, :], (30,30))
        #     # Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
        #     Sxx = (Sxx / np.max(Sxx))#.astype('f') # TRY * 100 & 256
        #     #Sxx = np.round(Sxx, 2) * 256
        #     labels_test.append(filename)
        #     images_test.append(Sxx)
        #     #plt.savefig("test" + '_' + str(i) + '_' + str(k))
        #     #plt.close()
        #     #labels[i, k] = i  # person identifier
    return np.array(images_test), np.array(labels_test) #np.array(images_train), np.array(images_test), np.array(labels_train), np.array(labels_test)

MODEL_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/bento"
os.chdir(MODEL_DIRECTORY)

images_test_orig, labels_test = create_spectrograms()
images_test_orig = images_test_orig.astype(np.float32)
print(images_test_orig.shape)

images_test = images_test_orig.reshape((images_test_orig.shape[0]*images_test_orig.shape[1], 1, 1, images_test_orig.shape[2], images_test_orig.shape[3]))
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
# model = Sequential()
# print(images_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
# model.add(ConvLSTM2D(30, kernel_size=(3,3), activation='relu', padding='same', input_shape=(1, 1, images_tr.shape[3], images_tr.shape[4])))#
# #model.add(Dropout(0.3))
# #model.add(ConvLSTM2D(30, kernel_size=(5,5), activation='relu', padding='same', input_shape=(1, images_tr.shape[3], images_tr.shape[4])))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,
#                               patience=5, min_lr=0.0001)
model = load_model('CLSTM_2014.h5')

#model.compile(loss='mean_squared_error', optimizer=sgd)#
#model.fit(images_tr[:-1], images_tr_t[1:], epochs=100, batch_size=32, callbacks=[reduce_lr])
#model.save('CLSTM_opt.h5')

#noise = 0.1*np.max(images_test[0])*np.random.normal(size=(30,30))
# First two dimensions of preds should be: number of subjects, n spectrograms of the subject
#preds = []
#for i in range(images_test.shape[0]):
preds = model.predict(images_test)#np.reshape(,(1, np.shape(images_test)[1], np.shape(images_test)[2], np.shape(images_test)[3], np.shape(images_test)[4]))))#images_test[0]

preds = preds.reshape(images_test_orig.shape)
# Try average between different filter shapes
# Try batch norm and max norm
# Try to remove outlier windows

plt.ion()

for i in range(images_test_orig.shape[0]):
    for j in range(images_test_orig.shape[1]):
        img_test = np.reshape(images_test_orig[i,j],(30,30))
        pred = np.reshape(preds[i,j],(30,30))

        print("SSIM pred/test"+" Subject "+str(i)+":", ssim(img_test,pred))
        print("MSE pred/test" + " Subject " + str(i) + ":", np.mean((img_test - pred)**2))
        plt.subplot(211)
        plt.title('Original')
        plt.imshow(img_test)  # [0,:,:,:30,:30]
        plt.subplot(212)
        plt.title('Prediction')
        plt.imshow(pred)
        plt.pause(1)
        plt.clf()
        plt.show()



#print(np.reshape(images_test[1],(30,30)).shape)
# plt.subplot(211)
# plt.title('Original')
# plt.imshow(img_test)#[0,:,:,:30,:30]
# plt.subplot(212)
# plt.title('Prediction')
# plt.imshow(pred)
# plt.show()

