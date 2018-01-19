from DeepLibphys.utils.functions.common import get_fantasia_full_paths
from DeepLibphys.sandbox.ConvNets import CNN
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import spectrogram
from novainstrumentation.peaks import peaks
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

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/Fantasia/ECG/mat'#ECG_2014_full'

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
    # plt.ion()
    for filename in sorted(glob.glob(os.path.join(DATASET_DIRECTORY, '*.mat'))):#for i in range(n_subjects):
        print(filename)
        original_signals = np.array(loadmat(os.path.join(DATASET_DIRECTORY, filename))['val'][0][:])#loadmat(get_fantasia_full_paths()[i])['val'][0][:n_samples]
        #original_signals = process_cnn_signal(original_signals)


        # Ignore signals with significant artifacts
        original_signals = (original_signals - np.min(original_signals)) / (np.max(original_signals) - np.min(original_signals))
        if len(peaks(original_signals, tol=0.92)) < 90:
            continue

        #print(len(peaks(original_signals, tol=0.92)))
        # plt.plot(original_signals)
        # plt.pause(1)
        # plt.clf()
        # plt.show()
        for k in range(train_windows):
            f, t, Sxx = spectrogram(original_signals[k * (nperseg - noverlap): k * (nperseg - noverlap) + window_size],
                                    250, nperseg=nperseg, noverlap=noverlap)#, mode='complex')

            #print(Sxx)#.shape)
            Sxx = resize(Sxx[:15, :], (30,30))
            #Sxx = cvtColor(Sxx, COLOR_GRAY2BGR)
            Sxx = (Sxx / np.max(Sxx))#.astype('df')
            #Sxx = np.round(Sxx, 2) * 256
            #plt.imshow(Sxx)
            #plt.axis('off')
            #plt.pause(0.8)
            #plt.show()
            #exit()
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

MODEL_DIRECTORY = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/bento"
os.chdir(MODEL_DIRECTORY)


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


images_tr = images_train.reshape((images_train.shape[0], 1, 1, images_train.shape[1], images_train.shape[2]))
images_tr_t = images_train.reshape((images_train.shape[0], 1, images_train.shape[1], images_train.shape[2]))
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
# print(images_tr_t[1:].shape)
# # Try selu, hard_sigmoid, linear
model.add(ConvLSTM2D(30, kernel_size=(3,3), activation='relu', padding='same', input_shape=(1, 1, images_tr.shape[3], images_tr.shape[4])))#
# #model.add(Dropout(0.3))
# #model.add(ConvLSTM2D(30, kernel_size=(5,5), activation='relu', padding='same', input_shape=(1, images_tr.shape[3], images_tr.shape[4])))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92,patience=5, min_lr=0.0001)
# model = load_model('CLSTM_2DB.h5')
# SSIM pred/test: 0.9611199541178623 -> 2 databases
# SSIM pred/control: 0.24364713657370382
# SSIM test/control: 0.22046930448271526

#model.compile(loss='mean_squared_error', optimizer=sgd)#
#model.fit(images_tr[:-1], images_tr_t[1:], epochs=100, batch_size=32, callbacks=[reduce_lr])
#model.save('CLSTM_2014.h5')
print("Max:",np.max(images_test[0]))
#noise = 0.1*np.max(images_test[0])*np.random.normal(size=(30,30))
pred = model.predict(np.reshape(images_test[0],(1, np.shape(images_test)[1], np.shape(images_test)[2], np.shape(images_test)[3], np.shape(images_test)[4])))#images_test[0]
# Try average between different filter shapes
# Try batch norm and max norm
# Try to remove outlier windows

img_test = np.reshape(images_test[1],(30,30)).astype(np.float32)
img_ctrl = np.reshape(images_train[300],(30,30)).astype(np.float32)
pred = np.reshape(pred,(30,30))

print("SSIM pred/test:", ssim(img_test,pred))

print("SSIM pred/control:", ssim(img_ctrl,pred))

print("SSIM test/control:", ssim(img_ctrl,img_test))


plt.subplot(211)
plt.title('Original')
plt.imshow(img_test)#[0,:,:,:30,:30]
plt.subplot(212)
plt.title('Prediction')
plt.imshow(pred)
plt.show()

# SSIM pred/test: 0.6461646698412297 -> 3x3 Relu + adam -> looks better
# SSIM pred/control: 0.1566060121491769
# SSIM test/control: 0.17051390322682425

# SSIM pred/test: 0.7581730109483535 3x3 Relu + sgd
# SSIM pred/control: 0.1835740493347146
# SSIM test/control: 0.17051390322682425

#print(np.reshape(images_test[1],(30,30)).shape)
# plt.subplot(311)
# plt.title('Original')
# plt.imshow(img_test)#[0,:,:,:30,:30]
# plt.subplot(312)
# plt.title('Noise')
# plt.imshow(img_noise)
# plt.subplot(313)
# plt.title('Prediction')
# plt.imshow(pred)
# plt.show()
# Como saber se os espetrogramas estao bem criados? Maior janela?
# plt.imshow(results[0]) # for predicted spectrograms
# plt.axis('off')
# plt.show()
