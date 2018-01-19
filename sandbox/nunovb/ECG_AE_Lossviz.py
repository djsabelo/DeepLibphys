from DeepLibphys.utils.functions.common import quantize_signal, remove_noise, segment_signal, segment_matrix, get_fantasia_dataset
from DeepLibphys.sandbox.nunovb.simple_viz_AE import Autoencoder
from novainstrumentation.peaks import peaks
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import loadmat
import glob
import tensorflow as tf
import seaborn
from mpl_toolkits.mplot3d import Axes3D

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/Fantasia/ECG/mat'

def get_files(signals):
    for filename in glob.glob(os.path.join(DATASET_DIRECTORY, '*.mat')):
        print(filename)
        signals.append(np.array(loadmat(filename)['val'][0][:5000]))
    return np.array(signals)

# Read all the signals
#signals = []
#sig = get_files(signals)
##sig = get_fantasia_dataset(15000)
##np.array([loadmat(get_fantasia_full_paths()[i])['val'][0] for i in range(len(get_fantasia_full_paths()))])

# Read one signal
sig = np.array(loadmat(os.path.join(DATASET_DIRECTORY, 'f1y01m.mat'))['val'][0][:])

# lower bound to ignore artifacts
for i in range(len(sig)):
    if sig[i] < 14670:
        sig[i] = 14670

# Normalize
#sig = (sig - np.mean(sig)) / np.std(sig)
sig = (sig - np.min(sig))/(np.max(sig) - np.min(sig)) * 2 - 1

# print(sig)
# plt.plot(sig)
# plt.show()
# exit()

peaks = peaks(sig, tol=0.65)#segment_signal(sig, 1024)[0].astype(np.float32)
#print(peaks)
# Put red circles on peaks
# plt.plot(sig)
# plt.title("Detected Peaks")
# plt.ylabel('Normalized Value')
# plt.xlabel('Samples')
# plt.plot(peaks,sig[peaks], "r.")
# plt.show()
# exit()
x = []#np.empty((len(sig),1024))
for i, val in enumerate(peaks):
    if(peaks[i] > 512 and peaks[i] < len(sig) - 512):
        #print(sig[i-512:i+512])
        #x[i] = sig[i-512:i+512]
        x.append(sig[val-512:val+512])
x = np.array(x).astype('f')
#x = segment_matrix(sig, 1024)[0].astype(np.float32)
# plt.ion()
print("X shape:", x.shape)

# for i in range(50):
#     plt.plot(x[i])
#     #plt.show()
#     plt.pause(0.05)
#     #plt.clf()
# exit()

costs = []
weights = []
model = Autoencoder()
costs, weights = model.fit(x[:4400], learning_rate=0.001, batch_size=256, load=False, save=False)
    #with tf.device('/device:GPU:1'):
print(weights.shape)
# print(weights[0].reshape(-1,10)[0])
# print(weights[1].reshape(-1,10)[0])
# print(weights[0].reshape(-1,10)[1])
# print(weights[1].reshape(-1,10)[1])
# exit()
#costs = costs.reshape(-1,10)
X = weights[0]#.reshape(-1,10)
Y = weights[1]#.reshape(-1,10)
# print(costs.shape)
# print(X.shape)
# print(Y.shape)
loss_surface = np.reshape(costs, (X.shape[0], Y.shape[0]))
X,Y = np.meshgrid(X,Y)#weights[0], weights[1])#[:,0:5]
print(X.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Loss Surface')
ax.set_zlabel('Cost')
ax.set_ylabel('X2')
ax.set_xlabel('X1')
#colors = np.random.rand_int()#cm.rainbow(np.linspace(0, 1, signals.shape[1]))
ax.plot_surface(X, Y, loss_surface, color=np.random.rand(3))#cm.coolwarm.reversed()
plt.show()
'''plt.figure()
plt.subplot(211)
plt.title("Adaptive Learning Rate")
plt.ylabel('log Cost')
plt.xlabel('Epoch')
plt.plot(costs, color="#32CD32", alpha=0.8)
plt.plot(np.arange(10,len(costs)-10),averages, color="#8B0000", alpha=0.6)
plt.subplot(212)
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.plot(lr, color="#6A5ACD", alpha=0.8)
plt.show()'''