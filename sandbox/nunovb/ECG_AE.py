from DeepLibphys.utils.functions.common import quantize_signal, remove_noise, segment_signal, segment_matrix, get_fantasia_dataset
from DeepLibphys.sandbox.nunovb.conv_AE_graph import Autoencoder
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
x = (sig - np.min(sig))/(np.max(sig) - np.min(sig)) * 2 - 1
x = x[:35000]
# print(sig)
# plt.plot(sig)
# plt.show()
# exit()

# peaks = peaks(sig, tol=0.65)#segment_signal(sig, 1024)[0].astype(np.float32)
#print(peaks)

# Put red circles on peaks
# plt.plot(sig)
# plt.title("Detected Peaks")
# plt.ylabel('Normalized Value')
# plt.xlabel('Samples')
# plt.plot(peaks,sig[peaks], "r.")
# plt.show()
# exit()

# Window cutting
# x = []#np.empty((len(sig),1024))
# for i, val in enumerate(peaks):
#     if(peaks[i] > 512 and peaks[i] < len(sig) - 512):
#         #print(sig[i-512:i+512])
#         #x[i] = sig[i-512:i+512]
#         x.append(sig[val-512:val+512])
# x = np.array(x).astype('f')

#print(x)
# plt.title("Original Signal")
# plt.ylabel('Normalized Voltage')
# plt.xlabel('Samples')
# plt.plot(x[0])
# plt.show()
# exit()
#x = segment_matrix(sig, 1024)[0].astype(np.float32)
# plt.ion()
print("X shape:", x.shape)

# for i in range(50):
#     plt.plot(x[i])
#     #plt.show()
#     plt.pause(0.05)
#     #plt.clf()
# exit()

#with tf.device('/device:GPU:0'):
x_train = x[:32768]
model = Autoencoder()
x_train = x_train.reshape(16,2048,1)
print(x_train.shape)

model.fit(x_train, n_epochs=2, learning_rate=0.0005, batch_size=8, load=True, save=True, name='1/CAE')

test = x[33000:34024].reshape(1,-1,1)


pred = model.reconstruct(test)
print(pred.shape)

#np.save('/home/bento/pred.npy', pred)
#pred = np.load('/home/bento/pred.npy')
#lr = model.get_adapt_lr()
#np.save('/home/bento/lr.npy', lr)
#lr = np.load('/home/bento/lr.npy')
#lr = (np.max(lr) - lr)/(np.max(lr) - np.min(lr))

plt.subplot(211)
plt.title("Test Signal")
plt.ylabel('Normalized Voltage')
plt.xlabel('Samples')
plt.plot(test.flatten())
plt.subplot(212)
plt.title("Predicted Signal")
plt.ylabel('Normalized Voltage')
plt.xlabel('Samples')
plt.plot(pred[0,0,:])#.flatten())
plt.pause(0.05)
plt.show()

plt.ion()
for i in range(1000):
    plt.title("Predicted Signal")
    plt.ylabel('Normalized Voltage')
    plt.xlabel('Samples')
    plt.plot(pred[0,i,:])#.flatten())
    plt.pause(0.05)
    #plt.show()
    plt.clf()
exit()
#np.save('/home/bento/costs3_10.npy', costs)
#np.save('/home/bento/weights3_10.npy', weights)

#costs = np.load('/home/bento/costs3_10.npy').flatten()
#costs = (np.max(costs) - costs)/(np.max(costs) - np.min(costs))
#averages = np.array([np.mean(costs[i-10:i+10]) for i in range(10, len(costs)-10)])

#weights = np.load('/home/bento/weights3_10.npy')
# print(weights[0].reshape(-1,10)[0])
# print(weights[1].reshape(-1,10)[0])
# print(weights[0].reshape(-1,10)[1])
# print(weights[1].reshape(-1,10)[1])
# exit()
'''costs = np.array([costs for i in range(10)]).reshape(10,10)
X = weights[0]#.reshape(-1,10)
Y = weights[1]#.reshape(-1,10)
print(costs.shape)
print(X.shape)
print(Y.shape)


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
ax.plot_surface(X, Y, costs, color=np.random.rand(3))#cm.coolwarm.reversed()
plt.show()'''
