from DeepLibphys.utils.functions.common import quantize_signal, remove_noise, segment_signal, segment_matrix, get_fantasia_dataset
from DeepLibphys.sandbox.nunovb.linear_AE import Autoencoder
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
import glob
import seaborn

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/Fantasia/ECG/mat'

def get_files(signals):
    for filename in glob.glob(os.path.join(DATASET_DIRECTORY, '*.mat')):
        print(filename)
        signals.append(np.array(loadmat(filename)['val'][0][:15000]))
    return np.array(signals)

# Read one signal
sig = np.array(loadmat(os.path.join(DATASET_DIRECTORY, 'f1y01m.mat'))['val'][0][:70000])

# Normalize
#sig = (sig - np.mean(sig)) / np.std(sig)
sig = (np.max(sig) - sig)/(np.max(sig) - np.min(sig))

print(sig)
#plt.plot(sig[0])
#plt.show()
#exit()
x = segment_signal(sig, 1024)[0].astype(np.float32)
#x = segment_matrix(sig, 1024)[0].astype(np.float32)

print(x.shape)

model = Autoencoder()
model.fit(x, n_epochs=200, save=False)
pred = model.reconstruct(x)[0]

#lr = model.get_adapt_lr()
#np.save('/home/bento/lr.npy', lr)
lr = np.load('/home/bento/lr.npy')
#lr = (np.max(lr) - lr)/(np.max(lr) - np.min(lr))
#costs = model.get_cost_vector()
#np.save('/home/bento/costs.npy', costs)
costs = np.log(np.load('/home/bento/costs.npy'))
#costs = (np.max(costs) - costs)/(np.max(costs) - np.min(costs))
averages = np.array([np.mean(costs[i-10:i+10]) for i in range(10, len(costs)-10)])

plt.figure()
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
plt.show()