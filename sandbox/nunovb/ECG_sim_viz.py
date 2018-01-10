from DeepLibphys.utils.functions.common import quantize_signal, remove_noise, segment_signal, segment_matrix, get_fantasia_dataset
from DeepLibphys.sandbox.nunovb.linear_AE import Autoencoder
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from scipy.io import loadmat
import glob
import seaborn as sns

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/Fantasia/ECG/mat'

def normalize(sig):
    sig = (sig - np.min(sig))/(np.max(sig) - np.min(sig))
    return sig

def get_files(signals):
    for filename in sorted(glob.glob(os.path.join(DATASET_DIRECTORY, '*.mat'))):
        print(filename)
        signals.append(np.array(loadmat(filename)['val'][0][10000:15000]))
    return np.array(signals)

# Read one signal
# sig = np.array(loadmat(os.path.join(DATASET_DIRECTORY, 'f1y01m.mat'))['val'][0][:5000])
signals = []
signals = get_files(signals)
print(signals.shape)
signals = normalize(signals)

print(signals.shape)
# Normalize
print(np.max(signals))
dists = []
# for i, sig in enumerate(signals):
#     signals[i] = normalize(sig)
print(signals.shape)
# plt.plot(signals[14])
# plt.show()
# exit()
for i, sig in enumerate(signals):
    dists.append(np.mean([np.abs(sig - sig2) for sig2 in signals]))
print(np.max(signals))
dists = np.array(dists)
print(dists)
print(dists.shape)
print(np.arange(1,41).shape)
#colors = cm.rainbow(np.linspace(0, 1, signals.shape[1]))
#for d in signals:
patch1 = mpatches.Patch(color='blue', label='Old t1')
patch2 = mpatches.Patch(color='green', label='Young t1')
patch3 = mpatches.Patch(color='red', label='Old t2')
patch4 = mpatches.Patch(color='y', label='Young t2')
plt.legend(handles=[patch1,patch2,patch3,patch4])
#plt.gca().add_patch(patch1)
plt.plot(np.arange(1,11), dists[0:10], 'b.', markersize=13)#np.arange(1,41),dists)#,color=colors)
plt.plot(np.arange(11,21),dists[10:20], 'g.', markersize=13)
plt.plot(np.arange(21,31),dists[20:30], 'r.', markersize=13)#np.arange(1,41),dists)#,color=colors)
plt.plot(np.arange(31,41),dists[30:40], 'y.', markersize=13)
#sns.regplot(x=dists[10:20], y=dists[30:40])
plt.show()