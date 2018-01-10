from DeepLibphys.utils.functions.common import quantize_signal, remove_noise, segment_signal, segment_matrix, get_fantasia_dataset
from DeepLibphys.sandbox.nunovb.linear_AE import Autoencoder
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import loadmat
import glob
import seaborn as sns

DATASET_DIRECTORY = '/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Signals/Fantasia/ECG/mat'

def get_files(signals):
    for filename in sorted(glob.glob(os.path.join(DATASET_DIRECTORY, '*.mat'))):
        print(filename)
        signals.append(np.array(loadmat(filename)['val'][0][:15000]))
    return np.array(signals)

# Read one signal
# sig = np.array(loadmat(os.path.join(DATASET_DIRECTORY, 'f1y01m.mat'))['val'][0][:5000])
signals = []
signals = get_files(signals)
print(signals.shape)
# Normalize
print(np.max(signals))
dists = []
for sig in signals:
    sig = (np.max(sig) - sig)/(np.max(sig) - np.min(sig))
    dists.append(np.mean([np.abs(sig - sig2) for sig2 in signals]))
print(np.max(signals))

#colors = cm.rainbow(np.linspace(0, 1, signals.shape[1]))
for d in dists:
    sns.regplot()#,color=colors)
sns.plt.show()