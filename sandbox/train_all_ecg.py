import time

import numpy as np

import DeepLibphys
import DeepLibphys.utils.functions.database as db
from novainstrumentation import smooth
from DeepLibphys.utils.functions.common import *
import matplotlib.pyplot as plt
from DeepLibphys.utils.functions.signal2model import Signal2Model
import scipy.io as sio
import seaborn
import DeepLibphys.models.LibphysMBGRU as GRU

model_info = db.ecg_1024_256_RAW[0]
signal_dim = model_info.Sd
hidden_dim = model_info.Hd
mini_batch_size = 16
batch_size = 128
window_size = 512
save_interval = 10000
signal_directory = "ECG_CLUSTER[256.512]"


print("Loading signals...")
mit_sinus = np.load('../data/processed/biometry_mit_sinus[256].npz')['signals']
mit_long_term = np.load('../data/processed/biometry_mit_long_term[256].npz')['signals']
cibhi_1, cibhi_2 = np.load('../data/processed/biometry_cybhi[256].npz')['train_signals'], \
                              np.load('../data/processed/biometry_cybhi[256].npz')['test_signals']
fantasia = np.load("../data/processed/FANTASIA_ECG[256].npz")['x_train']

signal2model = Signal2Model('ecg_26_SNR_12', signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim, batch_size=batch_size,
                            mini_batch_size=mini_batch_size, window_size=window_size,
                            save_interval=save_interval)


model = GRU.LibphysMBGRU(signal2model)
model.load(model.get_file_tag(), signal_directory)
model.model_name = "ecg_abstraction"

model.train_block(mit_sinus.tolist() + mit_long_term.tolist() + cibhi_1.tolist() + cibhi_2.tolist(), signal2model,
                  n_for_each=mini_batch_size)

