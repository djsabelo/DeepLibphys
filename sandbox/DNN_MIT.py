import numpy as np

import DeepLibphys
import DeepLibphys.classification.RLTC as RLTC
import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.database import ModelInfo
from DeepLibphys.utils.functions.signal2model import Signal2Model
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
import math
from itertools import repeat
import time
import seaborn
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool

GRU_DATA_DIRECTORY = "../data/trained/"
SNR_DIRECTORY = "../data/validation/June_MIT"


if __name__ == "__main__":
    N_Windows = 256
    W = 512
    signal_dim = 256
    hidden_dim = 256
    mini_batch_size = 16
    batch_size = 128
    window_size = 512
    save_interval = 1000
    fs = 250
    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

    mit_dir = RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia'
    processed_mit_data_path = '../data/processed/biometry_mit[256].npz'
    mit_dir = RAW_SIGNAL_DIRECTORY + 'MIT-Sinus'
    processed_data_path = '../data/processed/biometry_mit_sinus[256].npz'
    core_name = 'ecg_mit_sinus_'
    # mit_dir = RAW_SIGNAL_DIRECTORY + 'MIT-Long-Term'
    # processed_data_path = '../data/processed/biometry_mit_long_term[256].npz'
    # core_name = 'ecg_mit_mit_long_term_'

    signals = np.load(processed_data_path)['signals']
    processed_data_path = '../data/processed/biometry_mit_long_term[256].npz'
    signals = signals.tolist() + np.load(processed_data_path)['signals'].tolist() #\
              #+ np.load(processed_mit_data_path)['signals'].tolist()

    for i, s in enumerate(signals):
        signals[i] = s[int(len(s)*0.33):]

    signals = np.array(signals)
    all_models_info = [ModelInfo(Hd=256, Sd=256, dataset_name=core_name+str(i),
                        name="MIT "+str(i+1),
                        directory="ECG_BIOMETRY[256.512]")
              for i in range(0, 18)] \
                      + [ModelInfo(Hd=256, Sd=256, dataset_name='ecg_mit_long_term_'+str(i),
                        name="MIT "+str(i+19),
                        directory="ECG_BIOMETRY[256.512]")
              for i in range(0, 6)]
    # \
    #                 + [ModelInfo(Hd=256, Sd=256, dataset_name='ecg_mit_' + str(i),
    #                              name="MIT " + str(i + 25),
    #                              directory="ECG_BIOMETRY[256.1024]")
    #                    for i in range(0, 44)]

    filename = SNR_DIRECTORY + "/LOSS_FOR_ALL_MIT_512"
    loss_tensor = RLTC.get_or_save_loss_tensor(filename, N_Windows, W, all_models_info, signals, force_new=False,
                                               mean_tol=0.9, std_tol=0.5)

    EERs, thresholds = RLTC.process_eers(loss_tensor, W, SNR_DIRECTORY + "/img", "MIT_EER", save_pdf=True,
                                         batch_size=120, fs=250, decimals=4)

    for i in [1, 20, 50]:
        RLTC.identify_biosignals(loss_tensor, all_models_info, w_for_classification=i)
