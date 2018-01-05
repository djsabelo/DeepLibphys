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

VALIDATION_DIRECTORY = "../data/validation/Dec_MIT"


def get_variables(param):
    if param == "arr":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia', '../data/processed/biometry_mit[256].npz', 'ecg_mit_arrythmia_'
    if param == "sinus":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Sinus', '../data/processed/biometry_mit_sinus[256].npz', 'ecg_mit_sinus_'
    if param == "long":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Long-Term', '../data/processed/biometry_mit_long_term[256].npz', \
                'ecg_mit_long_term_'

    return None


if __name__ == "__main__":
    N_Windows = None
    W = 1024
    signal_dim = 256
    hidden_dim = 256
    mini_batch_size = 16
    batch_size = 256
    window_size = 512
    save_interval = 1000
    fs = 250
    loss_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

    signals = []
    all_models_info = []
    s_i = 1
    for s_index, db_name in enumerate(["arr", "sinus", "long"]):
        mit_dir, processed_data_path, core_name = get_variables(db_name)
        signals_aux = np.load(processed_data_path)['signals']
        signals += extract_test_part(signals_aux)
        all_models_info += [ModelInfo(Hd=256, Sd=256, dataset_name=core_name+str(i),
                                      name="MIT " + str(s_i + (i)),
                                      directory=loss_directory)
                            for i in range(0, np.shape(signals_aux)[0])]
        s_i += np.shape(signals_aux)[0]

    filename = VALIDATION_DIRECTORY + "/MIT_LOSS[{0}].npz".format(W)
    loss_tensor = RLTC.get_or_save_loss_tensor(filename, N_Windows, W, all_models_info, signals, force_new=True,
                                               mean_tol=0.5, std_tol=2)

    # X_list, Y_list = prepare_test_data(signal, signal2model, overlap=overlap,
    #                                    batch_percentage=batch_percentage, mean_tol=mean_tol, std_tol=std_tol,
    #                                    randomize=False)
    loss_tensor, all_models_info = RLTC.filter_loss_tensor(signals, loss_tensor, all_models_info, W,
                                                             min_windows=256, max_tol=0.9, std_tol=0.5)
    loss_tensor_ = np.copy(loss_tensor)
    # mask = mask_without_indexes(loss_tensor, [2, 30, 39, 40, 49, 51, 57, 63])
    # loss_tensor = loss_tensor[mask][:, mask]
    for i in [1, 60, 120, 240]:
        RLTC.identify_biosignals(loss_tensor, all_models_info, batch_size=i)

    EERs, thresholds = RLTC.process_eers(loss_tensor_, W, VALIDATION_DIRECTORY + "/img", "MIT_EER", save_pdf=True,
                                         batch_size=120, fs=250, decimals=4)




