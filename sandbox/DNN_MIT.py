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
from sklearn import svm

VALIDATION_DIRECTORY = "../data/validation/MIT"


def get_variables(param):
    if param == "arr":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia', '../data/processed/biometry_mit[256].npz', 'ecg_mit_arrythmia_'
    if param == "sinus":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Sinus', '../data/processed/biometry_mit_sinus[256].npz', 'ecg_mit_sinus_'
    if param == "long":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Long-Term', '../data/processed/biometry_mit_long_term[256].npz', \
                'ecg_mit_long_term_'

    return None

def get_processing_variables():
    a_dir, apdp, core_name0 = get_variables('arr')
    s_dir, spdp, core_name1 = get_variables('sinus')
    l_dir, lpdp, core_name2 = get_variables('long')

    # full_paths = os.listdir(mit_dir)
    filenames = [a_dir + "/" + full_path for full_path in os.listdir(a_dir) if full_path.endswith(".mat")] + \
                [s_dir + "/" + full_path for full_path in os.listdir(s_dir) if full_path.endswith(".mat")] + \
                [l_dir + "/" + full_path for full_path in os.listdir(l_dir) if full_path.endswith(".mat")]

    Ns = [len(np.load(apdp)["signals"]), len(np.load(spdp)["signals"].tolist()), len(np.load(lpdp)["signals"].tolist())]

    core_names = []
    for x, n in enumerate(Ns):
        for i in range(n):
            core_names.append(eval("core_name{0}".format(x)) + str(i))

    # print(Ns)
    processed_filenames = np.array(['../data/processed/MIT/{0}[256].npz'.format(core_name) for core_name in core_names])

    return filenames, Ns, np.array(core_names), processed_filenames


if __name__ == "__main__":
    N_Windows = None
    isnew = False
    W = 512
    overlap = 0.11
    signal_dim = 256
    hidden_dim = 256
    mini_batch_size = 256
    batch_size = 256
    window_size = 512
    fs = 250
    model_directory = 'ECG_BIOMETRY[MIT]'.format("MIT")
    # loss_filename = VALIDATION_DIRECTORY + "/MIT_LOSS[1024].npz".format(W)
    raw_filenames, Ns, core_names, processed_filenames = get_processing_variables()
    # print(core_names)
    # z1, z2 = 0, Ns[0]
    # name = "arr"
    # z1, z2 = Ns[0], Ns[0] + Ns[1]
    # name = "sinus"
    # z1, z2 = Ns[0] + Ns[1], len(core_names)
    # name = "long"
    z1, z2 = 0, len(core_names)
    name = "all"
    loss_filename = VALIDATION_DIRECTORY + "/MIT_LOSS[" + name + "].npz".format(W)
    # z = np.array(list(range(14)) + list(range(15, len(raw_filenames)-1)))
    z = np.arange(z1, z2)
    # z = np.array([0, 1, 2, 4, 5, 6, 7, 8, 10, 11])
    trained_epochs_1250 = []
    trained_epochs_1000 = []
    all_models_info, signals = [], []
    for i, filename in zip(z, processed_filenames[z]):
        signal, core_name = np.load(filename)["signal"], np.load(filename)["core_name"]
        signals.append(extract_test_part([signal])[0])
        if i in trained_epochs_1250:
            ds, epoch = 0, 1250
        elif i in trained_epochs_1000:
            ds, epoch = 0, 1000
        else:
            ds, epoch = -5, -5
        all_models_info.append(ModelInfo(Hd=256, Sd=256, dataset_name=core_name,
                                     name="MIT " + str(i), DS=ds, t=epoch,
                                     directory=model_directory))

    loss_tensor = RLTC.get_or_save_loss_tensor(loss_filename, N_Windows, W, all_models_info, signals,
                                               mini_batch=mini_batch_size, force_new=isnew, min_windows=256,
                                               mean_tol=0.5, std_tol=0.5, overlap=overlap)

    # svm_signals = signals.append(extr([signal])[0])


    # Ns = [45, 18, 6]
    # X_list, Y_list = prepare_test_data(signal, signal2model, overlap=overlap,
    #                                    batch_percentage=batch_percentage, mean_tol=mean_tol, std_tol=std_tol,
    #                                    randomize=False)
    # loss_tensor, all_models_info = RLTC.filter_loss_tensor(signals, loss_tensor, all_models_info, W,
    #                                                          min_windows=256, max_tol=0.9, std_tol=0.1)

    # indexes = np.array([1, 7, 11, 12, 20, 30, 32, 33, 42] + list(range(Ns[0] + 2, sum(Ns) + 1))) - 1
    # loss_tensor = loss_tensor[indexes][:, indexes]
    # mask = mask_without_indexes(loss_tensor, [2, 10, 15, 29, 43, 48, 49, 51, 54, 57, 58, 61, 64, 68])
    # mask = mask_without_indexes(loss_tensor, [0, 10, 43, 48, 50, 51, 54, 57, 58, 59, 64, 68]) # 93.2% -60 windows
    # mask = mask_without_indexes(loss_tensor,
    #                             [0, 6, 8, 11, 12, 36, 37, 38, 41, 51, 55])  # 82.2% - 60 windows 100% - 120 windows
    # mask = mask_without_indexes(loss_tensor, [0, 2, 6, 8, 10, 11, 12, 15, 29, 33, 36, 37, 38, 41, 48, 50, 51, 54, 55,
    #                                           57, 58, 59, 64, 68]) #99.3% - 60 windows 100% - 120 windnows
    # 0, 2, 6, 8, 10, 11, 12, 15, 29, 33, 36, 37, 38, 41, 48, 50, 51, 54, 55, 57, 58, 59, 64, 68
    #    2,       10,         15, 29, 33,                 48, 50,     54,     57, 58, 59, 64, 68
    # 0,    6, 8,    11, 12,              36, 37, 38, 41,         51,     55

    # mask = mask_without_indexes(loss_tensor, [10, 11, 29, 33] + [51, 55, 58, 59, 64 ]) #ALL
    # mask = mask_without_indexes(loss_tensor, [51, 55, 58, 59]) #SINUS
    # mask = mask_without_indexes(loss_tensor, [10, 11, 29, 33]) #ARRYTHMIA
    mask = mask_without_indexes(loss_tensor, [29])
    loss_tensor = loss_tensor[mask][:, mask]
    all_models_info = np.array(all_models_info)[mask]
    loss_tensor_ = np.copy(loss_tensor)
    for i in [120*3]:
        RLTC.identify_biosignals(loss_tensor, all_models_info, batch_size=i)
    name += "[-29]"
    EERs, thresholds, batch_size_array = RLTC.process_eers(loss_tensor_, W, VALIDATION_DIRECTORY, "_MIT_" + name +
                                                           "", save_pdf=True, batch_size=120, fs=250,
                                                           decimals=4, force_new=True)




