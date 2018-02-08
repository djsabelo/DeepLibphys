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
    W = 1024
    signal_dim = 256
    hidden_dim = 256
    mini_batch_size = 16
    batch_size = 256
    window_size = 512
    save_interval = 1000
    fs = 250
    model_directory = 'ECG_BIOMETRY[{0}]'.format("MIT")
    raw_filenames, Ns, core_names, processed_filenames = get_processing_variables()

    all_models_info, signals = [], []
    for i, filename in enumerate(processed_filenames):
        signal, core_name = np.load(filename)["signal"], np.load(filename)["core_name"]
        signals.append(extract_test_part([signal])[0])
        all_models_info.append(ModelInfo(Hd=256, Sd=256, dataset_name=core_name,
                                     name="MIT " + str(1 + i),
                                     directory=model_directory))

    filename = VALIDATION_DIRECTORY + "/MIT_LOSS[{0}].npz".format(W)
    loss_tensor = RLTC.get_or_save_loss_tensor(filename, N_Windows, W, all_models_info, signals, force_new=False)

    # X_list, Y_list = prepare_test_data(signal, signal2model, overlap=overlap,
    #                                    batch_percentage=batch_percentage, mean_tol=mean_tol, std_tol=std_tol,
    #                                    randomize=False)
    # loss_tensor, all_models_info = RLTC.filter_loss_tensor(signals, loss_tensor, all_models_info, W,
    #                                                          min_windows=256, max_tol=0.9, std_tol=0.5)
    loss_tensor_ = np.copy(loss_tensor)
    # mask = mask_without_indexes(loss_tensor, [2, 30, 39, 40, 49, 51, 57, 63])
    # loss_tensor = loss_tensor[mask][:, mask]
    for i in [1, 60, 120, 240]:
        RLTC.identify_biosignals(loss_tensor, all_models_info, batch_size=i)

    EERs, thresholds = RLTC.process_eers(loss_tensor_, W, VALIDATION_DIRECTORY + "/img", "MIT_EER", save_pdf=True,
                                         batch_size=120, fs=250, decimals=4)




