import numpy as np

import DeepLibphys
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import *
import DeepLibphys.classification.RLTC as RLTC

GRU_DATA_DIRECTORY = "../data/trained/"
SNR_DIRECTORY = "../data/validation/Sep_DNN_FANTASIA_1024_1024"
# SNR_DIRECTORY = "../data/validation/May_DNN_SNR_FANTASIA_1"


if __name__ == "__main__":
    N_Windows = 1024
    W = 1024
    signal_dim = 256
    hidden_dim = 256
    batch_size = 128
    window_size = 1024
    fs = 250

    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)
    dir_name = TRAINED_DATA_DIRECTORY + signal_directory

    signals = np.load("../data/processed/FANTASIA_ECG[256].npz")['x_train']

    s_models = db.ecg_1024_256_RAW
    SNRs = ["RAW"]

    loss_tensor = []
    # for i in range(1, 41):
    #     classify_biosignals(SNR_DIRECTORY +
    # "/LOSS_FOR_SNR_{0}_iteration_{1}".format("RAW", 0), w_for_classification=i)

    iterations = 1
    bs = 120
    all_EERs = []
    seconds = (W/fs) + (np.arange(1, bs) * W * 0.33) / fs
    # indexes = list(range(1,6))+list(range(7,20))
    iteration = 0
    SNR_DIRECTORY = "../data/validation/Nov_Fantasia"

    filename = SNR_DIRECTORY + "/LOSS_TENSOR.npz"
    signals = [signal[int(len(signal)*0.33):] for signal in signals]
    models = db.ecg_1024_256_RAW

    loss_tensor = RLTC.get_or_save_loss_tensor(full_path=filename, force_new=False, N_Windows=N_Windows, W=W,
                                               models=models, test_signals=signals, mean_tol=0.8, overlap=0.33,
                                               mini_batch=256, std_tol=0.1)
    # x = list(range(37))

    # for i in [15, 20]:
    #     RLTC.identify_biosignals(loss_tensor, s_models, i)
    # for i in range(np.shape(loss_tensor)[0]):
    #     for j in range(np.shape(loss_tensor)[1]):
    #         for k in range(np.shape(loss_tensor)[2]):
    #             loss_tensor[i, j, k] = loss_tensor[i, j, k] - np.min(loss_tensor[i, :, :])
    #             loss_tensor[i, j, k] = loss_tensor[i, j, k] / np.max(loss_tensor[i, :, :])

    for i in [1, 15, 60]:
    #     RLTC.identify_biosignals(loss_tensor, s_models,  batch_size=i)
        temp_loss_tensor = RLTC.calculate_batch_min_loss(loss_tensor, i)
        eer, thresholds_out, candidate_index = RLTC.calculate_smart_roc(temp_loss_tensor, decimals=5)

    print("Number of windows: {0}".format(np.shape(loss_tensor)[2]))
    EERs, thresholds, batch_size_array = RLTC.process_eers(loss_tensor, W, SNR_DIRECTORY, "RAW ECG", batch_size=bs,
                                                           decimals=5)

    file = np.load(SNR_DIRECTORY + "/" + "RAW ECG" + "_EER.npz")
    EERs, thresholds = file["EERs"], file["thresholds"]

    for i in [15, 20]:
        RLTC.identify_biosignals(loss_tensor, s_models, i)#, thresholds[i])