import numpy as np

import DeepLibphys
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import *
import DeepLibphys.classification.RLTC as RLTC

GRU_DATA_DIRECTORY = "../data/trained/"
SNR_DIRECTORY = "../data/validation/Sep_DNN_FANTASIA_1024_1024"
# SNR_DIRECTORY = "../data/validation/May_DNN_SNR_FANTASIA_1"

def load_noisy_fantasia_signals(SNRx=None):
    clean_list = np.load("../data/processed/FANTASIA_ECG[256].npz")['x_train']
    clean = np.zeros((1, len(clean_list), len(clean_list[0])))
    for i, c in enumerate(clean_list):
            clean[0, i, :] = np.array(c[-len(clean_list[0]):], dtype=np.int)

    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(128, 1024)
    dir_name = TRAINED_DATA_DIRECTORY + signal_directory
    noise1 = np.load(dir_name + "/signals_without_noise_[{0}].npz".format(256))['processed_noise_array']
    noise2 = np.load(dir_name + "/signals_with_noise_2_[{0}].npz".format(256))['processed_noise_array']
    SNRs = np.load(dir_name + "/signals_without_noise_[{0}].npz".format(256))['SNRs']

    if SNRx is not None:
        noise1 = noise1[np.where(np.logical_and(SNRs >= SNRx[0], SNRs <= SNRx[1]))[0]]
        noise2 = noise2[np.where(np.logical_and(SNRs >= SNRx[0], SNRs <= SNRx[1]))[0]]

    return np.vstack((clean, np.hstack((noise1, noise2))))

if __name__ == "__main__":
    N_Windows = None
    W = 512
    signal_dim = 256
    hidden_dim = 256
    batch_size = 256
    window_size = 1024
    fs = 250
    isnew = True

    signal_directory = 'BIOMETRY[{0}.{1}]'.format(batch_size, window_size)
    dir_name = TRAINED_DATA_DIRECTORY + signal_directory

    signal = np.load("../data/processed/FANTASIA_ECG[256].npz")['x_train']
    s_models = db.ecg_1024_256_RAW

    loss_tensor = []
    SNR_DIRECTORY = "../data/validation/FANTASIA"

    bs = 120
    filename = SNR_DIRECTORY + "/LOSS_TENSOR_[512]_FILTER.npz"
    # filename = SNR_DIRECTORY + "/LOSS_TENSOR_[256].npz"
    model = db.ecg_1024_256_RAW
    filenames = [filename]

    loss_quaternion = []
    passed_SNRs = []
    i = 0


    signal = extract_test_part(signal)
    loss_tensor = RLTC.get_or_save_loss_tensor(full_path=filename, force_new=isnew, N_Windows=N_Windows, W=W,
                                               models=model, test_signals=signal, mean_tol=0.9,
                                               overlap=0.33, mini_batch=256, std_tol=0.05)

    # loss_tensor = loss_tensor[list(range(14))+list(range(15, 40))][:, list(range(15))+list(range(16, 40))]

    for i in [1, 5, 15, 60, 120]:
        RLTC.identify_biosignals(loss_tensor, s_models, i)

    EERs, thresholds, batch_size_array = RLTC.process_eers(loss_tensor, W, SNR_DIRECTORY, "_NO_LIMIT", batch_size=bs,
                                                           decimals=4, save_pdf=True, force_new=True)

    # batch_size_array = np.arange(1, batch_size)
    # seconds = batch_size_array * 0.33 * W / fs
    # RLTC.plot_errs(np.mean(SNRs_EERs, axis=1), seconds, passed_SNRs, SNR_DIRECTORY, "SNR_ERRs", title="EER per SNR",
    #                savePdf=True, plot_mean=False)
    #
    # file = np.load(SNR_DIRECTORY + "/" + "RAW ECG" + "_EER.npz")
    # EERs, thresholds = file["EERs"], file["thresholds"]

    # for i in [60, 120]:
    #     RLTC.identify_biosignals(loss_tensor, s_models, i)#, thresholds[i])