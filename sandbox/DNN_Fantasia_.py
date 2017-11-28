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
    N_Windows = 1024
    W = 1024
    signal_dim = 256
    hidden_dim = 256
    batch_size = 256
    window_size = 1024
    fs = 250

    signal_directory = 'BIOMETRY[{0}.{1}]'.format(batch_size, window_size)
    dir_name = TRAINED_DATA_DIRECTORY + signal_directory

    # signals = np.load("../data/processed/FANTASIA_ECG[256].npz")['x_train']+np.load("../data/processed/FANTASIA_ECG[256].npz")['x_train']
    signals = load_noisy_fantasia_signals(SNRx=[9, 12])

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


    filename = SNR_DIRECTORY + "/LOSS_TENSOR_test.npz"
    models = [db.ecg_1024_256_RAW]
    filenames = [filename]
    # SNRs = ["RAW"] + [str(i) for i in range(12,7,-1)]
    # filenames = [SNR_DIRECTORY + "/LOSS_TENSOR_SNR_{0}.npz".format(SNR) for SNR in SNRs]
    # models = [db.ecg_1024_256_RAW, db.ecg_1024_256_SNR_12, db.ecg_1024_256_SNR_11, db.ecg_1024_256_SNR_10,
    #           db.ecg_1024_256_SNR_9]

    loss_quaternion = []
    passed_SNRs = []
    i = 0
    for filename, model, signal_batch in zip(filenames, models, signals):
        try:
            signalx = [signal[int(len(signal) * 0.33):] for signal in signal_batch]
            loss_tensor = RLTC.get_or_save_loss_tensor(full_path=filename, force_new=False, N_Windows=N_Windows, W=W,
                                                       models=model, test_signals=signalx, mean_tol=0.8,
                                                       overlap=0.33, mini_batch=256, std_tol=0.05)
            loss_quaternion.append(loss_tensor)
            passed_SNRs.append(SNRs[i])
        except:
            print("Could Not run {0}".format(filename))

        i += 1

    # x = list(range(37))

    # loss_tensor = loss_tensor / (np.max(loss_tensor, axis=0)-np.min(loss_tensor, axis=0))
    # for i in [60, 120]:
    #     RLTC.identify_biosignals(loss_tensor, s_models, i)
    # for i in [1, 15, 60]:
    # #     RLTC.identify_biosignals(loss_tensor, s_models,  batch_size=i)
    #     temp_loss_tensor = RLTC.calculate_batch_min_loss(loss_tensor, i)
    #     eer, thresholds_out, candidate_index = RLTC.calculate_smart_roc(temp_loss_tensor, decimals=5)
    #
    # print("Number of windows: {0}".format(np.shape(loss_tensor)[2]))
    SNRs_EERs = []
    for loss_tensor in loss_quaternion:
        EERs, thresholds, batch_size_array = RLTC.process_eers(loss_tensor, W, SNR_DIRECTORY, "RAW ECG", batch_size=bs,
                                                               decimals=5, save_pdf=False)
    #     SNRs_EERs.append(EERs)
    #
    # SNRs_EERs = np.array(SNRs_EERs)
    # np.savez(SNR_DIRECTORY + "/SNR_EERs.npz", SNRs_EERs=SNRs_EERs)
    # SNRs_EERs = np.load(SNR_DIRECTORY + "/SNR_EERs.npz")["SNRs_EERs"]
    # batch_size_array = np.arange(1, batch_size)
    # seconds = batch_size_array * 0.33 * W / fs
    # RLTC.plot_errs(np.mean(SNRs_EERs, axis=1), seconds, passed_SNRs, SNR_DIRECTORY, "SNR_ERRs", title="EER per SNR",
    #                savePdf=True, plot_mean=False)
    #
    # file = np.load(SNR_DIRECTORY + "/" + "RAW ECG" + "_EER.npz")
    # EERs, thresholds = file["EERs"], file["thresholds"]

    # for i in [60, 120]:
    #     RLTC.identify_biosignals(loss_tensor, s_models, i)#, thresholds[i])