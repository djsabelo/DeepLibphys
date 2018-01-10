import numpy as np
from DeepLibphys.utils.functions.signal2model import *

def get_cybhi_database_info(n=1):
    signal_dim = 256
    hidden_dim = 256
    window_size = 512
    W = 512

    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format('NEW', window_size)
    signals = np.load("../data/processed/CYBHi/cybhi_signals.npz")["signals"]
    signals = signals[list(range(12))+list(range(13, len(signals)))]
    all_models_info = []

    for s, signal_data in enumerate(signals):
        if n == 1:
            name = 'ecg_cybhi_' + signal_data.name
        elif n == 2:
            name = 'ecg_cybhi_2_' + signal_data.name
        else:
            return []

        all_models_info.append(ModelInfo(Sd=signal_dim, Hd=hidden_dim, dataset_name=name, directory=signal_directory,
                                         DS=-5, t=-5, W=W, name="CYBHi {0}".format(s)))

    return all_models_info


def get_mit_variables(param):
    RAW_SIGNAL_DIRECTORY = '/media/belo/Storage/owncloud/Research Projects/DeepLibphys/Signals/'
    if param == "arr":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia', '../data/processed/biometry_mit[256].npz', 'ecg_mit_arrythmia_'
    if param == "sinus":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Sinus', '../data/processed/biometry_mit_sinus[256].npz', 'ecg_mit_sinus_'
    if param == "long":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Long-Term', '../data/processed/biometry_mit_long_term[256].npz', \
                'ecg_mit_long_term_'

    return None


def get_mit_database_info():
    batch_size = 256
    window_size = 512
    loss_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

    all_models_info = []
    s_i = 1
    for s_index, db_name in enumerate(["arr", "sinus", "long"]):
        mit_dir, processed_data_path, core_name = get_mit_variables(db_name)
        signals_aux = np.load(processed_data_path)['signals']
        all_models_info += [ModelInfo(Hd=256, Sd=256, dataset_name=core_name + str(i),
                                      name="MIT " + str(s_i + (i)),
                                      directory=loss_directory)
                            for i in range(0, np.shape(signals_aux)[0])]
        s_i += np.shape(signals_aux)[0]

        return all_models_info




signal_models = [ModelInfo(64, 256, "eeg_all", "EEG_Attention[1000.256]",
                           -5, -5, 256, "EEG"),
                 ModelInfo(64, 128,"provadeesforco_emg", "EMG",
                           480, 2000, 256, "EMG"),
                 ModelInfo(64, 256, "ecg_1", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 1"),
                 ModelInfo(64, 256, "ecg_2", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 2"),
                 ModelInfo(64, 256, "ecg_3", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 3"),
                 ModelInfo(64, 256, "ecg_4", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 4"),
                 ModelInfo(64, 256, "ecg_5", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 5"),
                 ModelInfo(64, 256, "ecg_6", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 6"),
                 ModelInfo(64, 256, "ecg_7", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 7"),
                 ModelInfo(64, 256, "ecg_8", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 8"),
                 ModelInfo(64, 256, "ecg_9", "ECGs_FANTASIA_[256.256]",
                           -5, -5, 256, "ECG 9")
                 ]

signal_generic_models = [ModelInfo(64, 256, "eeg_all", "EEG_Attention[1000.256]",
                                   -5, -5, 256, "EEG"),
                         ModelInfo(64, 128, "provadeesforco_emg", "EMG",
                                   480, 2000, 256, "EMG"),
                         ModelInfo(64, 256, "ecg_all_fantasia", "ECGs_FANTASIA_[256.256]",
                                   -5, -5, 256, "ECG ALL"),
                         ]

ecg_models = [
    ModelInfo(64, 256, "ecg_1", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 1"),
    ModelInfo(64, 256, "ecg_2", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 2"),
    ModelInfo(64, 256, "ecg_3", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 3"),
    ModelInfo(64, 256, "ecg_4", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 4"),
    ModelInfo(64, 256, "ecg_5", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 5"),
    ModelInfo(64, 256, "ecg_6", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 6"),
    ModelInfo(64, 256, "ecg_7", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 7"),
    ModelInfo(64, 256, "ecg_8", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 8"),
    ModelInfo(64, 256, "ecg_9", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 9"),
    ModelInfo(64, 256, "ecg_10", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 10"),
    ModelInfo(64, 256, "ecg_11", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 11"),
    ModelInfo(64, 256, "ecg_12", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 12"),
    ModelInfo(64, 256, "ecg_13", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 13"),
    ModelInfo(64, 256, "ecg_14", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 14"),
    ModelInfo(64, 256, "ecg_15", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 15"),
    ModelInfo(64, 256, "ecg_16", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 16"),
    ModelInfo(64, 256, "ecg_17", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 17"),
    ModelInfo(64, 256, "ecg_18", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 18"),
    ModelInfo(64, 256, "ecg_19", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 19"),
    ModelInfo(64, 256, "ecg_all_fantasia", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG ALL"),
    ModelInfo(64, 256, "ecg_all_old", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG OLD"),
    ModelInfo(64, 256, "ecg_all_young", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG YOUNG"),
                 ]

ecg_models = [
    ModelInfo(64, 256, "ecg_1", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 1"),
    ModelInfo(64, 256, "ecg_2", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 2"),
    ModelInfo(64, 256, "ecg_3", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 3"),
    ModelInfo(64, 256, "ecg_4", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 4"),
    ModelInfo(64, 256, "ecg_5", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 5"),
    ModelInfo(64, 256, "ecg_6", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 6"),
    ModelInfo(64, 256, "ecg_7", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 7"),
    ModelInfo(64, 256, "ecg_8", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 8"),
    ModelInfo(64, 256, "ecg_9", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 9"),
    ModelInfo(64, 256, "ecg_10", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 10"),
    ModelInfo(64, 256, "ecg_11", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 11"),
    ModelInfo(64, 256, "ecg_12", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 12"),
    ModelInfo(64, 256, "ecg_13", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 13"),
    ModelInfo(64, 256, "ecg_14", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 14"),
    ModelInfo(64, 256, "ecg_15", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 15"),
    ModelInfo(64, 256, "ecg_16", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 16"),
    ModelInfo(64, 256, "ecg_17", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 17"),
    ModelInfo(64, 256, "ecg_18", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 18"),
    ModelInfo(64, 256, "ecg_19", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG 19"),
    ModelInfo(64, 256, "ecg_all_fantasia", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG ALL"),
    ModelInfo(64, 256, "ecg_all_old", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG OLD"),
    ModelInfo(64, 256, "ecg_all_young", "ECGs_FANTASIA_[256.256]",
              -5, -5, 256, "ECG YOUNG"),
                 ]


eeg_models = [ModelInfo(dataset_name="eeg_attention"+str(i),
                        name="EEG "+str(i+1),
                        directory="EEGs_SEQ_ATTENTION_[128.512]")
              for i in range(6)]

biometry_x_models = [ModelInfo(Hd=128, dataset_name="biometric_acc_x_"+str(i),
                        name="BIO X "+str(i+1),
                        directory="BIO_ACC_[256.64]")
              for i in range(30)]

biometry_y_models = [ModelInfo(Hd=128, dataset_name="biometric_acc_y_"+str(i),
                        name="BIO Y "+str(i+1),
                        directory="BIO_ACC_[256.64]")
              for i in range(30)]

biometry_z_models = [ModelInfo(Hd=128, dataset_name="biometric_acc_z_"+str(i),
                        name="BIO Z "+str(i+1),
                        directory="BIO_ACC_[256.64]")
              for i in range(30)]

ecg_biometry_models = [ModelInfo(Hd=256, dataset_name="bio_ecg_"+str(i),
                        name="BIO ECG "+str(i+1),
                        directory="BIOMETRIC_ECGs_[128.256]")
              for i in range(19)]

ecg_clean_models = [ModelInfo(Hd=256, dataset_name="clean_ecg"+str(i+1),
                        name="ECG "+str(i+1),
                        directory="CLEAN_ECG_BIOMETRY[128.256]")
              for i in range(0,20)]

ecg_64_models = [ModelInfo(Hd=256, Sd=64, dataset_name="ecg_"+str(i+1),
                        name="ECG "+str(i+1),
                        directory="BIOMETRY[128.512]")
              for i in range(0, 20)]

emg_64_models = [ModelInfo(Hd=512, Sd=64, W=512, dataset_name="emg_"+str(i),
                        name="EMG "+str(i),
                        directory="SYNTHESIS[256.512]")
              for i in (list(range(1, 15)))]

emg_64_32_models = [ModelInfo(Hd=512, Sd=64, W=512, dataset_name="emg_"+str(i),
                        name="EMG "+str(i),
                        directory="Synthesis")
              for i in (list(range(1, 15)))]

emg_64_1024_models = [ModelInfo(Hd=512, Sd=64, dataset_name="emg_"+str(i),
                        name="EMG "+str(i),
                        directory="SYNTHESIS[128.1024]")
              for i in (list(range(1, 15)))]

resp_64_models = [ModelInfo(Hd=256, Sd=64, dataset_name="resp_"+str(i+1),
                        name="RESP "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 20)]

ecg_1024_clean_models = [ModelInfo(Hd=256, dataset_name="clean_ecg"+str(i+1),
                        name="ECG "+str(i+1),
                        directory="CLEAN_ECG_BIOMETRY[256.1024]")
              for i in range(0, 20)]

ecg_1024_256_RAW = [ModelInfo(Hd=256, Sd=256, W=1024, dataset_name="ecg_"+str(i),
                              name="ECG "+str(i),
                              directory="BIOMETRY[256.1024]")
                    for i in range(1, 41)]

ecg_1024_256_SNR_12 = [ModelInfo(Hd=256, Sd=256, W=1024, dataset_name="ecg_"+str(i+1)+"_SNR_12",
                        name="ECG "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 40)]

ecg_1024_256_SNR_11 = [ModelInfo(Hd=256, Sd=256, W=1024, dataset_name="ecg_"+str(i+1)+"_SNR_11",
                        name="ECG "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 40)]

ecg_1024_256_SNR_10 = [ModelInfo(Hd=256, Sd=256, W=1024, dataset_name="ecg_"+str(i+1)+"_SNR_10",
                        name="ECG "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 40)]

ecg_1024_256_SNR_9 = [ModelInfo(Hd=256, Sd=256, W=1024, dataset_name="ecg_"+str(i+1)+"_SNR_9",
                        name="ECG "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 40)]

ecg_1024_256_SNR_8 = [ModelInfo(Hd=256, Sd=256, W=1024, dataset_name="ecg_"+str(i+1)+"_SNR_8",
                        name="ECG "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 40)]

ecg_1024_256_SNR_7 = [ModelInfo(Hd=256, Sd=256, W=1024, dataset_name="ecg_"+str(i+1)+"_SNR_7",
                        name="ECG "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 40)]

ecg_1024_256_SNR_6 = [ModelInfo(Hd=256, Sd=256, W=1024, dataset_name="ecg_"+str(i+1)+"_SNR_6",
                        name="ECG "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 40)]


ecg_noisy_models = [ModelInfo(Hd=256, dataset_name="noisy_ecg_"+str(i),
                        name="ECG "+str(i),
                        directory="NOISE_ECGs_[150.256]")
              for i in range(1, 21)]

ecg_SNR_12 = [ModelInfo(Hd=256, dataset_name="ecg_SNR_12"+str(i),
                        name="ECG "+str(i),
                        directory="ECG_BIOMETRY[256.256]")
              for i in range(1, 41)]

ecg_SNR_9 = [ModelInfo(Hd=256, dataset_name="ecg_SNR_9"+str(i),
                        name="ECG "+str(i),
                        directory="ECG_BIOMETRY[256.256]")
              for i in range(1, 41)]

ecg_SNR_8 = [ModelInfo(Hd=256, dataset_name="ecg_SNR_8"+str(i),
                        name="ECG "+str(i),
                        directory="ECG_BIOMETRY[256.256]")
              for i in range(1, 41)]

ecg_SNR_7 = [ModelInfo(Hd=256, dataset_name="ecg_SNR_7"+str(i),
                        name="ECG "+str(i),
                        directory="ECG_BIOMETRY[256.256]")
              for i in range(1, 41)]

ecg_SNR_6 = [ModelInfo(Hd=256, dataset_name="ecg_SNR_6"+str(i),
                        name="ECG "+str(i),
                        directory="ECG_BIOMETRY[256.256]")
              for i in range(1, 41)]

ecg_SNR_11 = [ModelInfo(Hd=256, dataset_name="ecg_SNR_11"+str(i),
                        name="ECG "+str(i),
                        directory="ECG_BIOMETRY[256.256]")
              for i in range(1, 41)]

ecg_SNR_10 = [ModelInfo(Hd=256, dataset_name="ecg_SNR_10"+str(i),
                        name="ECG "+str(i),
                        directory="ECG_BIOMETRY[256.256]")
              for i in range(1, 4)]

mit_256 = [ModelInfo(Hd=256, Sd=256, dataset_name="ecg_mit_"+str(i),
                        name="MIT "+str(i+1),
                        directory="ECG_BIOMETRY[128.1024]")
              for i in range(0, 45)]

mit_1024 = get_mit_database_info()

cybhi_256 = [ModelInfo(Hd=256, dataset_name="ecg_cybhi_"+str(i),
                        name="CYBHi "+str(i+1),
                        directory="ECG_BIOMETRY[112.256]")
              for i in range(0, 58)]

cybhi_1024 = [ModelInfo(Sd=256, Hd=256, dataset_name="ecg_cybhi_"+str(i),
                        name="CYBHi "+str(i+1),
                        directory="ECG_BIOMETRY[112.1024]")
              for i in range(0, 15)]

cybhi_512_M1 = get_cybhi_database_info()

cybhi_512_M2 = get_cybhi_database_info(2)

web_group_x_models = [ModelInfo(Hd=256, dataset_name=("web_group_x["+str(i)+"]"),
                        name="WEB GROUP X "+str(i),
                        directory="WEB_[64.256]")
              for i in range(4)]

web_group_y_models = [ModelInfo(Hd=256, dataset_name="web_group_y["+str(i)+"]",
                        name="WEB GROUP Y "+str(i),
                        directory="WEB_[64.256]")
              for i in range(4)]


web_group_x_models_128 = [ModelInfo(Hd=128, dataset_name=("web_group_x["+str(i)+"]"),
                        name="WEB GROUP "+str(i),
                        directory="WEB_[128.64]",
                        Sd = 128)
              for i in range(4)]

web_group_y_models_128 = [ModelInfo(Hd=128, dataset_name="web_group_y["+str(i)+"]",
                        name="WEB GROUP "+str(i),
                        directory="WEB_[128.64]",
                        Sd = 128)
              for i in range(4)]


gsr_models = [ModelInfo(dataset_name="driver_gsr"+str(i),
                        name="GSR X "+str(i+1),
                        directory="DRIVER_GSR_[256.256]")
              for i in range(14)]

rr_models = [ModelInfo(dataset_name='day_hrv_rr_{0}'.format(i),
                        name="GROUP "+str(i),
                        directory='DAY_HRV_HF_[256.2397]')
              for i in range(4)]

rr_256_models = [ModelInfo(dataset_name='day_rr_10Hz_{0}'.format(i+1),
                        name="RR "+str(i+1),
                        directory="DAY_RR_[128.256]")
              for i in range(4)]

rr_128_models = [ModelInfo(Hd=128, dataset_name='day_rr_10Hz_[20]{0}'.format(i+1),
                        name="RR "+str(i+1),
                        directory="DAY_RR_[128.256]")
              for i in range(4)]

lf_models = [ModelInfo(dataset_name='day_lf_10Hz_{0}'.format(i+1),
                        name="LF "+str(i+1),
                        directory="DAY_RR_[128.64]")
              for i in range(4)]

vlf_models = [ModelInfo(dataset_name='day_vlf_10Hz_{0}'.format(i+1),
                        name="VLF "+str(i+1),
                        directory="DAY_RR_[128.64]")
              for i in range(4)]

hf_models = [ModelInfo(dataset_name='day_vlf_10Hz_{0}'.format(i+1),
                        name="VLF "+str(i+1),
                        directory="DAY_RR_[128.64]")
              for i in range(4)]

signal_tests = [SignalInfo("emg", 'EMG Cycling', 0, 300000, "EMG BIKE"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 1, 900000, "ECG 1"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 2, 900000, "ECG 2"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 3, 900000, "ECG 3"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 4, 900000, "ECG 4"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 5, 900000, "ECG 5"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 6, 900000, "ECG 6"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 7, 900000, "ECG 7"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 8, 900000, "ECG 8"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 9, 900000, "ECG 9"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 10, 900000, "ECG 10"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 11, 900000, "ECG 11"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 12, 900000, "ECG 12"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 13, 900000, "ECG 13"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 14, 900000, "ECG 14"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 15, 900000, "ECG 15"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 16, 900000, "ECG 16"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 17, 900000, "ECG 17"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 18, 900000, "ECG 18"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 19, 900000, "ECG 19"),
                SignalInfo("gsr", 'DRIVER_GSR', 1, 55800, "GSR 1"),
                SignalInfo("gsr", 'DRIVER_GSR', 3, 55800, "GSR 2"),
                SignalInfo("gsr", 'DRIVER_GSR', 4, 55800, "GSR 3"),
                SignalInfo("gsr", 'DRIVER_GSR', 5, 55800, "GSR 4"),
                SignalInfo("gsr", 'DRIVER_GSR', 6, 55800, "GSR 5"),
                SignalInfo("gsr", 'DRIVER_GSR', 7, 55800, "GSR 6"),
                SignalInfo("gsr", 'DRIVER_GSR', 8, 55800, "GSR 7"),
                SignalInfo("gsr", 'DRIVER_GSR', 9, 55800, "GSR 8"),
                SignalInfo("gsr", 'DRIVER_GSR', 10, 55800, "GSR 9"),
                SignalInfo("gsr", 'DRIVER_GSR', 11, 55800, "GSR 10"),
                SignalInfo("gsr", 'DRIVER_GSR', 12, 55800, "GSR 11"),
                SignalInfo("gsr", 'DRIVER_GSR', 14, 55800, "GSR 12"),
                SignalInfo("gsr", 'DRIVER_GSR', 15, 55800, "GSR 13"),
                SignalInfo("gsr", 'DRIVER_GSR', 16, 55800, "GSR 14"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 1, 900000, "RESP 1"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 2, 900000, "RESP 2"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 3, 900000, "RESP 3"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 4, 900000, "RESP 4"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 5, 900000, "RESP 5"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 6, 900000, "RESP 6"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 7, 900000, "RESP 7"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 8, 900000, "RESP 8"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 9, 900000, "RESP 9"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 10, 900000, "RESP 10"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 11, 900000, "RESP 11"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 12, 900000, "RESP 12"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 13, 900000, "RESP 13"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 14, 900000, "RESP 14"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 15, 900000, "RESP 15"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 16, 900000, "RESP 16"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 17, 900000, "RESP 17"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 18, 900000, "RESP 18"),
                SignalInfo("resp", 'Fantasia/RESP/mat/', 19, 900000, "RESP 19"),
                SignalInfo("eeg", 'EEG_Attention', 0, 429348, "EEG ATT 0"),
                SignalInfo("eeg", 'EEG_Attention', 1, 429348, "EEG ATT 1"),
                SignalInfo("eeg", 'EEG_Attention', 2, 429348, "EEG ATT 2"),
                SignalInfo("eeg", 'EEG_Attention', 3, 429348, "EEG ATT 3"),
                SignalInfo("eeg", 'EEG_Attention', 4, 429348, "EEG ATT 4"),
                SignalInfo("eeg", 'EEG_Attention', 5, 429348, "EEG ATT 5"),
                ] + [SignalInfo("biometric", 'Biometry', i, 90000, "BIO "+str(i)) for i in range(300)]

fantasia_ecgs = [SignalInfo("ecg", 'Fantasia/ECG/mat/', 1, 900000, "ECG 1"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 2, 900000, "ECG 2"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 3, 900000, "ECG 3"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 4, 900000, "ECG 4"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 5, 900000, "ECG 5"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 6, 900000, "ECG 6"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 7, 900000, "ECG 7"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 8, 900000, "ECG 8"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 9, 900000, "ECG 9"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 10, 900000, "ECG 10"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 11, 900000, "ECG 11"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 12, 900000, "ECG 12"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 13, 900000, "ECG 13"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 14, 900000, "ECG 14"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 15, 900000, "ECG 15"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 16, 900000, "ECG 16"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 17, 900000, "ECG 17"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 18, 900000, "ECG 18"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 19, 900000, "ECG 19"),
                SignalInfo("ecg", 'Fantasia/ECG/mat/', 20, 900000, "ECG 20")]

fantasia_resp = [SignalInfo("ecg", 'Fantasia/RESP/mat/', 1, 900000, "ECG 1")]

ecg_noisy_signals = [[SignalInfo("ecg noise", 'Fantasia/Noise_ECG/', i, 900000, "ECG-NOISE-"+str(j)+"-"+str(i))
                      for i in range(1,20)] for j in range(4)]

day_rr = [SignalInfo("day", 'Day_HRV/RR/', 0, -1, "RR", 'ah_r-r_10Hz')]
day_hf = [SignalInfo("day hrv", 'Day_HRV/HRV/', 0, -1, "RR", 'hf_10Hz')]
day_lf = [SignalInfo("day hrv", 'Day_HRV/HRV/', 0, -1, "RR", 'lf_10Hz')]
day_vlf = [SignalInfo("day hrv", 'Day_HRV/HRV/', 0, -1, "RR", 'vlf_10Hz')]
