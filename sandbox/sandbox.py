# from DeepLibphys.utils.functions.common import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

import scipy.io as sio


xxx = "/media/belo/Storage/owncloud/Research Projects/DeepLibphys/Signals/FMH/#5_EMG.txt"
x = np.loadtxt(xxx)
plt.plot(x[:, 4], "r")
plt.show()
#
# noise_removed_path = "../data/processed/FANTASIA_ECG[256].npz"
# save_path = "../data/processed/ALL/"
# test_path = "../data/processed/ALL/test/"
#
# o_signals = np.load(noise_removed_path)["x_train"]
# names = ["fantasia_"+str(i) for i, s in enumerate(o_signals)]
#
# [np.savez(save_path + core_name+".npz", signal=signal, core_name=core_name) for signal, core_name in zip(extract_train_part(o_signals, 0.5), names)]
# [np.savez(test_path + core_name+".npz", signal=signal, core_name=core_name) for signal, core_name in zip(extract_test_part(o_signals, 0.5), names)]

# from DeepLibphys.utils.functions.common import *
# import numpy as np
#
#
# def get_variables(param):
#     if param == "arr":
#         return RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia', '../data/processed/biometry_mit[256].npz', 'ecg_mit_arrythmia_'
#     if param == "sinus":
#         return RAW_SIGNAL_DIRECTORY + 'MIT-Sinus', '../data/processed/biometry_mit_sinus[256].npz', 'ecg_mit_sinus_'
#     if param == "long":
#         return RAW_SIGNAL_DIRECTORY + 'MIT-Long-Term', '../data/processed/biometry_mit_long_term[256].npz', \
#                 'ecg_mit_long_term_'
#
#     return None
#
#
# def get_processing_variables():
#     a_dir, apdp, core_name0 = get_variables('arr')
#     s_dir, spdp, core_name1 = get_variables('sinus')
#     l_dir, lpdp, core_name2 = get_variables('long')
#
#     # full_paths = os.listdir(mit_dir)
#     filenames = [a_dir + "/" + full_path for full_path in os.listdir(a_dir) if full_path.endswith(".mat")] + \
#                 [s_dir + "/" + full_path for full_path in os.listdir(s_dir) if full_path.endswith(".mat")] + \
#                 [l_dir + "/" + full_path for full_path in os.listdir(l_dir) if full_path.endswith(".mat")]
#
#     Ns = [len(np.load(apdp)["signals"]), len(np.load(spdp)["signals"].tolist()), len(np.load(lpdp)["signals"].tolist())]
#
#     core_names = []
#     paths = []
#     new_paths = []
#     test_paths = []
#     for x, n in enumerate(Ns):
#         for i in range(n):
#             core_names.append(eval("core_name{0}".format(x)) + str(i))
#             paths.append('../data/processed/MIT/{0}[256].npz'.format(core_names[-1]))
#             new_paths.append('../data/processed/ALL/{0}.npz'.format(core_names[-1]))
#             test_paths.append('../data/processed/ALL/test/{0}.npz'.format(core_names[-1]))
#
#     return paths, new_paths, test_paths, core_names
#
#
#
# paths, new_paths, test_paths, core_names = get_processing_variables()
#
# for path, new, test, core in zip(paths, new_paths, test_paths, core_names):
#     s = np.load(path)["signal"]
#     np.savez(new, signal=extract_train_part([s], 0.5), core_name=core)
#     np.savez(test, signal=extract_test_part([s], 0.5), core_name=core)
#
#
# # [np.savez(save_path + core_name+".npz", signal=signal, core_name=core_name) for signal, core_name in zip(signals, names)]