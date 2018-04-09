from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.signal2model import *
import DeepLibphys.models.LibphysMBGRU as GRU


def prepare_data(windows, signal2model, overlap=0.11, batch_percentage=1):
    x__matrix, y__matrix = [], []
    window_size, max_batch_size, mini_batch_size = \
        signal2model.window_size, signal2model.batch_size, signal2model.mini_batch_size
    reject = 0
    total = 0
    for w, window in enumerate(windows):
        if len(window) > window_size+1:
            for s_w in range(0, len(window) - window_size - 1, int(window_size*overlap)):
                small_window = np.round(window[s_w:s_w + window_size])
                # print(np.max(small_window) - np.min(small_window))
                if (np.max(small_window) - np.min(small_window)) \
                        > 0.7 * signal2model.signal_dim:
                    x__matrix.append(window[s_w:s_w + window_size])
                    y__matrix.append(window[s_w + 1:s_w + window_size + 1])
                else:
                    reject += 1

                total += 1

    x__matrix = np.array(x__matrix)
    y__matrix = np.array(y__matrix)

    max_index = int(np.shape(x__matrix)[0]*batch_percentage)
    batch_size = int(max_batch_size) if max_batch_size < max_index else \
        max_index - max_index % mini_batch_size

    indexes = np.random.permutation(int(max_index))[:batch_size]

    print("Windows of {0}: {1}; Rejected: {2} of {3}".format(signal2model.model_name, batch_size, reject, total))
    return x__matrix[indexes], y__matrix[indexes]


def manual_extraction(x_train, y_train):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    l, = plt.plot(x_train[0], lw=2)

    class BooleanSwitcher(object):
        indexes = []
        ind = 0

        def yes(self, event):
            if self.ind < len(x_train):
                self.indexes.append(self.ind)
                self.ind += 1
            if self.ind < len(x_train):
                l.set_ydata(x_train[self.ind])
                plt.draw()
            else:
                self.crop()
                plt.close()

        def no(self, event):
            self.ind += 1
            if self.ind < len(x_train):
                l.set_ydata(x_train[self.ind])
                plt.draw()
            else:
                self.crop()
                plt.close()

        def crop(self):
            c = len(self.indexes) % 16
            self.indexes = self.indexes[:(len(self.indexes) - c)]
    callback = BooleanSwitcher()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    by = Button(axnext, 'Yes')
    by.on_clicked(callback.yes)
    bn = Button(axprev, 'No')
    bn.on_clicked(callback.no)
    plt.show()
    return x_train[callback.indexes], y_train[callback.indexes]


GRU_DATA_DIRECTORY = "/media/belo/Storage/owncloud/Research Projects/DeepLibphys/Current Trained/"


def try_to_load(signal2model):
    dir_name = "ECG_BIOMETRY[128.1024]"
    search_dir = GRU_DATA_DIRECTORY + dir_name + '/'
    files = os.listdir(search_dir)
    try:
        files.index(get_file_tag(signal2model.model_name))
        print("Found!")
        return False
    except ValueError:
        for x in range(5000, 0, -250):
            try:
                files.index(get_file_tag(signal2model.model_name))
                model = GRU.LibphysMBGRU(signal2model)
                model.load(dir_name=dir_name, file_tag=model.get_file_tag(0, x))
                print("Loaded! epoch {0}".format(x))
                return model
            except ValueError:
                pass

    return GRU.LibphysMBGRU(signal2model)


# def try_to_load(model):
#     try:
#         model.load(dir_name="ECG_BIOMETRY[CYBHi]")
#         return
#     except:
#         for i in range(2500, 0, -250):
#             try:
#                 model.load(dir_name="ECG_BIOMETRY[CYBHi]", file_tag=model.get_file_tag(0, i))
#                 print("Loaded! epoch {0}".format(i))
#                 return
#             except:
#                 try:
#                     model.load(dir_name="ECG_BIOMETRY[64.512.8]")
#                 except:
#                     pass
#
#     model.load(dir_name="ECG_BIOMETRY[32.512]")
#     return


def start(moment, x):
    signal_dim = 256
    hidden_dim = 256
    mini_batch_size = 8
    batch_size = 256
    window_size = 1024
    save_interval = 250
    overlap = 0.055

    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)
    # signal_directory = 'ECG_BIOMETRY[{0}.{1}.8]'.format(batch_size, window_size)
    # signal_directory = 'ECG_BIOMETRY[CYBHi]'
    noise_removed_path = "Data/CYBHi/signals_long_v2.npz"
    fileDir = "Data/CYBHi/3rd"

    # moment = 1
    _signals = np.load(noise_removed_path)["signals"]
    names = np.array([signal.name for signal in _signals])
    if moment == 1:
        signals = np.array(extract_train_part([signal.train_windows for signal in _signals], 0.5))
        signals_y = np.array(extract_test_part([signal.train_windows for signal in _signals], 0.5))
    else:
        signals = np.array(extract_train_part([signal.test_windows for signal in _signals], 0.5))
        signals_y = np.array(extract_test_part([signal.test_windows for signal in _signals], 0.5))

    # [print(DATASET_DIRECTORY + 'GRU_ecg_cybhi_M{0}_{1}[256.256.-1.-5.-5].npz'.format(moment, name)) for name in names]
    # [print(name + ":" + str(os.path.isfile(DATASET_DIRECTORY + signal_directory + '/GRU_ecg_cybhi_M{0}_{1}[256.256.-1.-5.-5].npz'.format(moment, name)))) for name in names]
    #
    #
    # exit()
    step = 1
    indexes = np.arange(0, 63, step)
    if moment == 1:
        indexes = np.array([0, 1, 2, 3, 8, 9, 10, 12, 16, 17, 19, 20, 22, 26, 27, 28] +
                           [30, 32, 33, 34, 40, 41, 42, 43, 44, 45, 46, 48, 53, 54, 58, 60, 61, 62])
    # if x == 6:
    #     z = np.arange(60, 63)
    # x = 0
    # else:
    # z = np.arange(indexes[x], indexes[x+1])
    # z = np.arange(names.tolist().index("CF") , int(len(signals)/2))#, len(signals))
    # z = np.arange(60, 63)
    # z = np.arange(0, 4)
    z = [indexes[x]]
    print(str(x)+": "+str(list(z)))
    for s, signal, name in zip(np.arange(len(signals))[z], signals[z], names[z]):
        name = 'ecg_cybhi_M{0}_{1}'.format(moment, name)
        print(name)
        signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                    batch_size=batch_size,
                                    mini_batch_size=mini_batch_size, window_size=window_size,
                                    save_interval=save_interval, tolerance=1e-9, count_to_break_max=15)

        # x_train, y_train = prepare_test_data([signal], signal2model, overlap=overlap, mean_tol=0.8, std_tol=0.1)
        std_tol = 0.1
        mean_tol = 0.05
        plt.plot(signal)
        plt.figure()
        plt.plot(signals_y[s])
        plt.figure()
        x_train, y_train = prepare_several_special_data([signal], signal2model, overlap=overlap,
                                                        mean_tol=mean_tol, std_tol=std_tol)
        # x_train, y_train = manual_extraction(x_train, y_train)
        print("Final number of windows: {0}".format(len(x_train)))
        print("Compiling Model {0} for {1}".format(s, name))

        path = fileDir + "/Signals"
        full_path = path + "/{0}.pdf".format(signal2model.model_name)
        # if savePdf:
        print("Saving {0}.pdf".format(full_path))
        if not os.path.exists(path):
            os.makedirs(path)

        fig = plt.figure()
        pdf = PdfPages(full_path)
        for x in x_train:
            plt.plot(x)
            pdf.savefig(fig)
            plt.clf()
        pdf.close()
        # model = GRU.LibphysMBGRU(signal2model)

        n_runs = 0
        running_ok = False
        while not running_ok:
            n_runs += 1

            # if n_runs < 3:
            #     model = try_to_load(signal2model)
            #     if model is False:
            #         break
            # else:
            # model = GRU.LibphysMBGRU(signal2model)
            # signal2model2 = Signal2Model("generic_ecg", signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
            #                             batch_size=batch_size,
            #                             mini_batch_size=mini_batch_size, window_size=window_size,
            #                             save_interval=save_interval, number_of_epochs=2000, lower_error=1e-9,
            #                             count_to_break_max=15, learning_rate_val=0.01)

            # model = GRU.LibphysMBGRU(signal2model2)
            model = GRU.LibphysMBGRU(signal2model)
            if name == "ARA":
                model.load(file_tag=model.get_file_tag(0, 250), dir_name=signal2model.signal_directory)
            # history_of_indexes = {}
            # model.load(dir_name="ECG_CLUSTER[128.1024]", file_tag=model.get_file_tag(999, 0))
            # model.model_name = signal2model.model_name
            print("Initiating training... ")
            running_ok = model.train_model(x_train, y_train, signal2model)

            if not running_ok:
                model = GRU.LibphysMBGRU(signal2model)
            else:
                model.save(dir_name=signal_directory)

start(1, 33)
# start(int(sys.argv[1]), int(sys.argv[2]))
# print("Ended process " + sys.argv[1])

# print("Training {0} of {1}".format(len(train_dates) - len(noisy_data), len(train_dates)))


# for i, signal, person_name in zip(range(z, len(train_signals)), train_signals[z:], train_names[z:]):
#     if noisy_data.count(person_name) == 0:
#         name = 'ecg_cybhi_' + person_name
#         signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
#                                     batch_size=batch_size,
#                                     mini_batch_size=mini_batch_size, window_size=window_size,
#                                     save_interval=save_interval)
#
#         running_ok = False
#         while not running_ok:
#             print("Compiling Model {0} for {1} and date {2}".format(name, person_name, train_dates[i]))
#             model = GRU.LibphysMBGRU(signal2model)
#             print("Initiating training... ")
#             n_for_each = batch_size if batch_size % signal2model.mini_batch_size == 0 \
#                 else signal2model.mini_batch_size * int(batch_size / signal2model.mini_batch_size)
#
#
#             x_train = signal[indexes, :-1]
#             y_train = signal[indexes, 1:]
#             running_ok = model.train_model(x_train, y_train, signal2model)
#             # running_ok = model.train(signal, signal2model, train_ratio=1)
#         model.save(dir_name=signal_directory)
