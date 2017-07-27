import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import scipy.interpolate as intr
# from DeepLibphys.utils.functions.common import data
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from DeepLibphys.sandbox.DNN_Single_SNR_DEV import calculate_min_windows_loss as CML

SNR_DIRECTORY = "../data/validation/May_DNN_SNR_FANTASIA_1"


def plot_errs(EERs, time, labels, title="", file="_iterations", savePdf=False, plot_mean=True):
    cmap = matplotlib.cm.get_cmap('rainbow')
    fig = plt.figure("fig", figsize=(900 / 96, 600 / 96), dpi=96)
    ax = plt.subplot2grid((1, 10), (0, 0), colspan=8)
    for i, EER in zip(range(len(EERs)), EERs):
        ax.plot(time, EER, '.', color=cmap(i / len(EERs)), alpha=0.2)
        ax.plot(time, EER, color=cmap(i / len(EERs)), alpha=0.2, label=labels[i])

    if plot_mean:
        mean_EERs = np.mean(EERs, axis=0).squeeze()
        index_min = np.argmin(mean_EERs)
        ax.plot(time, mean_EERs, 'b.', alpha=0.5)
        ax.plot(time, mean_EERs, 'b-', alpha=0.5, label="Mean")
        ax.plot(time[index_min], np.min(mean_EERs), 'ro', alpha=0.6)
    else:
        for i, EER in zip(range(len(EERs)), EERs):
            index_min = np.argmin(EER)
            ax.plot(time[index_min], np.min(EER), 'o', color=cmap(i / len(EERs)), alpha=0.6)

    indexi = 0.6 * np.max(time)
    indexj = 0.5 * (np.max(EER) - np.min(EER))

    step = (np.max(EER) - np.min(EER))
    if plot_mean:
        ax.annotate("EER MIN MEAN = {0:.4f}".format(mean_EERs[index_min]),
                     xy=(indexi, indexj))

        # plt.annotate("EER MIN STD = {0:.4f}".format(np.std(EER)),
        #              xy=(indexi, indexj - step))
        #
        # plt.annotate("TIME FOR MIN MEAN = {0:.4f}".format(time[index_min]),
        #              xy=(indexi, indexj - step * 2))
        #
        # plt.annotate("TIME FOR MIN STD = {0:.4f}".format(mean_EERs[index_min]),
        #              xy=(indexi, indexj - step * 3))
    else:
        for i, EER in zip(range(len(EERs)), EERs):
            index_min = np.argmin(EER)
            ax.annotate("EER MIN = {0:.3f}%".format(EER[index_min]*100),
                         xy=(time[index_min], EER[index_min] + 0.01), color=cmap(i / len(EERs)), ha='center')

    # plt.legend()
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)

    # if savePdf:
    print("Saving img_2/EER{0}.pdf".format(file))
    dir_name = SNR_DIRECTORY+"/img_2"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    pdf = PdfPages(dir_name + "/EER{0}.pdf".format(file))
    plt.savefig(dir_name + "/EER{0}.eps".format(file), format='eps', dpi=100)
    pdf.savefig(fig)
    plt.clf()
    pdf.close()
    # else:
    #     plt.show()
    return plt


def check_cyb():
    DATA_DIRECTORY = '../data/validation/CYBHI'
    loss_filename = DATA_DIRECTORY + "/LOSS_FOR_iteration_0"

    loss_tensor = np.load(loss_filename + ".npz")["loss_tensor"]
    labels = ["ECG {0}".format(i+1) for i in range(len(loss_tensor[0, :, 0]))]

    loss_temp_tensor = CML(loss_tensor, 2)
    step = 40
    scale = 1/step

    N = 15
    for m in range(N):
        print("Figure {0}".format(m))
        plt.figure()
        plt.title("Model ECG " + str(m+1))
        loss = np.zeros((N, step))
        for w in range(step):
            loss[:, w] = loss_temp_tensor[m, :, w]-np.min(loss_temp_tensor[m, :, w])
            loss[:, w] = loss[:, w]/np.max(loss)
        cmap = matplotlib.cm.get_cmap('rainbow')

        colors = [cmap(i / N) for i in range(N)]

        for i in range(N):
            plt.bar(np.ones(step)*i, loss[i, :] , color=cmap(i / N), alpha=scale)
        plt.xticks(range(len(loss_temp_tensor[:, 0])), labels)
        # plt.figure()
        sumi = np.sum(loss_temp_tensor[m, :, :], axis=1)
        f = intr.interp1d(np.arange(len(sumi)), sumi)
        xxx = f(np.arange(0, len(sumi)-1, 1/100))
        plt.plot(np.arange(0, len(sumi)-1, 1/100), xxx / np.max(xxx))
        plt.show()


check_cyb()
    # for filename in filenames:
    #     print("Printing: {0}".format(filename))
    #     all_data = np.load(filename)["all_data"]
    #
    #     # print(all_data[1]['EERs'])
    #     for score in all_data[1]['scores']:
    #         for s in score:
    #             print(s)
    # all_data = {
    #     "EERs": EERs,
    #     "iteration": iteration,
    #     "batch_size": batch_size,
    #     "scores": scores,
    #     "roc1": roc1,
    #     "roc2": roc2,
    #     "best": np.argmin(np.mean(EERs, axis=0)),
    #     "theshold": thresholds
    # }
