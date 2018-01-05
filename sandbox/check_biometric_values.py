import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import scipy.interpolate as intr
# from DeepLibphys.utils.functions.common import data
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from DeepLibphys.sandbox.DNN_Single_SNR_DEV import calculate_min_windows_loss as CML
import DeepLibphys.classification.RLTC as RLTC

SNR_DIRECTORY = "../data/validation/May_DNN_SNR_FANTASIA_1"

from matplotlib.colors import LinearSegmentedColormap as lsc


def cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: (lambda x: x[0], cdict[key]) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_list = np.unique(sum(step_dict.values(), []))
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(map(function, y0))
    y1 = np.array(map(function, y1))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    return lsc(name, step_dict, N=N, gamma=gamma)

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
    print("Saving img/EER{0}.pdf".format(file))
    dir_name = SNR_DIRECTORY+"/img"
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
    # loss_filename = DATA_DIRECTORY + "/LOSS_FOR_iteration_0"

    loss_filename = DATA_DIRECTORY + "/LOSS_CYBHi[64.512]"
    loss_tensor = np.load(loss_filename + ".npz")["loss_tensor"]

    check_model_loss(loss_tensor)


def check_fantasia(indexes=None):
    DATA_DIRECTORY = "../data/validation/Sep_FANTASIA_512"
    loss_tensor = np.load(DATA_DIRECTORY + "/LOSS_FOR_SNR_RAW_iteration_0" + ".npz")["loss_tensor"]

    if indexes is None:
        indexes = np.arange(np.shape(loss_tensor)[0])

    check_model_loss(loss_tensor, indexes)


def check_model_loss(loss_tensor, indexes, b=5):
    labels = ["ECG {0}".format(i+1) for i in range(len(loss_tensor[0, :, 0]))]

    loss_temp_tensor = RLTC.normalize_tensor(RLTC.calculate_batch_min_loss(loss_tensor, b), 0.001)
    step = np.shape(loss_temp_tensor)[2]
    # scale = 1/(step*100)
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)
    sns.set_context("notebook", font_scale=1.5)

    N = 40
    normalized_loss = []
    for m in indexes:
        print("Figure {0}".format(m+1))
        plt.figure()
        plt.title("Model ECG " + str(m+1))
        loss = loss_temp_tensor[m]
        cmap = matplotlib.cm.get_cmap('rainbow')
        mean_loss = np.mean(loss, axis=1)
        std_loss = np.std(loss, axis=1)


        for i in range(N):
            # plt.bar(np.ones(step)*i, loss[i, :] , color=cmap(i / N), alpha=scale)
            colorx = cmap(i / N)
            plt.bar(i, mean_loss[i], color=colorx, alpha=1)
            colorx = [x - x * 0.3 for x in colorx[:-1]]
            (_, caps, _) = plt.errorbar(i, mean_loss[i], yerr=std_loss[i], ecolor=colorx, elinewidth=1, capsize=5)
            for x, cap in enumerate(caps):
                caps[x].set_markeredgewidth(1)


        plt.xticks(range(len(loss_temp_tensor[:, 0])+1), labels, rotation=45)
        plt.xlim([-1, 41])
        # plt.figure()
        sumi = np.sum(loss_temp_tensor[m, :, :], axis=1)
        f = intr.interp1d(np.arange(len(sumi)), sumi)
        # xxx = f(np.arange(0, len(sumi)-1, 1/100))
        # plt.plot(np.arange(0, len(sumi)-1, 1/100), xxx / np.max(xxx))
        plt.show()

        # sns.boxplot(loss.T)
        # plt.show()
    colors = [cmap(i / N) for i in range(N)]





check_fantasia([39])
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
