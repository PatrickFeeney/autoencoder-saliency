import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
import torch
import torch.nn as nn

from polycraft_nov_det.data import torch_mnist
from polycraft_nov_det.models.lsa.LSA_mnist_no_est import load_model


def plot_sample_reconstructions():
    # define constants
    batch_size = 5
    loss_func = nn.MSELoss()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # construct datasets
    _, _, test_non_novel = torch_mnist(batch_size, [0, 1, 2, 3, 4], shuffle=False)
    _, _, test_novel = torch_mnist(batch_size, [5, 6, 7, 8, 9], shuffle=False)
    # construct model
    model_path = "models\\LSA_mnist_no_est_class_0_1_2_3_4\\500_lr_1e-2\\LSA_mnist_no_est_500.pt"
    model = load_model(model_path)
    model.to(device)
    model.eval()
    # construct plot data variables
    plot_data = torch.tensor([])
    plot_r_data = torch.tensor([])
    plot_saliency = torch.tensor([])
    # eval model
    for is_novel in [False, True]:
        # choose data loader
        if not is_novel:
            data_loader = test_non_novel
        else:
            data_loader = test_novel
        # generate figures
        data, target = next(iter(data_loader))
        data = data.to(device)
        data.requires_grad_()
        r_data, embedding = model(data)
        batch_loss = loss_func(data, r_data)
        batch_loss.backward()
        saliency = data.grad.data
        # select examples by set
        if not is_novel:
            index = 2
        else:
            index = 2
        plot_data = torch.cat((plot_data, data[index, None]))
        plot_r_data = torch.cat((plot_r_data, r_data[index, None]))
        plot_saliency = torch.cat((plot_saliency, saliency[index, None]))
    plot_reconstruction(plot_data, plot_r_data, plot_saliency)
    plt.savefig("figures/maps.pdf", bbox_inches='tight', pad_inches=0)


def calc_top_k(k=10):
    # define constants
    batch_size = 5
    loss_func = nn.MSELoss()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # construct datasets
    _, _, test_non_novel = torch_mnist(batch_size, [0, 1, 2, 3, 4], shuffle=False)
    _, _, test_novel = torch_mnist(batch_size, [5, 6, 7, 8, 9], shuffle=False)
    # construct model
    model_path = "models\\LSA_mnist_no_est_class_0_1_2_3_4\\500_lr_1e-2\\LSA_mnist_no_est_500.pt"
    model = load_model(model_path)
    model.to(device)
    model.eval()
    # eval model
    for is_novel in [False, True]:
        top_k_agreement = np.asarray([])
        # choose data loader
        if not is_novel:
            data_loader = test_non_novel
        else:
            data_loader = test_novel
        # generate data
        for data, target in data_loader:
            data = data.to(device)
            data.requires_grad_()
            r_data, embedding = model(data)
            batch_loss = loss_func(data, r_data)
            batch_loss.backward()
            saliency = data.grad.data
            # remove grad from tensors for numpy conversion
            data = data.detach().cpu()
            r_data = r_data.detach().cpu()
            saliency = saliency.detach().cpu()
            # regularize the saliency map
            saliency = torch.abs(saliency)
            saliency = saliency / torch.amax(saliency, (2, 3), keepdim=True)
            # regularized compute squared error
            r_error = torch.square(r_data - data)
            r_error = r_error / torch.amax(r_error, (2, 3), keepdim=True)
            # calc agreement between top k
            top_k_dif = np.sum(np.abs(top_k(saliency, k) - top_k(r_error, k)), axis=(1, 2, 3))
            top_k_dif /= 2
            top_k_agreement = np.hstack((top_k_agreement, k - top_k_dif))
        # save top k results
        fname = "novel_top_%i.npy" % (k,) if is_novel else "non_novel_top_%i.npy" % (k,)
        np.save(fname, top_k_agreement)


def calc_mse(square_saliency=False):
    # define constants
    batch_size = 5
    loss_func = nn.MSELoss()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # construct datasets
    _, _, test_non_novel = torch_mnist(batch_size, [0, 1, 2, 3, 4], shuffle=False)
    _, _, test_novel = torch_mnist(batch_size, [5, 6, 7, 8, 9], shuffle=False)
    # construct model
    model_path = "models\\LSA_mnist_no_est_class_0_1_2_3_4\\500_lr_1e-2\\LSA_mnist_no_est_500.pt"
    model = load_model(model_path)
    model.to(device)
    model.eval()
    # eval model
    for is_novel in [False, True]:
        mse = np.asarray([])
        # choose data loader
        if not is_novel:
            data_loader = test_non_novel
        else:
            data_loader = test_novel
        # generate data
        for data, target in data_loader:
            data = data.to(device)
            data.requires_grad_()
            r_data, embedding = model(data)
            batch_loss = loss_func(data, r_data)
            batch_loss.backward()
            saliency = data.grad.data
            # remove grad from tensors for numpy conversion
            data = data.detach().cpu()
            r_data = r_data.detach().cpu()
            saliency = saliency.detach().cpu()
            # regularize the saliency map
            saliency = torch.abs(saliency)
            saliency = saliency / torch.amax(saliency, (2, 3), keepdim=True)
            if square_saliency:
                saliency = torch.square(saliency)
            # regularized compute squared error
            r_error = torch.square(r_data - data)
            r_error = r_error / torch.amax(r_error, (2, 3), keepdim=True)
            # calc agreement between top k
            mse_batch = np.square(saliency - r_error).mean(axis=(1, 2, 3))
            mse = np.hstack((mse, mse_batch))
        # save top k results
        fname = "novel_mse.npy" if is_novel else "non_novel_mse.npy"
        if square_saliency:
            fname = "sqr_" + fname
        np.save(fname, mse)


def plot_top_k(k=10):
    novel_top_k = np.load("novel_top_%i.npy" % (k,))
    non_novel_top_k = np.load("non_novel_top_%i.npy" % (k,))
    labels = ["Novel", "Normal"]
    fig, ax = plt.subplots()
    ax.hist([novel_top_k, non_novel_top_k],
            bins=np.arange(-.5, k + 1),
            density=True,
            label=labels)
    ax.set_xticks(np.arange(0, k + 1))
    ax.set_xlabel("Top-%i Agreement" % (k,))
    ax.set_ylabel("Frequency Across Test Set")
    ax.legend()
    plt.show()


def plot_mse(square_saliency=False):
    prefix = "sqr_" if square_saliency else ""
    novel_mse = np.load(prefix + "novel_mse.npy")
    non_novel_mse = np.load(prefix + "non_novel_mse.npy")
    fig, ax = plt.subplots()
    ax.violinplot([novel_mse, non_novel_mse],
                  showmedians=True,
                  vert=False)
    ax.set_xlabel("Mean Squared Error")
    ax.set_xlim(0, .09)
    ax.set_ylabel("Dataset")
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Novel", "Normal"])
    if square_saliency:
        ax.set_title("MSE Between Squared Saliency and Reconstruction Error")
    else:
        ax.set_title("MSE Between Saliency and Reconstruction Error")
    plt.show()


def top_k_random_dist(k=10):
    n = 28**2
    dist = np.array([])
    for i in range(k, -1, -1):
        dist = np.hstack((np.array([comb(k, i) / comb(n, i) - dist.sum()]), dist))
    return dist


def bin_normed(data):
    bin_vals = [0, .25, .5, .75, 1]
    for i in range(1, len(bin_vals)):
        in_bin = np.logical_and(bin_vals[i - 1] < data, data <= bin_vals[i])
        data[in_bin] = bin_vals[i]
    return data


def top_k(data, k=10):
    top_k = np.zeros_like(data)
    for i in range(data.shape[0]):
        ind_tuple = np.unravel_index(np.argsort(data[i], axis=None), data[i].shape)
        top_ind_tuple = tuple([ind[-k:] for ind in ind_tuple])
        top_ind_tuple = (np.asarray([i] * k),) + top_ind_tuple
        top_k[top_ind_tuple] = 1
    return top_k


def plot_reconstruction(images, r_images, saliency):
    # remove grad from tensors for numpy conversion
    images = images.detach().cpu()
    r_images = r_images.detach().cpu()
    saliency = saliency.detach().cpu()
    # regularize the saliency map
    saliency = torch.abs(saliency)
    saliency = saliency / torch.amax(saliency, (2, 3), keepdim=True)
    # regularized compute squared error
    r_error = torch.square(r_images - images)
    r_error = r_error / torch.amax(r_error, (2, 3), keepdim=True)
    # plot the image, reconstruction, and saliency
    titles = ["Input", "Reconstruction", "Saliency Map", "Square Saliency", "Reconstruction Error"]
    num_images = 2
    num_plots = len(titles)
    figsize = (2 * num_plots, 2 * num_images)
    fig, ax = plt.subplots(nrows=num_images, ncols=num_plots, figsize=figsize)
    ax[0][0].set_ylabel("Normal")
    ax[1][0].set_ylabel("Novel")
    for i, title in enumerate(titles):
        ax[0][i].set_title(title)
    imshow_kwargs = {
        "cmap": "gray",
        "vmin": 0,
        "vmax": 1,
    }
    image_sets = [images, r_images, saliency, saliency**2, r_error]
    for i in range(num_images):
        for j, image_set in enumerate(image_sets):
            ax[i][j].imshow(image_set[i, 0], **imshow_kwargs)
    # disable tick marks
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    return fig
