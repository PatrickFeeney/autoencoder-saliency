import matplotlib.pyplot as plt
import numpy as np
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
    _, valid_non_novel, _ = torch_mnist(batch_size, [0, 1, 2, 3, 4], shuffle=False)
    _, valid_novel, _ = torch_mnist(batch_size, [5, 6, 7, 8, 9], shuffle=False)
    # construct model
    model_path = "models\\LSA_mnist_no_est_class_0_1_2_3_4\\500_lr_1e-2\\LSA_mnist_no_est_500.pt"
    model = load_model(model_path)
    model.to(device)
    model.eval()
    # eval model
    for is_novel in [False, True]:
        # choose data loader
        if not is_novel:
            data_loader = valid_non_novel
        else:
            data_loader = valid_novel
        # generate figures
        for data, target in data_loader:
            data = data.to(device)
            data.requires_grad_()
            r_data, embedding = model(data)
            batch_loss = loss_func(data, r_data)
            batch_loss.backward()
            saliency = data.grad.data
            plot_reconstruction(data, r_data, saliency)
            plt.show()
            break


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
    # set number of images for plot
    num_images = 5
    if images.shape[0] < num_images:
        num_images = images.shape[0]
    # remove grad from tensors for numpy conversion
    images = images.detach().cpu()[:num_images]
    r_images = r_images.detach().cpu()[:num_images]
    saliency = saliency.detach().cpu()[:num_images]
    # regularize the saliency map
    saliency = torch.abs(saliency)
    saliency = saliency / torch.amax(saliency, (2, 3), keepdim=True)
    # regularized compute squared error
    r_error = torch.square(r_images - images)
    r_error = r_error / torch.amax(r_error, (2, 3), keepdim=True)
    # compute where reconstruction greater than saliency
    dif = saliency - r_error
    # plot the image, reconstruction, and saliency
    titles = ["Input", "Reconstruction", "Saliency", "R. Error", "Dif"]
    num_plots = len(titles)
    fig, ax = plt.subplots(nrows=num_plots, ncols=num_images)
    for i, title in enumerate(titles):
        ax[i][num_images // 2].set_title(title)
    imshow_kwargs = {
        "cmap": "gray",
        "vmin": 0,
        "vmax": 1,
    }
    image_sets = [images, r_images, saliency, r_error, dif]
    for i in range(num_images):
        for j, image_set in enumerate(image_sets):
            if j < len(image_sets) - 1:
                ax[j][i].imshow(image_set[i, 0], **imshow_kwargs)
            else:
                ax[j][i].imshow(image_set[i, 0], cmap="RdBu", vmin=-1, vmax=1)
    # disable tick marks
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    return fig
