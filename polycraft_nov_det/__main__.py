import polycraft_nov_det.saliency as saliency


for k in [1, 3, 5, 10]:
    saliency.plot_top_k(k)
saliency.plot_mse()
saliency.plot_mse(True)
saliency.plot_sample_reconstructions()
