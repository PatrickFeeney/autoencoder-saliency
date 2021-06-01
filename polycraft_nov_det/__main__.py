import polycraft_nov_det.saliency as saliency


for k in [5, 10]:
    saliency.print_top_k_dist(k)
saliency.print_mean_mse()
for k in [1, 3]:
    saliency.print_top_k(k)
for k in [5, 10]:
    saliency.plot_top_k(k)
saliency.plot_sample_reconstructions()
