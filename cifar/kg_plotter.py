import os
import matplotlib.pyplot as plt
import math

GRID_COLS = 4
MAX_PLOTTABLE_IMGS = 8

def plot(img):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img.squeeze(), cmap='Greys_r')
    plt.show(block=True)

def plot_batch(img_batch, filenames=None, predictions_batch=None, labels_batch=None):
    num_imgs = img_batch.shape[0]
    num_plot_imgs = min(MAX_PLOTTABLE_IMGS, num_imgs)
    rows = math.ceil(float(num_plot_imgs)/GRID_COLS)

    fig = plt.figure()
    for i in range(0, num_plot_imgs):
        img = img_batch[i]
        sub_plot = fig.add_subplot(rows, GRID_COLS, i+1)
        sub_plot.axes.get_xaxis().set_visible(False)
        sub_plot.axes.get_yaxis().set_visible(False)
        title = ''
        if predictions_batch is not None:
            title += ' p:' + str(predictions_batch[i])
        if labels_batch is not None:
            title += ' l:' + str(labels_batch[i])
        plt.imshow(img.squeeze(), cmap='Greys_r')
        if filenames is not None:
           actual_filename = os.path.basename(filenames[i])
           title += '\n' + actual_filename
        plt.title(title)

    plt.show(block=True)
