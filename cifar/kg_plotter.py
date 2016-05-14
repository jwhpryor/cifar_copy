import matplotlib.pyplot as plt
import math

GRID_COLS = 4
MAX_PLOTTABLE_IMGS = 6

def plot(img):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img.squeeze(), cmap='Greys_r')
    plt.show(block=True)

def plot_batch(img_batch, label_batch):
    num_imgs = img_batch.shape[0]
    num_plot_imgs = min(MAX_PLOTTABLE_IMGS, num_imgs)
    rows = math.ceil(float(num_plot_imgs)/GRID_COLS)

    fig = plt.figure()
    for i in range(0, num_plot_imgs):
        img = img_batch[i]
        fig.add_subplot(rows, GRID_COLS, i+1)
        plt.imshow(img.squeeze(), cmap='Greys_r')

    plt.show(block=True)
