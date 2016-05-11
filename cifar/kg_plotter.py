import matplotlib.pyplot as plt
import math

GRID_COLS = 4
MAX_PLOTTABLE_IMGS = 28

def plot(img):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img.squeeze(), cmap='Greys_r')
    plt.show(block=True)

def plot_batch(img_batch, label_batch):
    num_imgs = img_batch.shape[0]
    rows = math.floor(num_imgs/GRID_COLS)

    fig = plt.figure()
    for i in range(0, min(MAX_PLOTTABLE_IMGS, num_imgs)):
        img = img_batch[i]
        fig.add_subplot(rows, GRID_COLS, i+1)
        plt.imshow(img.squeeze(), cmap='Greys_r')

    plt.show(block=True)
