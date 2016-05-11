import matplotlib.pyplot as plt

def plot(img):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img.squeeze(), cmap='Greys_r')
    plt.show(block=True)
