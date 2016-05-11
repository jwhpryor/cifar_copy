from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'data/imgs/train/c0',
                           """Images train dir""")
tf.app.flags.DEFINE_string('logs_dir', 'logs',
                            """Where to write logs output.""")
tf.app.flags.DEFINE_integer('max_steps', 5,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('plot_imgs', False,
                            """Whether to plot images.""")

def get_images_and_labels():
    path = os.path.join(os.getcwd(), FLAGS.train_dir)
    imgs = [os.path.join(path, filename) for filename in os.listdir(path)]

    return imgs, None

def inference():
   pass

def train():
    # get the images from disk
    imgs, labels = get_images_and_labels()
    filenames = [imgs[0], imgs[1], imgs[2]]

    # setup our read and tell tf how to read the images
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # plot the images
    if FLAGS.plot_imgs is True:
        my_img = tf.image.decode_jpeg(value, channels=1, ratio=4)
        fig = plt.figure()
        for i in range(1, 10):
            fig.add_subplot(3, 3, i)
            image = my_img.eval()
            tf.image_summary('image', image)   # y u no work?
            plt.imshow(image.squeeze(), cmap='Greys_r')
        plt.show()

    # build inference graph

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    print('Starting Training.')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
            train()

    print('\n\nTraining complete.')
