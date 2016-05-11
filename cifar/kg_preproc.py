from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kg
import kg_plotter

import os.path
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'data/imgs/train',
                           """Images train dir""")
tf.app.flags.DEFINE_string('output_dir', 'output direcotory',
                           """Where to write TF records.""")
tf.app.flags.DEFINE_boolean('plot_imgs', False,
                            """Whether to plot images.""")

def get_train_images_and_labels():
    path = os.path.join(os.getcwd(), FLAGS.train_dir)
    imgs = []
    img_dirs = [os.path.join(path, filename) for filename in os.listdir(path)]
    labels = {}

    # kaggle sorts images in top level dir by their classes
    for i in range(0, len(img_dirs)):
        # (this is lazy and non-resilient since the order of the dirs listed could be arbitrary but fix it later)
        dir_name = img_dirs[i]
        for filename in [os.path.join(dir_name, x) for x in os.listdir(dir_name)]:
            filename = filename
            imgs.append(filename)
            labels[filename] = i
    return imgs, labels

def preproc():
    img_filenames, label_dic = get_train_images_and_labels()
    np.random.shuffle(img_filenames)

    img_filenames = img_filenames[0:20]
    labels = [label_dic[x] for x in img_filenames]

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # setup our read and tell tf how to read the images
            reader = tf.WholeFileReader()
            filename_queue = tf.train.string_input_producer(img_filenames, num_epochs=1)

            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            init_op = tf.initialize_all_variables()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(init_op)
            writer = tf.python_io.TFRecordWriter(kg.PROTO_FILE)

            key, img_raw = reader.read(filename_queue)
            img = tf.image.decode_jpeg(img_raw, channels=kg.CHANNELS, ratio=kg.DOWNSAMPLE)

            try:
                while not coord.should_stop():
                    filename, img_vals = sess.run([key, img])
                    label = label_dic[filename]

                    example = tf.train.Example(
                       features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[label])),
                                'image': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=img_vals.tostring()))
                            }))
                    serialized = example.SerializeToString()
                    writer.write(serialized)


            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

if __name__ == '__main__':
    print('Starting Preproc.')
    preproc()