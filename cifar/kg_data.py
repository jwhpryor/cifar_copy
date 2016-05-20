from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kg

import os.path
import csv
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('label_dictionary', 'data/imgs/driver_imgs_list.csv',
                           """Dictionary of all filenames and their respective class info""")
tf.app.flags.DEFINE_string('train_dir', 'data/imgs/train',
                           """Images train dir""")
tf.app.flags.DEFINE_string('eval_dir', 'data/imgs/test',
                           """Images train dir""")
tf.app.flags.DEFINE_boolean('plot_imgs', False,
                            """Whether to plot images.""")
tf.app.flags.DEFINE_float('holdout', 0.1,
                          """Percent of samples to holdout.""")

# parses the csv provided by kg for training
def get_label_dic():
    output_dic = {}
    if not tf.gfile.Exists(FLAGS.label_dictionary):
        raise Exception('No label dictionary found')
    with open(FLAGS.label_dictionary, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            driver_id = int(row[0][1:])
            class_id = int(row[1][1:])
            filename = row[2]
            output_dic[filename] = kg.DriverRecord(filename, driver_id, class_id)

    return output_dic

# breaks apart set into train and eval holdout (accomodating batch_size)
def get_train_eval_sets():
    path = os.path.join(os.getcwd(), FLAGS.train_dir)
    filenames = []
    img_dirs = [os.path.join(path, filename) for filename in os.listdir(path)]

    # kaggle sorts images in top level dir by their classes
    for i in range(0, len(img_dirs)):
        # (this is lazy and non-resilient since the order of the dirs listed could be arbitrary but fix it later)
        dir_name = img_dirs[i]
        for filename in [os.path.join(dir_name, x) for x in os.listdir(dir_name)]:
            filename = filename
            filenames.append(filename)
    np.random.shuffle(filenames)

    return filenames[0:kg.NUM_TRAIN_SAMPLES], filenames[-kg.NUM_EVAL_SAMPLES:]

