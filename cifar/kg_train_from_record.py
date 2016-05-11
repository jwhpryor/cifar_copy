from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kg
import kg_plotter

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('plot_imgs', True,
                            """Whether to plot images.""")

def read_and_decode_single_example(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature(kg.IMG_BYTES, tf.string)
        })

    label = features['label']
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [kg.IMG_HEIGHT, kg.IMG_WIDTH, kg.CHANNELS])

    return label, image

if __name__ == '__main__':
    print('Starting Training.')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer([kg.PROTO_FILE], shuffle=False)
            label, image = read_and_decode_single_example(filename_queue)
            chars = tf.to_float(image)

            coord = tf.train.Coordinator()
            init_op = tf.initialize_all_variables()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(init_op)

            for i in range(0,20):
                l, c = sess.run([label, chars])
                if FLAGS.plot_imgs:
                    print('Label: ' + str(l))
                    kg_plotter.plot(c)
