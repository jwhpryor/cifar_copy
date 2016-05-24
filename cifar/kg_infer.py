from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os

import numpy as np
import tensorflow as tf

import kg
import kg_plotter

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('infer_dir', 'data/imgs/test/',
                           """Directory To run inference on.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'logs',
                           """Directory where to read from.""")
tf.app.flags.DEFINE_boolean('plot_imgs', True,
                           """Plot the images as you evaluate.""")
tf.app.flags.DEFINE_integer('top_k', 3,
                            """The # of indices from the logits to report.""")

def infer(filenames):
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            # build the model
            filename_queue = tf.train.string_input_producer(filenames, shuffle=False,
                                                             name='infer_filename_queue')
            image_batch, filename_t, img_t = kg.img_from_jpeg(filename_queue)
            logits_t = kg.inference(image_batch, batch_size=1)
            #logits_squeezed = tf.squeeze(logits_t)
            #logits_squeezed = tf.convert_to_tensor(logits_squeezed)
            #top_indices_t = tf.nn.top_k(logits_squeezed)
            top_indices_t = tf.nn.top_k(logits_t, k=FLAGS.top_k)[1]

            # restore our model
            variable_averages = tf.train.ExponentialMovingAverage(kg.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Found checkpoint...')
                saver.restore(sess, os.path.join(checkpoint_dir, ckpt.model_checkpoint_path))
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # setup our reader and tell tf how to read the images
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            tf.train.start_queue_runners(sess=sess, coord=coord)

            #while not coord.should_stop():
            while True:
                if FLAGS.plot_imgs:
                    #logits, filenames, imgs = sess.run([logits_t, filename_t, img_t])
                    logits, filenames, imgs = sess.run([top_indices_t, filename_t, img_t])
                    print(logits)
                    kg_plotter.plot(imgs)
                else:
                    inference_logits, filenames = sess.run([logits, filename_t])

            #coord.request_stop()
            #coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
    if not tf.gfile.Exists(FLAGS.infer_dir):
        raise Exception('No inference directory found.')
    filenames = [os.path.join(os.path.join(os.getcwd(), FLAGS.infer_dir), x) for x in os.listdir(FLAGS.infer_dir)]
    #filenames = filenames[0:1]
    infer(filenames)

if __name__ == '__main__':
    tf.app.run()
