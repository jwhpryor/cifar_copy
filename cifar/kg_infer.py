from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
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

def eval_once(saver, summary_writer, top_k_op, summary_op, images, labels, logits ):
    with tf.Session() as sess:
        checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Found checkpoint...')
            # Restores from checkpoint
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt.model_checkpoint_path))
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / kg.BATCH_SIZE))
            #num_iter = min(10, int(math.ceil(FLAGS.num_examples / kg.BATCH_SIZE)))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * kg.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                if False:
                    predictions, imgs, labels, logits = sess.run([top_k_op, images, labels, logits])
                    kg_plotter.plot_batch(imgs, None)
                else:
                    print('Evaluating... (step ' + str(step) + ' of ' + str(num_iter) + ')')
                    predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def infer(filenames):
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            # build the model
            filename_queue = tf.train.string_input_producer(filenames, shuffle=False,
                                                             name='infer_filename_queue')
            images = kg.img_from_jpeg(filename_queue)
            logits = kg.inference(images, batch_size=1)

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

                inference_logits = sess.run([logits])
                print(inference_logits)

            #coord.request_stop()
            #coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
    if not tf.gfile.Exists(FLAGS.infer_dir):
        raise Exception('No inference directory found.')
    filenames = [os.path.join(os.getcwd(), x) for x in os.listdir(FLAGS.infer_dir)]
    infer(filenames)

if __name__ == '__main__':
    tf.app.run()
