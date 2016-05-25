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
import kg_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_logs_dir', 'logs/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'logs',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('batch_size', kg.BATCH_SIZE,
                            """Size of batch to process with.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 108000,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_boolean('plot_imgs', True,
                           """Whether to plot the images as we evaluate.""")

def eval_once(num_examples, saver, summary_writer, top_k_op, summary_op, images_t, labels_t, logits_t, filenames_t):
    with tf.Session() as sess:
        checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Found checkpoint...')
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt.model_checkpoint_path))
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * kg.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                if FLAGS.plot_imgs:
                    predictions, images, labels, logits, filenames = sess.run([top_k_op, images_t, labels_t, logits_t,
                                                                            filenames_t])
                    kg_plotter.plot_batch(images, predictions_batch=predictions, labels_batch=labels,
                                          filenames=filenames)
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
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def infer_once(num_examples, saver, filenames_t, top_k_op, images_t):
    with tf.Session() as sess:
        checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Found checkpoint...')
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt.model_checkpoint_path))
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
            step = 0
            while step < num_iter and not coord.should_stop():
                if FLAGS.plot_imgs:
                    filenames, predictions, imgs = sess.run([filenames_t, top_k_op, images_t])
                    kg_plotter.plot_batch(imgs, filenames=filenames, predictions_batch=predictions)
                else:
                    print('Evaluating... (step ' + str(step) + ' of ' + str(num_iter) + ')')
                    predictions = sess.run([top_k_op])
                step += 1

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def infer_folder(folder):
    filenames = [os.path.join(folder, x) for x in os.listdir(folder)]

    with tf.Graph().as_default() as g:
        infer_filename_queue = tf.train.string_input_producer(filenames, shuffle=False, name='infer_filename_queue')
        num_examples = len(filenames)

        images_t, filenames_t = kg.img_labels_from_jpeg(infer_filename_queue, batch_size=FLAGS.batch_size)
        logits = kg.inference(images_t)
        top_k_op = tf.nn.top_k(logits, 1)[1]

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(kg.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            infer_once(num_examples, saver, filenames_t, top_k_op, images_t)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def evaluate():
    with tf.Graph().as_default() as g:
        filenames, labels = kg_data.get_train_eval_and_label(eval=True)
        eval_filename_queue = tf.train.string_input_producer(filenames, shuffle=False, name='eval_filename_queue')
        eval_label_queue = tf.train.input_producer(labels, shuffle=False, name='eval_label_queue')
        num_examples = len(filenames)

        images_t, labels_t, filenames_t = kg.img_labels_from_jpeg(eval_filename_queue, eval_label_queue,
                                                            batch_size=kg.BATCH_SIZE)
        logits_t = kg.inference(images_t)
        top_k_op = tf.nn.in_top_k(logits_t, labels_t, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(kg.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_logs_dir, g)

        while True:
            eval_once(num_examples, saver, summary_writer, top_k_op, summary_op, images_t, labels_t, logits_t,
                      filenames_t)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_logs_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_logs_dir)
    tf.gfile.MakeDirs(FLAGS.eval_logs_dir)

    #infer_folder('/Users/jwhpryor/ml/data/imgs/test')

    evaluate()

if __name__ == '__main__':
    tf.app.run()
