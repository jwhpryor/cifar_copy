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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'logs/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'logs',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', kg.NUM_EVAL_SAMPLES,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

def eval_once(saver, summary_writer, top_k_op, summary_op, images, labels, logits ):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        checkpoint_dir = os.path.join(os.getcwd(), FLAGS.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
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
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * kg.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                if True:
                    predictions, imgs, labels, logits = sess.run([top_k_op, images, labels, logits])
                else:
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


def evaluate():
    with tf.Graph().as_default() as g:
        eval_filename_queue = tf.train.string_input_producer([kg.EVAL_PROTO_FILE], shuffle=False,
                                                              name='eval_filename_queue')
        labels, images, _ = kg.img_label_pairs(eval_filename_queue, eval=True)
        logits = kg.inference(images)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(kg.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op, images, labels, logits)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()
