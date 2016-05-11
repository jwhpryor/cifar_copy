from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kg
import kg_plotter

import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Size of training batches.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of steps for training.""")
tf.app.flags.DEFINE_boolean('plot_imgs', False,
                            """Whether to plot images.""")
tf.app.flags.DEFINE_string('log_dir', 'logs',
                            """Where to emit logs for tensorboard.""")

def img_label_pairs(filename_queue):
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
    image = tf.to_float(image)

    # the shuffling is done in preprocessing
    image_batch, labels_batch = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=1)

    return labels_batch, image_batch

if __name__ == '__main__':
    print('Starting Training.')

    with tf.Graph().as_default():
        with tf.Session() as sess:
            global_step = tf.Variable(0, trainable=False)

            train_filename_queue = tf.train.string_input_producer([kg.TRAIN_PROTO_FILE], shuffle=False)
            labels, images = img_label_pairs(train_filename_queue)

            logits = kg.inference(images)
            loss = kg.loss(logits, labels)
            train_op = kg.train(loss, global_step)

            saver = tf.train.Saver(tf.all_variables())
            summary_op = tf.merge_all_summaries()

            coord = tf.train.Coordinator()
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                #_, loss_value, imgs_s, labels_s = sess.run([train_op, loss, images, labels])   use for plotting
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                #if step % 10 == 0:
                if True:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                         examples_per_sec, sec_per_batch))

                if FLAGS.plot_imgs:
                    kg_plotter.plot_batch(imgs_s, labels_s)

                if step % 1 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
