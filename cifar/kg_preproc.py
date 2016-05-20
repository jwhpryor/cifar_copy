from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import csv
import tensorflow as tf
import numpy as np

import kg
import kg_data

FLAGS = tf.app.flags.FLAGS

def write_example_from_img(writer, label, img_vals):
    # TODO: not using driver id, not sure we want to?
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

def preproc_files(out_proto, label_dic, filenames):
    num_files = len(filenames)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # setup our read and tell tf how to read the images
            reader = tf.WholeFileReader()
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)

            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            init_op = tf.initialize_all_variables()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(init_op)
            writer = tf.python_io.TFRecordWriter(out_proto)

            key, img_raw = reader.read(filename_queue)
            img = tf.image.decode_jpeg(img_raw, channels=kg.CHANNELS, ratio=kg.DOWNSAMPLE)

            i = 0
            try:
                while not coord.should_stop():
                    filename, img_vals = sess.run([key, img])
                    driver_data = label_dic[filename.split('/')[-1]]
                    label = driver_data.class_id
                    write_example_from_img(writer, label, img)

                    print('[' + str(i) + ' of ' + str(num_files) + '] Processing ' + filename + '...')
                    i = i +1

            except tf.errors.OutOfRangeError as e:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    print('Collecting file sets...\n')
    driver_dic = kg_data.get_label_dic()
    train_filenames, eval_filenames = kg_data.get_train_eval_sets()

    # train proto
    print('Starting Train Preprocessing...')
    preproc_files('proto/test.record', driver_dic, train_filenames)
    print('')

    # eval proto
    print('Starting Eval Preprocessing.')
    preproc_files(kg.EVAL_PROTO_FILE, driver_dic, eval_filenames)
