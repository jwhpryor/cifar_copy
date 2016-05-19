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

                    print('[' + str(i) + ' of ' + str(num_files) + '] Processing ' + filename + '...')
                    i = i +1

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


            except tf.errors.OutOfRangeError as e:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

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

if __name__ == '__main__':
    print('Collecting file sets...\n')
    driver_dic = get_label_dic()
    train_filenames, eval_filenames = get_train_eval_sets()

    # train proto
    print('Starting Train Preprocessing...')
#    preproc_files('proto/test.record', driver_dic, train_filenames)
    print('')

    # eval proto
    print('Starting Eval Preprocessing.')
    preproc_files(kg.EVAL_PROTO_FILE, driver_dic, eval_filenames)
