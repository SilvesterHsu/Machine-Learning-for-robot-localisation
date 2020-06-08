# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import tensorflow as tf
import os


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = '/home/kevin/data/michigan_gt/2012_11_17/global_localization.tfrecords'
index_dir = '/home/kevin/data/michigan_gt/2012_11_17/index.txt'
image_dir = '/home/kevin/data/michigan_gt/2012_11_17/images'
target_dir = '/home/kevin/data/michigan_gt/2012_11_17/poses'

text_file = open(index_dir, "r")
indices = text_file.readlines()

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []

for index in indices:
    index = index[:-1]
    img_file = os.path.join(image_dir, index+".png")
    target_file = os.path.join(target_dir, index+".txt")

    img = np.array(Image.open(img_file))
    target = np.loadtxt(target_file)

    # Feature contains a map of string to feature proto objects
    feature = {}
    feature['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
    feature['target'] = tf.train.Feature(float_list=tf.train.FloatList(value=target.tolist()))
    feature['height'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[0]]))
    feature['width'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[1]]))
    feature['target_dim'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[6]))

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize the example to a string
    serialized = example.SerializeToString()

    # write the serialized objec to the disk
    writer.write(serialized)
writer.close()
