"""
Usage:
    python3 view_record.py --record=data.record
"""
import argparse

import cv2
import numpy as np
import tensorflow as tf


# FLAGS = None
# IMG_SIZE = 128
# MARK_SIZE = 68 * 2


def parse_tfrecord(record_path):
  """Try to extract a image from the record file as jpg file."""
  dataset = tf.data.TFRecordDataset(record_path)

  # Create a dictionary describing the features. This dict should be
  # consistent with the one used while generating the record file.
  feature_description = {
    'image/height': tf.FixedLenFeature([], tf.int64),
    'image/width': tf.FixedLenFeature([], tf.int64),
    'image/depth': tf.FixedLenFeature([], tf.int64),
    'image/filename': tf.FixedLenFeature([], tf.string),
    'image/encoded': tf.FixedLenFeature([], tf.string),
    'image/format': tf.FixedLenFeature([], tf.string),
    'image/object/class/text': tf.VarLenFeature(tf.string)
    #'label/marks': tf.FixedLenFeature([MARK_SIZE], tf.float32),
  }

  def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.parse_single_example(example_proto, feature_description)

  parsed_dataset = dataset.map(_parse_function)
  return parsed_dataset

def _extract_feature(element):
  """
  Extract features from a single example from dataset.
  """
  features = tf.parse_single_example(
    element,
    # Defaults are not specified since both keys are required.
    features={
      'image/height': tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64),
      # 'image/depth': tf.FixedLenFeature([], tf.int64),
      'image/filename': tf.FixedLenFeature([], tf.string),
      'image/encoded': tf.FixedLenFeature([], tf.string),
      'image/format': tf.FixedLenFeature([], tf.string),
      'image/object/class/label': tf.VarLenFeature(tf.int64),
      'image/object/difficult': tf.VarLenFeature(tf.int64),
      'image/object/class/text': tf.VarLenFeature(tf.string)
    })
  return features

def show_record(filenames):
  """Show the TFRecord contents"""
  # Generate dataset from TFRecord file.
  dataset = tf.data.TFRecordDataset(filenames)

  # Make dataset iterateable.
  iterator = dataset.make_one_shot_iterator()
  next_example = iterator.get_next()

  # Extract features from single example
  features = _extract_feature(next_example)
  # image_decoded = tf.image.decode_image(features['image/encoded'])
  filename = tf.cast(features['image/filename'], tf.string)
  labels = tf.cast(features['image/object/class/label'], tf.int64)
  difficult = tf.cast(features['image/object/difficult'], tf.int64)
  label_text = tf.cast(features['image/object/class/text'], tf.string)

  # Use openCV for preview
  # cv2.namedWindow("image", cv2.WINDOW_NORMAL)

  total = 0
  stat = { 'shoe': 0, 'fake poop a': 0, 'fake poop b': 0,
           'pet feces': 0 }
  stat_diff = { 'shoe': 0, 'fake poop a': 0, 'fake poop b': 0,
           'pet feces': 0 }
  # Actual session to run the graph.
  with tf.Session() as sess:
    while True:
      try:
        # image_tensor, fname = sess.run(
        #   [image_decoded, filename])

        # Use OpenCV to preview the image.
        # image = np.array(image_tensor, np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        # Show the result
        # cv2.imshow(fname, image)
        # if cv2.waitKey() == 27:
        #   break

        name, lbl, diff, text = sess.run([filename, labels, difficult, label_text])
        # if name == '2003584.jpg':
        #   print ('gotta')
        diff_array, text_array = sess.run ([tf.sparse.to_dense(tf.sparse.reorder(diff)),
                                            tf.sparse.to_dense(tf.sparse.reorder(text), default_value="")])

        print ("cp {}{} {}".format("/workspace/downloads/rockrobo_data/det_testset/neice_final/VOC2007/ann-text/"
                                    ,name.decode("utf-8").replace('jpeg', 'txt')
                                    ,"/workspace/downloads/rockrobo_data/det_testset/neice_final/VOC2007/ann-text-0524/"
                                    ))

        for txt in text_array:
          if txt in stat:
            stat[txt] += 1

        if np.any(diff_array > 0):
          diff_txt = text_array[diff_array > 0]
          for txt in diff_txt:
            # print ("  {} is diffcult".format(txt))
            # if 'fake poop' in txt:
            #   print ("lalalala")
            if txt in stat_diff:
              stat_diff[txt] += 1


        total += 1

      except tf.errors.OutOfRangeError:
        break

    # # Use OpenCV to preview the image.
    # image = np.array(image_decoded, np.uint8)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #
    #
    # # Show the result
    # cv2.imshow("image", image)
    # if cv2.waitKey() == 27:
    #   break

    print ("total = %d" % total)

    print (stat)
    print (stat_diff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        type=str,
        default="rio29_day_test_20200524.tfrecord",
        help="The record file."
    )
    args = parser.parse_args()
    show_record(args.record)