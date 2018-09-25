"""Create the input data pipeline using `tf.data`"""

import functools
import tensorflow as tf
import tensorflow.contrib as tfcontrib


def _process_pathnames(fname, label_path):
    # We map this function onto each pathname pair
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)

    label_img_str = tf.read_file(label_path)
    # These are gif images so they return as (num_frames, h, w, c)
    label_img = tf.image.decode_gif(label_img_str)[0]
    # The label image should only have values of 1 or 0, indicating pixel wise
    # object (car) or not (background). We take the first channel only.
    label_img = label_img[:, :, 0]
    label_img = tf.expand_dims(label_img, axis=-1)
    return img, label_img


def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([],
                                                  -width_shift_range,
                                                  width_shift_range)
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                                   -height_shift_range,
                                                   height_shift_range)
        # Translate both
        output_img = tfcontrib.image.translate(output_img,
                                               [width_shift_range, height_shift_range])
        label_img = tfcontrib.image.translate(label_img,
                                              [width_shift_range, height_shift_range])
    return output_img, label_img


def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(
                                        tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
    return tr_img, label_img


def _augment(img, label_img, params):

    label_img = tf.image.resize_images(label_img, params.resize)
    img = tf.image.resize_images(img, params.resize)

    if params.hue_delta:
        img = tf.image.random_hue(img, params.hue_delta)

    img, label_img = flip_img(params.horizontal_flip, img, label_img)
    img, label_img = shift_img(
        img, label_img, params.width_shift_range * params.resize[1], params.height_shift_range * params.resize[0])
    label_img = tf.to_float(label_img) * params.scale / tf.convert_to_tensor(255.0)
    img = tf.to_float(img) * params.scale / tf.convert_to_tensor(255.0)
    return img, label_img


def input_fn(filenames,
             labels,
             params):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames,
                          num_parallel_calls=params.num_parallel_calls)
    if 'resize' not in params:
        assert params.batch_size == 1, "Batching images must be of the same size"

    preproc_fn = functools.partial(_augment, params=params)
    dataset = dataset.map(
        preproc_fn, num_parallel_calls=params.num_parallel_calls)

    if params.shuffle:
        dataset = dataset.shuffle(num_x)
    # prefetch a batch (for efficiency).
    dataset = dataset.batch(params.batch_size).prefetch(1)

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels,
              'iterator_init_op': iterator_init_op}
    return inputs
