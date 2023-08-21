"""Dataset utilities."""

import tensorflow as tf
import tensorflow_datasets as tfds


def convert(image, label):
    # Cast and normalize the image to [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def load_dataset(name, split, *, is_training, batch_size):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load(name, split=split, as_supervised=True).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.map(convert)
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)
