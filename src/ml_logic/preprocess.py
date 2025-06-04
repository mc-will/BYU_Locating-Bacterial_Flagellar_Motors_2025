import tensorflow as tf
import numpy as np
import pandas as pd

import os
import glob
import sys
sys.path.append('../src')

from src.utils.data import get_csv_from_bq,select_tomo_ids

def selection_images_labels(df, dir_images, num_slices=[300], num_motors=[1],
                       y_shape_range=(960, 960), x_shape_range=(928, 928)):

     # Selecting tomos according to our whishes
    tomo_ids_1 = select_tomo_ids(df, number_of_slices=num_slices, number_of_motors=num_motors,
                    y_shape_range=y_shape_range, x_shape_range=x_shape_range)

    # Creating database with our selected tomos
    df_select = df[df['tomo_id'].isin(tomo_ids_1)]

    # Selecting directory with images we want to feed to our model (e.x., mean, adaptequal)
    dir_mean_image = f'../data/pictures_process/{dir_images}'
    keywords = set(tomo_ids_1)

    # Find all jpg image paths recursively
    all_images = glob.glob(os.path.join(dir_mean_image, '**', '*.jpg'), recursive=True)

    # Filter image paths where the filename contains any of the keywords
    filtered_image_paths = [
        path for path in all_images
        if any(kw in os.path.basename(path) for kw in keywords)
    ]

    # Defining the motor coordinates as a tuple, to then use as a target (we will call them labels)
    df_select['motor_coord'] = df_select.apply(lambda row: (row['Motor_axis_2'], row['Motor_axis_1']), axis=1).copy()


    # Prepare labels as float32 NumPy array
    labels = np.array(df_select['motor_coord'].tolist(), dtype=np.float32)

    return filtered_image_paths,labels


# Define image reading function
def read_img_jpg(path, label):
    """
    Reads a JPEG image from a file path, decodes it as a grayscale image (1 channel),
    normalizes pixel values to the range [0, 1], and returns it along with its label.

    Parameters:
    ----------
    path : tf.Tensor
        A scalar string tensor representing the file path to the JPEG image.

    label : tf.Tensor or any
        The label associated with the image (e.g., coordinates or class ID).

    Returns:
    -------
    img : tf.Tensor
        A 3D float32 tensor of shape (height, width, 1) representing the normalized image.

    label : same as input
        The original label passed in, unchanged.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.cast(img, tf.float32) / 255.0  # normalize to [0, 1]
    return img, label



def batches_images_ram(
    read_img_jpg,
    filtered_image_paths,
    labels,
    shuffle=True,
    batch_size=32,
    split=False,
    val_fraction=0.2,
    test_fraction=0.2,
    seed=42
):
    """
    Load images and labels as tf.data.Dataset, optionally shuffle and batch,
    and optionally split into train/val/test datasets.

    Parameters:
    -----------
    read_img_jpg : function
        Function to load and preprocess image from path.

    filtered_image_paths : list or np.array
        List of image paths.

    labels : np.array or list
        Corresponding labels.

    shuffle : bool, default=True
        Whether to shuffle the dataset.

    batch_size : int, default=32
        Batch size.

    split : bool, default=False
        Whether to split dataset into train/val/test.

    val_fraction : float, default=0.2
        Fraction of data for validation.

    test_fraction : float, default=0.2
        Fraction of data for testing.

    seed : int, default=42
        Random seed for shuffling.

    Returns:
    --------
    If split=False:
        dataset : tf.data.Dataset
            Dataset with (image, label) pairs, batched and optionally shuffled.

    If split=True:
        train_ds, val_ds, test_ds : tf.data.Dataset
            The three splits, all batched and shuffled as specified.
    """

    dataset_size = len(filtered_image_paths)

    # Create dataset from (path, label)
    dataset = tf.data.Dataset.from_tensor_slices((filtered_image_paths, labels))

    # Shuffle dataset if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed)

    if split:
        # Compute sizes
        val_size = int(val_fraction * dataset_size)
        test_size = int(test_fraction * dataset_size)
        train_size = dataset_size - val_size - test_size

        # Split datasets carefully
        test_ds = dataset.take(test_size)
        val_ds = dataset.skip(test_size).take(val_size)
        train_ds = dataset.skip(test_size + val_size)

        # Map and batch each split
        train_ds = train_ds.map(read_img_jpg).batch(batch_size)
        val_ds = val_ds.map(read_img_jpg).batch(batch_size)
        test_ds = test_ds.map(read_img_jpg).batch(batch_size)

        return train_ds, val_ds, test_ds

    else:
        # Just map, batch and prefetch full dataset
        dataset = dataset.map(read_img_jpg).batch(batch_size)
        return dataset
