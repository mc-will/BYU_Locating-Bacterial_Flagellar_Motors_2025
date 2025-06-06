import tensorflow as tf
import numpy as np
import pandas as pd

import os
import glob
import sys
sys.path.append('../src')

from utils.data import get_csv_from_bq,select_tomo_ids
# from src.utils.data import get_csv_from_bq,select_tomo_ids

def selection_images_labels(df, dir_images, num_slices=[300], num_motors=[1]):

    ''''
    function to return the path to the selected images (which type, which tomos, how many motors,
    shape of the images)
    Parameters:
    ----------
    df = database (train)
    dir_images(str) = directory with the images we want to feed to the model
    num_slices, num_motors, y_shape_range, x_shape_range = params for the select_tomo_ids function

    Returns:
    -------
    filtered_image_paths (list or np.array): List of image paths.

    labels (np.array or list): Corresponding labels.
    '''

    # Step 1: Filter tomos
    tomo_ids = select_tomo_ids(df, number_of_slices=num_slices, number_of_motors=num_motors)
    df_select = df[df['tomo_id'].isin(tomo_ids)].copy()

    # Step 2: Set up labels
    df_select['motor_coord'] = df_select.apply(lambda row: (row['Motor_axis_2'], row['Motor_axis_1']), axis=1)

    # Step 3: Load all images
    dir_mean_image = f'../data/pictures_process/{dir_images}'
    all_images = glob.glob(os.path.join(dir_mean_image, '**', '*.jpg'), recursive=True)

    print(f"Found {len(all_images)} images in {dir_mean_image}")

    # Step 4: Match images using substring matching
    filtered_image_paths = []
    labels = []

    for _, row in df_select.iterrows():
        tomo_id = row['tomo_id']
        matched = [p for p in all_images if tomo_id in os.path.basename(p)]

        if matched:
            filtered_image_paths.append(matched[0])  # If multiple, take the first
            labels.append(row['motor_coord'])
        else:
            print(f"⚠️ No image found for tomo_id: {tomo_id}")

    print(f"Matched {len(filtered_image_paths)} image-label pairs")

    labels = np.array(labels, dtype=np.float32)
    return filtered_image_paths, labels


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
        # Just map and batch
        dataset = dataset.map(read_img_jpg).batch(batch_size)
        return dataset
