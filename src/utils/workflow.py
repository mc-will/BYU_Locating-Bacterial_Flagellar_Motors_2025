import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd

import os
import glob
import keras

from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

from keras.preprocessing.image import load_img, img_to_array

from PIL import Image

MODEL_CLASSIF_URI = 'https://storage.cloud.google.com/models-wagon1992-group-project/7/06ba2fef2e314aba94bdf6286c0440cc/artifacts/model/data/model.keras'
MODEL_X_Y_URI = 'https://storage.cloud.google.com/models-wagon1992-group-project/8/fcaa4e2b1d574cc1beea21aa251ce7f2/artifacts/model/data/model.keras'
MODEL_Z_URI_WILL = 'https://storage.cloud.google.com/byu19922/willreg_z.keras'


test_tomos = ['tomo_dae195', 'tomo_f2fa4a', 'tomo_cabaa0', 'tomo_f7f28b', 'tomo_ed1c97', 'tomo_ff505c', 'tomo_8f4d60', 'tomo_2aeb29', 'tomo_651ecd', 'tomo_e96200', 'tomo_0d4c9e', 'tomo_2dcd5c', 'tomo_983fce', 'tomo_7b1ee3', 'tomo_8b6795', 'tomo_dcb9b4', 'tomo_e764a7', 'tomo_e26c6b', 'tomo_331130', 'tomo_f8b835', 'tomo_746d88', 'tomo_9cd09e', 'tomo_b9eb9a', 'tomo_cf0875', 'tomo_7cf523', 'tomo_fd41c4', 'tomo_54e1a7', 'tomo_ca472a', 'tomo_6478e5', 'tomo_e9b7f2', 'tomo_247826', 'tomo_675583', 'tomo_f0adfc', 'tomo_378f43', 'tomo_19a313', 'tomo_172f08', 'tomo_f3e449', 'tomo_3b83c7', 'tomo_8c13d9', 'tomo_2c607f', 'tomo_c11e12', 'tomo_412d88', 'tomo_4b124b', 'tomo_38c2a6', 'tomo_ec1314', 'tomo_1c38fd', 'tomo_e63ab4', 'tomo_f07244', 'tomo_210371', 'tomo_d6e3c7', 'tomo_935f8a', 'tomo_a4c52f', 'tomo_a46b26', 'tomo_fadbe2', 'tomo_b28579', 'tomo_35ec84', 'tomo_369cce', 'tomo_6c203d', 'tomo_b80310', 'tomo_640a74', 'tomo_22976c', 'tomo_d21396', 'tomo_ecbc12', 'tomo_040b80', 'tomo_85708b', 'tomo_b98cf6', 'tomo_e1e5d3', 'tomo_138018', 'tomo_3264bc', 'tomo_e50f04', 'tomo_d723cd', 'tomo_2a6ca2', 'tomo_1f0e78', 'tomo_67565e', 'tomo_fd5b38', 'tomo_05b39c', 'tomo_372a5c', 'tomo_c3619a', 'tomo_ba76d8', 'tomo_a67e9f', 'tomo_a6646f', 'tomo_db656f', 'tomo_4102f1', 'tomo_bb5ac1', 'tomo_4ed9de', 'tomo_61e947', 'tomo_1da0da', 'tomo_821255', 'tomo_3e7783', 'tomo_c84b46', 'tomo_974fd4', 'tomo_444829', 'tomo_b50c0f', 'tomo_2a6091', 'tomo_fa5d78', 'tomo_bdd3a0', 'tomo_1c2534', 'tomo_d916dc', 'tomo_bdc097', 'tomo_7036ee', 'tomo_cacb75', 'tomo_5b359d', 'tomo_7fa3b1', 'tomo_049310', 'tomo_dd36c9', 'tomo_e3864f', 'tomo_0a8f05', 'tomo_ff7c20', 'tomo_0fab19', 'tomo_1c75ac', 'tomo_d0699e', 'tomo_1e9980', 'tomo_4ee35e', 'tomo_6943e6', 'tomo_99a3ce']

df = pd.read_csv('../data/csv_raw/train_labels.csv')

#### TODO utiliser le code packager ####
def select_tomo_ids(df, number_of_slices=[300], number_of_motors=[0, 1], y_shape_range=(924, 960), x_shape_range=(924, 960)) -> pd.Series:
    '''
    Return the list of the tomo_ids obtained by filtering the DataFrame base on the given parameters

            Parameters:
                    df (pd.Dataframe): the dataset to filter
                    number_of_slices (list:int): number of slices per tomogram
                    max_number_of_motors (list:int): max number of motors
                    y_shape_range(tuple:int): tuple of the (min, max) y size of pictures
                    x_shape_range(tuple:int): tuple of the (min, max) x size of pictures

            Returns:
                    pd.Series: pandas Series of the tomo_ids corresponding to the filter
    '''
    df = df[(df['Array_shape_axis_1'] >= y_shape_range[0]) & (df['Array_shape_axis_2'] <= y_shape_range[1])]
    df = df[(df['Array_shape_axis_1'] >= x_shape_range[0]) & (df['Array_shape_axis_2'] <= x_shape_range[1])]
    df = df[(df['Array_shape_axis_0'].isin(number_of_slices)) & (df['Number_of_motors'].isin(number_of_motors))]


    return df.tomo_id
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
            labels.append(row['Number_of_motors'])
        else:
            print(f"⚠️ No image found for tomo_id: {tomo_id}")

    print(f"Matched {len(filtered_image_paths)} image-label pairs")

    labels = np.array(labels, dtype=np.float32)
    return filtered_image_paths, labels
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
    # Combine and optionally shuffle the data as a list of tuples
    data = list(zip(filtered_image_paths, labels))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(data)

    # Unzip the shuffled data back
    filtered_image_paths, labels = zip(*data)

    # Convert back to lists or arrays
    filtered_image_paths = list(filtered_image_paths)
    labels = list(labels)

    if split:
        # Compute sizes
        val_size = int(val_fraction * dataset_size)
        test_size = int(test_fraction * dataset_size)
        train_size = dataset_size - val_size - test_size

        # Split into slices
        test_paths = filtered_image_paths[:test_size]
        print(test_paths)
        test_labels = labels[:test_size]

        val_paths = filtered_image_paths[test_size:test_size + val_size]
        print(val_paths)
        val_labels = labels[test_size:test_size + val_size]

        train_paths = filtered_image_paths[test_size + val_size:]
        print(train_paths)
        train_labels = labels[test_size + val_size:]

        # Create tf.data.Dataset for each
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).map(read_img_jpg).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).map(read_img_jpg).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels)).map(read_img_jpg).batch(batch_size)

        return train_ds, val_ds, test_ds #, test_paths, test_labels

    else:
        # Single dataset
        dataset = tf.data.Dataset.from_tensor_slices((filtered_image_paths, labels))
        dataset = dataset.map(read_img_jpg).batch(batch_size)
        return dataset, filtered_image_paths, labels


####### Motor Detection #######

### TODO à packager ###
def generate_base_data():
  all_slices_number = df['Array_shape_axis_0'].unique()

  filtered_image_paths,labels = selection_images_labels(df, 'adaptequal_1_padded', num_slices=list(all_slices_number), num_motors=[0, 1])

  train_ds, val_ds, test_ds = batches_images_ram(
      read_img_jpg,
      filtered_image_paths,
      labels,
      shuffle=True,
      batch_size=32,
      split=True,
      val_fraction=0.2,
      test_fraction=0.2,
      seed=42)

  return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = generate_base_data()

X_test = []
y_test = []

for batch_x, batch_y in test_ds:
    X_test.append(batch_x.numpy())
    y_test.append(batch_y.numpy())

X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

classif_model = keras.saving.load_model('../models/classif.keras')

y_pred = classif_model.predict(X_test)

y_pred_labels = (y_pred > 0.5).astype(int)

sklearn_score = fbeta_score(y_test, y_pred_labels, beta=2)
print(f'fbeta_score before range check: {sklearn_score}')

pred_dict = {
    'tomo_id': test_tomos,
    'pred': y_pred_labels.tolist()
}

prediction_df = pd.DataFrame.from_dict(pred_dict)

prediction_df['pred'] = prediction_df['pred'].apply(lambda x: x[0])

df_preds = pd.merge(df, prediction_df, on='tomo_id', how='inner')

sklearn_score = fbeta_score(y_test, y_pred_labels, beta=2)
print(f'Fbeta score on motor presence prediction: {sklearn_score}')

####### Motor Position #######
df_regression = df_preds[(df_preds["pred"] == 1) & (df_preds["Number_of_motors"] == 1)]

### X, Y ###
reg_x_y = keras.saving.load_model('../models/reg_x_y.keras', compile=False)

################# A Packager #################
IMG_SIZE_ORIG = 960
IMG_SIZE = 240
MASK_RADIUS = 20  # pixels

def get_image_path(tomo_id):
    img_dir = '../data/pictures_process/adaptequal_1_padded'
    return os.path.join(img_dir, f'{tomo_id}.jpg')

def get_mask_path(tomo_id):
    img_dir = '../data/pictures_process/mask_multiclass/mask_multiclass'
    return os.path.join(img_dir, f'{tomo_id}.png')

def get_tomo_ids():
    folder = '../data/pictures_process/mask_multiclass/mask_multiclass'
    file_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.endswith('.png')
    ]
    return file_list

def get_xy(tomo_id):
    csv_path = '../data/csv_raw/train_labels.csv'
    df = pd.read_csv(csv_path)
    y = df[df['tomo_id'] == tomo_id]['Motor_axis_1'].values[0]
    x = df[df['tomo_id'] == tomo_id]['Motor_axis_2'].values[0]
    return x, y

def rgba_mask_to_class_indices(mask_path, target_size=(IMG_SIZE, IMG_SIZE)):
    mask = Image.open(mask_path).convert('RGBA').resize(target_size, resample=Image.NEAREST)
    mask_np = np.array(mask)
    class_indices = np.zeros(mask_np.shape[:2], dtype=np.uint8)
    # Transparent (alpha == 0)
    class_indices[mask_np[..., 3] == 0] = 0
    # Rouge
    red = (mask_np[..., 0] > 127) & (mask_np[..., 1] < 127) & (mask_np[..., 2] < 127) & (mask_np[..., 3] > 0)
    class_indices[red] = 1
    # Vert
    green = (mask_np[..., 0] < 127) & (mask_np[..., 1] > 127) & (mask_np[..., 2] < 127) & (mask_np[..., 3] > 0)
    class_indices[green] = 2
    # Bleu
    blue = (mask_np[..., 0] < 127) & (mask_np[..., 1] < 127) & (mask_np[..., 2] > 127) & (mask_np[..., 3] > 0)
    class_indices[blue] = 3
    return class_indices

def load_data(ids):
    X = []
    Y_mask = []
    Y_motor_xy = []
    tomo_ids = []  # <-- Ajout ici

    for tomo_id in ids:
        if tomo_id in get_tomo_ids():
            image_path = get_image_path(tomo_id)
            mask_path = get_mask_path(tomo_id)
            img = load_img(image_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            img_rgb = np.repeat(img_array, 3, axis=-1)
            X.append(img_rgb)

            mask_array = rgba_mask_to_class_indices(mask_path, target_size=(IMG_SIZE, IMG_SIZE))
            Y_mask.append(mask_array)

            x, y = get_xy(tomo_id)
            Y_motor_xy.append([x*IMG_SIZE/IMG_SIZE_ORIG, y*IMG_SIZE/IMG_SIZE_ORIG])

            tomo_ids.append(tomo_id)  # <-- Ajout ici

    # Conversion en np.array
    npX = np.array(X, dtype=np.float32)
    npY_mask = np.array(Y_mask, dtype=np.float32)
    npY_motor_xy = np.array(Y_motor_xy, dtype=np.float32)

    result = {
        'X': npX,
        'Y': npY_mask,
        'motor_xy': npY_motor_xy,
        'tomo_ids': tomo_ids
    }

    return result
################# A Packager #################


tomo_id_for_reg = df_regression.tomo_id

data = load_data(tomo_id_for_reg)

x_dict = {}
y_dict = {}

for i, X in enumerate(data['X']):
    pred = reg_x_y.predict(np.expand_dims(data["X"][i], axis=0))  # shape: (1, H, W, 4)
    mask_pred = np.argmax(pred[0], axis=-1)  # shape: (H, W)
    # Coordonnées prédites "moteur"
    ys, xs = np.where(mask_pred == 3)
    if len(xs) > 0:
        x_center = np.mean(xs)
        y_center = np.mean(ys)
        print(f"Moteur centre prédit : x = {x_center:.1f}, y = {y_center:.1f}")
        x_dict.update({data["tomo_ids"][i]: x_center*4})
        y_dict.update({data["tomo_ids"][i]: y_center*4})
    else:
        print("Aucune zone moteur détectée")
        ### x_center, y_center = None, None
        x_dict.update({data["tomo_ids"][i]: -1})
        y_dict.update({data["tomo_ids"][i]: -1})


x_df = pd.DataFrame.from_dict(x_dict, orient='index', columns=['Pred_motor_axis_2']).reset_index()
y_df = pd.DataFrame.from_dict(y_dict, orient='index', columns=['Pred_motor_axis_1']).reset_index()

motors_not_found = x_df[x_df['Pred_motor_axis_2'] == -1].shape[0]
total_motors = x_df.shape[0]
print(f'{motors_not_found} out of {total_motors} motors not found')


df_regression = pd.merge(df_regression, x_df, left_on='tomo_id', right_on='index', how='inner')
df_regression = pd.merge(df_regression, y_df, left_on='tomo_id', right_on='index', how='inner')
df_regression = df_regression.drop(columns=['index_x', 'index_y'])

df_regression['euclid_dist'] = tf.sqrt((df_regression['Motor_axis_2'] - df_regression['Pred_motor_axis_2'])**2 + (df_regression['Motor_axis_1'] - df_regression['Pred_motor_axis_1'])**2)

df_regression.loc[df_regression['Pred_motor_axis_1'] == -1, 'euclid_dist'] = np.nan


# preprocessing preds with no motors or no motors predicted
tmp = df_preds[(df_preds['Number_of_motors'] == 0) | ((df_preds['Number_of_motors'] == 1) & (df_preds['pred'] == 0))]
tmp.loc[:,'Pred_motor_axis_2'] = -1
tmp.loc[:,'Pred_motor_axis_1'] = -1
tmp.loc[:,'euclid_dist'] = np.nan

df_full = pd.concat([df_regression, tmp], axis=0)

assert df_full.shape[0] == df_preds.shape[0]
### Z ###


####### Range check #######
df_processed = df_full.copy()

df_processed['range'] = df_processed['Voxel_spacing'] * df_processed['euclid_dist']

# histplot
plt.hist(df_processed['range'], bins=40)
plt.vlines(1000, ymin=0, ymax=8, color='r', label='Kaggle range limit')
plt.vlines(2000, ymin=0, ymax=8, color='r', linestyles='dashed', label='Custom range limit')
plt.title('Histogram of euclidean distance between y_true and y_pred')
plt.legend()


# boxplot
df_processed.boxplot(column='range')
plt.hlines(1000, xmin=0.5, xmax=1.5, color='r', label='Kaggle range limit')
plt.hlines(2000, xmin=0.5, xmax=1.5, color='r', linestyles='dashed', label='Custom range limit')
plt.title('Boxplot of euclidean distance between y_true and y_pred')
plt.legend()





####### Final fbetascore #######
ANGSTROM_RANGE = 2000

t = df_processed.copy()

t.loc[t['range'] > ANGSTROM_RANGE, 'pred'] = 0
t.loc[(t['Number_of_motors'] == 1) & (t['pred'] == 1) & (t['Pred_motor_axis_2'] == -1), 'pred'] = 0

final_fbeta_score = fbeta_score(t['Number_of_motors'], t['pred'], beta=2)
print(f'final_fbeta_score: {final_fbeta_score}')



# test_tomos = ['tomo_dae195', 'tomo_f2fa4a', 'tomo_cabaa0', 'tomo_f7f28b', 'tomo_ed1c97', 'tomo_ff505c', 'tomo_8f4d60', 'tomo_2aeb29', 'tomo_651ecd', 'tomo_e96200', 'tomo_0d4c9e', 'tomo_2dcd5c', 'tomo_983fce', 'tomo_7b1ee3', 'tomo_8b6795', 'tomo_dcb9b4', 'tomo_e764a7', 'tomo_e26c6b', 'tomo_331130', 'tomo_f8b835', 'tomo_746d88', 'tomo_9cd09e', 'tomo_b9eb9a', 'tomo_cf0875', 'tomo_7cf523', 'tomo_fd41c4', 'tomo_54e1a7', 'tomo_ca472a', 'tomo_6478e5', 'tomo_e9b7f2', 'tomo_247826', 'tomo_675583', 'tomo_f0adfc', 'tomo_378f43', 'tomo_19a313', 'tomo_172f08', 'tomo_f3e449', 'tomo_3b83c7', 'tomo_8c13d9', 'tomo_2c607f', 'tomo_c11e12', 'tomo_412d88', 'tomo_4b124b', 'tomo_38c2a6', 'tomo_ec1314', 'tomo_1c38fd', 'tomo_e63ab4', 'tomo_f07244', 'tomo_210371', 'tomo_d6e3c7', 'tomo_935f8a', 'tomo_a4c52f', 'tomo_a46b26', 'tomo_fadbe2', 'tomo_b28579', 'tomo_35ec84', 'tomo_369cce', 'tomo_6c203d', 'tomo_b80310', 'tomo_640a74', 'tomo_22976c', 'tomo_d21396', 'tomo_ecbc12', 'tomo_040b80', 'tomo_85708b', 'tomo_b98cf6', 'tomo_e1e5d3', 'tomo_138018', 'tomo_3264bc', 'tomo_e50f04', 'tomo_d723cd', 'tomo_2a6ca2', 'tomo_1f0e78', 'tomo_67565e', 'tomo_fd5b38', 'tomo_05b39c', 'tomo_372a5c', 'tomo_c3619a', 'tomo_ba76d8', 'tomo_a67e9f', 'tomo_a6646f', 'tomo_db656f', 'tomo_4102f1', 'tomo_bb5ac1', 'tomo_4ed9de', 'tomo_61e947', 'tomo_1da0da', 'tomo_821255', 'tomo_3e7783', 'tomo_c84b46', 'tomo_974fd4', 'tomo_444829', 'tomo_b50c0f', 'tomo_2a6091', 'tomo_fa5d78', 'tomo_bdd3a0', 'tomo_1c2534', 'tomo_d916dc', 'tomo_bdc097', 'tomo_7036ee', 'tomo_cacb75', 'tomo_5b359d', 'tomo_7fa3b1', 'tomo_049310', 'tomo_dd36c9', 'tomo_e3864f', 'tomo_0a8f05', 'tomo_ff7c20', 'tomo_0fab19', 'tomo_1c75ac', 'tomo_d0699e', 'tomo_1e9980', 'tomo_4ee35e', 'tomo_6943e6', 'tomo_99a3ce']
# val_tomos = ['tomo_6f2c1f', 'tomo_dfc627', 'tomo_8d5995', 'tomo_cc2b5c', 'tomo_50cbd9', 'tomo_a72a52', 'tomo_9ae65f', 'tomo_9c0253', 'tomo_66285d', 'tomo_47d380', 'tomo_98686a', 'tomo_4077d8', 'tomo_97a2c6', 'tomo_ba9b3d', 'tomo_e2a336', 'tomo_aaa1fd', 'tomo_e8db69', 'tomo_532d49', 'tomo_f94504', 'tomo_5e2a91', 'tomo_2fc82d', 'tomo_16fce8', 'tomo_401341', 'tomo_0333fa', 'tomo_a81e01', 'tomo_b87c8e', 'tomo_e61cdf', 'tomo_b2ebbc', 'tomo_10c564', 'tomo_f71c16', 'tomo_47ac94', 'tomo_fea6e8', 'tomo_c00ab5', 'tomo_823bc7', 'tomo_278194', 'tomo_2fb12d', 'tomo_a537dd', 'tomo_19a4fd', 'tomo_417e5f', 'tomo_81445c', 'tomo_317656', 'tomo_7fbc49', 'tomo_806a8f', 'tomo_ab804d', 'tomo_957567', 'tomo_8634ee', 'tomo_fc1665', 'tomo_63e635', 'tomo_2645a0', 'tomo_5984bf', 'tomo_fc3c39', 'tomo_101279', 'tomo_08a6d6', 'tomo_0c2749', 'tomo_6607ec', 'tomo_23ce49', 'tomo_ca1d13', 'tomo_e55f81', 'tomo_bfd5ea', 'tomo_d7475d', 'tomo_136c8d', 'tomo_c4db00', 'tomo_ea3f3a', 'tomo_ef1a1a', 'tomo_2dd6bd', 'tomo_82d780', 'tomo_bede89', 'tomo_d5465a', 'tomo_e71210', 'tomo_9f1828', 'tomo_7550f4', 'tomo_efe1f8', 'tomo_bd42fa', 'tomo_01a877', 'tomo_59b470', 'tomo_0c3d78', 'tomo_d0c025', 'tomo_0eb41e', 'tomo_ca8be0', 'tomo_dbc66d', 'tomo_84997e', 'tomo_5dd63d', 'tomo_b9088c', 'tomo_24795a', 'tomo_6521dc', 'tomo_676744', 'tomo_cff77a', 'tomo_6f83d4', 'tomo_f78e91', 'tomo_6303f0', 'tomo_997437', 'tomo_cae587', 'tomo_9aee96', 'tomo_be9b98', 'tomo_97876d', 'tomo_e2da77', 'tomo_081a2d', 'tomo_cb5ec6', 'tomo_fc5ae4', 'tomo_4925ee', 'tomo_38d285', 'tomo_79a385', 'tomo_4469a7', 'tomo_05f919', 'tomo_568537', 'tomo_71ece1', 'tomo_85fa87', 'tomo_bcb115', 'tomo_2cace2', 'tomo_b4d92b', 'tomo_cc3fc4', 'tomo_94c173', 'tomo_a2a928', 'tomo_375513', 'tomo_40b215']
# train_tomos = ['tomo_c6f50a', 'tomo_288d4f', 'tomo_229f0a', 'tomo_decb81', 'tomo_39b15b', 'tomo_466489', 'tomo_d8c917', 'tomo_736dfa', 'tomo_03437b', 'tomo_066095', 'tomo_935ae0', 'tomo_c10f64', 'tomo_8e4919', 'tomo_2bb588', 'tomo_5bb31c', 'tomo_692081', 'tomo_ba37ec', 'tomo_20a9ed', 'tomo_b2b342', 'tomo_d396b5', 'tomo_b54396', 'tomo_122a02', 'tomo_4e3e37', 'tomo_9f918e', 'tomo_6cb0f0', 'tomo_f36495', 'tomo_e81143', 'tomo_6acb9e', 'tomo_56b9a3', 'tomo_dfdc32', 'tomo_98d455', 'tomo_2483bb', 'tomo_e5a091', 'tomo_91beab', 'tomo_b18127', 'tomo_73173f', 'tomo_221a47', 'tomo_5f1f0c', 'tomo_24a095', 'tomo_60d478', 'tomo_4f5a7b', 'tomo_975287', 'tomo_072a16', 'tomo_a8bf76', 'tomo_399bd9', 'tomo_512f98', 'tomo_4e38b8', 'tomo_146de2', 'tomo_423d52', 'tomo_711fad', 'tomo_b03f81', 'tomo_62dbea', 'tomo_a2bf30', 'tomo_25780f', 'tomo_8f5995', 'tomo_191bcb', 'tomo_372690', 'tomo_cf5bfc', 'tomo_f1bf2f', 'tomo_c678d9', 'tomo_cf53d0', 'tomo_c13fbf', 'tomo_e0739f', 'tomo_6d22d1', 'tomo_57c814', 'tomo_13973d', 'tomo_518a1f', 'tomo_dee783', 'tomo_556257', 'tomo_b11ddc', 'tomo_8e8368', 'tomo_e9fa5f', 'tomo_c649f8', 'tomo_517f70', 'tomo_622ca9', 'tomo_ac9fef', 'tomo_46250a', 'tomo_7eb641', 'tomo_bfdf19', 'tomo_6bb452', 'tomo_616f0b', 'tomo_f672c0', 'tomo_4c2e4e', 'tomo_9d3a0e', 'tomo_225d8f', 'tomo_d96d6e', 'tomo_5f34b3', 'tomo_93c0b4', 'tomo_464108', 'tomo_8554af', 'tomo_b24f1a', 'tomo_648adf', 'tomo_b7d94c', 'tomo_9ed470', 'tomo_db4517', 'tomo_3b8291', 'tomo_f427b3', 'tomo_f76529', 'tomo_5b087f', 'tomo_bc143f', 'tomo_774aae', 'tomo_b0e5c6', 'tomo_2f3261', 'tomo_b93a2d', 'tomo_ab78d0', 'tomo_c7b008', 'tomo_5764d6', 'tomo_2b996c', 'tomo_5308e8', 'tomo_e1a034', 'tomo_672101', 'tomo_0e9757', 'tomo_134bb0', 'tomo_643b20', 'tomo_d3bef7', 'tomo_f6de9b', 'tomo_8e58f1', 'tomo_4d528f', 'tomo_e72e60', 'tomo_0fe63f', 'tomo_a0cb00', 'tomo_221c8e', 'tomo_e57baf', 'tomo_02862f', 'tomo_738500', 'tomo_17143f', 'tomo_1446aa', 'tomo_0c3a99', 'tomo_8e4f7d', 'tomo_891afe', 'tomo_d0d9b6', 'tomo_aff073', 'tomo_0308c5', 'tomo_db2a10', 'tomo_d634b7', 'tomo_813916', 'tomo_30d4e5', 'tomo_6f0ee4', 'tomo_23c8a4', 'tomo_4f379f', 'tomo_37076e', 'tomo_37c426', 'tomo_fbb49b', 'tomo_e7c195', 'tomo_455dcd', 'tomo_fd9357', 'tomo_3eb9c8', 'tomo_cd1a7c', 'tomo_16136a', 'tomo_6a84b7', 'tomo_918e2b', 'tomo_c36b4b', 'tomo_f871ad', 'tomo_df866a', 'tomo_6733fa', 'tomo_3c6038', 'tomo_85edfd', 'tomo_4e41c2', 'tomo_9fc2b6', 'tomo_05df8a', 'tomo_7f0184', 'tomo_1cc887', 'tomo_06e11e', 'tomo_16efa8', 'tomo_8d231b', 'tomo_78b03d', 'tomo_32aaa7', 'tomo_1fb6a7', 'tomo_ac4f0d', 'tomo_971966', 'tomo_fb6ce6', 'tomo_50f0bf', 'tomo_285454', 'tomo_4b59a2', 'tomo_69d7c9', 'tomo_3183d2', 'tomo_2c9da1', 'tomo_d23087', 'tomo_3b7a22', 'tomo_4baff0', 'tomo_adc026', 'tomo_e5ac94', 'tomo_307f33', 'tomo_94a841', 'tomo_285d15', 'tomo_6bc974', 'tomo_51a47f', 'tomo_1dc5f9', 'tomo_7e3494', 'tomo_08bf73', 'tomo_0eb994', 'tomo_7dc063', 'tomo_79756f', 'tomo_6c5a26', 'tomo_76a42b', 'tomo_fb08b5', 'tomo_139d9e', 'tomo_e32b81', 'tomo_12f896', 'tomo_1af88d', 'tomo_603e40', 'tomo_f82a15', 'tomo_0f9df0', 'tomo_539259', 'tomo_fe85f6', 'tomo_fe050c', 'tomo_d31c96', 'tomo_c38e83', 'tomo_b10aa4', 'tomo_d83ff4', 'tomo_d5aa20', 'tomo_bbe766', 'tomo_305c97', 'tomo_5b8db4', 'tomo_5d798e', 'tomo_c8f3ce', 'tomo_881d84', 'tomo_da38ea', 'tomo_d0aa3b', 'tomo_abac2e', 'tomo_183270', 'tomo_51a77e', 'tomo_516cdd', 'tomo_c9d07c', 'tomo_656915', 'tomo_7dcfb8', 'tomo_a4f419', 'tomo_c4bfe2', 'tomo_eb4fd4', 'tomo_1ab322', 'tomo_6b1fd3', 'tomo_9dbc12', 'tomo_8e30f5', 'tomo_033ebe', 'tomo_a3ed10', 'tomo_4555b6', 'tomo_6e237a', 'tomo_5b34b2', 'tomo_3e6ead', 'tomo_891730', 'tomo_6df2d6', 'tomo_769126', 'tomo_95c0eb', 'tomo_bebadf', 'tomo_5f235a', 'tomo_3a3519', 'tomo_23a8e8', 'tomo_651ec2', 'tomo_3b1cc9', 'tomo_91c84c', 'tomo_a1a9a3', 'tomo_8ee8fd', 'tomo_28f9c1', 'tomo_67ff4e', 'tomo_acadd7', 'tomo_d2b1bc', 'tomo_53e048', 'tomo_a75c98', 'tomo_abb45a', 'tomo_493bea', 'tomo_0363f2', 'tomo_2c8ea2', 'tomo_30b580', 'tomo_a37a5c', 'tomo_569981', 'tomo_d84544', 'tomo_e685b8', 'tomo_ede779', 'tomo_b8595d', 'tomo_180bfd', 'tomo_4e478f', 'tomo_c596be', 'tomo_89d156', 'tomo_9f424e', 'tomo_71d2c0', 'tomo_646049', 'tomo_13484c', 'tomo_95e699', 'tomo_c36baf', 'tomo_9f222a', 'tomo_72b187', 'tomo_6a6a3b', 'tomo_91031e', 'tomo_abbd3b', 'tomo_7a9b64', 'tomo_868255', 'tomo_1efc28', 'tomo_319f79', 'tomo_256717', 'tomo_72763e', 'tomo_385eb6', 'tomo_3a0914', 'tomo_e34af8', 'tomo_4c1ca8', 'tomo_2acf68', 'tomo_24fda8', 'tomo_bad724', 'tomo_e51e5e', 'tomo_4e1b18', 'tomo_374ca7', 'tomo_9674bf', 'tomo_098751', 'tomo_c925ee', 'tomo_bde7f3', 'tomo_122c46', 'tomo_53c71b', 'tomo_e2ccab', 'tomo_68e123', 'tomo_9986f0', 'tomo_8e90f9', 'tomo_cc65a9', 'tomo_57592d', 'tomo_80bf0f', 'tomo_d9a2af', 'tomo_00e047', 'tomo_2a89bb', 'tomo_c4a4bb', 'tomo_0da370', 'tomo_6c4df3', 'tomo_47c399', 'tomo_9997b3', 'tomo_db6051', 'tomo_510f4e', 'tomo_60ddbd', 'tomo_bb9df3', 'tomo_79d622', 'tomo_499ee0', 'tomo_087d64', 'tomo_49f4ee', 'tomo_f8b46e', 'tomo_d2339b', 'tomo_2e1f4c', 'tomo_2c9f35', 'tomo_0a180f', 'tomo_a910fe', 'tomo_9cde9d', 'tomo_b7becf', 'tomo_d6c63f', 'tomo_be4a3a', 'tomo_c46d3c', 'tomo_161683', 'tomo_a020d7', 'tomo_b4d9da', 'tomo_513010', 'tomo_8f063a', 'tomo_b8f096', 'tomo_e77217', 'tomo_d26fcb']
