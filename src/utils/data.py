import pandas as pd

from google.cloud import bigquery

from utils.render_motor_position import get_motor_coordinates, get_slice_file_path


def get_csv_from_bq():
    '''
    Fetch the 'train_labels' table from bq dataset 'spatial-encoder-456811-u6.datasets_wagon1992_group_project' and saves it in 'data/csv_raw'
    '''
    gcp_project = 'spatial-encoder-456811-u6'
    query = 'SELECT * FROM `spatial-encoder-456811-u6.datasets_wagon1992_group_project.train_labels`'

    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    df = query_job.result().to_dataframe().reset_index().drop(columns='index')

    df.to_csv('../data/csv_raw/train_labels.csv')


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


def get_best_slice(df, tomogram_id):
    '''
    Find the coordinates of the motor (only work with 1 motor tomogram), the use the z to get the picture_path where
    the motor is located, i.e. the 'best slice'

            Parameters:
                df (pd.Dataframe): the dataset to find coordinates
                tomogram_id (string): the tomogram id

            Returns:
                best_slice_path (string): the path of the picture of the 'best slice' (where the motor is located)

    '''
    x, y, z = get_motor_coordinates(df, tomogram_id)
    best_slice_path = get_slice_file_path(tomogram_id, int(z))
    return best_slice_path
