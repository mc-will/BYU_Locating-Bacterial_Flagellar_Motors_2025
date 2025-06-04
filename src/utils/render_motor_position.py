# accès au chemin d'une slice en fonction de l'identifiant du tomogramme et de l'indice de la slice
import os
from PIL import Image, ImageDraw
import pandas as pd

def get_slice_file_path(tomogram_id, z):
    '''
    accès au chemin d'une slice en fonction de l'identifiant du tomogramme et de l'indice de la slice
    Parameters:
        tomogram_id (str): l'identifiant du tomogramme
        z (int): l'indice de la slice
    Returns:
        str: le chemin de l'image
    '''
    tomogrammes_train_dir = './data/pictures_raw/train/'
    file_name = f'slice_{str(z).zfill(4)}.jpg'
    image_path = os.path.join(tomogrammes_train_dir, tomogram_id, file_name)
    return image_path

def get_motor_coordinates(df, tomogram_id):
    '''
    Récupération des coordonnées du moteur dans le tomogramme
    Parameters:
        df (pd.Dataframe): le dataframe des données
        tomogram_id (str): l'identifiant du tomogramme
    Returns:
        tuple: les coordonnées du moteur
    '''
    df_tomogram = df[df['tomo_id'] == tomogram_id]
    x = df_tomogram['Motor_axis_2'].values[0]
    y = df_tomogram['Motor_axis_1'].values[0]
    z = df_tomogram['Motor_axis_0'].values[0]
    return x, y, z


def draw_on_image(path_image_source, dir_path_destination, x, y, x_pred=-1, y_pred=-1):
    '''
    Dessine un moteur et un moteur prédit sur une image
    Parameters:
        path_image_source (str): le chemin de l'image source
        dir_path_destination (str): le chemin de destination
        x (int): l'abscisse du moteur
        y (int): l'ordonnée du moteur
        x_pred (int): l'abscisse du moteur prédit
        y_pred (int): l'ordonnée du moteur prédit
    Returns:
        None
    '''
    def draw_marker(dessin, x, y, rayon, color):
        rayon = 25
        left_up = (x - rayon, y - rayon)
        right_down = (x + rayon, y + rayon)
        dessin.ellipse([left_up, right_down], outline=color, width=2)

        # dessin d'un point au centre du cercle
        dessin.ellipse(
            [(x - 1, y - 1), (x + 1, y + 1)],
            fill=color, outline=color)

    # Ouvre l'image
    img = Image.open(path_image_source).convert('RGB')
    # Prépare le dessin
    dessin = ImageDraw.Draw(img)

    draw_marker(dessin, x, y, 25, 'green')
    if x_pred != -1 and y_pred != -1:
        draw_marker(dessin, x_pred, y_pred, 25, 'red')

    # Sauvegarde
    name_image_source = path_image_source.split('/')[-1]
    output_file = os.path.join(dir_path_destination, name_image_source)
    if not os.path.exists(dir_path_destination):
        print(f'create directory: {dir_path_destination}')
        os.makedirs(dir_path_destination)
    img.save(output_file)

def _render_tomogramme_to_file(tomogram_id, z, y, x):
    '''
    Render d'un tomogramme avec un moteur dans le repertoire "output_path"
    Parameters:
        tomogram_id (str): l'identifiant du tomogramme
        z (int): l'indice de la slice
        y (int): l'ordonnée du moteur
        x (int): l'abscisse du moteur
    '''
    # recherche du chemin de l'image
    image_path = get_slice_file_path(tomogram_id, z)
    output_path = './data/pictures_process/motor_position/'
    draw_on_image(image_path, output_path, x, y)

def render_all_tomogrammes():
    '''
    Render tous les tomogrammes avec un seul moteur
    '''
    df_train = pd.read_csv('./data/csv_raw/train_labels.csv')
    # cast des coordonnées en int depuis float
    df_train['Motor_axis_0'] = df_train['Motor_axis_0'].astype(int)
    df_train['Motor_axis_1'] = df_train['Motor_axis_1'].astype(int)
    df_train['Motor_axis_2'] = df_train['Motor_axis_2'].astype(int)

    # liste des tomogrammes à 1 seul moteur
    df_train_1 = df_train[df_train['Number_of_motors']==1]
    id_1_moteurs = df_train_1['tomo_id'].unique()

    # render des tomogrammes avec un moteur
    for tomogram_id in id_1_moteurs:
        x, y, z = get_motor_coordinates(df_train, tomogram_id)
        _render_tomogramme_to_file(tomogram_id,z,y,x)

if __name__ == '__main__':
    render_all_tomogrammes()
