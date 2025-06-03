# accès au chemin d'une slice en fonction de l'identifiant du tomogramme et de l'indice de la slice
import os
from PIL import Image, ImageDraw
import pandas as pd

def _get_slice_file_path(tomogram_id, z):
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
    print(f'get_slice_file_path: {tomogram_id} {z} {file_name}')
    image_path = os.path.join(tomogrammes_train_dir, tomogram_id, file_name)
    return image_path

def _get_motor_coordinates(df, tomogram_id):
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
    print(f'get_motor_coordinates: {tomogram_id} {x} {y} {z}')
    return x, y, z


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
    image_path = _get_slice_file_path(tomogram_id, z)

    # Ouvre l'image
    img = Image.open(image_path).convert('RGB')

    # Prépare le dessin
    draw = ImageDraw.Draw(img)

    # Dessine un cercle (défini par son bounding box)
    rayon = 25
    left_up = (x - rayon, y - rayon)
    right_down = (x + rayon, y + rayon)
    draw.ellipse([left_up, right_down], outline='red', width=2)

    # dessin d'un point au centre du cercle
    draw.ellipse(
        [(x - 1, y - 1), (x + 1, y + 1)],
        fill='red', outline='red'
    )

    # Sauvegarde
    output_path = './data/pictures_process/motor_position/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img.save(f'{output_path}/{tomogram_id}_{z}.jpg')


def render_all_tomogrammes():
    '''
    Render tous les tomogrammes avec un seul moteur
    '''
    # afficher le repertoire courant
    print("---------------------------------")
    print(os.getcwd())

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
        print(f'{tomogram_id}')
        x, y, z = _get_motor_coordinates(df_train, tomogram_id)
        print(f'{x} {y} {z}')
        _render_tomogramme_to_file(tomogram_id,z,y,x)

if __name__ == '__main__':
    render_all_tomogrammes()