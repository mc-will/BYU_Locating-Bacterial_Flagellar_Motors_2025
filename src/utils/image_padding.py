import os
from PIL import Image


def padd_picture(image_source_path, image_destination_path, size):
    '''
    Pads une image avec des pixels blancs pour la rendre à la taille donnée.
    Parameters:
        image_source_path (str): le chemin de l'image source
        image_destination_path (str): le chemin de l'image destination
        size (int): la taille de la nouvelle image
    '''
    img = Image.open(image_source_path).convert('RGB')
    new_img = Image.new('RGB', (size, size), (255, 255, 255))
    new_img.paste(img, (0, 0))

    # creation si necessaire du repertoire de destination
    if not os.path.exists(os.path.dirname(image_destination_path)):
        os.makedirs(os.path.dirname(image_destination_path))
    new_img.save(image_destination_path)

def list_all_pictures_in_path(dir_path):
    '''
    Liste tous les fichiers dans un repertoire.
    Parameters:
        dir_path (str): le chemin du repertoire
    Returns:
        list: la liste des chemins des fichiers
    '''
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

def padd_mean_picture(size = 960):
    '''
    Pads toutes les images du repertoire mean_image avec des pixels blancs pour les rendre à la taille de 960x960.
    Parameters:
        size: taille finale de l'image avec padding (carrée)
    Returns:
        None
    '''
    input_dir = './data/pictures_process/mean_image/'
    output_dir = './data/pictures_process/mean_image_padded/'
    list_of_pictures_path = list_all_pictures_in_path(input_dir)
    for picture_path in list_of_pictures_path:
        file_name = os.path.basename(picture_path)
        padd_picture(picture_path, f'{output_dir}/{file_name}', size)


if __name__ == '__main__':
    padd_mean_picture()
