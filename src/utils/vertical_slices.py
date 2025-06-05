import numpy as np
from PIL import Image
import os

# Dossier contenant les images, triées dans l'ordre des couches
#folder = "./data/pictures_raw/train/tomo_5f34b3/"
folder = "./data/pictures_raw/train/tomo_1cc887/"
files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])

# Chargement des images dans une liste, puis stack en un seul array 3D
images = [np.array(Image.open(os.path.join(folder, f))) for f in files]
volume = np.stack(images, axis=0) 


x = 500  # colonne à extraire
slice_x = volume[:, :, x]  # shape: (n, hauteur)

y = 500
slice_y = volume[:, y, :]

z = 100
slice_z = volume[z, :, :]

# Supposons que slice_2d est ton array 2D (par exemple vertical_slice ou sagittal_slice)
# Si besoin, normaliser/convertir en uint8
img = Image.fromarray(slice_x.astype(np.uint8))
img.save("ma_tranche.png")  # ou img.show() pour l’afficher


mean = volume.mean(axis=1)  # shape: (n, largeur)
img = Image.fromarray(mean.astype(np.uint8))
img.save("mean.png")

def compute_vertical_slice(tomo_id, axe='x', indice_slice=150, output_dir='./data/pictures_process/vertical_slices/'):
    folder_source = f"./data/pictures_raw/train/{tomo_id}/"
    files = sorted([f for f in os.listdir(folder_source) if f.endswith('.jpg')])
    images = [np.array(Image.open(os.path.join(folder_source, f))) for f in files]
    volume = np.stack(images, axis=0) 
    if axe == 'x':
        slice = volume[:, :, indice_slice]
    elif axe == 'y':
        slice = volume[:, indice_slice, :]
    elif axe == 'z':
        slice = volume[indice_slice, :, :]
    
    img = Image.fromarray(slice.astype(np.uint8))
    img.save(f"{output_dir}/{tomo_id}_{axe}_{indice_slice}.png")

    