import numpy as np
from utils.config import THRESHOLD, PERSON_LABEL, MASK_COLOR, ALPHA
from PIL import Image
from torch import Tensor

def apply_saved_mask(image, image_name, folder_name):

    # Convertir l'image en tableau numpy
    img_np = np.array(image)
    masks = np.load(f'saved_masks/{folder_name}/{image_name}.npy')
    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for i, mask in enumerate(masks):  
        for c in range(3):
            img_np[:, :, c] = np.where(mask, 
                                    (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                    img_np[:, :, c])
    
    # Convertir en image à partir d'un tableau numpy et le renvoyer            
    return Image.fromarray(img_np.astype(np.uint8))

def get_masks(image_name, folder_name):
    masks = np.load(f'saved_masks/{folder_name}/{image_name}.npy')
    return masks
