import numpy as np

# Seuil pour la classification ou la segmentation
THRESHOLD = 0.5
# ID de label pour la classification des personnes
PERSON_LABEL = 1  
# Couleur du masque pour les visualisations (rouge dans ce cas)
MASK_COLOR = np.array([255, 0, 0]) 
# Valeur alpha pour la transparence ou le mélange
ALPHA = 0.4 