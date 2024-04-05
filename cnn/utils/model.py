from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# Fonction pour charger le modèle pré-entraîné et ses transformations
def get_model():
    
    # Charger les poids par défaut pour le modèle Mask R-CNN avec ResNet-50
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    
    # Obtenir les transformations associées aux poids
    transformsations = weights.transforms()
    
    # Initialiser le modèle avec les poids et le mettre en mode évaluation
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    # Retourner le modèle et les transformations
    return model, transformsations