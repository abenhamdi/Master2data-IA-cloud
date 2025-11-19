"""
Fonctions utilitaires pour le pretraitement des images
"""
import numpy as np
from PIL import Image
import cv2
import io

# Configuration
IMG_SIZE = 224  # Taille standard pour les CNN (224x224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Ordre alphabetique

def preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Pretraite une image pour le modele
    
    Args:
        image_path: chemin vers l'image
        target_size: taille cible (width, height)
    
    Returns:
        numpy array: image pretraitee normalisee
    """
    # TODO: Charger l'image
    # img = Image.open(image_path)
    
    # TODO: Convertir en RGB si necessaire
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    
    # TODO: Redimensionner a target_size
    # img = img.resize(target_size)
    
    # TODO: Convertir en numpy array
    # img_array = np.array(img)
    
    # TODO: Normaliser les valeurs (0-1)
    # img_array = img_array / 255.0
    
    # TODO: Ajouter dimension batch si necessaire
    # img_array = np.expand_dims(img_array, axis=0)
    
    pass

def preprocess_image_from_upload(file_storage, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Pretraite une image uploadee (Flask FileStorage)
    
    Args:
        file_storage: objet FileStorage de Flask
        target_size: taille cible
    
    Returns:
        numpy array: image pretraitee
    """
    # TODO: Lire l'image depuis FileStorage
    # image_bytes = file_storage.read()
    
    # TODO: Convertir en PIL Image
    # img = Image.open(io.BytesIO(image_bytes))
    
    # TODO: Convertir en RGB
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    
    # TODO: Redimensionner
    # img = img.resize(target_size)
    
    # TODO: Convertir en array et normaliser
    # img_array = np.array(img) / 255.0
    
    # TODO: Ajouter dimension batch
    # img_array = np.expand_dims(img_array, axis=0)
    
    pass

def get_class_name(class_index):
    """
    Retourne le nom de la classe a partir de l'index
    
    Args:
        class_index: index de la classe (0-3)
    
    Returns:
        str: nom de la classe
    """
    return CLASSES[class_index]

def get_class_description(class_name):
    """
    Retourne une description de la classe
    
    Args:
        class_name: nom de la classe
    
    Returns:
        str: description
    """
    descriptions = {
        'glioma': 'Tumeur gliale - Tumeur cerebrale debutant dans les cellules gliales',
        'meningioma': 'Meningiome - Tumeur des meninges entourant le cerveau',
        'pituitary': 'Tumeur pituitaire - Tumeur de la glande pituitaire',
        'notumor': 'Aucune tumeur detectee - IRM normale'
    }
    return descriptions.get(class_name, 'Description non disponible')

