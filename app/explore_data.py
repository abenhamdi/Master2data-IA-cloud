"""
Script d'exploration du dataset d'IRM cerebrales
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

def count_images_per_class(base_path='dataset/Training'):
    """
    Compte le nombre d'images par classe
    
    Returns:
        dict: {classe: nombre_images}
    """
    classes = {}
    
    # TODO: Parcourir les dossiers et compter les images
    # Indice: os.listdir(), len()
    # Pour chaque classe: glioma, meningioma, pituitary, notumor
    
    # classes['glioma'] = len([f for f in os.listdir(...) if f.endswith(('.jpg', '.png'))])
    
    return classes

def visualize_samples(base_path='dataset/Training', samples_per_class=3):
    """
    Affiche des exemples d'images pour chaque classe
    
    Args:
        base_path: chemin vers le dossier Training
        samples_per_class: nombre d'echantillons a afficher par classe
    """
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    
    # TODO: Creer une grille d'images avec matplotlib
    # Indice: plt.subplots(), Image.open()
    
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(15, 12))
    fig.suptitle('Echantillons d\'IRM par classe de tumeur', fontsize=16)
    
    # TODO: Pour chaque classe
    #   Pour chaque echantillon
    #     Charger l'image
    #     Afficher dans la grille
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    plt.show()

def analyze_image_sizes(base_path='dataset/Training'):
    """
    Analyse les dimensions des images
    
    Returns:
        dict: statistiques sur les dimensions
    """
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    widths = []
    heights = []
    
    # TODO: Parcourir les images et collecter les dimensions
    # Indice: Image.open(path).size
    
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        # Votre code ici
        # Pour chaque image:
        #   img = Image.open(image_path)
        #   width, height = img.size
        #   widths.append(width)
        #   heights.append(height)
    
    if widths and heights:
        print(f"\nStatistiques des dimensions:")
        print(f"   Largeur moyenne: {np.mean(widths):.0f}px")
        print(f"   Hauteur moyenne: {np.mean(heights):.0f}px")
        print(f"   Largeur min-max: {np.min(widths)}-{np.max(widths)}px")
        print(f"   Hauteur min-max: {np.min(heights)}-{np.max(heights)}px")
    
    return {'widths': widths, 'heights': heights}

if __name__ == "__main__":
    print("Exploration du dataset d'IRM cerebrales")
    print("=" * 60)
    
    # Compter les images
    print("\nNombre d'images par classe:")
    counts = count_images_per_class()
    for classe, count in counts.items():
        print(f"   {classe}: {count} images")
    
    # Analyser les dimensions
    analyze_image_sizes()
    
    # Visualiser des echantillons
    print("\nGeneration des visualisations...")
    visualize_samples()
    
    print("\nExploration terminee!")

