"""
Chargement et adaptation de modeles Vision Transformer depuis Hugging Face

Usage:
    from load_model_hf import load_anomaly_detection_model, adapt_model_for_binary
    
    model, processor = load_anomaly_detection_model()
    model = adapt_model_for_binary(model, num_classes=2)
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests


def load_anomaly_detection_model(model_name="microsoft/swin-tiny-patch4-window7-224"):
    """
    Charge un modele Vision Transformer depuis Hugging Face
    
    Args:
        model_name: Nom du modele sur Hugging Face Hub
                   Options:
                   - microsoft/swin-tiny-patch4-window7-224 (28M params)
                   - google/vit-base-patch16-224 (86M params)
                   - facebook/deit-base-distilled-patch16-224 (87M params)
    
    Returns:
        tuple: (model, processor)
    """
    print("="*70)
    print("CHARGEMENT MODELE DEPUIS HUGGING FACE")
    print("="*70)
    print(f"\nModele : {model_name}")
    
    # Charger le processeur d'images
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Charger le modele
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Informations sur le modele
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModele charge : {model.__class__.__name__}")
    print(f"Nombre de parametres : {num_params:,}")
    print(f"Taille d'entree : {processor.size}")
    
    return model, processor


def adapt_model_for_binary(model, num_classes=2, freeze_backbone=True):
    """
    Adapte un modele pre-entraine pour la classification binaire
    
    Args:
        model: Modele pre-entraine depuis Hugging Face
        num_classes: Nombre de classes (2 pour binaire : normal/anomalie)
        freeze_backbone: Geler les couches pre-entrainees
    
    Returns:
        model: Modele adapte
    """
    print("\n" + "="*70)
    print("ADAPTATION DU MODELE POUR DETECTION D'ANOMALIES")
    print("="*70)
    
    # Geler les couches pre-entrainees (optionnel)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("\nCouches pre-entrainees gelees")
    
    # Remplacer la tete de classification
    # Pour Swin Transformer : model.classifier
    # Pour ViT : model.classifier
    
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        print(f"Tete de classification remplacee : {in_features} -> {num_classes}")
    else:
        print("ATTENTION : Structure de modele non reconnue")
    
    # Compter les parametres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParametres totaux : {total_params:,}")
    print(f"Parametres entrainables : {trainable_params:,}")
    print(f"Parametres geles : {total_params - trainable_params:,}")
    
    return model


def test_huggingface_api(image_path, model_name="microsoft/swin-tiny-patch4-window7-224", api_token=None):
    """
    Teste l'inference via l'API Hugging Face
    
    Args:
        image_path: Chemin vers une image de test
        model_name: Nom du modele sur HF Hub
        api_token: Token API Hugging Face (optionnel)
    
    Returns:
        dict: Resultats de l'inference
    """
    print("\n" + "="*70)
    print("TEST API HUGGING FACE INFERENCE")
    print("="*70)
    
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # Lire l'image
    with open(image_path, "rb") as f:
        data = f.read()
    
    # Appeler l'API
    print(f"\nEnvoi de la requete a l'API...")
    response = requests.post(API_URL, headers=headers, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Reponse recue : {len(result)} predictions")
        
        # Afficher les top 3 predictions
        if isinstance(result, list) and len(result) > 0:
            print("\nTop 3 predictions :")
            for i, pred in enumerate(result[:3]):
                print(f"  {i+1}. {pred['label']} : {pred['score']:.4f}")
        
        return result
    else:
        print(f"Erreur API : {response.status_code}")
        print(f"Message : {response.text}")
        return None


if __name__ == "__main__":
    # Test du chargement
    model, processor = load_anomaly_detection_model()
    
    # Adaptation pour detection binaire
    model = adapt_model_for_binary(model, num_classes=2, freeze_backbone=True)
    
    # Test avec une entree aleatoire
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nTest forward pass : {output.logits.shape}")
    
    # Test API (optionnel - necessite une image)
    # test_huggingface_api("./data/mvtec/bottle/test/good/000.png")

