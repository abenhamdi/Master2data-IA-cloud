"""
Demonstration de la quantization pour reduction de taille

Compare un modele Vision Transformer original avec sa version quantifiee.

Usage:
    python quantization_demo.py --category bottle --model-path model_cpu_bottle.pth
"""

import torch
import torch.nn as nn
import torch.quantization
import argparse
from load_model_hf import load_anomaly_detection_model, adapt_model_for_binary
from mvtec_dataset import create_mvtec_dataloaders
from train_utils import evaluate_model, measure_inference_time, get_model_size


def apply_dynamic_quantization(model):
    """
    Applique la quantization dynamique sur les couches Linear
    
    Args:
        model: Modele PyTorch/Hugging Face
    
    Returns:
        model: Modele quantifie
    """
    print("\nApplication de la quantization dynamique...")
    
    # Le modele doit etre sur CPU
    model_cpu = model.cpu()
    
    # Quantization dynamique (poids en int8, activations en float32)
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Quantifier toutes les couches Linear
        dtype=torch.qint8
    )
    
    print("Quantization appliquee sur les couches Linear")
    
    return quantized_model


def compare_quantization(category='bottle', model_path=None):
    """
    Compare le modele original avec le modele quantifie
    
    Args:
        category: Categorie MVTec AD
        model_path: Chemin vers le modele entraine (optionnel)
    
    Returns:
        dict: Resultats de la comparaison
    """
    print("="*70)
    print("COMPARAISON MODELE ORIGINAL vs QUANTIFIE")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Charger le modele
    print("\nChargement du modele...")
    model_original, processor = load_anomaly_detection_model()
    model_original = adapt_model_for_binary(model_original, num_classes=2, freeze_backbone=False)
    
    # Charger les poids entraines si disponibles
    if model_path and torch.cuda.is_available():
        try:
            model_original.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Poids charges depuis {model_path}")
        except:
            print("Impossible de charger les poids, utilisation du modele pre-entraine")
    
    model_original.to(device)
    model_original.eval()
    
    # Charger les donnees
    _, test_loader = create_mvtec_dataloaders(
        root_dir='./data/mvtec',
        category=category,
        batch_size=32,
        processor=processor
    )
    
    # ===== MODELE ORIGINAL =====
    print("\n" + "="*70)
    print("ANALYSE DU MODELE ORIGINAL")
    print("="*70)
    
    original_size = get_model_size(model_original, "temp_original.pth")
    original_time = measure_inference_time(model_original, test_loader, device, num_batches=50)
    original_metrics = evaluate_model(model_original, test_loader, device)
    
    print(f"\nModele original :")
    print(f"  Taille : {original_size:.2f} MB")
    print(f"  Temps inference : {original_time:.2f} ms/batch")
    print(f"  Accuracy : {original_metrics['accuracy']:.2f}%")
    print(f"  F1-Score : {original_metrics['f1']:.2f}%")
    
    # ===== QUANTIZATION =====
    print("\n" + "="*70)
    print("QUANTIZATION DYNAMIQUE")
    print("="*70)
    
    model_quantized = apply_dynamic_quantization(model_original)
    
    quantized_size = get_model_size(model_quantized, "temp_quantized.pth")
    quantized_time = measure_inference_time(model_quantized, test_loader, device, num_batches=50)
    quantized_metrics = evaluate_model(model_quantized, test_loader, device)
    
    print(f"\nModele quantifie :")
    print(f"  Taille : {quantized_size:.2f} MB")
    print(f"  Temps inference : {quantized_time:.2f} ms/batch")
    print(f"  Accuracy : {quantized_metrics['accuracy']:.2f}%")
    print(f"  F1-Score : {quantized_metrics['f1']:.2f}%")
    
    # ===== COMPARAISON =====
    print("\n" + "="*70)
    print("COMPARAISON DETAILLEE")
    print("="*70)
    
    compression_ratio = original_size / quantized_size
    speedup = original_time / quantized_time
    accuracy_loss = original_metrics['accuracy'] - quantized_metrics['accuracy']
    f1_loss = original_metrics['f1'] - quantized_metrics['f1']
    
    print(f"\nGains :")
    print(f"  Compression : {compression_ratio:.2f}x")
    print(f"  Reduction taille : {((original_size - quantized_size) / original_size) * 100:.1f}%")
    print(f"  Speedup : {speedup:.2f}x")
    print(f"  Gain temps : {((original_time - quantized_time) / original_time) * 100:.1f}%")
    
    print(f"\nImpact precision :")
    print(f"  Perte accuracy : {accuracy_loss:.2f}%")
    print(f"  Perte F1-Score : {f1_loss:.2f}%")
    
    # Validation objectif metier
    print(f"\n" + "="*70)
    print("VALIDATION OBJECTIF METIER")
    print("="*70)
    
    target_size = 30  # MB
    target_retention = 95  # %
    
    retention = (quantized_metrics['accuracy'] / original_metrics['accuracy']) * 100
    
    print(f"\nObjectif : Taille < {target_size} MB, Retention > {target_retention}%")
    print(f"Resultat :")
    print(f"  Taille : {quantized_size:.2f} MB {'OK' if quantized_size < target_size else 'KO'}")
    print(f"  Retention : {retention:.1f}% {'OK' if retention > target_retention else 'KO'}")
    
    if quantized_size < target_size and retention > target_retention:
        print(f"\nOBJECTIF ATTEINT : Deploiement edge possible")
    else:
        print(f"\nOBJECTIF NON ATTEINT : Optimisations supplementaires necessaires")
    
    # Sauvegarder
    torch.save(model_quantized.state_dict(), f'model_quantized_{category}.pth')
    print(f"\nModele quantifie sauvegarde : model_quantized_{category}.pth")
    
    return {
        'original': {'size': original_size, 'time': original_time, 'metrics': original_metrics},
        'quantized': {'size': quantized_size, 'time': quantized_time, 'metrics': quantized_metrics},
        'gains': {'compression': compression_ratio, 'speedup': speedup}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstration quantization')
    parser.add_argument('--category', default='bottle', help='Categorie MVTec AD')
    parser.add_argument('--model-path', default=None, help='Chemin vers modele entraine')
    
    args = parser.parse_args()
    
    results = compare_quantization(args.category, args.model_path)

