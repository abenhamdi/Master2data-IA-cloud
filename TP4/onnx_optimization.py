"""
Export ONNX et optimisation de l'inference

Compare PyTorch vs ONNX Runtime pour l'inference.

Usage:
    python onnx_optimization.py --category bottle --model-path model_cpu_bottle.pth
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import time
import argparse
import os
from load_model_hf import load_anomaly_detection_model, adapt_model_for_binary
from mvtec_dataset import create_mvtec_dataloaders
from train_utils import evaluate_model, measure_inference_time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def export_to_onnx(model, dummy_input, onnx_path="model_anomaly.onnx", opset_version=14):
    """
    Exporte un modele vers ONNX
    
    Args:
        model: Modele PyTorch
        dummy_input: Exemple d'entree
        onnx_path: Chemin de sauvegarde
        opset_version: Version ONNX (14 recommande pour Transformers)
    
    Returns:
        str: Chemin du fichier ONNX
    """
    print("="*70)
    print("EXPORT VERS ONNX")
    print("="*70)
    
    model.eval()
    model.cpu()
    
    print(f"\nExport vers {onnx_path}...")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Opset version : {opset_version}")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    # Verification
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nModele ONNX exporte")
    print(f"  Fichier : {onnx_path}")
    print(f"  Taille : {file_size:.2f} MB")
    print(f"  Operations : {len(onnx_model.graph.node)}")
    
    return onnx_path


def run_onnx_inference(onnx_path, data_loader, num_batches=100):
    """
    Execute l'inference avec ONNX Runtime
    
    Args:
        onnx_path: Chemin vers le modele ONNX
        data_loader: DataLoader
        num_batches: Nombre de batches
    
    Returns:
        tuple: (temps_moyen_ms, metrics)
    """
    print("\n" + "="*70)
    print("INFERENCE ONNX RUNTIME")
    print("="*70)
    
    # Creer session ONNX
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    print(f"\nProvider : {session.get_providers()[0]}")
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    times = []
    all_preds = []
    all_labels = []
    
    print(f"\nInference sur {num_batches} batches...")
    
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        
        inputs_numpy = inputs.numpy()
        labels_numpy = labels.numpy()
        
        # Mesure temps
        start = time.time()
        ort_outputs = session.run([output_name], {input_name: inputs_numpy})
        end = time.time()
        
        times.append(end - start)
        
        # Predictions
        logits = ort_outputs[0]
        preds = np.argmax(logits, axis=1)
        
        all_preds.extend(preds)
        all_labels.extend(labels_numpy)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{num_batches}")
    
    # Metriques
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    avg_time = (sum(times) / len(times)) * 1000
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
    
    print(f"\nResultats ONNX :")
    print(f"  Temps : {avg_time:.2f} ms/batch")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  F1-Score : {metrics['f1']:.2f}%")
    
    return avg_time, metrics


def compare_pytorch_onnx(category='bottle', model_path=None, batch_size=32):
    """
    Compare PyTorch vs ONNX Runtime
    
    Args:
        category: Categorie MVTec AD
        model_path: Chemin vers modele entraine
        batch_size: Taille des batches
    
    Returns:
        dict: Resultats de la comparaison
    """
    print("="*70)
    print("COMPARAISON PYTORCH vs ONNX RUNTIME")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Charger modele
    print("\nChargement du modele...")
    model, processor = load_anomaly_detection_model()
    model = adapt_model_for_binary(model, num_classes=2, freeze_backbone=False)
    
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Poids charges depuis {model_path}")
        except:
            print("Utilisation du modele pre-entraine")
    
    model.to(device)
    model.eval()
    
    # Charger donnees
    _, test_loader = create_mvtec_dataloaders(
        root_dir='./data/mvtec',
        category=category,
        batch_size=batch_size,
        processor=processor
    )
    
    # Export ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = f"model_{category}_optimized.onnx"
    export_to_onnx(model, dummy_input, onnx_path)
    
    # ===== BENCHMARK PYTORCH =====
    print("\n" + "="*70)
    print("BENCHMARK PYTORCH")
    print("="*70)
    
    pytorch_time = measure_inference_time(model, test_loader, device, num_batches=50)
    pytorch_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\nResultats PyTorch :")
    print(f"  Temps : {pytorch_time:.2f} ms/batch")
    print(f"  Accuracy : {pytorch_metrics['accuracy']:.2f}%")
    print(f"  F1-Score : {pytorch_metrics['f1']:.2f}%")
    
    # ===== BENCHMARK ONNX =====
    onnx_time, onnx_metrics = run_onnx_inference(onnx_path, test_loader, num_batches=50)
    
    # ===== COMPARAISON =====
    print("\n" + "="*70)
    print("COMPARAISON DETAILLEE")
    print("="*70)
    
    speedup = pytorch_time / onnx_time
    accuracy_diff = abs(pytorch_metrics['accuracy'] - onnx_metrics['accuracy'])
    
    print(f"\nPerformance :")
    print(f"  Speedup ONNX : {speedup:.2f}x")
    print(f"  Gain temps : {((pytorch_time - onnx_time) / pytorch_time) * 100:.1f}%")
    
    print(f"\nPrecision :")
    print(f"  Difference accuracy : {accuracy_diff:.3f}%")
    
    if accuracy_diff < 0.5:
        print(f"  Precision preservee (< 0.5%)")
    
    # Analyse cout
    print(f"\n" + "="*70)
    print("ANALYSE COUT PRODUCTION")
    print("="*70)
    
    images_per_day = 10000
    batches_per_day = images_per_day / batch_size
    
    pytorch_hours = (pytorch_time / 1000 / 3600) * batches_per_day
    onnx_hours = (onnx_time / 1000 / 3600) * batches_per_day
    
    cost_per_hour = 0.50  # Azure ML CPU
    
    pytorch_cost = pytorch_hours * cost_per_hour
    onnx_cost = onnx_hours * cost_per_hour
    savings = pytorch_cost - onnx_cost
    
    print(f"\nHypothese : {images_per_day:,} images/jour")
    print(f"Cout quotidien :")
    print(f"  PyTorch : ${pytorch_cost:.2f}/jour")
    print(f"  ONNX : ${onnx_cost:.2f}/jour")
    print(f"  Economie : ${savings:.2f}/jour (${savings * 30:.2f}/mois)")
    
    if speedup > 1.5:
        print(f"\nRECOMMANDATION : Deployer avec ONNX Runtime")
    else:
        print(f"\nRECOMMANDATION : Evaluer le trade-off")
    
    return {
        'pytorch': {'time': pytorch_time, 'metrics': pytorch_metrics},
        'onnx': {'time': onnx_time, 'metrics': onnx_metrics},
        'speedup': speedup,
        'savings_per_month': savings * 30
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimisation ONNX')
    parser.add_argument('--category', default='bottle', help='Categorie MVTec AD')
    parser.add_argument('--model-path', default=None, help='Chemin vers modele entraine')
    parser.add_argument('--batch-size', type=int, default=32, help='Taille des batches')
    
    args = parser.parse_args()
    
    results = compare_pytorch_onnx(args.category, args.model_path, args.batch_size)

