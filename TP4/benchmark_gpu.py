"""
Benchmark CPU vs GPU pour detection d'anomalies

Compare les performances d'entrainement sur CPU et GPU avec un modele
Vision Transformer fine-tune pour la detection d'anomalies industrielles.

Usage:
    python benchmark_gpu.py --category bottle --epochs 3
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from load_model_hf import load_anomaly_detection_model, adapt_model_for_binary
from mvtec_dataset import create_mvtec_dataloaders
from train_utils import train_model, evaluate_model


def run_benchmark(category='bottle', num_epochs=3, batch_size=16):
    """
    Execute le benchmark sur CPU et GPU
    
    Args:
        category: Categorie MVTec AD
        num_epochs: Nombre d'epoques
        batch_size: Taille des batches
    
    Returns:
        dict: Resultats du benchmark
    """
    print("="*70)
    print("BENCHMARK CPU vs GPU - DETECTION D'ANOMALIES")
    print("="*70)
    
    # Verifier GPU
    has_gpu = torch.cuda.is_available()
    print(f"\nGPU disponible : {has_gpu}")
    if has_gpu:
        print(f"GPU detecte : {torch.cuda.get_device_name(0)}")
        print(f"Memoire GPU : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Charger le modele et les donnees
    print("\nChargement du modele et des donnees...")
    model_base, processor = load_anomaly_detection_model()
    model_base = adapt_model_for_binary(model_base, num_classes=2, freeze_backbone=True)
    
    train_loader, test_loader = create_mvtec_dataloaders(
        root_dir='./data/mvtec',
        category=category,
        batch_size=batch_size,
        processor=processor
    )
    
    results = {}
    
    # ===== BENCHMARK CPU =====
    print("\n" + "="*70)
    print("BENCHMARK SUR CPU")
    print("="*70)
    
    device_cpu = torch.device('cpu')
    model_cpu = model_base.to(device_cpu)
    criterion = nn.CrossEntropyLoss()
    optimizer_cpu = optim.AdamW(model_cpu.parameters(), lr=1e-4)
    
    print("\nEntrainement sur CPU...")
    cpu_time = train_model(model_cpu, train_loader, criterion, optimizer_cpu, device_cpu, num_epochs)
    
    print("\nEvaluation sur CPU...")
    cpu_metrics = evaluate_model(model_cpu, test_loader, device_cpu)
    
    results['cpu'] = {
        'time': cpu_time,
        'metrics': cpu_metrics
    }
    
    print(f"\nResultats CPU :")
    print(f"  Temps : {cpu_time:.2f}s ({cpu_time/num_epochs:.2f}s/epoch)")
    print(f"  Accuracy : {cpu_metrics['accuracy']:.2f}%")
    print(f"  F1-Score : {cpu_metrics['f1']:.2f}%")
    
    # Sauvegarder le modele
    torch.save(model_cpu.state_dict(), f'model_cpu_{category}.pth')
    print(f"\nModele sauvegarde : model_cpu_{category}.pth")
    
    # ===== BENCHMARK GPU =====
    if has_gpu:
        print("\n" + "="*70)
        print("BENCHMARK SUR GPU")
        print("="*70)
        
        device_gpu = torch.device('cuda')
        model_gpu, _ = load_anomaly_detection_model()
        model_gpu = adapt_model_for_binary(model_gpu, num_classes=2, freeze_backbone=True)
        model_gpu = model_gpu.to(device_gpu)
        
        optimizer_gpu = optim.AdamW(model_gpu.parameters(), lr=1e-4)
        
        # Warmup GPU
        print("\nWarmup GPU...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device_gpu)
        _ = model_gpu(dummy_input)
        torch.cuda.synchronize()
        
        print("\nEntrainement sur GPU...")
        gpu_time = train_model(model_gpu, train_loader, criterion, optimizer_gpu, device_gpu, num_epochs)
        
        print("\nEvaluation sur GPU...")
        gpu_metrics = evaluate_model(model_gpu, test_loader, device_gpu)
        
        results['gpu'] = {
            'time': gpu_time,
            'metrics': gpu_metrics
        }
        
        print(f"\nResultats GPU :")
        print(f"  Temps : {gpu_time:.2f}s ({gpu_time/num_epochs:.2f}s/epoch)")
        print(f"  Accuracy : {gpu_metrics['accuracy']:.2f}%")
        print(f"  F1-Score : {gpu_metrics['f1']:.2f}%")
        
        # Sauvegarder
        torch.save(model_gpu.state_dict(), f'model_gpu_{category}.pth')
        print(f"\nModele sauvegarde : model_gpu_{category}.pth")
        
        # ===== COMPARAISON =====
        print("\n" + "="*70)
        print("COMPARAISON CPU vs GPU")
        print("="*70)
        
        speedup = cpu_time / gpu_time
        print(f"\nAcceleration GPU : {speedup:.2f}x plus rapide")
        
        # Analyse economique
        print("\nAnalyse economique (tarifs Azure 2025) :")
        cost_cpu = (cpu_time / 3600) * 0.90
        cost_gpu = (gpu_time / 3600) * 3.50
        
        print(f"  Cout CPU : ${cost_cpu:.4f}")
        print(f"  Cout GPU : ${cost_gpu:.4f}")
        
        if cost_gpu < cost_cpu:
            savings = cost_cpu - cost_gpu
            print(f"  Economie : ${savings:.4f} ({(savings/cost_cpu)*100:.1f}%)")
        else:
            extra = cost_gpu - cost_cpu
            print(f"  Surcout : ${extra:.4f}")
    
    return results


def plot_results(results):
    """
    Visualise les resultats du benchmark
    """
    if 'gpu' not in results:
        print("\nGPU non disponible, pas de graphique")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    devices = ['CPU', 'GPU']
    colors = ['#3498db', '#2ecc71']
    
    # Temps
    times = [results['cpu']['time'], results['gpu']['time']]
    axes[0].bar(devices, times, color=colors, alpha=0.7)
    axes[0].set_ylabel('Temps (secondes)')
    axes[0].set_title('Temps d\'entrainement')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Accuracy
    accs = [results['cpu']['metrics']['accuracy'], results['gpu']['metrics']['accuracy']]
    axes[1].bar(devices, accs, color=colors, alpha=0.7)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Precision')
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)
    
    # F1-Score
    f1s = [results['cpu']['metrics']['f1'], results['gpu']['metrics']['f1']]
    axes[2].bar(devices, f1s, color=colors, alpha=0.7)
    axes[2].set_ylabel('F1-Score (%)')
    axes[2].set_title('F1-Score')
    axes[2].set_ylim([0, 100])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_cpu_gpu.png', dpi=300, bbox_inches='tight')
    print("\nGraphique sauvegarde : benchmark_cpu_gpu.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark CPU vs GPU')
    parser.add_argument('--category', default='bottle', help='Categorie MVTec AD')
    parser.add_argument('--epochs', type=int, default=3, help='Nombre d\'epoques')
    parser.add_argument('--batch-size', type=int, default=16, help='Taille des batches')
    
    args = parser.parse_args()
    
    results = run_benchmark(args.category, args.epochs, args.batch_size)
    plot_results(results)

