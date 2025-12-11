"""
Fonctions utilitaires pour l'entrainement et l'evaluation

Usage:
    from train_utils import train_model, evaluate_model
    
    train_time = train_model(model, train_loader, criterion, optimizer, device)
    metrics = evaluate_model(model, test_loader, device)
"""

import torch
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=3):
    """
    Entraine le modele et retourne le temps total d'entrainement
    
    Args:
        model: Modele a entrainer
        train_loader: DataLoader pour l'entrainement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device (cpu ou cuda)
        num_epochs: Nombre d'epoques
    
    Returns:
        float: Temps total d'entrainement en secondes
    """
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradient
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Gerer les sorties Hugging Face (outputs.logits) ou PyTorch standard
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Calcul de la perte
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Affichage periodique
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Statistiques de l'epoque
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    return total_time


def evaluate_model(model, test_loader, device):
    """
    Evalue le modele sur le dataset de test avec metriques detaillees
    
    Args:
        model: Modele a evaluer
        test_loader: DataLoader pour le test
        device: Device (cpu ou cuda)
    
    Returns:
        dict: Metriques (accuracy, precision, recall, f1, auc)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Gerer les sorties Hugging Face ou PyTorch
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calcul des metriques
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # AUC (si probabilites disponibles)
    try:
        all_probs_array = np.array(all_probs)
        if all_probs_array.shape[1] > 1:
            auc = roc_auc_score(all_labels, all_probs_array[:, 1])
        else:
            auc = 0.0
    except:
        auc = 0.0
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc
    }
    
    return metrics


def measure_inference_time(model, data_loader, device, num_batches=100):
    """
    Mesure le temps d'inference moyen
    
    Args:
        model: Modele a evaluer
        data_loader: DataLoader
        device: Device (cpu ou cuda)
        num_batches: Nombre de batches a mesurer
    
    Returns:
        float: Temps moyen par batch en millisecondes
    """
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            inputs = inputs.to(device)
            
            # Synchroniser si GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            
            # Forward pass
            _ = model(inputs)
            
            # Synchroniser si GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append(end - start)
    
    avg_time_ms = (sum(times) / len(times)) * 1000
    return avg_time_ms


def get_model_size(model, filename="temp_model.pth"):
    """
    Retourne la taille du modele en MB
    
    Args:
        model: Modele PyTorch
        filename: Nom du fichier temporaire
    
    Returns:
        float: Taille en MB
    """
    import os
    
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    os.remove(filename)
    
    return size_mb


if __name__ == "__main__":
    print("Module train_utils charge")
    print("Fonctions disponibles :")
    print("  - train_model()")
    print("  - evaluate_model()")
    print("  - measure_inference_time()")
    print("  - get_model_size()")

