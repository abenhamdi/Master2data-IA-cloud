"""
Dataset PyTorch pour MVTec Anomaly Detection

Usage:
    from mvtec_dataset import MVTecDataset, create_mvtec_dataloaders
    
    train_loader, test_loader = create_mvtec_dataloaders(
        root_dir='./data/mvtec',
        category='bottle',
        batch_size=32
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms


class MVTecDataset(Dataset):
    """
    Dataset pour MVTec Anomaly Detection
    
    Structure attendue:
        root_dir/category/train/good/*.png
        root_dir/category/test/good/*.png
        root_dir/category/test/broken_large/*.png
        root_dir/category/test/contamination/*.png
    """
    
    def __init__(self, root_dir, category='bottle', split='train', transform=None):
        """
        Args:
            root_dir: Repertoire racine du dataset
            category: Categorie (bottle, cable, capsule, hazelnut, metal_nut)
            split: 'train' ou 'test'
            transform: Transformations a appliquer
        """
        self.root_dir = Path(root_dir) / category / split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Charger les images
        if split == 'train':
            # Train : seulement des images "good" (label 0)
            good_dir = self.root_dir / 'good'
            if good_dir.exists():
                for img_path in good_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(0)  # 0 = normal
        else:
            # Test : images "good" (0) et anomalies (1)
            for defect_type in self.root_dir.iterdir():
                if defect_type.is_dir():
                    label = 0 if defect_type.name == 'good' else 1
                    for img_path in defect_type.glob('*.png'):
                        self.images.append(str(img_path))
                        self.labels.append(label)
        
        print(f"Dataset {category} {split} : {len(self.images)} images")
        if split == 'test':
            num_normal = sum(1 for l in self.labels if l == 0)
            num_anomaly = sum(1 for l in self.labels if l == 1)
            print(f"  Normal : {num_normal}, Anomalies : {num_anomaly}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Charger l'image
        image = Image.open(img_path).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_mvtec_dataloaders(root_dir, category='bottle', batch_size=32, 
                             image_size=224, processor=None, num_workers=2):
    """
    Cree les DataLoaders pour train et test
    
    Args:
        root_dir: Repertoire racine du dataset
        category: Categorie MVTec AD
        batch_size: Taille des batches
        image_size: Taille des images (224 pour Vision Transformers)
        processor: Processeur Hugging Face (optionnel)
        num_workers: Nombre de workers pour le chargement
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Definir les transformations
    if processor is not None:
        # Utiliser les parametres du processeur Hugging Face
        transform = transforms.Compose([
            transforms.Resize((processor.size['height'], processor.size['width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        ])
    else:
        # Transformations par defaut (ImageNet)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Creer les datasets
    train_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split='train',
        transform=transform
    )
    
    test_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split='test',
        transform=transform
    )
    
    # Creer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test du dataset
    print("Test du dataset MVTec AD")
    
    train_loader, test_loader = create_mvtec_dataloaders(
        root_dir='./data/mvtec',
        category='bottle',
        batch_size=16
    )
    
    # Afficher un batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape : {images.shape}")
    print(f"Labels shape : {labels.shape}")
    print(f"Labels : {labels}")

