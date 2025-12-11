"""
Script de telechargement du dataset MVTec Anomaly Detection

Le dataset MVTec AD contient des images de pieces industrielles avec anomalies.
Site officiel : https://www.mvtec.com/company/research/datasets/mvtec-ad

Usage:
    python download_mvtec.py --category bottle --output ./data/mvtec
"""

import os
import argparse
from pathlib import Path


def download_mvtec_category(category='bottle', output_dir='./data/mvtec'):
    """
    Telecharge une categorie du dataset MVTec AD
    
    Args:
        category: Categorie a telecharger (bottle, cable, capsule, hazelnut, metal_nut)
        output_dir: Repertoire de destination
    
    Note:
        Le telechargement automatique n'est pas disponible.
        Veuillez telecharger manuellement depuis :
        https://www.mvtec.com/company/research/datasets/mvtec-ad
    """
    
    print("="*70)
    print("TELECHARGEMENT DATASET MVTEC AD")
    print("="*70)
    
    print(f"\nCategorie demandee : {category}")
    print(f"Destination : {output_dir}")
    
    # Creer le repertoire de destination
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nLe dataset MVTec AD necessite un telechargement manuel.")
    print("\nEtapes :")
    print("1. Visitez : https://www.mvtec.com/company/research/datasets/mvtec-ad")
    print(f"2. Telechargez la categorie : {category}")
    print(f"3. Extrayez le fichier ZIP dans : {output_dir}")
    
    print("\nStructure attendue :")
    print(f"{output_dir}/")
    print(f"  {category}/")
    print("    train/")
    print("      good/")
    print("    test/")
    print("      good/")
    print("      broken_large/")
    print("      contamination/")
    print("    ground_truth/")
    
    print("\nAlternative : Dataset simplifie disponible sur")
    print("https://github.com/openvinotoolkit/anomalib/tree/main/datasets")
    
    # Verifier si le dataset existe deja
    category_path = Path(output_dir) / category
    if category_path.exists():
        print(f"\nLe repertoire {category_path} existe deja.")
        
        # Compter les images
        train_good = category_path / 'train' / 'good'
        if train_good.exists():
            num_train = len(list(train_good.glob('*.png')))
            print(f"Images d'entrainement trouvees : {num_train}")
        
        test_path = category_path / 'test'
        if test_path.exists():
            num_test = sum(len(list(d.glob('*.png'))) for d in test_path.iterdir() if d.is_dir())
            print(f"Images de test trouvees : {num_test}")
        
        return True
    
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telecharger dataset MVTec AD')
    parser.add_argument('--category', default='bottle', 
                       choices=['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut'],
                       help='Categorie MVTec AD')
    parser.add_argument('--output', default='./data/mvtec', 
                       help='Repertoire de sortie')
    
    args = parser.parse_args()
    
    download_mvtec_category(args.category, args.output)

