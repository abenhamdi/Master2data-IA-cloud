# TP Jour 4 - Optimisation des modeles IA

## Cas d'usage : Detection d'anomalies industrielles

Vous travaillez pour une entreprise de maintenance industrielle qui souhaite automatiser la detection d'anomalies sur des pieces manufacturees. Le systeme doit analyser des milliers d'images par jour pour identifier des defauts de production.

**Objectif** : Optimiser un modele Vision Transformer pour reduire le temps d'inference de 80% et les couts de 70%.

## Fichiers fournis

```
TD_Jour4_Etudiant/
├── README.md                    # Ce fichier
├── requirements.txt             # Dependances Python
├── download_mvtec.py            # Telechargement dataset
├── mvtec_dataset.py             # Dataset PyTorch
├── load_model_hf.py             # Chargement modele Hugging Face
├── train_utils.py               # Fonctions d'entrainement
├── benchmark_gpu.py             # Benchmark CPU vs GPU
├── quantization_demo.py         # Demonstration quantization
└── onnx_optimization.py         # Export et optimisation ONNX
```

## Installation

```bash
# Creer environnement virtuel
python -m venv venv_tp
source venv_tp/bin/activate  # Windows: venv_tp\Scripts\activate

# Installer dependances
pip install -r requirements.txt

# Verifier installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers OK')"
```

## Preparation du dataset

Le dataset MVTec AD doit etre telecharge manuellement :

1. Visitez : https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Telechargez la categorie "bottle" ou "capsule"
3. Extrayez dans `./data/mvtec/`

Structure attendue :
```
data/mvtec/bottle/
├── train/good/*.png
├── test/good/*.png
├── test/broken_large/*.png
└── test/contamination/*.png
```

Ou executez :
```bash
python download_mvtec.py --category bottle --output ./data/mvtec
```

## Deroulement du TP

### Partie 1 : Benchmark CPU vs GPU (60 min)

```bash
python benchmark_gpu.py --category bottle --epochs 3
```

**Ce que fait le script** :
- Charge un Vision Transformer depuis Hugging Face
- Fine-tune pour detection d'anomalies (normal/anomalie)
- Compare performances CPU vs GPU
- Analyse cout/performance

**Resultats attendus** :
- Speedup GPU : 10-20x
- Modeles sauvegardes : `model_cpu_bottle.pth`, `model_gpu_bottle.pth`
- Graphique : `benchmark_cpu_gpu.png`

### Partie 2 : Quantization (45 min)

```bash
python quantization_demo.py --category bottle --model-path model_cpu_bottle.pth
```

**Ce que fait le script** :
- Applique quantization dynamique (int8)
- Compare taille et vitesse
- Valide objectif metier (< 30 MB, > 95% precision)

**Resultats attendus** :
- Compression : 2-4x
- Speedup : 1.5-3x
- Perte precision : < 1%

### Partie 3 : ONNX Runtime (45 min)

```bash
python onnx_optimization.py --category bottle --model-path model_cpu_bottle.pth
```

**Ce que fait le script** :
- Exporte modele vers ONNX
- Benchmark PyTorch vs ONNX Runtime
- Analyse cout production

**Resultats attendus** :
- Speedup ONNX : 1.5-3x
- Precision preservee (< 0.5% difference)
- Economie : 40-60%

### Partie 4 : Azure ML Deployment (optionnel, 60 min)

Necessite compte Azure (Azure for Students ou Free Trial).

Voir documentation Azure ML pour deployer le modele ONNX sur un endpoint manage.

### Partie 5 : Tests de charge (optionnel, 30 min)

Tests de performance en conditions reelles avec mesure latence p95/p99 et throughput.

## Conseils pratiques

### Si vous n'avez pas de GPU

- Tout le TP peut etre realise sur CPU
- Les speedups seront differents mais les concepts identiques
- Vous pouvez utiliser Google Colab pour tester avec GPU gratuitement

### Gestion du temps

- Commencez par la Partie 1 (fondamentale)
- Les Parties 2 et 3 sont independantes (vous pouvez les faire dans l'ordre que vous voulez)
- La Partie 4 necessite d'avoir complete les parties precedentes
- La Partie 5 est optionnelle si vous manquez de temps

### Debugging

**Erreur de memoire GPU** :
```python
# Reduire le batch size
train_loader = DataLoader(dataset, batch_size=64)  # Au lieu de 128
```

**Modele non trouve** :
```python
# Verifier que vous avez bien sauvegarde le modele
torch.save(model.state_dict(), 'model_cpu.pth')
```

**Erreur ONNX** :
```python
# Verifier la version d'opset
torch.onnx.export(..., opset_version=11)
```

## Ressources supplementaires

### Documentation officielle

- PyTorch Quantization : https://pytorch.org/docs/stable/quantization.html
- ONNX Runtime : https://onnxruntime.ai/docs/
- TorchVision Models : https://pytorch.org/vision/stable/models.html

### Tutoriels recommandes

- PyTorch Performance Tuning : https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- ONNX Model Zoo : https://github.com/onnx/models
- Quantization Best Practices : https://pytorch.org/blog/quantization-in-practice/

### Datasets

- CIFAR-10 : Telechargement automatique via torchvision
- Taille : ~170 MB
- 60,000 images 32x32 en couleur (10 classes)

## Evaluation

### Criteres (sur 100 points)

1. **Qualite du code (20 points)**
   - Proprete et organisation (5 pts)
   - Commentaires pertinents (5 pts)
   - Gestion des erreurs (5 pts)
   - Bonnes pratiques (5 pts)

2. **Completude (30 points)**
   - Partie 1 : 8 points
   - Partie 2 : 8 points
   - Partie 3 : 7 points
   - Partie 4 : 7 points

3. **Analyse (30 points)**
   - Interpretation des resultats (10 pts)
   - Comparaisons pertinentes (10 pts)
   - Recommandations justifiees (10 pts)

4. **Optimisation (20 points)**
   - Gains de performance (10 pts)
   - Qualite des optimisations (10 pts)

### Livrables finaux

A rendre dans une archive ZIP contenant :

```
nom_prenom_tp_jour4.zip
├── code/
│   ├── benchmark_gpu.py
│   ├── quantization_demo.py
│   ├── onnx_optimization.py
│   ├── optimization_pipeline.py
│   └── (autres fichiers Python)
├── resultats/
│   ├── optimization_report.csv
│   ├── benchmark_cpu_gpu.png
│   ├── optimization_complete.png
│   └── (autres graphiques)
├── modeles/
│   ├── model_cpu.pth
│   ├── model_quantized.pth
│   └── model_final.onnx
└── rapport.md  # Votre analyse et recommandations
```

## Support

### Questions frequentes

**Q : Mon entrainement est tres lent, c'est normal ?**
R : Oui, sur CPU l'entrainement peut prendre 10-20 minutes. Reduisez a 2-3 epochs si necessaire.

**Q : Je n'ai pas de GPU, puis-je faire le TP ?**
R : Absolument ! Tout le TP fonctionne sur CPU. Les speedups seront differents mais les concepts identiques.

**Q : La quantization reduit trop ma precision, que faire ?**
R : C'est normal, une perte de 1-2% est acceptable. Si > 5%, verifiez votre implementation.

**Q : ONNX Runtime n'est pas plus rapide que PyTorch ?**
R : Sur GPU, le gain est souvent limite. Sur CPU, vous devriez voir 1.5-3x de speedup.

### Contact

Pour toute question pendant le TP :
- Levez la main pour appeler le formateur
- Consultez la documentation officielle
- Echangez avec vos camarades (entraide encouragee !)

## Bon courage !

N'oubliez pas : l'optimisation est un processus iteratif. Testez, mesurez, analysez, et recommencez !

