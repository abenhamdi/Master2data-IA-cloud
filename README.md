# Master 2 Data - Industrialisation de l'IA dans le Cloud

Repository des travaux pratiques pour le cours "Industrialisation de l'IA dans le Cloud" - Master 2 Data.

## Structure du repository

```
Master2data-IA-cloud/
├── TP3/                    # Deploiement d'un modele IA sur Azure
│   ├── app/               # Application Flask
│   ├── Dockerfile         # Configuration Docker
│   ├── requirements.txt   # Dependances Python
│   └── README.md          # Documentation TP3
│
└── TP4/                    # Optimisation des modeles IA
    ├── benchmark_gpu.py    # Benchmark CPU vs GPU
    ├── quantization_demo.py # Demonstration quantization
    ├── onnx_optimization.py # Export et optimisation ONNX
    ├── requirements.txt    # Dependances Python
    └── README.md           # Documentation TP4
```

## TP3 : Deploiement d'un modele IA sur Azure

**Objectif** : Deployer une application de classification d'images medicales sur Azure App Service.

**Technologies** :
- Python, TensorFlow/Keras
- Flask pour l'API REST
- Docker pour la conteneurisation
- Azure App Service pour l'hebergement

**Voir** : [TP3/README.md](TP3/README.md) pour les instructions detaillees.

## TP4 : Optimisation des modeles IA

**Objectif** : Optimiser un modele Vision Transformer pour reduire le temps d'inference de 80% et les couts de 70%.

**Technologies** :
- PyTorch, Transformers (Hugging Face)
- Quantization dynamique
- ONNX Runtime
- Azure Machine Learning (optionnel)

**Cas d'usage** : Detection d'anomalies industrielles sur le dataset MVTec AD.

**Voir** : [TP4/README.md](TP4/README.md) pour les instructions detaillees.

## Installation

Chaque TP possede son propre environnement virtuel et ses dependances.

### TP3
```bash
cd TP3
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### TP4
```bash
cd TP4
python -m venv venv_tp
source venv_tp/bin/activate  # Windows: venv_tp\Scripts\activate
pip install -r requirements.txt
```

## Prerequis

- Python 3.8+
- Compte Azure (Azure for Students ou Free Trial)
- Git
- Docker (optionnel pour TP3)
- GPU recommande mais non obligatoire pour TP4

## Ressources

### Documentation officielle
- [Azure App Service](https://learn.microsoft.com/azure/app-service/)
- [Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [ONNX Runtime](https://onnxruntime.ai/docs/)

### Datasets
- TP3 : Brain Tumor Classification (Kaggle)
- TP4 : MVTec Anomaly Detection

## Support

Pour toute question :
- Consultez la documentation de chaque TP
- Consultez la documentation officielle des technologies utilisees
- Contactez le formateur

## Licence

Materiel pedagogique - YNOV Montpellier - Master 2 Data

---

**Formateur** : Abdelkader Ben Hamdi  
**Annee** : 2025-2026
