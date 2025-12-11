# TP3 - Deploiement d'un modele IA sur Azure

## Objectif

Deployer une application de classification d'images medicales (tumeurs cerebrales) sur Azure App Service.

## Description

Cette application utilise un modele CNN (Convolutional Neural Network) pour classifier les images IRM cerebrales en 4 categories:
- **Glioma**: Tumeur gliale
- **Meningioma**: Tumeur des meninges
- **Pituitary**: Tumeur de la glande pituitaire
- **No Tumor**: Absence de tumeur

## Installation locale

### Prerequis

- Python 3.9+
- pip
- Compte Kaggle (pour telecharger le dataset)

### Etapes d'installation

1. **Creer un environnement virtuel**

```bash
cd TP3
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. **Installer les dependances**

```bash
pip install -r requirements.txt
```

3. **Telecharger le dataset depuis Kaggle**

```bash
# Configurer Kaggle CLI
pip install kaggle
mkdir ~/.kaggle
# Copier votre kaggle.json dans ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Telecharger
kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri
unzip brain-tumor-classification-mri.zip -d dataset/
```

4. **Entrainer le modele** (40-90 minutes)

```bash
python -m app.model
```

5. **Lancer l'application**

```bash
python app.py
```

L'application sera accessible sur `http://localhost:8000`

## API Documentation

### Endpoints

#### GET `/`
Page d'accueil avec formulaire d'upload

#### POST `/predict`
Effectue une prediction sur une image IRM

**Request:** multipart/form-data avec image

**Reponse:**
```json
{
  "class_name": "glioma",
  "confidence": 0.95,
  "probabilities": {
    "glioma": 0.95,
    "meningioma": 0.03,
    "notumor": 0.01,
    "pituitary": 0.01
  }
}
```

#### GET `/health`
Health check de l'application

#### GET `/model/info`
Informations sur le modele

#### GET `/classes`
Liste des classes avec descriptions

## Docker

### Construire l'image

```bash
docker build -t brain-tumor-classifier:v1 .
```

### Lancer le conteneur

```bash
docker run -p 8000:8000 brain-tumor-classifier:v1
```

## Deploiement sur Azure

### Prerequis Azure

- Compte Azure actif
- Azure CLI installe

### Etapes de deploiement

1. **Se connecter a Azure**

```bash
az login
```

2. **Creer un groupe de ressources**

```bash
az group create --name rg-brain-tumor --location westeurope
```

3. **Creer un App Service Plan**

```bash
az appservice plan create \
  --name plan-brain-tumor \
  --resource-group rg-brain-tumor \
  --sku B1 \
  --is-linux
```

4. **Creer l'App Service**

```bash
az webapp create \
  --resource-group rg-brain-tumor \
  --plan plan-brain-tumor \
  --name brain-tumor-[votre-nom] \
  --runtime "PYTHON|3.9"
```

5. **Configurer le demarrage**

```bash
az webapp config set \
  --resource-group rg-brain-tumor \
  --name brain-tumor-[votre-nom] \
  --startup-file "gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 1 app:app"
```

6. **Deployer l'application**

```bash
zip -r app.zip . -x "*.git*" "venv/*" "__pycache__/*" "dataset/*"
az webapp deployment source config-zip \
  --resource-group rg-brain-tumor \
  --name brain-tumor-[votre-nom] \
  --src app.zip
```

## Structure du projet

```
TP3/
├── app/
│   ├── __init__.py           # Package Python
│   ├── model.py              # Modele CNN
│   ├── api.py                # API Flask
│   ├── utils.py              # Fonctions utilitaires
│   └── explore_data.py       # Exploration dataset
├── dataset/
│   ├── Training/             # Images d'entrainement
│   └── Testing/              # Images de test
├── models/
│   └── brain_tumor_classifier.h5  # Modele entraine
├── uploads/                  # Images uploadees (temporaire)
├── app.py                    # Point d'entree
├── requirements.txt          # Dependances
├── Dockerfile                # Configuration Docker
└── README.md                 # Documentation
```

## Technologies utilisees

- **Python 3.9**
- **TensorFlow/Keras** - Deep Learning
- **Flask** - Framework web
- **OpenCV & Pillow** - Traitement d'images
- **pandas & numpy** - Manipulation de donnees
- **gunicorn** - Serveur WSGI
- **Docker** - Conteneurisation
- **Azure App Service** - Hebergement

## Avertissement Medical

Cette application est un outil d'aide au diagnostic uniquement. Elle ne remplace PAS l'expertise d'un medecin radiologue qualifie. Toute decision medicale doit etre prise par un professionnel de sante apres examen complet du patient.

