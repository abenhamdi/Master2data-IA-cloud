# TP Jour 5 - Architecture Microservices IA & Gouvernance
## Système de Détection d'Objets Automobiles
**Master 2 - Industrialisation IA dans le Cloud**  
**Durée : 3 heures** 

---

## Contexte du projet

Vous êtes data scientist dans une entreprise spécialisée dans les systèmes d'aide à la conduite (ADAS - Advanced Driver Assistance Systems). Votre mission est de déployer un système de détection d'objets automobiles en production, en respectant les principes d'architecture microservices et les exigences de gouvernance IA.

Le système doit :
- Détecter et classifier les véhicules dans des images (voitures, camions, bus)
- Être déployé sous forme de microservices indépendants
- Respecter les principes de gouvernance IA et de conformité RGPD
- Être auditable et traçable

**Dataset** : [Car Object Detection - Kaggle](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)

---

## Objectifs pédagogiques

À l'issue de ce TP, vous serez capable de :

1. Concevoir et déployer une architecture microservices pour un système IA
2. Implémenter différents patterns de communication (REST, gRPC)
3. Mettre en place un système de logging et monitoring centralisé
4. Appliquer un cadre de gouvernance IA (transparence, traçabilité, conformité)
5. Générer des rapports d'audit et de conformité RGPD

---

## 📦 Prérequis techniques

### Logiciels requis
- Docker Desktop (version 20.10+)
- Docker Compose (version 2.0+)
- Python 3.9+
- kubectl (pour la partie Kubernetes)
- minikube ou kind (cluster Kubernetes local)
- Git
- Un éditeur de code (VS Code recommandé)

### Connaissances requises
- Bases de Docker et conteneurisation
- Python et bibliothèques ML (scikit-learn, TensorFlow/PyTorch)
- API REST
- Notions de Kubernetes (niveau débutant acceptable)

### Vérification de l'environnement

```bash
# Vérifier Docker
docker --version
docker-compose --version

# Vérifier Python
python --version
pip --version

# Vérifier Kubernetes
kubectl version --client
minikube version  # ou kind version

# Vérifier Git
git --version
```

---

##  Architecture du système

Le système est composé de 5 microservices :

```
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway (Kong)                      │
│            Routage • Auth • Rate Limiting                    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
│   Model      │  │  Feature    │  │  Results   │
│   Serving    │  │  Service    │  │  Service   │
│   (gRPC)     │  │  (REST)     │  │  (REST)    │
└───────┬──────┘  └──────┬──────┘  └─────┬──────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                ┌────────▼────────┐
                │  Logging &      │
                │  Monitoring     │
                │  (ELK Stack)    │
                └─────────────────┘
```

### Description des services

1. **API Gateway** : Point d'entrée unique, gestion de l'authentification et du rate limiting
2. **Model Serving** : Service d'inférence du modèle de détection (communication gRPC)
3. **Feature Service** : Prétraitement et extraction de features des images (REST)
4. **Results Service** : Stockage et récupération des résultats de prédiction (REST)
5. **Logging & Monitoring** : Centralisation des logs et métriques (ELK Stack)

---

## Partie 1 : Préparation des données et du modèle (30 min)

### Étape 1.1 : Récupération et exploration du dataset

1. Téléchargez le dataset depuis Kaggle :
   ```bash
   # Installer kaggle CLI
   pip install kaggle
   
   # Configurer les credentials Kaggle (fichier ~/.kaggle/kaggle.json)
   # Télécharger le dataset
   kaggle datasets download -d sshikamaru/car-object-detection
   unzip car-object-detection.zip -d data/
   ```

2. Explorez la structure du dataset :
   ```bash
   data/
   ├── images/           # Images de véhicules
   ├── annotations/      # Annotations au format YOLO ou COCO
   └── classes.txt       # Liste des classes
   ```

3. **Question de réflexion** : 
   - Combien d'images contient le dataset ?
   - Quelles sont les classes d'objets présentes ?
   - Quelle est la distribution des classes (équilibrée ou déséquilibrée) ?

### Étape 1.2 : Entraînement d'un modèle de détection

Pour ce TP, nous utiliserons un modèle pré-entraîné (transfer learning) pour gagner du temps.

Créez le fichier `model/train_model.py` :

```python
"""
Script d'entraînement du modèle de détection d'objets automobiles
Utilise un modèle pré-entraîné (MobileNetV2) avec transfer learning
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import json
from datetime import datetime

# TODO : Implémenter la fonction de chargement des données
def load_dataset(data_path, img_size=(224, 224)):
    """
    Charge et prétraite le dataset
    
    Args:
        data_path: Chemin vers le dossier de données
        img_size: Taille des images (hauteur, largeur)
    
    Returns:
        X_train, y_train, X_val, y_val, class_names
    """
    # À COMPLÉTER
    pass

# TODO : Implémenter la fonction de création du modèle
def create_model(num_classes, input_shape=(224, 224, 3)):
    """
    Crée un modèle de classification basé sur MobileNetV2
    
    Args:
        num_classes: Nombre de classes à prédire
        input_shape: Forme des images d'entrée
    
    Returns:
        model: Modèle Keras compilé
    """
    # À COMPLÉTER
    # Indice : Utiliser MobileNetV2 pré-entraîné sur ImageNet
    # Ajouter une couche de classification personnalisée
    pass

# TODO : Implémenter la fonction d'entraînement
def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    """
    Entraîne le modèle
    
    Args:
        model: Modèle Keras
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        epochs: Nombre d'époques
    
    Returns:
        history: Historique d'entraînement
    """
    # À COMPLÉTER
    pass

# TODO : Implémenter la sauvegarde du modèle avec métadonnées
def save_model_with_metadata(model, history, output_path="models/"):
    """
    Sauvegarde le modèle avec ses métadonnées (pour la gouvernance)
    
    Args:
        model: Modèle entraîné
        history: Historique d'entraînement
        output_path: Chemin de sauvegarde
    """
    # À COMPLÉTER
    # Sauvegarder :
    # - Le modèle (.h5 ou SavedModel)
    # - Les métriques d'entraînement
    # - Les métadonnées (date, version, hyperparamètres)
    pass

if __name__ == "__main__":
    print(" Démarrage de l'entraînement du modèle...")
    
    # Configuration
    DATA_PATH = "data/"
    NUM_CLASSES = 3  # voiture, camion, bus
    EPOCHS = 10
    
    # Pipeline d'entraînement
    # À COMPLÉTER
```

**Livrables Partie 1** :
- [ ] Script `train_model.py` complété et fonctionnel
- [ ] Modèle entraîné sauvegardé dans `models/car_detector_v1.h5`
- [ ] Fichier `models/model_metadata.json` contenant les métadonnées

**Aide** : Consultez `AIDE_PARTIE1.md` pour des indices sur l'implémentation.

---

## Partie 2 : Conteneurisation des microservices (45 min)

### Étape 2.1 : Service de Model Serving (gRPC)

Créez le service d'inférence qui expose le modèle via gRPC.

**Fichier : `services/model_serving/server.py`**

```python
"""
Service de Model Serving - Communication gRPC
Expose le modèle de détection via une API gRPC haute performance
"""

import grpc
from concurrent import futures
import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
import json

# TODO : Importer les proto générés
# import model_serving_pb2
# import model_serving_pb2_grpc

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelServingService:
    """Service gRPC pour l'inférence du modèle"""
    
    def __init__(self, model_path):
        """
        Initialise le service avec le modèle
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        # TODO : Charger le modèle
        # self.model = ...
        # self.class_names = ...
        logger.info(f" Modèle chargé depuis {model_path}")
    
    def Predict(self, request, context):
        """
        Méthode gRPC pour la prédiction
        
        Args:
            request: Requête contenant l'image encodée
            context: Contexte gRPC
        
        Returns:
            PredictionResponse avec les résultats
        """
        try:
            # TODO : Implémenter la logique de prédiction
            # 1. Décoder l'image depuis request.image_data
            # 2. Prétraiter l'image
            # 3. Faire la prédiction
            # 4. Logger la prédiction (pour l'audit)
            # 5. Retourner le résultat
            
            logger.info(f"📊 Prédiction effectuée - ID: {request.request_id}")
            pass
            
        except Exception as e:
            logger.error(f" Erreur lors de la prédiction: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return None

def serve():
    """Démarre le serveur gRPC"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # TODO : Enregistrer le service
    # model_serving_pb2_grpc.add_ModelServingServicer_to_server(
    #     ModelServingService(model_path="models/car_detector_v1.h5"), 
    #     server
    # )
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("🚀 Serveur gRPC démarré sur le port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

**Fichier : `services/model_serving/model_serving.proto`**

```protobuf
syntax = "proto3";

package modelserving;

// Service de prédiction
service ModelServing {
  rpc Predict (PredictionRequest) returns (PredictionResponse);
  rpc HealthCheck (HealthCheckRequest) returns (HealthCheckResponse);
}

// Requête de prédiction
message PredictionRequest {
  string request_id = 1;
  bytes image_data = 2;  // Image encodée en base64
  string user_id = 3;    // Pour la traçabilité
}

// Réponse de prédiction
message PredictionResponse {
  string request_id = 1;
  repeated Detection detections = 2;
  string model_version = 3;
  double inference_time_ms = 4;
}

// Détection d'un objet
message Detection {
  string class_name = 1;
  double confidence = 2;
  BoundingBox bbox = 3;
}

// Boîte englobante
message BoundingBox {
  double x_min = 1;
  double y_min = 2;
  double x_max = 3;
  double y_max = 4;
}

// Health check
message HealthCheckRequest {}
message HealthCheckResponse {
  string status = 1;
}
```

**Fichier : `services/model_serving/Dockerfile`**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code du service
COPY . .

# Générer les proto
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model_serving.proto

# Exposer le port gRPC
EXPOSE 50051

# Commande de démarrage
CMD ["python", "server.py"]
```

**Fichier : `services/model_serving/requirements.txt`**

```
tensorflow==2.15.0
grpcio==1.60.0
grpcio-tools==1.60.0
numpy==1.24.3
pillow==10.2.0
```

### Étape 2.2 : Service de Features (REST)

Créez le service de prétraitement des images.

**Fichier : `services/feature_service/app.py`**

```python
"""
Feature Service - API REST
Prétraite les images et extrait les features avant l'inférence
"""

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import logging
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO : Implémenter les fonctions de prétraitement

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Prétraite une image pour l'inférence
    
    Args:
        image_bytes: Image en bytes
        target_size: Taille cible
    
    Returns:
        Image prétraitée encodée en base64
    """
    # À COMPLÉTER
    pass

def extract_features(image_array):
    """
    Extrait des features basiques de l'image (pour le monitoring)
    
    Args:
        image_array: Image sous forme de numpy array
    
    Returns:
        dict: Dictionnaire de features
    """
    # À COMPLÉTER
    # Exemples de features :
    # - Dimensions
    # - Luminosité moyenne
    # - Contraste
    # - Format
    pass

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({
        "status": "healthy",
        "service": "feature-service",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """
    Endpoint de prétraitement d'image
    
    Body:
        {
            "image": "base64_encoded_image",
            "request_id": "unique_id"
        }
    
    Returns:
        {
            "preprocessed_image": "base64_encoded",
            "features": {...},
            "request_id": "unique_id"
        }
    """
    try:
        # TODO : Implémenter la logique
        # 1. Récupérer l'image depuis la requête
        # 2. Décoder l'image
        # 3. Prétraiter l'image
        # 4. Extraire les features
        # 5. Logger l'opération
        # 6. Retourner le résultat
        
        logger.info(f"Image prétraitée - ID: {data.get('request_id')}")
        pass
        
    except Exception as e:
        logger.error(f" Erreur de prétraitement: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
```

**Fichier : `services/feature_service/Dockerfile`**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port
EXPOSE 5001

# Commande de démarrage
CMD ["python", "app.py"]
```

**Fichier : `services/feature_service/requirements.txt`**

```
flask==3.0.0
pillow==10.2.0
numpy==1.24.3
```

### Étape 2.3 : Service de Results (REST)

Créez le service de stockage des résultats.

**Fichier : `services/results_service/app.py`**

```python
"""
Results Service - API REST
Stocke et récupère les résultats de prédiction
"""

from flask import Flask, request, jsonify
from datetime import datetime
import json
import sqlite3
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO : Initialiser la base de données SQLite
def init_db():
    """Initialise la base de données"""
    # À COMPLÉTER
    # Créer une table 'predictions' avec les colonnes :
    # - id (PRIMARY KEY)
    # - request_id (UNIQUE)
    # - user_id (pour la traçabilité RGPD)
    # - predictions (JSON)
    # - model_version
    # - inference_time_ms
    # - timestamp
    pass

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({
        "status": "healthy",
        "service": "results-service",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/predictions', methods=['POST'])
def store_prediction():
    """
    Stocke un résultat de prédiction
    
    Body:
        {
            "request_id": "unique_id",
            "user_id": "anonymized_user_id",
            "predictions": [...],
            "model_version": "v1",
            "inference_time_ms": 45.2
        }
    """
    try:
        # TODO : Implémenter la logique de stockage
        # 1. Récupérer les données
        # 2. Valider les données
        # 3. Stocker dans la base de données
        # 4. Logger l'opération
        
        logger.info(f" Prédiction stockée - ID: {data.get('request_id')}")
        pass
        
    except Exception as e:
        logger.error(f"Erreur de stockage: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predictions/<request_id>', methods=['GET'])
def get_prediction(request_id):
    """Récupère un résultat de prédiction par son ID"""
    try:
        # TODO : Implémenter la récupération
        pass
        
    except Exception as e:
        logger.error(f"Erreur de récupération: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predictions/user/<user_id>', methods=['GET'])
def get_user_predictions(user_id):
    """
    Récupère toutes les prédictions d'un utilisateur
    (Important pour le droit d'accès RGPD)
    """
    try:
        # TODO : Implémenter la récupération
        pass
        
    except Exception as e:
        logger.error(f"Erreur de récupération: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predictions/user/<user_id>', methods=['DELETE'])
def delete_user_predictions(user_id):
    """
    Supprime toutes les prédictions d'un utilisateur
    (Important pour le droit à l'oubli RGPD)
    """
    try:
        # TODO : Implémenter la suppression
        # Logger l'opération pour l'audit
        
        logger.info(f"🗑️ Données supprimées pour l'utilisateur: {user_id}")
        pass
        
    except Exception as e:
        logger.error(f"Erreur de suppression: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5002, debug=False)
```

**Fichier : `services/results_service/Dockerfile`**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Créer le dossier pour la base de données
RUN mkdir -p /app/data

# Exposer le port
EXPOSE 5002

# Commande de démarrage
CMD ["python", "app.py"]
```

**Fichier : `services/results_service/requirements.txt`**

```
flask==3.0.0
```

### Étape 2.4 : Docker Compose

Créez le fichier `docker-compose.yml` pour orchestrer tous les services.

```yaml
version: '3.8'

services:
  # Service de Model Serving (gRPC)
  model-serving:
    build: ./services/model_serving
    container_name: model-serving
    ports:
      - "50051:50051"
    volumes:
      - ./models:/app/models:ro
    environment:
      - MODEL_PATH=/app/models/car_detector_v1.h5
      - LOG_LEVEL=INFO
    networks:
      - ia-network
    restart: unless-stopped

  # Service de Features (REST)
  feature-service:
    build: ./services/feature_service
    container_name: feature-service
    ports:
      - "5001:5001"
    environment:
      - LOG_LEVEL=INFO
    networks:
      - ia-network
    restart: unless-stopped

  # Service de Results (REST)
  results-service:
    build: ./services/results_service
    container_name: results-service
    ports:
      - "5002:5002"
    volumes:
      - ./data/results:/app/data
    environment:
      - LOG_LEVEL=INFO
      - DB_PATH=/app/data/predictions.db
    networks:
      - ia-network
    restart: unless-stopped

  # TODO : Ajouter les services de logging (ELK Stack)
  # elasticsearch:
  #   ...
  
  # logstash:
  #   ...
  
  # kibana:
  #   ...

networks:
  ia-network:
    driver: bridge

volumes:
  elasticsearch-data:
```

**Livrables Partie 2** :
- [ ] Services conteneurisés et fonctionnels
- [ ] `docker-compose.yml` complété
- [ ] Tous les services démarrent sans erreur
- [ ] Tests de communication entre services

**Commandes de test** :
```bash
# Construire et démarrer les services
docker-compose up --build -d

# Vérifier les logs
docker-compose logs -f

# Tester le health check
curl http://localhost:5001/health
curl http://localhost:5002/health
```

---

## Partie 3 : Logging et Monitoring centralisés (30 min)

### Étape 3.1 : Configuration de l'ELK Stack

Ajoutez les services Elasticsearch, Logstash et Kibana à votre `docker-compose.yml`.

```yaml
  # Elasticsearch - Stockage des logs
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - ia-network
    restart: unless-stopped

  # Logstash - Pipeline de traitement des logs
  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    container_name: logstash
    volumes:
      - ./config/logstash/logstash.conf:/usr/share/logstash/pipeline/logstash.conf:ro
    ports:
      - "5000:5000"
      - "9600:9600"
    environment:
      - "LS_JAVA_OPTS=-Xmx256m -Xms256m"
    depends_on:
      - elasticsearch
    networks:
      - ia-network
    restart: unless-stopped

  # Kibana - Visualisation des logs
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - ia-network
    restart: unless-stopped
```

### Étape 3.2 : Configuration de Logstash

Créez le fichier `config/logstash/logstash.conf` :

```ruby
input {
  # Réception des logs depuis les services
  tcp {
    port => 5000
    codec => json
  }
}

filter {
  # TODO : Ajouter des filtres pour enrichir les logs
  # Exemples :
  # - Ajouter un timestamp
  # - Extraire des champs spécifiques
  # - Classifier les logs par niveau (INFO, WARNING, ERROR)
  
  # Anonymisation des données sensibles (RGPD)
  if [user_id] {
    mutate {
      # Hasher le user_id pour l'anonymisation
      add_field => { "user_id_hash" => "%{user_id}" }
      remove_field => [ "user_id" ]
    }
  }
}

output {
  # Envoi vers Elasticsearch
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "ia-logs-%{+YYYY.MM.dd}"
  }
  
  # Debug (optionnel)
  stdout {
    codec => rubydebug
  }
}
```

### Étape 3.3 : Intégration du logging dans les services

Modifiez vos services pour envoyer les logs vers Logstash.

**Exemple pour le Feature Service** :

```python
import logging
import logstash
import socket

# Configuration du logger avec Logstash
logger = logging.getLogger('feature-service')
logger.setLevel(logging.INFO)

# Handler vers Logstash
logstash_handler = logstash.TCPLogstashHandler(
    host='logstash',
    port=5000,
    version=1
)
logger.addHandler(logstash_handler)

# Utilisation
logger.info('Image prétraitée', extra={
    'request_id': request_id,
    'service': 'feature-service',
    'operation': 'preprocess',
    'duration_ms': duration
})
```

**Livrables Partie 3** :
- [ ] ELK Stack déployé et fonctionnel
- [ ] Logs centralisés depuis tous les services
- [ ] Dashboard Kibana configuré
- [ ] Visualisations des métriques clés

**Questions de réflexion** :
- Pourquoi est-il important de centraliser les logs ?
- Comment l'anonymisation des données contribue-t-elle à la conformité RGPD ?
- Quelles métriques sont essentielles pour le monitoring d'un système IA ?

---

## Partie 4 : Gouvernance IA et conformité RGPD (45 min)

### Étape 4.1 : Création d'une checklist de gouvernance

Créez le fichier `governance/CHECKLIST_GOUVERNANCE.md` :

```markdown
# Checklist de Gouvernance IA
## Système de Détection d'Objets Automobiles

### 1. Transparence des données

- [ ] **Origine des données** : Dataset Kaggle documenté
- [ ] **Qualité des données** : Vérification de la distribution des classes
- [ ] **Biais potentiels** : Analyse des biais (géographiques, temporels, etc.)
- [ ] **Documentation** : Métadonnées complètes du dataset

**Notes** :
- Dataset : Car Object Detection - Kaggle
- Nombre d'images : [À COMPLÉTER]
- Classes : [À COMPLÉTER]
- Biais identifiés : [À COMPLÉTER]

### 2. Transparence du modèle

- [ ] **Architecture** : Documentation de l'architecture (MobileNetV2)
- [ ] **Hyperparamètres** : Liste complète des hyperparamètres
- [ ] **Métriques de performance** : Accuracy, Precision, Recall, F1-Score
- [ ] **Limitations** : Documentation des limitations connues

**Métriques** :
- Accuracy : [À COMPLÉTER]
- Precision : [À COMPLÉTER]
- Recall : [À COMPLÉTER]
- F1-Score : [À COMPLÉTER]

### 3. Équité et non-discrimination

- [ ] **Analyse des biais** : Test sur différents sous-groupes
- [ ] **Fairness metrics** : Calcul des métriques d'équité
- [ ] **Mitigation** : Stratégies de réduction des biais
- [ ] **Monitoring continu** : Surveillance des biais en production

**Actions** :
- [À COMPLÉTER]

### 4. Conformité RGPD

- [ ] **Base légale** : Intérêt légitime / Consentement
- [ ] **Minimisation des données** : Collecte uniquement des données nécessaires
- [ ] **Droit d'accès** : API pour récupérer les données utilisateur
- [ ] **Droit à l'oubli** : API pour supprimer les données utilisateur
- [ ] **Anonymisation** : Pseudonymisation des identifiants
- [ ] **Chiffrement** : Données en transit (TLS) et au repos
- [ ] **Registre des traitements** : Documentation RGPD complète

**Endpoints RGPD** :
- GET /predictions/user/<user_id> : Droit d'accès
- DELETE /predictions/user/<user_id> : Droit à l'oubli

### 5. Sécurité

- [ ] **Authentification** : Mécanisme d'authentification en place
- [ ] **Autorisation** : Contrôle d'accès basé sur les rôles
- [ ] **Chiffrement** : TLS pour toutes les communications
- [ ] **Audit** : Journalisation de tous les accès
- [ ] **Gestion des incidents** : Procédure de réponse aux incidents

**Mesures** :
- [À COMPLÉTER]

### 6. Auditabilité et traçabilité

- [ ] **Versioning** : Gestion des versions du modèle
- [ ] **Logging** : Journalisation de toutes les prédictions
- [ ] **Métriques** : Collecte des métriques de performance
- [ ] **Alertes** : Système d'alertes pour les anomalies
- [ ] **Rapports** : Génération automatique de rapports d'audit

**Outils** :
- Versioning : MLflow / DVC
- Logging : ELK Stack
- Monitoring : Kibana Dashboards

### 7. Responsabilité

- [ ] **Propriétaire du modèle** : [À COMPLÉTER]
- [ ] **Responsable des données** : [À COMPLÉTER]
- [ ] **DPO (Data Protection Officer)** : [À COMPLÉTER]
- [ ] **Chaîne de responsabilité** : Documentée et claire

### 8. Explicabilité

- [ ] **Documentation utilisateur** : Guide d'utilisation du système
- [ ] **Interprétabilité** : Méthodes d'explication des prédictions (LIME, SHAP)
- [ ] **Feedback** : Mécanisme de retour utilisateur
- [ ] **Recours** : Procédure de contestation des décisions

**Méthodes** :
- [À COMPLÉTER]

---

**Date de création** : [DATE]  
**Dernière mise à jour** : [DATE]  
**Version** : 1.0  
**Validé par** : [NOM]
```

### Étape 4.2 : Implémentation du versioning du modèle

Créez le fichier `governance/model_registry.py` :

```python
"""
Registre des modèles - Gestion des versions et métadonnées
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path

class ModelRegistry:
    """Registre centralisé des modèles ML"""
    
    def __init__(self, registry_path="governance/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Charge le registre depuis le fichier JSON"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": []}
    
    def _save_registry(self):
        """Sauvegarde le registre dans le fichier JSON"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_path, metadata):
        """
        Enregistre un nouveau modèle avec ses métadonnées
        
        Args:
            model_path: Chemin vers le fichier du modèle
            metadata: Dict contenant les métadonnées
                - name: Nom du modèle
                - version: Version
                - description: Description
                - metrics: Métriques de performance
                - hyperparameters: Hyperparamètres
                - training_data: Info sur les données d'entraînement
                - author: Auteur
        
        Returns:
            model_id: Identifiant unique du modèle
        """
        # TODO : Implémenter l'enregistrement
        # 1. Calculer le hash du modèle (pour l'intégrité)
        # 2. Générer un ID unique
        # 3. Ajouter les métadonnées au registre
        # 4. Sauvegarder le registre
        pass
    
    def get_model(self, model_id):
        """Récupère les métadonnées d'un modèle"""
        # TODO : Implémenter
        pass
    
    def list_models(self, name=None, version=None):
        """Liste les modèles enregistrés"""
        # TODO : Implémenter
        pass
    
    def get_latest_version(self, name):
        """Récupère la dernière version d'un modèle"""
        # TODO : Implémenter
        pass
    
    def deprecate_model(self, model_id, reason):
        """Marque un modèle comme déprécié"""
        # TODO : Implémenter
        pass

# Exemple d'utilisation
if __name__ == "__main__":
    registry = ModelRegistry()
    
    # Enregistrer un modèle
    metadata = {
        "name": "car_detector",
        "version": "1.0.0",
        "description": "Modèle de détection d'objets automobiles basé sur MobileNetV2",
        "metrics": {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.91,
            "f1_score": 0.90
        },
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "adam"
        },
        "training_data": {
            "dataset": "Car Object Detection - Kaggle",
            "num_samples": 5000,
            "num_classes": 3,
            "split": "80/20 train/val"
        },
        "author": "Équipe Data Science"
    }
    
    model_id = registry.register_model("models/car_detector_v1.h5", metadata)
    print(f" Modèle enregistré avec l'ID: {model_id}")
```

### Étape 4.3 : Génération de rapports d'audit

Créez le fichier `governance/audit_report.py` :

```python
"""
Générateur de rapports d'audit et de conformité
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

class AuditReportGenerator:
    """Génère des rapports d'audit pour la conformité"""
    
    def __init__(self, db_path="data/results/predictions.db"):
        self.db_path = db_path
    
    def generate_usage_report(self, period_days=30):
        """
        Génère un rapport d'utilisation du système
        
        Args:
            period_days: Période en jours
        
        Returns:
            dict: Rapport d'utilisation
        """
        # TODO : Implémenter
        # Métriques à inclure :
        # - Nombre total de prédictions
        # - Nombre d'utilisateurs uniques
        # - Temps d'inférence moyen
        # - Distribution des classes prédites
        # - Taux d'erreur
        pass
    
    def generate_compliance_report(self, model_id):
        """
        Génère un rapport de conformité RGPD
        
        Args:
            model_id: ID du modèle à auditer
        
        Returns:
            dict: Rapport de conformité
        """
        report = {
            "report_date": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "compliance_checks": {}
        }
        
        # TODO : Implémenter les vérifications
        # 1. Vérifier la base légale du traitement
        report["compliance_checks"]["legal_basis"] = {
            "status": "compliant",
            "details": "Intérêt légitime - Système d'aide à la conduite"
        }
        
        # 2. Vérifier la minimisation des données
        # À COMPLÉTER
        
        # 3. Vérifier l'anonymisation
        # À COMPLÉTER
        
        # 4. Vérifier les droits des personnes
        # À COMPLÉTER
        
        # 5. Vérifier la sécurité
        # À COMPLÉTER
        
        return report
    
    def generate_performance_report(self, period_days=30):
        """
        Génère un rapport de performance du modèle
        
        Args:
            period_days: Période en jours
        
        Returns:
            dict: Rapport de performance
        """
        # TODO : Implémenter
        # Métriques à inclure :
        # - Temps d'inférence (min, max, moyenne, p95, p99)
        # - Distribution des scores de confiance
        # - Détection de drift (si implémenté)
        # - Disponibilité du service
        pass
    
    def export_report(self, report, output_path):
        """
        Exporte un rapport au format JSON et Markdown
        
        Args:
            report: Rapport à exporter
            output_path: Chemin de sortie (sans extension)
        """
        # Export JSON
        json_path = Path(f"{output_path}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Export Markdown
        md_path = Path(f"{output_path}.md")
        with open(md_path, 'w') as f:
            f.write(self._report_to_markdown(report))
        
        print(f"✅ Rapport exporté: {json_path} et {md_path}")
    
    def _report_to_markdown(self, report):
        """Convertit un rapport en format Markdown"""
        # TODO : Implémenter la conversion
        pass

# Exemple d'utilisation
if __name__ == "__main__":
    generator = AuditReportGenerator()
    
    # Générer un rapport de conformité
    compliance_report = generator.generate_compliance_report("car_detector_v1")
    generator.export_report(
        compliance_report,
        "governance/reports/compliance_report_" + datetime.now().strftime("%Y%m%d")
    )
    
    # Générer un rapport d'utilisation
    usage_report = generator.generate_usage_report(period_days=30)
    generator.export_report(
        usage_report,
        "governance/reports/usage_report_" + datetime.now().strftime("%Y%m%d")
    )
```

**Livrables Partie 4** :
- [ ] Checklist de gouvernance complétée
- [ ] Registre des modèles implémenté
- [ ] Rapports d'audit générés
- [ ] Documentation de conformité RGPD


## Partie 5 : Déploiement sur Kubernetes (30 min - BONUS)

### Étape 5.1 : Création des manifestes Kubernetes

Créez les fichiers de déploiement Kubernetes pour chaque service.

**Fichier : `k8s/model-serving-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
  labels:
    app: model-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
    spec:
      containers:
      - name: model-serving
        image: model-serving:v1
        ports:
        - containerPort: 50051
          name: grpc
        env:
        - name: MODEL_PATH
          value: "/app/models/car_detector_v1.h5"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import grpc; print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import grpc; print('OK')"
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc

apiVersion: v1
kind: Service
metadata:
  name: model-serving
spec:
  selector:
    app: model-serving
  ports:
  - port: 50051
    targetPort: 50051
    name: grpc
  type: ClusterIP
```

**Fichier : `k8s/feature-service-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feature-service
  labels:
    app: feature-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: feature-service
  template:
    metadata:
      labels:
        app: feature-service
    spec:
      containers:
      - name: feature-service
        image: feature-service:v1
        ports:
        - containerPort: 5001
          name: http
        env:
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 5
          periodSeconds: 5

apiVersion: v1
kind: Service
metadata:
  name: feature-service
spec:
  selector:
    app: feature-service
  ports:
  - port: 5001
    targetPort: 5001
    name: http
  type: ClusterIP
```

**Fichier : `k8s/results-service-deployment.yaml`**

```yaml
# TODO : À compléter par les étudiants
# Similaire aux autres déploiements
```

### Étape 5.2 : Configuration de l'Ingress

**Fichier : `k8s/ingress.yaml`**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ia-system-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  rules:
  - host: ia-system.local
    http:
      paths:
      - path: /api/features
        pathType: Prefix
        backend:
          service:
            name: feature-service
            port:
              number: 5001
      - path: /api/results
        pathType: Prefix
        backend:
          service:
            name: results-service
            port:
              number: 5002
```

### Étape 5.3 : Déploiement

```bash
# Démarrer minikube
minikube start --cpus=4 --memory=8192

# Construire les images dans minikube
eval $(minikube docker-env)
docker build -t model-serving:v1 ./services/model_serving
docker build -t feature-service:v1 ./services/feature_service
docker build -t results-service:v1 ./services/results_service

# Créer le namespace
kubectl create namespace ia-system

# Déployer les services
kubectl apply -f k8s/ -n ia-system

# Vérifier le déploiement
kubectl get pods -n ia-system
kubectl get services -n ia-system

# Accéder aux services
minikube service feature-service -n ia-system
```

**Livrables Partie 5** :
- [ ] Manifestes Kubernetes créés
- [ ] Services déployés sur minikube
- [ ] Ingress configuré
- [ ] Tests de scalabilité effectués

## Tests et validation

### Tests fonctionnels

Créez le fichier `tests/test_system.py` :

```python
"""
Tests d'intégration du système complet
"""

import requests
import grpc
import base64
from pathlib import Path

# TODO : Implémenter les tests

def test_feature_service_health():
    """Test du health check du feature service"""
    response = requests.get("http://localhost:5001/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_results_service_health():
    """Test du health check du results service"""
    # À COMPLÉTER
    pass

def test_end_to_end_prediction():
    """Test end-to-end d'une prédiction"""
    # 1. Charger une image de test
    # 2. Appeler le feature service
    # 3. Appeler le model serving
    # 4. Vérifier le résultat
    # 5. Vérifier le stockage dans results service
    # À COMPLÉTER
    pass

def test_rgpd_user_data_access():
    """Test du droit d'accès RGPD"""
    # À COMPLÉTER
    pass

def test_rgpd_user_data_deletion():
    """Test du droit à l'oubli RGPD"""
    # À COMPLÉTER
    pass

if __name__ == "__main__":
    print(" Exécution des tests...")
    test_feature_service_health()
    test_results_service_health()
    test_end_to_end_prediction()
    test_rgpd_user_data_access()
    test_rgpd_user_data_deletion()
    print("✅ Tous les tests sont passés!")
```

### Script de test complet

```bash
#!/bin/bash
# tests/run_tests.sh

echo "Démarrage des tests du système IA..."

# Test 1 : Health checks
echo "Test 1: Health checks..."
curl -s http://localhost:5001/health | jq
curl -s http://localhost:5002/health | jq

# Test 2: Prétraitement d'image
echo "Test 2: Prétraitement d'image..."
# À COMPLÉTER

# Test 3: Prédiction complète
echo "Test 3: Prédiction complète..."
# À COMPLÉTER

# Test 4: Vérification des logs dans Kibana
echo "Test 4: Vérification des logs..."
# À COMPLÉTER

# Test 5: Tests RGPD
echo "Test 5: Tests RGPD..."
# À COMPLÉTER

echo "Tests terminés!"
```

## Livrables finaux

### Livrables obligatoires

1. **Code fonctionnel** :
   - [ ] Tous les services implémentés et fonctionnels
   - [ ] Docker Compose opérationnel
   - [ ] Tests passants

2. **Documentation de gouvernance** :
   - [ ] Checklist de gouvernance complétée
   - [ ] Registre des modèles avec métadonnées
   - [ ] Documentation RGPD (base légale, mesures techniques)

3. **Architecture** :
   - [ ] Diagramme d'architecture mis à jour
   - [ ] Documentation des choix techniques
   - [ ] README.md complet

### Livrables bonus

4. **Rapport d'audit** (BONUS) :
   - [ ] Rapport de conformité RGPD généré
   - [ ] Rapport d'utilisation du système
   - [ ] Rapport de performance du modèle
   - [ ] Recommandations d'amélioration

5. **Déploiement Kubernetes** (BONUS) :
   - [ ] Manifestes Kubernetes
   - [ ] Déploiement sur minikube
   - [ ] Tests de scalabilité


## Critères d'évaluation

| Critère | Points | Description |
|---------|--------|-------------|
| **Architecture microservices** | 25 | Services indépendants, communication REST/gRPC, découplage |
| **Implémentation technique** | 25 | Code propre, fonctionnel, bonnes pratiques |
| **Logging et monitoring** | 15 | ELK Stack configuré, logs centralisés, dashboards |
| **Gouvernance IA** | 20 | Checklist complète, registre des modèles, traçabilité |
| **Conformité RGPD** | 10 | Droits des personnes, anonymisation, documentation |
| **Documentation** | 5 | README, commentaires, architecture |
| **Bonus** | 10 | Rapport d'audit, Kubernetes, fonctionnalités avancées |
| **TOTAL** | 110 | (100 + 10 bonus) |


## Ressources et aide

### Documentation officielle

- **Docker** : https://docs.docker.com/
- **Kubernetes** : https://kubernetes.io/docs/
- **gRPC** : https://grpc.io/docs/
- **Flask** : https://flask.palletsprojects.com/
- **TensorFlow** : https://www.tensorflow.org/
- **ELK Stack** : https://www.elastic.co/guide/

### Fichiers d'aide fournis

- `AIDE_PARTIE1.md` : Indices pour l'entraînement du modèle
- `AIDE_PARTIE2.md` : Exemples de code pour les microservices
- `AIDE_PARTIE3.md` : Configuration ELK Stack
- `AIDE_PARTIE4.md` : Exemples de gouvernance IA
- `FAQ.md` : Questions fréquentes

### Commandes utiles

```bash
# Docker
docker-compose up --build -d
docker-compose logs -f [service]
docker-compose down

# Kubernetes
kubectl get pods -n ia-system
kubectl logs -f [pod-name] -n ia-system
kubectl describe pod [pod-name] -n ia-system

# Tests
python tests/test_system.py
bash tests/run_tests.sh

# Génération de rapports
python governance/audit_report.py
```


## Planning suggéré (3h)

| Temps | Activité |
|-------|----------|
| 0h00 - 0h30 | Partie 1 : Préparation des données et modèle |
| 0h30 - 1h15 | Partie 2 : Conteneurisation des microservices |
| 1h15 - 1h45 | Partie 3 : Logging et monitoring (ELK Stack) |
| 1h45 - 2h30 | Partie 4 : Gouvernance IA et conformité RGPD |
| 2h30 - 3h00 | Tests, validation et documentation |



## Questions de réflexion finale

1. **Architecture** :
   - Quels sont les avantages et inconvénients de l'architecture microservices pour un système IA ?
   - Pourquoi avons-nous choisi gRPC pour le model serving et REST pour les autres services ?

2. **Gouvernance** :
   - Comment la traçabilité des prédictions contribue-t-elle à la gouvernance IA ?
   - Quelles sont les principales exigences RGPD pour un système d'IA en production ?

3. **Production** :
   - Quelles métriques sont essentielles pour monitorer un système IA en production ?
   - Comment détecter et gérer le drift d'un modèle en production ?

4. **Éthique** :
   - Comment garantir l'équité d'un système de détection d'objets ?
   - Quels biais potentiels peuvent affecter ce type de système ?


