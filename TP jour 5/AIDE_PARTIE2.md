# Aide - Partie 2 : Conteneurisation des microservices

## Indices pour l'implémentation

### 1. Service de Model Serving (gRPC)

#### Génération des fichiers proto

```bash
# Installer grpcio-tools
pip install grpcio-tools

# Générer les fichiers Python depuis le .proto
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    model_serving.proto

# Cela génère :
# - model_serving_pb2.py (messages)
# - model_serving_pb2_grpc.py (services)
```

#### Implémentation du serveur gRPC

```python
import grpc
from concurrent import futures
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import logging

import model_serving_pb2
import model_serving_pb2_grpc

logger = logging.getLogger(__name__)

class ModelServingService(model_serving_pb2_grpc.ModelServingServicer):
    
    def __init__(self, model_path):
        # Charger le modèle TensorFlow
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['car', 'truck', 'bus']
        logger.info(f"✅ Modèle chargé depuis {model_path}")
    
    def Predict(self, request, context):
        try:
            import time
            start_time = time.time()
            
            # 1. Décoder l'image depuis base64
            image_bytes = base64.b64decode(request.image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # 2. Prétraiter l'image
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # 3. Faire la prédiction
            predictions = self.model.predict(image_array, verbose=0)
            
            # 4. Créer les détections
            detections = []
            for idx, confidence in enumerate(predictions[0]):
                if confidence > 0.1:  # Seuil de confiance
                    detection = model_serving_pb2.Detection(
                        class_name=self.class_names[idx],
                        confidence=float(confidence),
                        bbox=model_serving_pb2.BoundingBox(
                            x_min=0.0, y_min=0.0,
                            x_max=1.0, y_max=1.0
                        )
                    )
                    detections.append(detection)
            
            # 5. Calculer le temps d'inférence
            inference_time = (time.time() - start_time) * 1000  # en ms
            
            # 6. Logger pour l'audit
            logger.info(f"📊 Prédiction - ID: {request.request_id}, "
                       f"User: {request.user_id}, "
                       f"Time: {inference_time:.2f}ms")
            
            # 7. Retourner la réponse
            return model_serving_pb2.PredictionResponse(
                request_id=request.request_id,
                detections=detections,
                model_version="v1.0.0",
                inference_time_ms=inference_time
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_serving_pb2.PredictionResponse()
    
    def HealthCheck(self, request, context):
        return model_serving_pb2.HealthCheckResponse(status="healthy")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_serving_pb2_grpc.add_ModelServingServicer_to_server(
        ModelServingService(model_path="/app/models/car_detector_v1.h5"),
        server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("🚀 Serveur gRPC démarré sur le port 50051")
    server.wait_for_termination()
```

#### Client gRPC pour tester

```python
import grpc
import base64
import model_serving_pb2
import model_serving_pb2_grpc

def test_grpc_prediction(image_path):
    # Créer un canal gRPC
    channel = grpc.insecure_channel('localhost:50051')
    stub = model_serving_pb2_grpc.ModelServingStub(channel)
    
    # Lire et encoder l'image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Créer la requête
    request = model_serving_pb2.PredictionRequest(
        request_id="test-001",
        image_data=image_data,
        user_id="user-123"
    )
    
    # Appeler le service
    response = stub.Predict(request)
    
    # Afficher les résultats
    print(f"Request ID: {response.request_id}")
    print(f"Model Version: {response.model_version}")
    print(f"Inference Time: {response.inference_time_ms:.2f}ms")
    print("Detections:")
    for detection in response.detections:
        print(f"  - {detection.class_name}: {detection.confidence:.2%}")

if __name__ == "__main__":
    test_grpc_prediction("test_image.jpg")
```

### 2. Service de Features (REST)

#### Implémentation complète

```python
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

def preprocess_image(image_bytes, target_size=(224, 224)):
    """Prétraite une image pour l'inférence"""
    # Décoder l'image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionner
    image = image.resize(target_size)
    
    # Convertir en bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    preprocessed_bytes = buffer.getvalue()
    
    # Encoder en base64
    encoded = base64.b64encode(preprocessed_bytes).decode('utf-8')
    
    return encoded, image

def extract_features(image):
    """Extrait des features basiques de l'image"""
    image_array = np.array(image)
    
    features = {
        'width': image.width,
        'height': image.height,
        'format': image.format or 'JPEG',
        'mode': image.mode,
        'mean_brightness': float(np.mean(image_array)),
        'std_brightness': float(np.std(image_array)),
        'min_pixel': int(np.min(image_array)),
        'max_pixel': int(np.max(image_array))
    }
    
    return features

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "feature-service",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Image data required"}), 400
        
        request_id = data.get('request_id', 'unknown')
        
        # Décoder l'image
        image_bytes = base64.b64decode(data['image'])
        
        # Prétraiter
        preprocessed_image, image = preprocess_image(image_bytes)
        
        # Extraire les features
        features = extract_features(image)
        
        # Logger
        logger.info(f"✅ Image prétraitée - ID: {request_id}, "
                   f"Size: {features['width']}x{features['height']}")
        
        return jsonify({
            "preprocessed_image": preprocessed_image,
            "features": features,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur de prétraitement: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
```

#### Test du Feature Service

```bash
# Test avec curl
curl -X POST http://localhost:5001/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -i test_image.jpg)'",
    "request_id": "test-001"
  }'
```

```python
# Test avec Python
import requests
import base64

def test_feature_service(image_path):
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    response = requests.post(
        'http://localhost:5001/preprocess',
        json={
            'image': image_data,
            'request_id': 'test-001'
        }
    )
    
    print(response.json())

test_feature_service('test_image.jpg')
```

### 3. Service de Results (REST)

#### Implémentation complète avec SQLite

```python
from flask import Flask, request, jsonify
from datetime import datetime
import json
import sqlite3
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv('DB_PATH', '/app/data/predictions.db')

def get_db_connection():
    """Crée une connexion à la base de données"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
    return conn

def init_db():
    """Initialise la base de données"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            predictions TEXT NOT NULL,
            model_version TEXT NOT NULL,
            inference_time_ms REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    # Index pour les recherches par user_id
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_id ON predictions(user_id)
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✅ Base de données initialisée")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "results-service",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/predictions', methods=['POST'])
def store_prediction():
    try:
        data = request.get_json()
        
        # Validation
        required_fields = ['request_id', 'user_id', 'predictions', 
                          'model_version', 'inference_time_ms']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Connexion à la DB
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insertion
        cursor.execute('''
            INSERT INTO predictions 
            (request_id, user_id, predictions, model_version, inference_time_ms, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['request_id'],
            data['user_id'],
            json.dumps(data['predictions']),
            data['model_version'],
            data['inference_time_ms'],
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Prédiction stockée - ID: {data['request_id']}, "
                   f"User: {data['user_id']}")
        
        return jsonify({
            "status": "success",
            "request_id": data['request_id']
        }), 201
        
    except sqlite3.IntegrityError:
        return jsonify({"error": "Request ID already exists"}), 409
    except Exception as e:
        logger.error(f"❌ Erreur de stockage: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predictions/<request_id>', methods=['GET'])
def get_prediction(request_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions WHERE request_id = ?
        ''', (request_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return jsonify({"error": "Prediction not found"}), 404
        
        return jsonify({
            "request_id": row['request_id'],
            "user_id": row['user_id'],
            "predictions": json.loads(row['predictions']),
            "model_version": row['model_version'],
            "inference_time_ms": row['inference_time_ms'],
            "timestamp": row['timestamp']
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur de récupération: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predictions/user/<user_id>', methods=['GET'])
def get_user_predictions(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append({
                "request_id": row['request_id'],
                "predictions": json.loads(row['predictions']),
                "model_version": row['model_version'],
                "inference_time_ms": row['inference_time_ms'],
                "timestamp": row['timestamp']
            })
        
        logger.info(f"📊 Récupération de {len(predictions)} prédictions "
                   f"pour l'utilisateur: {user_id}")
        
        return jsonify({
            "user_id": user_id,
            "count": len(predictions),
            "predictions": predictions
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur de récupération: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predictions/user/<user_id>', methods=['DELETE'])
def delete_user_predictions(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Compter les enregistrements avant suppression
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,))
        count = cursor.fetchone()[0]
        
        # Supprimer
        cursor.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"🗑️ {count} prédictions supprimées pour l'utilisateur: {user_id} "
                   f"(Droit à l'oubli RGPD)")
        
        return jsonify({
            "status": "success",
            "user_id": user_id,
            "deleted_count": count
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur de suppression: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5002, debug=False)
```

### 4. Docker Compose - Configuration complète

```yaml
version: '3.8'

services:
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
    healthcheck:
      test: ["CMD", "python", "-c", "import grpc; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3

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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

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
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5002/health"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  ia-network:
    driver: bridge
```

### 5. Commandes de test

```bash
# Construire et démarrer
docker-compose up --build -d

# Vérifier les logs
docker-compose logs -f

# Tester les health checks
curl http://localhost:5001/health
curl http://localhost:5002/health

# Tester le feature service
curl -X POST http://localhost:5001/preprocess \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image", "request_id": "test-001"}'

# Tester le results service
curl -X POST http://localhost:5002/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "user_id": "user-123",
    "predictions": [{"class": "car", "confidence": 0.95}],
    "model_version": "v1",
    "inference_time_ms": 45.2
  }'

# Récupérer une prédiction
curl http://localhost:5002/predictions/test-001

# Arrêter les services
docker-compose down
```

## Problèmes courants et solutions

### Problème 1 : Erreur de build Docker

```bash
# Vérifier les logs de build
docker-compose build --no-cache

# Vérifier l'espace disque
docker system df

# Nettoyer les images inutilisées
docker system prune -a
```

### Problème 2 : Service ne démarre pas

```bash
# Vérifier les logs
docker-compose logs [service-name]

# Vérifier les ports
netstat -an | grep LISTEN

# Redémarrer un service spécifique
docker-compose restart [service-name]
```

### Problème 3 : Communication entre services

```bash
# Tester la connectivité réseau
docker-compose exec feature-service ping model-serving

# Vérifier le réseau Docker
docker network ls
docker network inspect [network-name]
```

## Checklist Partie 2

- [ ] Service Model Serving implémenté et fonctionnel
- [ ] Service Feature Service implémenté et fonctionnel
- [ ] Service Results Service implémenté et fonctionnel
- [ ] Dockerfiles créés pour chaque service
- [ ] docker-compose.yml configuré
- [ ] Tous les services démarrent sans erreur
- [ ] Health checks fonctionnels
- [ ] Communication entre services testée
- [ ] Logs visibles et informatifs

## Temps estimé

- Implémentation Model Serving : 15 min
- Implémentation Feature Service : 10 min
- Implémentation Results Service : 15 min
- Configuration Docker : 5 min
