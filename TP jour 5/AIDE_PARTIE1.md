# Aide - Partie 1 : Préparation des données et du modèle

## Indices pour l'implémentation

### 1. Chargement du dataset

```python
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(data_path, img_size=(224, 224)):
    """
    Indice : Structure du dataset
    
    data/
    ├── images/
    │   ├── car_001.jpg
    │   ├── truck_001.jpg
    │   └── bus_001.jpg
    └── classes.txt
    """
    
    images = []
    labels = []
    
    # Indice 1 : Lire le fichier classes.txt pour obtenir les noms des classes
    # with open(os.path.join(data_path, 'classes.txt'), 'r') as f:
    #     class_names = [line.strip() for line in f.readlines()]
    
    # Indice 2 : Parcourir le dossier images/
    # for filename in os.listdir(os.path.join(data_path, 'images')):
    #     - Charger l'image avec cv2.imread() ou PIL.Image.open()
    #     - Redimensionner à img_size
    #     - Normaliser les pixels (diviser par 255.0)
    #     - Extraire le label depuis le nom du fichier
    
    # Indice 3 : Convertir en numpy arrays
    # X = np.array(images)
    # y = np.array(labels)
    
    # Indice 4 : Split train/validation
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_val, y_val, class_names
```

### 2. Création du modèle

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def create_model(num_classes, input_shape=(224, 224, 3)):
    """
    Indice : Utiliser le transfer learning avec MobileNetV2
    """
    
    # Indice 1 : Charger MobileNetV2 pré-entraîné
    # base_model = MobileNetV2(
    #     input_shape=input_shape,
    #     include_top=False,  # Exclure la couche de classification
    #     weights='imagenet'   # Poids pré-entraînés
    # )
    
    # Indice 2 : Geler les couches du modèle de base
    # base_model.trainable = False
    
    # Indice 3 : Ajouter des couches personnalisées
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # predictions = Dense(num_classes, activation='softmax')(x)
    
    # Indice 4 : Créer le modèle final
    # model = Model(inputs=base_model.input, outputs=predictions)
    
    # Indice 5 : Compiler le modèle
    # model.compile(
    #     optimizer='adam',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )
    
    return model
```

### 3. Entraînement du modèle

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    """
    Indice : Utiliser des callbacks pour améliorer l'entraînement
    """
    
    # Indice 1 : Définir les callbacks
    # callbacks = [
    #     EarlyStopping(
    #         monitor='val_loss',
    #         patience=3,
    #         restore_best_weights=True
    #     ),
    #     ModelCheckpoint(
    #         'models/best_model.h5',
    #         monitor='val_accuracy',
    #         save_best_only=True
    #     )
    # ]
    
    # Indice 2 : Entraîner le modèle
    # history = model.fit(
    #     X_train, y_train,
    #     validation_data=(X_val, y_val),
    #     epochs=epochs,
    #     batch_size=32,
    #     callbacks=callbacks,
    #     verbose=1
    # )
    
    return history
```

### 4. Sauvegarde avec métadonnées

```python
import json
from datetime import datetime

def save_model_with_metadata(model, history, output_path="models/"):
    """
    Indice : Sauvegarder le modèle et ses métadonnées pour la gouvernance
    """
    
    os.makedirs(output_path, exist_ok=True)
    
    # Indice 1 : Sauvegarder le modèle
    # model_file = os.path.join(output_path, 'car_detector_v1.h5')
    # model.save(model_file)
    
    # Indice 2 : Extraire les métriques
    # final_metrics = {
    #     'train_accuracy': float(history.history['accuracy'][-1]),
    #     'val_accuracy': float(history.history['val_accuracy'][-1]),
    #     'train_loss': float(history.history['loss'][-1]),
    #     'val_loss': float(history.history['val_loss'][-1])
    # }
    
    # Indice 3 : Créer les métadonnées
    # metadata = {
    #     'model_name': 'car_detector',
    #     'version': '1.0.0',
    #     'created_at': datetime.utcnow().isoformat(),
    #     'framework': 'tensorflow',
    #     'architecture': 'MobileNetV2',
    #     'input_shape': [224, 224, 3],
    #     'num_classes': 3,
    #     'class_names': ['car', 'truck', 'bus'],
    #     'metrics': final_metrics,
    #     'hyperparameters': {
    #         'learning_rate': 0.001,
    #         'batch_size': 32,
    #         'epochs': len(history.history['loss']),
    #         'optimizer': 'adam'
    #     },
    #     'training_data': {
    #         'dataset': 'Car Object Detection - Kaggle',
    #         'num_train_samples': len(X_train),
    #         'num_val_samples': len(X_val)
    #     }
    # }
    
    # Indice 4 : Sauvegarder les métadonnées
    # metadata_file = os.path.join(output_path, 'model_metadata.json')
    # with open(metadata_file, 'w') as f:
    #     json.dump(metadata, f, indent=2)
    
    print(f"✅ Modèle sauvegardé: {model_file}")
    print(f"✅ Métadonnées sauvegardées: {metadata_file}")
```

## Exemple de dataset simplifié

Si vous avez des difficultés avec le dataset Kaggle, vous pouvez créer un dataset simplifié :

```python
def create_dummy_dataset(num_samples=1000):
    """
    Crée un dataset factice pour tester le code
    """
    import numpy as np
    
    # Générer des images aléatoires
    X = np.random.rand(num_samples, 224, 224, 3).astype(np.float32)
    
    # Générer des labels aléatoires (3 classes)
    y = np.random.randint(0, 3, size=num_samples)
    
    # Split train/val
    split_idx = int(0.8 * num_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    class_names = ['car', 'truck', 'bus']
    
    return X_train, y_train, X_val, y_val, class_names
```

## Vérifications importantes

### 1. Vérifier la forme des données

```python
print(f"X_train shape: {X_train.shape}")  # Devrait être (N, 224, 224, 3)
print(f"y_train shape: {y_train.shape}")  # Devrait être (N,)
print(f"Nombre de classes: {len(np.unique(y_train))}")  # Devrait être 3
```

### 2. Vérifier le modèle

```python
model.summary()  # Afficher l'architecture
print(f"Nombre de paramètres: {model.count_params()}")
```

### 3. Visualiser l'historique d'entraînement

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("✅ Graphique sauvegardé: models/training_history.png")
```

## Problèmes courants et solutions

### Problème 1 : Out of Memory (OOM)

**Solution** : Réduire la taille du batch ou utiliser un modèle plus léger

```python
# Au lieu de batch_size=32
history = model.fit(..., batch_size=16)

# Ou utiliser un générateur de données
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow(X_train, y_train, batch_size=16)
```

### Problème 2 : Modèle qui n'apprend pas

**Solution** : Vérifier que les données sont normalisées et que le learning rate n'est pas trop élevé

```python
# Normaliser les données
X_train = X_train / 255.0
X_val = X_val / 255.0

# Ajuster le learning rate
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Problème 3 : Overfitting

**Solution** : Ajouter du dropout et de la régularisation

```python
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
```

## Checklist Partie 1

- [ ] Dataset chargé correctement
- [ ] Données normalisées (valeurs entre 0 et 1)
- [ ] Split train/validation effectué
- [ ] Modèle créé avec MobileNetV2
- [ ] Modèle compilé avec les bonnes métriques
- [ ] Entraînement lancé avec callbacks
- [ ] Modèle sauvegardé (.h5)
- [ ] Métadonnées sauvegardées (JSON)
- [ ] Graphiques de performance générés
- [ ] Validation des performances (accuracy > 70%)

## Temps estimé

- Chargement et exploration du dataset : 10 min
- Création du modèle : 5 min
- Entraînement : 10 min (dépend du matériel)
- Sauvegarde et validation : 5 min
