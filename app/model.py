"""
Modele de classification de tumeurs cerebrales par CNN
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from app.utils import IMG_SIZE, CLASSES

class BrainTumorClassifier:
    """Classe pour gerer le modele de classification"""
    
    def __init__(self, img_size=IMG_SIZE, num_classes=4):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = CLASSES
        
    def build_model(self, architecture='mobilenet'):
        """
        Construit l'architecture du modele CNN
        
        Args:
            architecture: 'custom' ou 'mobilenet'
        """
        if architecture == 'custom':
            # TODO: Creer une architecture CNN personnalisee
            # Indice: Sequential, Conv2D, MaxPooling2D, Dropout, Dense
            
            self.model = models.Sequential([
                layers.Input(shape=(self.img_size, self.img_size, 3)),
                
                # TODO: Ajouter des couches de convolution + pooling
                # Bloc 1
                # layers.Conv2D(32, (3,3), activation='relu', padding='same'),
                # layers.MaxPooling2D((2,2)),
                # layers.Dropout(0.25),
                
                # Bloc 2
                # layers.Conv2D(64, (3,3), activation='relu', padding='same'),
                # layers.MaxPooling2D((2,2)),
                # layers.Dropout(0.25),
                
                # Bloc 3
                # layers.Conv2D(128, (3,3), activation='relu', padding='same'),
                # layers.MaxPooling2D((2,2)),
                # layers.Dropout(0.25),
                
                # TODO: Flatten et couches denses
                # layers.Flatten(),
                # layers.Dense(128, activation='relu'),
                # layers.Dropout(0.5),
                # layers.Dense(self.num_classes, activation='softmax')
            ])
            
        elif architecture == 'mobilenet':
            # TODO: Utiliser MobileNetV2 pre-entraine (Transfer Learning)
            
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False  # Geler les poids pre-entraines
            
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        print("Architecture du modele construite")
        self.model.summary()
        
    def compile_model(self, learning_rate=0.001):
        """
        Compile le modele
        
        Args:
            learning_rate: taux d'apprentissage
        """
        # TODO: Compiler le modele
        # Indice: optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
        
        # self.model.compile(
        #     optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        #     loss='categorical_crossentropy',
        #     metrics=['accuracy']
        # )
        
        pass
    
    def prepare_data_generators(self, train_dir, test_dir, batch_size=32):
        """
        Prepare les generateurs de donnees avec augmentation
        
        Args:
            train_dir: dossier d'entrainement
            test_dir: dossier de test
            batch_size: taille des batchs
        
        Returns:
            train_generator, test_generator
        """
        # TODO: Creer des ImageDataGenerator avec augmentation
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            # TODO: Ajouter des augmentations
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # horizontal_flip=True,
            # zoom_range=0.2,
            # fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # TODO: Creer les generateurs depuis les dossiers
        # train_generator = train_datagen.flow_from_directory(
        #     train_dir,
        #     target_size=(self.img_size, self.img_size),
        #     batch_size=batch_size,
        #     class_mode='categorical'
        # )
        
        train_generator = None
        test_generator = None
        
        return train_generator, test_generator
    
    def train(self, train_generator, val_generator, epochs=30):
        """
        Entraine le modele
        
        Args:
            train_generator: generateur de donnees d'entrainement
            val_generator: generateur de donnees de validation
            epochs: nombre d'epoques
        
        Returns:
            history: historique d'entrainement
        """
        # TODO: Definir les callbacks
        callbacks = [
            # EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            # ModelCheckpoint('models/best_model.h5', save_best_only=True),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        # TODO: Entrainer le modele
        # self.history = self.model.fit(
        #     train_generator,
        #     epochs=epochs,
        #     validation_data=val_generator,
        #     callbacks=callbacks
        # )
        
        print("Entrainement termine!")
        return self.history
    
    def evaluate(self, test_generator):
        """
        Evalue le modele sur le jeu de test
        
        Args:
            test_generator: generateur de donnees de test
        
        Returns:
            dict: metriques (loss, accuracy)
        """
        # TODO: Evaluer le modele
        # loss, accuracy = self.model.evaluate(test_generator)
        
        loss, accuracy = 0, 0
        
        print(f"\nPerformances sur le jeu de test:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def save_model(self, filepath='models/brain_tumor_classifier.h5'):
        """
        Sauvegarde le modele entraine
        
        Args:
            filepath: chemin de sauvegarde
        """
        # TODO: Sauvegarder le modele
        # os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # self.model.save(filepath)
        
        pass
    
    def load_model(self, filepath='models/brain_tumor_classifier.h5'):
        """
        Charge un modele pre-entraine
        
        Args:
            filepath: chemin du modele
        """
        # TODO: Charger le modele
        # self.model = keras.models.load_model(filepath)
        
        pass
    
    def predict(self, image):
        """
        Fait une prediction sur une image
        
        Args:
            image: image pretraitee (numpy array)
        
        Returns:
            dict: {class_name, confidence, probabilities}
        """
        # TODO: Faire la prediction
        # predictions = self.model.predict(image)
        # class_index = np.argmax(predictions[0])
        # confidence = np.max(predictions[0])
        # class_name = self.class_names[class_index]
        
        # return {
        #     'class_name': class_name,
        #     'confidence': float(confidence),
        #     'probabilities': {
        #         self.class_names[i]: float(predictions[0][i])
        #         for i in range(len(self.class_names))
        #     }
        # }
        
        pass

def train_and_save_model():
    """Script principal pour entrainer et sauvegarder le modele"""
    print("Demarrage de l'entrainement du modele CNN")
    print("=" * 60)
    
    # TODO: Creer une instance du modele
    # classifier = BrainTumorClassifier()
    
    # TODO: Construire l'architecture
    # classifier.build_model(architecture='mobilenet')
    
    # TODO: Compiler le modele
    # classifier.compile_model()
    
    # TODO: Preparer les donnees
    # train_gen, val_gen = classifier.prepare_data_generators(
    #     'dataset/Training',
    #     'dataset/Testing'
    # )
    
    # TODO: Entrainer
    # classifier.train(train_gen, val_gen, epochs=30)
    
    # TODO: Evaluer
    # classifier.evaluate(val_gen)
    
    # TODO: Sauvegarder
    # classifier.save_model()
    
    pass

if __name__ == "__main__":
    train_and_save_model()
