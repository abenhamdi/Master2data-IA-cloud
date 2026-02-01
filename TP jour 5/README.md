# TP Jour 5 - Architecture Microservices IA & Gouvernance
## Système de Détection d'Objets Automobiles

**Master 2 - Industrialisation IA dans le Cloud**  
**YNOV Campus Montpellier - 2025-2026**

---

## Documents disponibles

###  Énoncé principal
- **`ENONCE_TP_JOUR5.md`** : Énoncé complet du TP avec toutes les parties

###  Fichiers d'aide
- **`AIDE_PARTIE1.md`** : Indices pour l'entraînement du modèle
- **`AIDE_PARTIE2.md`** : Exemples de code pour les microservices
- **`FAQ.md`** : Questions fréquentes et solutions aux problèmes courants

---

##  Objectifs du TP

Ce TP vous permettra de :

1. Déployer un système IA en architecture microservices
2. Implémenter des communications REST et gRPC
3. Mettre en place un système de logging centralisé (ELK Stack)
4. Appliquer un cadre de gouvernance IA
5. Assurer la conformité RGPD

---

## Durée et organisation

- **Durée totale** : 3 heures
- **Travail** : Individuel ou en binôme
- **Modalités** : Présentiel avec support du formateur

### Planning suggéré

| Temps | Partie | Activité |
|-------|--------|----------|
| 0h00 - 0h30 | Partie 1 | Préparation des données et entraînement du modèle |
| 0h30 - 1h15 | Partie 2 | Conteneurisation des microservices (Docker) |
| 1h15 - 1h45 | Partie 3 | Logging et monitoring centralisés (ELK Stack) |
| 1h45 - 2h30 | Partie 4 | Gouvernance IA et conformité RGPD |
| 2h30 - 3h00 | Tests | Validation et documentation finale |


##  Prérequis

### Logiciels requis

```bash
# Vérifier les installations
docker --version          # >= 20.10
docker-compose --version  # >= 2.0
python --version          # >= 3.9
kubectl version --client  # Pour la partie bonus
```

### Bibliothèques Python

```bash
pip install tensorflow numpy pillow opencv-python scikit-learn
pip install flask grpcio grpcio-tools
pip install matplotlib  # Pour les graphiques
```

### Dataset

- **Source** : [Car Object Detection - Kaggle](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)
- **Alternative** : Dataset factice fourni dans `AIDE_PARTIE1.md`

---

## Architecture du système

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

### Services à implémenter

1. **Model Serving** (gRPC) : Service d'inférence du modèle
2. **Feature Service** (REST) : Prétraitement des images
3. **Results Service** (REST) : Stockage des résultats
4. **Logging** (ELK) : Centralisation des logs
5. **API Gateway** (Bonus) : Point d'entrée unique


## Livrables attendus

### Obligatoires

- [ ] **Code fonctionnel** :
  - Modèle entraîné et sauvegardé
  - Services microservices implémentés
  - Docker Compose opérationnel
  - Tests passants

- [ ] **Documentation de gouvernance** :
  - Checklist de gouvernance complétée
  - Registre des modèles avec métadonnées
  - Documentation RGPD

- [ ] **README.md** :
  - Architecture du système
  - Instructions de déploiement
  - Choix techniques justifiés

### Bonus

- [ ] **Rapport d'audit** :
  - Conformité RGPD
  - Métriques de performance
  - Recommandations

- [ ] **Déploiement Kubernetes** :
  - Manifestes K8s
  - Tests de scalabilité

---

## Démarrage rapide

### 1. Cloner/Créer la structure

```bash
mkdir tp-jour5 && cd tp-jour5
mkdir -p data/images models services/{model_serving,feature_service,results_service}
mkdir -p governance config/logstash k8s tests
```

### 2. Commencer par la Partie 1

```bash
# Télécharger le dataset (ou utiliser le dataset factice)
# Créer model/train_model.py
# Entraîner le modèle
python model/train_model.py
```

### 3. Passer à la Partie 2

```bash
# Implémenter les services
# Créer les Dockerfiles
# Tester localement
docker-compose up --build
```

### 4. Consulter les aides si besoin

- Bloqué sur le modèle ? → `AIDE_PARTIE1.md`
- Problème Docker ? → `AIDE_PARTIE2.md`
- Question générale ? → `FAQ.md`


### Pendant le TP

1. **Consultez les fichiers d'aide** :
   - `AIDE_PARTIE1.md` pour la partie modèle
   - `AIDE_PARTIE2.md` pour les microservices
   - `FAQ.md` pour les questions courantes

2. **Collaborez**

### Ressources en ligne

- **Docker** : https://docs.docker.com/
- **gRPC** : https://grpc.io/docs/languages/python/
- **Flask** : https://flask.palletsprojects.com/
- **TensorFlow** : https://www.tensorflow.org/guide
- **ELK Stack** : https://www.elastic.co/guide/


## Conseils

### Pour réussir le TP

1. **Lisez l'énoncé en entier** avant de commencer
2. **Testez régulièrement** : ne codez pas tout d'un coup
3. **Documentez au fur et à mesure** : commentaires, README
4. **Utilisez Git** : commitez régulièrement
5. **Demandez de l'aide** : ne restez pas bloqué

### Gestion du temps

- Ne passez pas trop de temps sur l'entraînement du modèle (30 min max)
- Privilégiez un système fonctionnel à un système parfait
- Le bonus Kubernetes est optionnel : concentrez-vous sur les parties principales

### Travail en équipe

- Répartissez les tâches : un sur le modèle, un sur les services
- Synchronisez-vous régulièrement
- Utilisez Git pour collaborer


## Compétences développées

À l'issue de ce TP, vous aurez acquis :

### Compétences techniques
- Déploiement de modèles ML en production
- Architecture microservices
- Communication inter-services (REST/gRPC)
- Conteneurisation avec Docker
- Orchestration avec Kubernetes (bonus)
- Logging et monitoring

### Compétences transversales
- Gouvernance IA
- Conformité RGPD
- Documentation technique
- Résolution de problèmes
- Travail en équipe



*N'oubliez pas : l'objectif est d'apprendre, pas de tout finir parfaitement.*  
*Un système partiellement fonctionnel mais bien compris vaut mieux qu'un système complet sans compréhension.*
