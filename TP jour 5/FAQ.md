# FAQ - Questions Fréquentes

## Questions Générales

### Q1 : Combien de temps dois-je consacrer à chaque partie ?

**R** : Le planning suggéré est :
- Partie 1 (Modèle) : 30 min
- Partie 2 (Microservices) : 45 min
- Partie 3 (Logging) : 30 min
- Partie 4 (Gouvernance) : 45 min
- Tests et documentation : 30 min

Mais adaptez selon vos difficultés. L'important est de comprendre les concepts.

### Q2 : Puis-je travailler en binôme ?

**R** : Oui, c'est même recommandé ! Le travail en équipe reflète la réalité professionnelle.

### Q3 : Que faire si je n'ai pas assez de ressources (RAM/CPU) ?

**R** : 
- Réduisez le nombre de replicas dans Kubernetes
- Utilisez un modèle plus léger (MobileNetV2 est déjà léger)
- Réduisez la taille du batch lors de l'entraînement
- Limitez le nombre de services démarrés simultanément

---

## Questions Techniques - Partie 1 (Modèle)

### Q4 : Je n'arrive pas à télécharger le dataset Kaggle, que faire ?

**R** : Plusieurs options :
1. Utilisez le dataset factice fourni dans `AIDE_PARTIE1.md`
2. Créez votre propre mini-dataset avec quelques images de voitures
3. Utilisez un autre dataset de détection d'objets (COCO, Pascal VOC)

### Q5 : Mon modèle n'apprend pas (accuracy reste faible), pourquoi ?

**R** : Vérifiez :
- Les données sont-elles normalisées (valeurs entre 0 et 1) ?
- Le learning rate n'est-il pas trop élevé ?
- Y a-t-il assez de données d'entraînement ?
- Les labels sont-ils corrects ?

```python
# Vérifications
print(f"Min pixel value: {X_train.min()}")  # Devrait être ~0
print(f"Max pixel value: {X_train.max()}")  # Devrait être ~1
print(f"Unique labels: {np.unique(y_train)}")  # Devrait être [0, 1, 2]
```

### Q6 : L'entraînement prend trop de temps, comment accélérer ?

**R** :
- Réduisez le nombre d'époques (5 au lieu de 10)
- Utilisez un subset des données pour tester
- Utilisez un modèle pré-entraîné sans fine-tuning
- Activez le GPU si disponible :

```python
# Vérifier si GPU disponible
import tensorflow as tf
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
```

---

## Questions Techniques - Partie 2 (Microservices)

### Q7 : Erreur "ModuleNotFoundError" lors du build Docker, que faire ?

**R** : Vérifiez que tous les packages sont dans `requirements.txt` :

```txt
# services/model_serving/requirements.txt
tensorflow==2.15.0
grpcio==1.60.0
grpcio-tools==1.60.0
numpy==1.24.3
pillow==10.2.0
```

Puis reconstruisez sans cache :
```bash
docker-compose build --no-cache
```

### Q8 : Le service gRPC ne démarre pas, erreur de port ?

**R** : Vérifiez qu'aucun autre processus n'utilise le port 50051 :

```bash
# Sur macOS/Linux
lsof -i :50051

# Sur Windows
netstat -ano | findstr :50051

# Tuer le processus si nécessaire
kill -9 [PID]
```

### Q9 : Comment tester la communication gRPC ?

**R** : Utilisez le client Python fourni dans `AIDE_PARTIE2.md` ou l'outil `grpcurl` :

```bash
# Installer grpcurl
brew install grpcurl  # macOS
# ou télécharger depuis https://github.com/fullstorydev/grpcurl

# Tester le health check
grpcurl -plaintext localhost:50051 modelserving.ModelServing/HealthCheck
```

### Q10 : Erreur "Connection refused" entre services Docker ?

**R** : Vérifiez :
1. Les services sont sur le même réseau Docker
2. Utilisez le nom du service (pas `localhost`) :
   ```python
   # ❌ Incorrect
   channel = grpc.insecure_channel('localhost:50051')
   
   # ✅ Correct (dans Docker)
   channel = grpc.insecure_channel('model-serving:50051')
   ```

### Q11 : La base de données SQLite est vide après redémarrage ?

**R** : Assurez-vous que le volume est bien monté :

```yaml
# docker-compose.yml
results-service:
  volumes:
    - ./data/results:/app/data  # Persiste les données
```

---

## Questions Techniques - Partie 3 (Logging)

### Q12 : Elasticsearch ne démarre pas (Out of Memory) ?

**R** : Réduisez la mémoire allouée :

```yaml
elasticsearch:
  environment:
    - "ES_JAVA_OPTS=-Xms256m -Xmx256m"  # Au lieu de 512m
```

### Q13 : Je ne vois pas les logs dans Kibana ?

**R** : Vérifiez :
1. Elasticsearch est accessible : `curl http://localhost:9200`
2. Logstash reçoit les logs : `docker-compose logs logstash`
3. Créez un index pattern dans Kibana :
   - Allez dans Management > Index Patterns
   - Créez un pattern `ia-logs-*`
   - Sélectionnez `@timestamp` comme champ temporel

### Q14 : Comment envoyer des logs vers Logstash depuis Python ?

**R** : Installez `python-logstash` :

```bash
pip install python-logstash
```

```python
import logging
import logstash

logger = logging.getLogger('my-service')
logger.setLevel(logging.INFO)

# Handler Logstash
logger.addHandler(logstash.TCPLogstashHandler(
    host='logstash',  # ou 'localhost' hors Docker
    port=5000,
    version=1
))

# Utilisation
logger.info('Message de log', extra={'custom_field': 'value'})
```

---

## Questions Techniques - Partie 4 (Gouvernance)

### Q15 : Qu'est-ce qu'une "base légale" pour le RGPD ?

**R** : C'est la justification légale pour traiter des données personnelles. Les principales bases sont :
- **Consentement** : La personne a donné son accord explicite
- **Contrat** : Nécessaire pour exécuter un contrat
- **Obligation légale** : Imposé par la loi
- **Intérêt légitime** : Intérêt légitime de l'entreprise (notre cas)

Pour notre système ADAS, la base légale est l'**intérêt légitime** (sécurité routière).

### Q16 : Comment anonymiser les données utilisateur ?

**R** : Plusieurs techniques :
1. **Pseudonymisation** : Remplacer l'identifiant par un hash
   ```python
   import hashlib
   user_id_hash = hashlib.sha256(user_id.encode()).hexdigest()
   ```

2. **Suppression** : Ne pas stocker l'identifiant
3. **Agrégation** : Stocker uniquement des statistiques agrégées

### Q17 : Que doit contenir un rapport d'audit ?

**R** : Un rapport d'audit complet doit inclure :
- **Conformité RGPD** : Vérification de toutes les exigences
- **Métriques de performance** : Temps d'inférence, disponibilité
- **Utilisation** : Nombre de prédictions, utilisateurs
- **Sécurité** : Incidents, tentatives d'accès non autorisées
- **Qualité du modèle** : Métriques ML, détection de drift
- **Recommandations** : Améliorations suggérées

### Q18 : Comment gérer le versioning des modèles ?

**R** : Utilisez un registre de modèles avec :
- **Identifiant unique** : Hash du fichier modèle
- **Version sémantique** : v1.0.0, v1.1.0, v2.0.0
- **Métadonnées** : Métriques, hyperparamètres, dataset
- **Statut** : active, deprecated, archived

```python
# Exemple de structure
{
  "model_id": "abc123",
  "name": "car_detector",
  "version": "1.0.0",
  "status": "active",
  "created_at": "2025-02-01T10:00:00Z",
  "metrics": {"accuracy": 0.92},
  "file_hash": "sha256:..."
}
```

---

## Questions Techniques - Partie 5 (Kubernetes - BONUS)

### Q19 : Minikube ne démarre pas, que faire ?

**R** : Essayez :
```bash
# Supprimer le cluster existant
minikube delete

# Redémarrer avec plus de ressources
minikube start --cpus=4 --memory=8192 --driver=docker

# Vérifier le statut
minikube status
```

### Q20 : Comment accéder aux services dans Kubernetes ?

**R** : Plusieurs options :

```bash
# Option 1 : Port-forward
kubectl port-forward svc/feature-service 5001:5001 -n ia-system

# Option 2 : Minikube service
minikube service feature-service -n ia-system

# Option 3 : Ingress (si configuré)
# Ajouter à /etc/hosts : 
# 127.0.0.1 ia-system.local
curl http://ia-system.local/api/features/health
```

### Q21 : Comment scaler un service dans Kubernetes ?

**R** :
```bash
# Scaler manuellement
kubectl scale deployment feature-service --replicas=5 -n ia-system

# Vérifier
kubectl get pods -n ia-system

# Auto-scaling (HPA)
kubectl autoscale deployment feature-service \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n ia-system
```

---

## Questions sur l'Évaluation

### Q22 : Quels sont les critères d'évaluation principaux ?

**R** : Les critères sont (voir énoncé pour détails) :
1. Architecture microservices (25 pts)
2. Implémentation technique (25 pts)
3. Logging et monitoring (15 pts)
4. Gouvernance IA (20 pts)
5. Conformité RGPD (10 pts)
6. Documentation (5 pts)
7. Bonus (10 pts)

### Q23 : Est-ce grave si je ne termine pas tout ?

**R** : Non ! L'important est :
- De comprendre les concepts
- D'avoir un système partiellement fonctionnel
- De documenter vos choix et difficultés
- De proposer des améliorations

Un système incomplet mais bien documenté vaut mieux qu'un système complet sans explication.

### Q24 : Dois-je rendre un rapport écrit ?

**R** : Oui, un README.md complet avec :
- Architecture du système
- Choix techniques justifiés
- Instructions de déploiement
- Difficultés rencontrées
- Améliorations possibles

---

## Questions sur les Bonnes Pratiques

### Q25 : Quelles sont les bonnes pratiques pour les microservices ?

**R** :
1. **Un service = une responsabilité** (Single Responsibility)
2. **Communication asynchrone** quand possible
3. **Idempotence** : Une requête peut être rejouée sans effet de bord
4. **Circuit breaker** : Gérer les pannes en cascade
5. **Health checks** : Toujours implémenter /health
6. **Logging structuré** : JSON avec contexte
7. **Versioning des API** : /v1/predict, /v2/predict

### Q26 : Comment gérer les secrets (API keys, passwords) ?

**R** : **JAMAIS** dans le code ou les Dockerfiles !

```bash
# ❌ MAUVAIS
ENV API_KEY=secret123

# ✅ BON - Variables d'environnement
docker run -e API_KEY=secret123 my-service

# ✅ MEILLEUR - Fichier .env (gitignored)
# .env
API_KEY=secret123

# docker-compose.yml
services:
  my-service:
    env_file: .env

# ✅ OPTIMAL - Secrets management (Kubernetes)
kubectl create secret generic api-keys --from-literal=api-key=secret123
```

### Q27 : Comment tester un système distribué ?

**R** : Plusieurs niveaux de tests :

1. **Tests unitaires** : Chaque fonction isolée
2. **Tests d'intégration** : Communication entre services
3. **Tests end-to-end** : Scénario complet utilisateur
4. **Tests de charge** : Performance sous charge
5. **Tests de chaos** : Résilience aux pannes

```python
# Exemple de test d'intégration
def test_prediction_pipeline():
    # 1. Prétraiter l'image
    response1 = requests.post('http://localhost:5001/preprocess', ...)
    assert response1.status_code == 200
    
    # 2. Faire la prédiction
    response2 = grpc_call_predict(...)
    assert len(response2.detections) > 0
    
    # 3. Vérifier le stockage
    response3 = requests.get(f'http://localhost:5002/predictions/{request_id}')
    assert response3.status_code == 200
```

---

## Ressources Complémentaires

### Documentation officielle
- **Docker** : https://docs.docker.com/
- **Kubernetes** : https://kubernetes.io/docs/
- **gRPC** : https://grpc.io/docs/languages/python/
- **Flask** : https://flask.palletsprojects.com/
- **TensorFlow** : https://www.tensorflow.org/guide
- **ELK Stack** : https://www.elastic.co/guide/

### Tutoriels recommandés
- **Microservices avec Python** : https://realpython.com/python-microservices-grpc/
- **Docker Compose** : https://docs.docker.com/compose/gettingstarted/
- **Kubernetes pour débutants** : https://kubernetes.io/docs/tutorials/kubernetes-basics/

### Outils utiles
- **Postman** : Tester les API REST
- **BloomRPC** : Tester les API gRPC (GUI)
- **k9s** : Interface CLI pour Kubernetes
- **Lens** : Interface graphique pour Kubernetes

---

## Contact et Support

### Pendant le TP
- Levez la main pour appeler le formateur
- Consultez les fichiers AIDE_*.md
- Collaborez avec vos camarades

### Après le TP
- Email : [email du formateur]
- Forum du cours : [lien]
- Office hours : [horaires]

---

**Dernière mise à jour** : 01/02/2025  
**Version** : 1.0
