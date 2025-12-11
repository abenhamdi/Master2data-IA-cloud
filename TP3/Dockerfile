# Dockerfile pour l'application de classification de tumeurs cerebrales
# TODO: Completez ce Dockerfile

FROM tensorflow/tensorflow:2.13.0-gpu  
# ou tensorflow:2.13.0-slim sans GPU

# TODO: Utiliser une image TensorFlow (plus lourde mais necessaire)
# FROM tensorflow/tensorflow:2.13.0
# OU pour version plus legere sans GPU:
# FROM python:3.9-slim

# TODO: Definir le repertoire de travail
# WORKDIR /app

# TODO: Copier le fichier requirements.txt
# COPY requirements.txt .

# TODO: Installer les dependances systeme si necessaire
# RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# TODO: Installer les dependances Python
# RUN pip install --no-cache-dir -r requirements.txt

# TODO: Copier le reste de l'application
# COPY . .

# TODO: Creer les dossiers necessaires
# RUN mkdir -p models uploads

# TODO: Exposer le port 8000
# EXPOSE 8000

# TODO: Definir les variables d'environnement
# ENV PORT=8000
# ENV PYTHONUNBUFFERED=1

# TODO: Commande de demarrage avec gunicorn
# Note: timeout augmente a 600s car l'inference CNN est plus lente
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "600", "--workers", "1", "app:app"]

