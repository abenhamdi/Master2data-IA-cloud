"""
API Flask pour la classification de tumeurs cerebrales
"""
from flask import Flask, request, jsonify, render_template_string
import os
import numpy as np
from werkzeug.utils import secure_filename
from app.model import BrainTumorClassifier
from app.utils import preprocess_image_from_upload, get_class_description

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Initialisation de l'application Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Creer le dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chargement du modele
classifier = BrainTumorClassifier()
model_path = 'models/brain_tumor_classifier.h5'

try:
    classifier.load_model(model_path)
    print("Modele charge avec succes")
except Exception as e:
    print(f"Erreur lors du chargement du modele : {e}")

def allowed_file(filename):
    """Verifie si l'extension du fichier est autorisee"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Template HTML pour l'interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification de Tumeurs Cerebrales - IRM</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .upload-section {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-section:hover {
            background: #f8f9ff;
            border-color: #5568d3;
        }
        input[type="file"] {
            display: none;
        }
        .upload-button {
            background: #667eea;
            color: white;
            padding: 15px 30px;
            border-radius: 5px;
            cursor: pointer;
            display: inline-block;
            transition: background 0.3s;
        }
        .upload-button:hover {
            background: #5568d3;
        }
        #preview {
            margin: 20px 0;
            text-align: center;
        }
        #preview img {
            max-width: 400px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        #result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .result-success {
            background: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .result-error {
            background: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        .class-result {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .confidence {
            font-size: 18px;
            color: #666;
        }
        .probabilities {
            margin-top: 15px;
        }
        .prob-bar {
            margin: 10px 0;
        }
        .prob-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .prob-value {
            background: #e9ecef;
            height: 30px;
            border-radius: 5px;
            overflow: hidden;
        }
        .prob-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-weight: bold;
            transition: width 0.5s;
        }
        .disclaimer {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
        }
        .disclaimer-title {
            font-weight: bold;
            color: #856404;
            margin-bottom: 10px;
        }
        .info-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Classification de Tumeurs Cerebrales</h1>
        <p class="subtitle">Systeme d'aide au diagnostic par IRM</p>
        
        <div class="upload-section" onclick="document.getElementById('imageInput').click()">
            <div class="upload-button">Selectionner une IRM</div>
            <p style="margin-top: 15px; color: #666;">ou glissez-deposez une image ici</p>
            <p style="font-size: 12px; color: #999;">Formats acceptes: JPG, PNG (max 16MB)</p>
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="imageInput" name="image" accept="image/*" required>
        </form>
        
        <div id="preview"></div>
        <div id="result"></div>
        
        <div class="disclaimer">
            <div class="disclaimer-title">AVERTISSEMENT MEDICAL</div>
            <p>Ce systeme est un outil d'aide au diagnostic uniquement. Il ne remplace PAS l'expertise d'un medecin radiologue qualifie. Toute decision medicale doit etre prise par un professionnel de sante apres examen complet du patient.</p>
        </div>
        
        <div class="info-section">
            <h3>A propos de l'application</h3>
            <p>Cette application utilise un reseau de neurones convolutionnel (CNN) pour classifier les images IRM cerebrales en 4 categories:</p>
            <ul>
                <li><strong>Gliome:</strong> Tumeur gliale</li>
                <li><strong>Meningiome:</strong> Tumeur des meninges</li>
                <li><strong>Tumeur pituitaire:</strong> Tumeur de la glande pituitaire</li>
                <li><strong>Pas de tumeur:</strong> IRM normale</li>
            </ul>
        </div>
    </div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const uploadForm = document.getElementById('uploadForm');
        const preview = document.getElementById('preview');
        const result = document.getElementById('result');
        
        imageInput.addEventListener('change', handleFileSelect);
        
        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            // Previsualisation
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.innerHTML = '<img src="' + e.target.result + '" alt="IRM selectionnee">';
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            // Envoi automatique
            uploadImage(file);
        }
        
        async function uploadImage(file) {
            const formData = new FormData();
            formData.append('image', file);
            
            result.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResult(data);
                } else {
                    displayError(data.error || 'Erreur lors de la prediction');
                }
            } catch (error) {
                displayError('Erreur de connexion: ' + error.message);
            }
        }
        
        function displayResult(data) {
            const className = data.class_name;
            const confidence = (data.confidence * 100).toFixed(2);
            const probabilities = data.probabilities;
            
            let html = '<div class="result-success">';
            html += '<div class="class-result">Classe detectee: ' + className + '</div>';
            html += '<div class="confidence">Confiance: ' + confidence + '%</div>';
            html += '<div class="probabilities"><strong>Probabilites par classe:</strong>';
            
            for (const [cls, prob] of Object.entries(probabilities)) {
                const percentage = (prob * 100).toFixed(1);
                html += '<div class="prob-bar">';
                html += '<div class="prob-label">' + cls + '</div>';
                html += '<div class="prob-value">';
                html += '<div class="prob-fill" style="width: ' + percentage + '%">' + percentage + '%</div>';
                html += '</div></div>';
            }
            
            html += '</div></div>';
            
            result.innerHTML = html;
            result.style.display = 'block';
        }
        
        function displayError(message) {
            result.innerHTML = '<div class="result-error"><strong>Erreur:</strong> ' + message + '</div>';
            result.style.display = 'block';
        }
        
        // Drag and drop
        const uploadSection = document.querySelector('.upload-section');
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.style.background = '#f8f9ff';
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.style.background = '';
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.style.background = '';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                imageInput.files = files;
                handleFileSelect({target: {files: files}});
            }
        });
    </script>
</body>
</html>
"""

# TODO: Route GET "/" - Page d'accueil
@app.route('/')
def home():
    """Page d'accueil avec le formulaire d'upload"""
    # TODO: Retourner le template HTML
    # return render_template_string(HTML_TEMPLATE)
    pass

# TODO: Route POST "/predict" - Upload et prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour upload d'image et prediction
    
    Attendu: multipart/form-data avec 'image'
    Retourne: JSON avec {class_name, confidence, probabilities}
    """
    try:
        # TODO: Verifier qu'un fichier a ete envoye
        # if 'image' not in request.files:
        #     return jsonify({'error': 'No image file provided'}), 400
        
        # file = request.files['image']
        
        # TODO: Verifier que le fichier a un nom
        # if file.filename == '':
        #     return jsonify({'error': 'No selected file'}), 400
        
        # TODO: Verifier l'extension
        # if not allowed_file(file.filename):
        #     return jsonify({'error': 'Invalid file type. Allowed: JPG, PNG'}), 400
        
        # TODO: Pretraiter l'image
        # img_array = preprocess_image_from_upload(file)
        
        # TODO: Faire la prediction
        # prediction = classifier.predict(img_array)
        
        # TODO: Retourner le resultat
        # return jsonify(prediction), 200
        
        pass
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# TODO: Route GET "/health" - Health check
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # TODO: Verifier que le modele est charge
    # model_loaded = classifier.model is not None
    # return jsonify({
    #     'status': 'healthy' if model_loaded else 'degraded',
    #     'model_loaded': model_loaded
    # }), 200
    pass

# TODO: Route GET "/model/info" - Informations sur le modele
@app.route('/model/info', methods=['GET'])
def model_info():
    """Retourne des informations sur le modele"""
    # TODO: Retourner les infos du modele
    # return jsonify({
    #     'model_type': 'CNN (MobileNetV2)',
    #     'classes': classifier.class_names,
    #     'input_size': [classifier.img_size, classifier.img_size, 3]
    # }), 200
    pass

# TODO: Route GET "/classes" - Liste des classes
@app.route('/classes', methods=['GET'])
def get_classes():
    """Retourne la liste des classes avec descriptions"""
    # TODO: Retourner les classes et descriptions
    # classes_info = {
    #     cls: get_class_description(cls)
    #     for cls in classifier.class_names
    # }
    # return jsonify(classes_info), 200
    pass

# Gestion des erreurs
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

