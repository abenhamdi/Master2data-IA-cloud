"""
Point d'entree de l'application de classification de tumeurs cerebrales
"""
from app.api import app
import os

if __name__ == "__main__":
    # Port pour le developpement local
    port = int(os.environ.get('PORT', 8000))
    
    print("Demarrage de l'application Brain Tumor Classifier")
    print(f"   Port: {port}")
    print(f"   URL: http://localhost:{port}")
    print("   Appuyez sur CTRL+C pour arreter\n")
    
    # Lancement de l'application
    app.run(debug=True, host='0.0.0.0', port=port)

