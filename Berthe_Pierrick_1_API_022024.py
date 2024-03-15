# Importation des bibliothèques
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Création de l'application Flask
app = Flask(__name__)
app.config["DEBUG"] = True

# Récupération du chemin absolu et du répertoire du fichier api.py
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)

# Chargement du modèle pré-entraîné
MODEL_PATH = 'mlflow_model/model.pkl'
model = joblib.load(MODEL_PATH)

# Définition de la route de la page d'accueil
@app.route('/', methods=['GET'])
def home():
    description = f'''<h1>test de l'API de Pierrick</h1>
                    <p>abspath: {abspath}</p>
                    <p>dirname: {dirname}</p>'''
    return description

# Définition de la route de test de l'état de l'API
@app.route('/health', methods=['GET'])
def health():
    """
    Retourne un message indiquant que le serveur est opérationnel.
    """
    return jsonify({'status': 'ca maRche!'})

# Définition de la route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit la sortie en fonction des données d'entrée en utilisant le modèle
    pré-entraîné. Les données d'entrée sont reçues au format JSON.
    """
    print('request:')
    try:
        # Récupération des données au format JSON
        data = request.get_json()

        # Convertir les données en DataFrame Pandas
        df = pd.DataFrame(data)

        # Prédiction à l'aide du modèle pré-entraîné
        predictions = model.predict(df)

        # Retour de la prédiction au format JSON
        return jsonify({'prediction': str(predictions[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

# Exécution de l'application Flask si le script est exécuté directement
if __name__ == '__main__':
    app.run()
