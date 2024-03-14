# Importation des bibliothèques
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import mlflow.pyfunc

# Création de l'application Flask
app = Flask(__name__)
app.config["DEBUG"] = True

# # Chargement du modèle pré-entraîné
# MODEL_PATH = 'best_model.joblib'
# model = joblib.load(MODEL_PATH)

# Chargement du modèle pré-entraîné depuis le registre de modèles MLflow
MODEL_NAME = "model_light_gbm_best "
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/production")


# Définition de la route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit la sortie en fonction des données d'entrée en utilisant le modèle
    pré-entraîné. Les données d'entrée sont reçues au format JSON.
    """
    try:
        # Récupération des données au format JSON
        data_json = request.get_json()

        # Transformation des données JSON en DataFrame pandas
        data = pd.DataFrame(...)  

        # Prédiction à l'aide du modèle pré-entraîné
        predictions = model.predict(data)

        # Retour de la prédiction au format JSON
        return jsonify({'prediction': str(predictions[0])})

    # En cas d'erreur, retourne l'erreur au format JSON
    except Exception as e:
        return jsonify({'error': str(e)})


# Exécution de l'application Flask si le script est exécuté directement
if __name__ == '__main__':
    app.run()
