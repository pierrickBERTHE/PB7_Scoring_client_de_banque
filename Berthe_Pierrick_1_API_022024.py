# Importation des bibliothèques
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import shap

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
    try:
        # Récupération des données au format JSON
        data = request.get_json()

        # Convertir les données en DataFrame Pandas
        df = pd.DataFrame(data)

        # Prédictions à l'aide du modèle pré-entraîné
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        proba_class_1 = prediction_proba[1]

        # Extraire le dernier estimateur du pipeline
        final_estimator = model[-1]

        # Expliquer les prédictions à l'aide de SHAP et son TreeExplainer
        explainer = shap.TreeExplainer(final_estimator)
        shap_values = explainer.shap_values(df)
        shap_values_class_1 = shap_values[1][0]

        # Convertir le tableau 1D en tableau 2D pour créer un DataFrame
        shap_values_class_1_2d = shap_values_class_1.reshape(1, -1)

        # DataFrame avec les valeurs SHAP et les noms de colonnes
        shap_values_df = pd.DataFrame(
            shap_values_class_1_2d,
            columns=df.columns
        )

        # Nombre de features à afficher
        nbr_feature=5

        # Calculer la valeur absolue des valeurs SHAP
        abs_shap_values = shap_values_df.abs()

        # Obtenir les noms des nbr_feature ayant les valeurs SHAP les plus élevées
        top_features = (
            abs_shap_values.sum()
            .sort_values(ascending=False)
            .head(nbr_feature)
            .index
        )

        # Sélectionner les top_features (arrondis)
        client_data_subset = df[top_features].round(2)
        shap_values_subset = shap_values_df[top_features].values.round(2)[0]

        # Préparer la réponse
        response = {
            'prediction': {
                'explainer' : explainer.expected_value[1],
                'prediction': prediction.tolist(),
                'probabilité': (proba_class_1 * 100).tolist()
            },
            'feature_importance': {
                'shap_values': shap_values_class_1_2d[0].tolist(),
                'feature_names': df.columns.tolist(),
                'feature_values': df.values.tolist()[0],
            },
            'feature_importance_locale': {
                'top_features': client_data_subset.columns.tolist(),
                'shap_values_subset' : shap_values_subset.tolist(),
                'client_data_subset': client_data_subset.values.tolist()[0]
            }
        }

        # Retour de la prédiction au format JSON
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

# Exécution de l'application Flask si le script est exécuté directement
if __name__ == '__main__':
    app.run()
