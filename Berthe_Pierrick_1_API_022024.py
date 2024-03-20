# ============== étape 1 : Importation des librairies ====================

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import shap

# ====================== étape 2 : Généralités ============================

# Création de l'application Flask
app = Flask(__name__)
app.config["DEBUG"] = True

# Récupération du chemin absolu et du répertoire du fichier api.py
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)

# Chargement du modèle pré-entraîné
MODEL_PATH = 'mlflow_model/model.pkl'
model = joblib.load(MODEL_PATH)

# ====================== étape 3 : Fonctions ============================

def get_prediction(df):
    """
    Prédit la classe de l'instance en utilisant le modèle pré-entraîné.
    """
    # Prédire la classe de l'instance
    prediction = model.predict(df)[0]

    # Prédire la probabilité de la classe 1
    prediction_proba = model.predict_proba(df)[0]

    # Sélectionner la probabilité de la classe 1
    proba_class_1 = prediction_proba[1]

    return prediction, proba_class_1


def get_shap_values(df, final_estimator):
    """
    Calcule les valeurs SHAP pour l'instance donnée.
    """
    # Créer un explainer SHAP pour le dernier estimateur du pipeline
    explainer = shap.TreeExplainer(final_estimator)

    # Calculer les valeurs SHAP pour l'instance donnée
    shap_values = explainer.shap_values(df)

    # Sélectionner les valeurs SHAP pour la classe 1
    shap_values_class_1 = shap_values[1][0]

    return explainer, shap_values_class_1


def get_top_features(df, shap_values_class_1_2d, nbr_feature=5):
    """
    Retourne les caractéristiques les plus importantes pour l'instance donnée.
    """
    # Créer un DataFrame à partir des valeurs SHAP
    shap_values_df = pd.DataFrame(
        shap_values_class_1_2d,
        columns=df.columns
    )

    # Calculer les valeurs absolues des valeurs SHAP
    abs_shap_values = shap_values_df.abs()

    # Sélectionner les caractéristiques les plus importantes
    top_features = (
        abs_shap_values.sum()
        .sort_values(ascending=False)
        .head(nbr_feature)
        .index
    )

    return top_features, shap_values_df

# ======================== étape 4 : Routes ==========================

@app.route('/', methods=['GET'])
def home():
    """
    Retourne la page d'accueil de l'API.
    """
    description = f'''<h1>Bienvenue sur l'API de Pierrick</h1>
                    <p>Cette API est dédiée au projet 7 de ma formation Openclassrooms</p>
                    <p>Chemins :</p>
                    <p>abspath: {abspath}</p>
                    <p>dirname: {dirname}</p>'''
    return description


@app.route('/health', methods=['GET'])
def health():
    """
    Retourne un message indiquant que le serveur est opérationnel.
    """
    return jsonify({'status': 'API fonctionnelle'})


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

        # Prédire la classe de l'instance
        prediction, proba_class_1 = get_prediction(df)

        # Extraire le dernier estimateur du pipeline
        final_estimator = model[-1]

        # Calculer les valeurs SHAP pour l'instance donnée
        explainer, shap_values_class_1 = get_shap_values(df, final_estimator)

        # Convertir le tableau 1D en tableau 2D pour créer un DataFrame
        shap_values_class_1_2d = shap_values_class_1.reshape(1, -1)

        # Extraire les caractéristiques les plus importantes
        top_features, shap_values_df = get_top_features(
            df,
            shap_values_class_1_2d
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

# =================== étape 5 : Run de l'API ==========================

# Exécution de l'application Flask si le script est exécuté directement
if __name__ == '__main__':
    app.run()
