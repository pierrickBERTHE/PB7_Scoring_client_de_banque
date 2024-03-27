"""
Description: Ce fichier contient l'API Flask pour le projet 7 de ma formation # Openclassrooms.

Author: Pierrick Berthe
Date: 2024-03-21
"""

# ============== étape 1 : Importation des librairies ====================

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import shap
import mlflow.pyfunc
import git

# ====================== étape 2 : Généralités ============================

# Création de l'application Flask
app = Flask(__name__)
app.config["DEBUG"] = True

# Choix du répertoire racine (local ou distant)
environment = os.getenv('ENVIRONMENT', 'distant')

if environment == 'local':
    dirname = "C:\\Users\\pierr\\VSC_Projects\\Projet7_OCR_DataScientist"
else:
    dirname = "/home/pierrickberthe/mysite"

print(f'dirname: {dirname}')

# Chargement du modèle pré-entraîné
MODEL_PATH = os.path.join(dirname, "mlflow_model", "model.pkl")
model = joblib.load(MODEL_PATH)

# ====== étape 3 : Wrapper pour prediction avec seuil personalisé ===========

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Enveloppe personnalisée pour un modèle de machine learning, avec un seuil
    de prédiction personnalisé.

    Attributs:
    model (object): Le modèle de machine learning à envelopper.
    threshold (float): Le seuil de prédiction. Par défaut à 0.5.
    """
    def __init__(self, model, threshold=0.5):
        """
        Initialise l'enveloppe du modèle avec le modèle et le seuil donnés.

        Paramètres:
        model (object): Le modèle de machine learning à envelopper.
        threshold (float): Le seuil de prédiction. Par défaut à 0.5.
        """
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input):
        """
        Prédit les classes des échantillons en utilisant le seuil personnalisé.

        Paramètres:
        context (object): Le contexte de la prédiction. Non utilisé dans
        cette méthode.
        model_input (DataFrame): Les échantillons à prédire.

        Retourne:
        array: Les prédictions de classe pour les échantillons.
        """
        probabilities = self.model.predict_proba(model_input)
        predictions = (probabilities[:, 1] >= self.threshold).astype(int)
        return predictions

    def predict_proba(self, model_input, context=None):
        """
        Prédit les probabilités des classes pour les échantillons.

        Paramètres:
        model_input (DataFrame): Les échantillons à prédire.
        context (object, optionnel): Le contexte de la prédiction. Non
        utilisé dans cette méthode.

        Retourne:
        array: Les probabilités de classe pour les échantillons.
        """
        return self.model.predict_proba(model_input)

# ====================== étape 4 : Fonctions ============================

def get_prediction(df, seuil_predict=0.08):
    """
    Prédit la classe de l'instance en utilisant le modèle pré-entraîné.
    """
    # Créer un wrapper pour le modèle avec un seuil personnalisé
    wrapper = CustomModelWrapper(model, threshold=seuil_predict)

    # Prédire la classe de l'instance
    prediction = wrapper.predict(None, df)[0]

    # Prédire la probabilité de la classe 1
    prediction_proba = wrapper.predict_proba(df)[0]

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

# ======================== étape 5 : Routes ==========================

@app.route('/', methods=['GET'])
def home():
    """
    Retourne la page d'accueil de l'API.
    """
    description = (
        f'''<h1>Bienvenue sur l'API de Pierrick BERTHE</h1>
        <p>Cette API est dédiée au projet 7 de ma formation Openclassrooms</p>
        <p>Chemins :</p>
        <p>dirname: {dirname}</p>'''
    )
    return description


@app.route('/health', methods=['GET'])
def health():
    """
    Retourne un message indiquant que le serveur est opérationnel.
    """
    return jsonify({
        'status': 'API fonctionnelle',
        'webhooks': 'normalement configurés + MAJ auto avec post-merge hook',
        })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit la sortie en fonction des données d'entrée en utilisant le modèle
    pré-entraîné. Les données d'entrée sont reçues au format JSON.
    """

    # Récupération des données au format JSON
    data = request.get_json()

    # Vérification de la présence de données (erreur 415 si non présentes)
    if data is None:
        return jsonify(
            {'error': 'Bad Request', 'message': 'No input data provided'}
        ), 415

    try:
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
                'probabilité': round((proba_class_1 * 100), 2).tolist()
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

    # Gestion des erreurs (erreur 500 si erreur interne)
    except Exception as e:
        return jsonify(
            {'error': 'Internal Server Error', 'message': str(e)}
        ), 500


@app.route('/git_update', methods=['POST'])
def git_update():
    """
    Mise à jour du dépôt git.
    """
    # ESSAI de MAJ dépôt git
    try:
        # Chemin du dépôt git
        GIT_PATH = os.path.join(dirname, "Projet7_OCR_DataScientist")

        # Récupération du dépôt git
        repo = git.Repo(GIT_PATH)

        # Mise à jour du dépôt
        origin = repo.remotes.origin

        # Création de la branche main si elle n'existe pas
        repo.create_head('main', origin.refs.main).set_tracking_branch(
            origin.refs.main
            ).checkout()
        
        # Pull des modifications
        origin.pull()

        return '', 200

    except git.GitCommandError as e:
        return str(e), 500


# =================== étape 6 : Run de l'API ==========================

# Exécution de l'application Flask si le script est exécuté directement
if __name__ == '__main__':
    app.run()
