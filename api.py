"""
Description: Ce fichier contient l'API Flask pour le projet 7 de ma formation
Openclassrooms.

Author: Pierrick Berthe
Date: 2024-04-04

URL de l'API : https://api-pb-e85d72620dec.herokuapp.com/

"""
# ============== étape 1 : Importation des librairies ====================

from unittest import result
import flask
from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
import os
import shap
import mlflow.pyfunc
import zipfile
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np
matplotlib.use('Agg')

# Versions
print("\nVersion des librairies utilisees :")
print("Python        : " + sys.version)
print("Flask         : " + flask.__version__)
print("io            : No module version")
print("Joblib        : " + joblib.__version__)
print("Matplotlib    : " + matplotlib.__version__)
print("MlFlow        : " + mlflow.__version__)
print("os            : No module version")
print("Pandas        : " + pd.__version__)
print("Shap          : " + shap.__version__)
print("zipfile       : No module version")
print("\n")


# ====================== étape 2 : Lancement API ============================

def create_app():
    """
    Crée une application Flask.
    """
    app = Flask(__name__)
    app.config["DEBUG"] = True
    return app

# Création de l'application Flask
app = create_app()
print('API Flask démarrée\n')
print("getcwd:",os.getcwd(), "\n")


# ====================== étape 3 : Fichier data ============================

# Nom du fichier de données
file_name = "application_train_cleaned_frac_1%"


# ====================== étape 4 : chemins ============================

def create_path(directory, filename):
    path = os.path.join(directory, filename)
    print(f"{filename} path: {path}\n")
    return path

# Chemin du modèle pré-entraîné
MODEL_PATH = create_path("mlflow_model_RF", "model.pkl")

# Chemin du fichier  contenant les données
DATA_PATH_ZIP = create_path("data_heroku", file_name + ".zip")
DATA_FILE_CSV = file_name + ".csv"


# ================== étape 5 : Chargement du modèle ========================

# Chargement du modèle pré-entraîné
model = joblib.load(MODEL_PATH)
print('Modèle chargé\n')


# ==================== étape 6 : chargement data ==========================

def load_data(file_name_zip, file_name_csv, _model):
    """
    Charge les données à partir d'un fichier CSV et les transforme en
    utilisant un modèle donné.
    """

    # Ouvrir le fichier zip en mode lecture ('r')
    with zipfile.ZipFile(file_name_zip, 'r') as z:
        with z.open(file_name_csv) as f:
            cols = pd.read_csv(f, nrows=0).columns

    # Supprimer la première colonne
    cols = cols[1:]

    # Lire le fichier CSV sans la première colonne
    data_df = pd.read_csv(file_name_zip, usecols=cols)

    # Isolement de la colonne SK_ID_CURR et TARGET
    sk_id_curr = data_df['SK_ID_CURR']
    target = data_df['TARGET']

    # Suppression des colonnes TARGET et SK_ID_CURR
    data_df_dropped = data_df.drop(columns=["SK_ID_CURR", "TARGET"])

    # Imputation des valeurs manquantes (preprocessing du modele)
    data_array = _model.named_steps['preprocess'].transform(data_df_dropped)

    # Création d'un DataFrame à partir du tableau numpy
    data_preprocessed_df = pd.DataFrame(
        data_array,
        columns=data_df_dropped.columns
    )

    # Re-insertion des colonne SK_ID_CURR et TARGET 
    data_preprocessed_df['SK_ID_CURR'] = sk_id_curr
    data_preprocessed_df['TARGET'] = target

    data_df_dropped['SK_ID_CURR'] = sk_id_curr
    data_df_dropped['TARGET'] = target

    return data_preprocessed_df, data_df_dropped


# Chargement des données
data, data_brutes = load_data(DATA_PATH_ZIP, DATA_FILE_CSV, model)
print('chargement des données terminé\n')


# ====== étape 7 : Wrapper pour prediction avec seuil personalisé ===========

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Enveloppe personnalisée pour un modèle de machine learning, avec un seuil
    de prédiction personnalisé.
    """
    def __init__(self, model, threshold=0.5):
        """
        Initialise l'enveloppe du modèle avec le modèle et le seuil donnés.
        """
        self.model = model
        self.threshold = threshold

    def predict(self, model_input):
        """
        Prédit les classes des échantillons en utilisant le seuil personnalisé.
        """
        probabilities = self.model.predict_proba(model_input)
        prediction = (probabilities[:, 1] >= self.threshold).astype(int)
        return prediction

    def predict_proba(self, model_input):
        """
        Prédit les probabilités des classes pour les échantillons.
        """
        probabilities = self.model.predict_proba(model_input)[0]
        return probabilities


# ====================== étape 8 : Fonctions ============================

def get_prediction(df, seuil_predict):
    """
    Prédit la classe de l'instance en utilisant le modèle pré-entraîné.
    """
    # Créer un wrapper pour le modèle avec un seuil personnalisé
    wrapper = CustomModelWrapper(model, threshold=seuil_predict)

    # Prédire la classe de l'instance
    prediction = wrapper.predict(df)[0]

    # Prédire la probabilité de la classe 1
    prediction_proba = wrapper.predict_proba(df)

    return prediction, prediction_proba


def get_shap_values(df, final_estimator):
    """
    Calcule les valeurs SHAP pour l'instance donnée.
    """
    # Créer un explainer SHAP pour le dernier estimateur du pipeline
    explainer = shap.TreeExplainer(final_estimator, n_jobs=1)

    # Calculer les valeurs SHAP pour l'instance donnée
    shap_values = explainer.shap_values(df)

    return explainer, shap_values


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


# ======================== étape 9 : Routes ==========================

@app.route('/', methods=['GET'])
def home():
    """
    Retourne la page d'accueil de l'API.
    """
    description = (
        f'''<h1>Bienvenue sur l'API de Pierrick BERTHE</h1>
        <p>Cette API est dédiée au projet 7 de ma formation Openclassrooms</p>
        <p>Chemins :</p>
        <p>getcwd: {os.getcwd()}</p>
        <p>MODEL_PATH: {MODEL_PATH}</p>
        <p>DATA_PATH_ZIP: {DATA_PATH_ZIP}</p>
        <p>DATA_FILE_CSV: {DATA_FILE_CSV}</p>'''
    )
    return description


@app.route('/health', methods=['GET'])
def health():
    """
    Retourne un message indiquant que le serveur est opérationnel.
    """
    return jsonify({'status': 'API fonctionnelle'})


@app.route('/client_selection', methods=['POST'])
def client_selection():
    """
    Retourne les ID des clients pour la sélection
    """
    sk_id_curr_all = data['SK_ID_CURR']
    return sk_id_curr_all.to_json(orient='records')


@app.route('/feature_selection', methods=['POST'])
def feature_selection():
    """
    Retourne les noms des features pour la sélection (sauf les feat TARGET et
    id des clients)
    """
    data_dropped = data.drop(columns=['SK_ID_CURR', 'TARGET'])
    feat_name_all = data_dropped.columns
    return feat_name_all.to_list()


@app.route('/client_extraction', methods=['POST'])
def client_data_extraction():
    """
    Retourne les données du client en fonction de l'ID du client
    """ 
    data_json = request.get_json()
    if 'client_id' not in data_json:
        return jsonify({'error': 'No client_id provided'}), 400

    client_id = data_json["client_id"]
    client_data = data[data['SK_ID_CURR'] == client_id]
    client_data_brutes = data_brutes[data_brutes['SK_ID_CURR'] == client_id]

    if client_data.empty:
        return jsonify({'error': 'No data found for this client_id'}), 404

    nan_client = int(client_data.isna().sum().sum())
    return jsonify({
        'client_data': client_data.to_json(orient='records'),
        'client_data_brutes': client_data_brutes.to_json(orient='records'),
        'nan_client': nan_client
    })


@app.route('/feat_extraction', methods=['POST'])
def feat_data_extraction():
    """
    Retourne les données de la feature en fonction du nom de la feature
    """ 
    data_json = request.get_json()
    if 'feature_name' not in data_json:
        return jsonify({'error': 'No feature_name provided'}), 400

    feature_name = data_json["feature_name"]
    feat_data_brutes = data_brutes[feature_name]

    return jsonify({
        'feat_data_brutes': feat_data_brutes.to_json(orient='records'),
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
        seuil_predict = 0.43
        prediction, prediction_proba = get_prediction(
            df,
            seuil_predict=seuil_predict
        )

        # Préparer la réponse
        response = {
            'prediction': prediction.tolist(),
            'proba_0': round((prediction_proba[0]), 2).tolist(),
            'proba_1': round((prediction_proba[1]), 2).tolist(),
            'seuil_predict': seuil_predict
    }
        # Retour de la prédiction au format JSON
        return jsonify(response)

    # Gestion des erreurs (erreur 500 si erreur interne)
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return jsonify(
            {'error': 'Internal Server Error', 'message': str(e)}
        ), 500


@app.route('/feature_importance_locale', methods=['POST'])
def feature_importance_locale():
    """
    Retourne l'image de feature importance locale.
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

        # Extraire le dernier estimateur du pipeline
        final_estimator = model[-1]

        # Calculer les valeurs SHAP pour l'instance donnée
        explainer, shap_values = get_shap_values(df, final_estimator)

        # Convertir le tableau 1D en tableau 2D pour créer un DataFrame
        shap_values_class_1_2d = shap_values[1].reshape(1, -1)

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
            'explainer' : explainer.expected_value.tolist(),
            'fi_locale_subset': {
                'top_features': client_data_subset.columns.tolist(),
                'shap_values_subset' : shap_values_subset.tolist(),
                'client_data_subset': client_data_subset.values.tolist()[0]
            }
        }

        # Retour de la prédiction au format JSON
        return jsonify(response)

    # Gestion des erreurs (erreur 500 si erreur interne)
    except Exception as e:
        print(f"Erreur lors de la feature importance locale: {e}")
        return jsonify(
            {'error': 'Internal Server Error', 'message': str(e)}
        ), 500


@app.route('/feature_importance_globale', methods=['POST'])
def feature_importance_globale():
    """
    Retourne l'image de feature importance globale.
    """
    # Vérification de la présence de données (erreur 400 si non présentes)
    if data is None:
        return jsonify(
            {'error': 'Bad Request', 'message': 'No input data provided'}
        ), 400

    # Vérification de la présence de modèle (erreur 400 si non présentes)
    if model is None:
        return jsonify(
            {'error': 'Bad Request', 'message': 'No model provided'}
        ), 400

    # Calculer les valeurs SHAP pour toutes les données
    explainer = shap.TreeExplainer(model[-1], n_jobs=1)
    data_dropped = data.drop(columns=['SK_ID_CURR', 'TARGET'])
    shap_values_all = explainer.shap_values(
        data_dropped,
        check_additivity=False
    )

    # Affichage feature importance globale
    shap.summary_plot(
        shap_values_all[1],
        data_dropped,
        plot_type='dot',
        show=False
    )

    # Récupérer la fig actuelle et sauvegarder en tant qu'image PNG
    fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format='png')

    # Fermer la figure et réinitialiser le buffer
    plt.close(fig)
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


@app.route('/feature_plot', methods=['POST'])
def feature_plot():
    """
    Retourne les données du client en fonction de l'ID du client
    """ 

    # Récupération des données au format JSON
    data_json = request.get_json()

    # Vérification de la présence de données (erreur 400 si non présentes)
    if 'client_id' not in data_json:
        return jsonify({'error': 'No client_id provided'}), 400
    
    if 'feat_to_display' not in data_json:
        return jsonify({'error': 'No feat_to_display provided'}), 400

    # Récupération des données
    client_id = data_json["client_id"]
    feat_to_display = data_json["feat_to_display"]

    # Filtrer les données pour le client donné pour la feature donnée
    client_data = (
        data_brutes[data_brutes['SK_ID_CURR'] == client_id][feat_to_display]
    )

    # Filtrer les données pour les clients par classe pour la feature donnée
    client_0_data = data_brutes[data_brutes['TARGET'] == 0][feat_to_display]
    client_1_data = data_brutes[data_brutes['TARGET'] == 1][feat_to_display]

    # Créer un dictionnaire pour stocker les données
    result = {
        'client_data': client_data.tolist(),
        'client_0_data': client_0_data.tolist(),
        'client_1_data': client_1_data.tolist()
    }

    return jsonify(result)


# =================== étape 10 : Run de l'API ==========================

# Exécution de l'application Flask si le script est exécuté directement
if __name__ == '__main__':
    try:
        app.run()
    except SystemExit as e:
        print(f"SystemExit exception: {e}")
        print("Le programme n'a pas pu démarrer le serveur Flask.")
