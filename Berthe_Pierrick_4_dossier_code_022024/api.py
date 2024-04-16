"""
Description: Ce fichier contient l'API Flask pour le projet 7 de ma formation
Openclassrooms.

Author: Pierrick Berthe
Date: 2024-04-04

URL de l'API : http://pierrickberthe.eu.pythonanywhere.com/

"""

# ============== étape 1 : Importation des librairies ====================

from flask import Flask, request, jsonify, send_file
from matplotlib import backend_bases
import pandas as pd
import joblib
import os
import shap
import mlflow.pyfunc
import zipfile
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Afficher le nombre de coeurs de la machine
print("\n---Nombre CPU:----")
print(joblib.cpu_count())

# Afficher toutes les variables d'environnement
print("\n---Variables d'environnement:----")
for key, value in os.environ.items():
    print(f"{key}: {value}")
print("---FIN Variables d'environnement:----\n")

# Nombre de cœurs utilisés par joblib
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
print("LOKY_MAX_CPU_COUNT: ", os.environ['LOKY_MAX_CPU_COUNT'])

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
file_name = "application_train_cleaned_frac_10%"

# ====================== étape 3 : chemins ============================

def create_path(directory, filename):
    path = os.path.join(directory, filename)
    print(f"{filename} path: {path}\n")
    return path

# Chemin du modèle pré-entraîné
MODEL_PATH = create_path("mlflow_model", "model.pkl")

# Chemin du fichier  contenant les données
DATA_PATH_ZIP = create_path("data/cleaned", file_name + ".zip")
DATA_FILE_CSV = file_name + ".csv"

# ================== étape 3 : Chargement du modèle ========================

# Chargement du modèle pré-entraîné
model = joblib.load(MODEL_PATH)
print('Modèle chargé\n')

# ==================== étape 4 : chargement data ==========================

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

    # Isolement de la colonne SK_ID_CURR
    sk_id_curr = data_df['SK_ID_CURR']

    # Suppression des colonnes TARGET et SK_ID_CURR
    data_df_dropped = data_df.drop(columns=["TARGET", "SK_ID_CURR"])

    # Imputation des valeurs manquantes (preprocessing du modele)
    data_array = _model.named_steps['preprocess'].transform(data_df_dropped)

    # Création d'un DataFrame à partir du tableau numpy
    data_df_new = pd.DataFrame(data_array, columns=data_df_dropped.columns)

    # Re-insertion de la colonne SK_ID_CURR
    data_df_new['SK_ID_CURR'] = sk_id_curr

    return data_df_new


# Chargement des données
data = load_data(DATA_PATH_ZIP, DATA_FILE_CSV, model)
print('chargement des données terminé\n')

# ====== étape 5 : Wrapper pour prediction avec seuil personalisé ===========

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
        print("lancement méthode predict")
        probabilities = self.model.predict_proba(model_input)
        print("probabilities dans methode predict: ", probabilities)
        prediction = (probabilities[:, 1] >= self.threshold).astype(int)
        print("prediction: ", prediction)
        return prediction

    def predict_proba(self, model_input):
        """
        Prédit les probabilités des classes pour les échantillons.
        """
        print("lancement méthode predict_proba")
        probabilities = self.model.predict_proba(model_input)[0]
        print("probabilities dans methode predict_proba: ", probabilities)
        return probabilities

# ====================== étape 6 : Fonctions ============================

def get_prediction(df, seuil_predict=0.08):
    """
    Prédit la classe de l'instance en utilisant le modèle pré-entraîné.
    """
    print("Création du wrapper pour le modèle avec un seuil personnalisé")
    wrapper = CustomModelWrapper(model, threshold=seuil_predict)
    print("Wrapper créé")

    print("Prédiction de la classe de l'instance")
    print("Données d'entrée : ", df)
    prediction = wrapper.predict(df)
    print("Prédiction effectuée : ", prediction)

    # Prédire la probabilité de la classe 1
    print("Prédiction de la probabilité de la classe 1")
    # prediction_proba = wrapper.predict_proba(df)[0]
    proba_class_1 = wrapper.predict_proba(df)[1]
    print("Prédiction de la proba de la classe 1 effectuée : ", proba_class_1)

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
    # shap_values_class_1 = shap_values[1][0]
    shap_values_class_1 = shap_values[0]

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

# ======================== étape 7 : Routes ==========================

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

    if client_data.empty:
        return jsonify({'error': 'No data found for this client_id'}), 404

    nan_client = int(client_data.isna().sum().sum())
    return jsonify({
        'client_data': client_data.to_json(orient='records'),
        'nan_client': nan_client
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
        print("Conversion des données en DataFrame Pandas")
        df = pd.DataFrame(data)

        # Prédire la classe de l'instance
        print("prediction en cours")
        prediction, proba_class_1 = get_prediction(df)

        # Extraire le dernier estimateur du pipeline
        final_estimator = model[-1]

        # Calculer les valeurs SHAP pour l'instance donnée
        print("Calcul des valeurs SHAP")
        explainer, shap_values_class_1 = get_shap_values(df, final_estimator)

        # Convertir le tableau 1D en tableau 2D pour créer un DataFrame
        print("reshape des valeurs SHAP")
        shap_values_class_1_2d = shap_values_class_1.reshape(1, -1)

        # Extraire les caractéristiques les plus importantes
        print("Extraction des caractéristiques les plus importantes")
        top_features, shap_values_df = get_top_features(
            df,
            shap_values_class_1_2d
        )

        # Sélectionner les top_features (arrondis)
        print("Selection des top_features et des shap_values correspondants")
        client_data_subset = df[top_features].round(2)
        shap_values_subset = shap_values_df[top_features].values.round(2)[0]

        # Préparer la réponse
        print("Préparation de la réponse")
        response = {
            'prediction': {
                # 'explainer' : explainer.expected_value[1],
                'explainer' : explainer.expected_value,
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
        print(f"Erreur lors de la prédiction: {e}")
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
    explainer = shap.TreeExplainer(model[-1])
    data_dropped = data.drop(columns=['SK_ID_CURR'])
    shap_values_all = explainer.shap_values(
        data_dropped,
        check_additivity=False
    )

    # Affichage feature importance globale
    shap.summary_plot(
        shap_values_all[1],
        data.drop(columns=['SK_ID_CURR']),
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

# =================== étape 8 : Run de l'API ==========================

# Exécution de l'application Flask si le script est exécuté directement
if __name__ == '__main__':
    try:
        app.run(port=6000)
        # app.run()
    except SystemExit as e:
        print(f"SystemExit exception: {e}")
        print("Le programme n'a pas pu démarrer le serveur Flask.")
